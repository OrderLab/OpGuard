"""Pinned Megatron #1327 dispatchers: identical forward/output-grad, input-grad check."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher, MoEAlltoAllTokenDispatcher)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--exact-grad", action="store_true")
    a = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    parallel_state.initialize_model_parallel(expert_model_parallel_size=world)
    torch.manual_seed(1234 + rank)
    dtype = getattr(torch, a.dtype)
    experts, hidden, tokens = 8, 256, 256
    config = TransformerConfig(num_layers=1, hidden_size=hidden, num_attention_heads=8,
        num_moe_experts=experts, expert_model_parallel_size=world,
        moe_router_topk=a.topk, add_bias_linear=False, moe_token_dispatcher_type="alltoall")
    local = experts // world
    indices = list(range(rank * local, (rank + 1) * local))
    x = torch.randn(tokens, 1, hidden, dtype=dtype, device="cuda")
    # Routing to experts 0..6 leaves expert 7 empty on every rank.
    scores = torch.randn(tokens, experts-1, device="cuda")
    chosen = scores.topk(a.topk, dim=1).indices
    routing = torch.zeros(tokens, experts, device="cuda", dtype=torch.bool)
    routing.scatter_(1, chosen, True)
    probs = routing.to(dtype) / a.topk
    snapshots = []
    for cls in (MoEAllGatherTokenDispatcher, MoEAlltoAllTokenDispatcher):
        dispatcher = cls(local, indices, config)
        inputs = x.clone().requires_grad_(True)
        output, counts = dispatcher.token_permutation(inputs, probs, routing)
        # Same tensor at the semantic boundary for both dispatchers.
        gen = torch.Generator(device="cuda").manual_seed(5678 + rank)
        dy = torch.randn(output.shape, generator=gen, dtype=dtype, device="cuda")
        if a.exact_grad:
            dy = dy.round()
        dx, = torch.autograd.grad(output, inputs, dy)
        snapshots.append({"forward": output.detach().cpu(), "counts": counts.cpu(),
                          "dy": dy.cpu(), "dx": dx.cpu()})
        dist.barrier()
    left, right = snapshots
    result = {"rank": rank, "topk": a.topk, "dtype": a.dtype,
              "counts": left["counts"].tolist(),
              "forward_equal": torch.equal(left["forward"], right["forward"]),
              "output_grad_equal": torch.equal(left["dy"], right["dy"]),
              "input_grad_equal": torch.equal(left["dx"], right["dx"]),
              "input_grad_different": (left["dx"] != right["dx"]).sum().item(),
              "input_grad_max_abs": (left["dx"]-right["dx"]).abs().max().item()}
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save({"allgather": left, "alltoall": right}, a.output / f"rank{rank}.pt")
    (a.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
