"""Compare one pinned dispatcher per process, retaining identical boundary inputs."""
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher, MoEAlltoAllTokenDispatcher,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatcher", choices=["allgather", "alltoall"], required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    parallel_state.initialize_model_parallel(expert_model_parallel_size=world)
    torch.manual_seed(1234 + rank)
    experts, hidden, tokens = 8, 256, 256
    config = TransformerConfig(num_layers=1, hidden_size=hidden, num_attention_heads=8,
        num_moe_experts=experts, expert_model_parallel_size=world,
        moe_router_topk=args.topk, add_bias_linear=False, moe_token_dispatcher_type="alltoall")
    local = experts // world
    local_indices = list(range(rank * local, (rank + 1) * local))
    x = torch.randn(tokens, 1, hidden, device="cuda").requires_grad_(True)
    scores = torch.randn(tokens, experts - 1, device="cuda")
    chosen = scores.topk(args.topk, dim=1).indices
    routing = torch.zeros(tokens, experts, device="cuda", dtype=torch.bool)
    routing.scatter_(1, chosen, True)
    probs = routing.float() / args.topk
    cls = MoEAllGatherTokenDispatcher if args.dispatcher == "allgather" else MoEAlltoAllTokenDispatcher
    dispatcher = cls(local, local_indices, config)
    output, counts = dispatcher.token_permutation(x, probs, routing)
    generator = torch.Generator(device="cuda").manual_seed(5678 + rank)
    dy = torch.randn(output.shape, generator=generator, device="cuda")
    dx, = torch.autograd.grad(output, x, dy)
    snapshot = {"input": x.detach().cpu(), "routing": routing.cpu(), "probs": probs.cpu(),
                "forward": output.detach().cpu(), "counts": counts.cpu(), "dy": dy.cpu(), "dx": dx.cpu()}
    Path("observed").mkdir(exist_ok=True)
    torch.save(snapshot, f"observed/rank{rank}.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
