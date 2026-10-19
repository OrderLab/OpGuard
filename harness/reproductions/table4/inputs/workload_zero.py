"""Train a small embedding model with the configured ZeRO optimizer."""
import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.elastic.agent.server.api as elastic_api
from torch.distributed.elastic.utils.distributed import get_socket_with_port

# Import-only compatibility with PyTorch 2.9; optimizer code is unmodified.
elastic_api._get_socket_with_port = get_socket_with_port
elastic_api.log = logging.getLogger("elastic")
import deepspeed


class DummyModule(torch.nn.Module):
    def __init__(self, n_layer=16, hidden_size=1024, vocab_size=1024, weighted=False):
        super().__init__()
        self._vocab_size = vocab_size
        self.weighted = weighted
        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, hidden_size) for _ in range(n_layer)])

    def forward(self, input):
        idx = torch.arange(self._vocab_size, device=input.device, dtype=torch.long)
        hidden_states = input
        for index, emb in enumerate(self.embs):
            activation = emb(idx)
            if self.weighted:
                activation = activation * ((index + 1) / len(self.embs))
            hidden_states = hidden_states + activation
        return hidden_states.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--overlap", type=int, default=1)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--delay-cycles", type=int, default=0)
    p.add_argument("--loss-scale", type=float, default=0)
    p.add_argument("--weighted", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.manual_seed(42 + rank)
    torch.use_deterministic_algorithms(True)
    model = DummyModule(weighted=a.weighted)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cfg = {"train_micro_batch_size_per_gpu": 6, "gradient_accumulation_steps": 1,
           "steps_per_print": 10000, "zero_allow_untested_optimizer": True,
           "fp16": {"enabled": True}, "gradient_clipping": 1.0,
           "zero_optimization": {"stage": a.stage, "allgather_partitions": True,
               "reduce_scatter": True, "allgather_bucket_size": 100000000,
               "reduce_bucket_size": 1048576, "overlap_comm": bool(a.overlap),
               "contiguous_gradients": True}}
    if a.loss_scale:
        cfg["fp16"]["loss_scale"] = a.loss_scale
    engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                                  config=cfg, dist_init_required=False)
    if a.delay_cycles:
        original = optimizer.average_tensor
        def average_tensor(tensor):
            if optimizer.overlap_comm:
                with torch.cuda.stream(optimizer.reduction_stream):
                    torch.cuda._sleep(a.delay_cycles)
            return original(tensor)
        optimizer.average_tensor = average_tensor
    losses, norms = [], []
    for step in range(a.steps):
        input = torch.rand(1, dtype=torch.float16, device="cuda")
        label = torch.rand(1, dtype=torch.float16, device="cuda")
        output = engine(input)
        loss = torch.nn.functional.mse_loss(output.float(), label.float())
        engine.backward(loss)
        engine.step()
        losses.append(loss.detach().cpu())
        norm = optimizer._global_grad_norm
        norms.append(float(norm) if norm is not None else None)
        print(json.dumps({"rank": rank, "step": step, "loss": losses[-1].item(),
                          "grad_norm": norms[-1]}), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save({"loss": torch.stack(losses), "grad_norm": norms,
                "weights": [p.detach().cpu() for p in engine.module.parameters()]},
               a.output / f"rank{rank}.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
