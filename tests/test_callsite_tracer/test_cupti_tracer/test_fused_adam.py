#!/usr/bin/env python3
"""
Mock training script that emits CUDA kernel events into /tmp/cupti_events.jsonl
via the opguard_cupti extension (CUPTI correlation ID bridge).

Build opguard_cupti first, then run:
    python mock_train.py --steps 50 --bsz 64
"""

# --- start CUPTI BEFORE any torch.cuda calls ---
# import opguard.callsite_tracer.cupti_tracer.opguard_cupti as opguard_cupti
# opguard_cupti.start("./cupti_events.jsonl")

# Everything below can freely use CUDA.
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# Try APEX fused optimizer to exercise multi-tensor apply kernels
try:
    from apex.optimizers import FusedAdam  # pip install apex (built with --cuda_ext)
    _HAS_APEX = True
except Exception:
    _HAS_APEX = False

def make_model():
    # Small MLP to keep things simple but non-trivial
    return nn.Sequential(
        nn.Linear(4096, 4096, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4096, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 1024, bias=False),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp", action="store_true", help="use torch.cuda.amp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mock] device = {device}")

    model = make_model().to(device)
    if _HAS_APEX:
        optimizer = FusedAdam(model.parameters(), lr=args.lr)
        print("[mock] using APEX FusedAdam")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print("[mock] using torch.optim.AdamW")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Dummy data
    x = torch.randn(args.bsz, 4096, device=device)
    target = torch.randn(args.bsz, 1024, device=device)

    model.train()
    t0 = time.time()
    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad()
            if args.amp:
                with torch.cuda.amp.autocast():
                    y = model(x)
                    loss = F.mse_loss(y, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y = model(x)
                loss = F.mse_loss(y, target)
                loss.backward()
                optimizer.step()

            if step % 5 == 0:
                torch.cuda.synchronize(device)  # make timings/kernels visible
                print(f"[mock] step {step:04d}  loss={loss.item():.4f}")
    finally:
        # Always stop CUPTI to flush the activity buffers
        # opguard_cupti.stop()
        dt = time.time() - t0
        print(f"[mock] done in {dt:.2f}s, CUPTI events at /tmp/cupti_events.jsonl")

if __name__ == "__main__":
    from opguard.callsite_tracer.trace import CallsiteTracer
    with CallsiteTracer(
        dump_path=os.path.join("./", "callsite_tracer.json"),
        dump_every=10,
    ):
        main()
