#!/usr/bin/env python3
"""
Mock CUDA training (no time tweaks, real compute only).

- Requires: PyTorch with CUDA available.
- Does: A few hundred SGD steps on random data using a small MLP.
- Intent: Roughly ~15s on many GPUs, but duration is whatever the GPU delivers.

Run:
  python mock_cuda_train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this mock training (torch.cuda.is_available() == False).")
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    # Model: small-ish MLP (pure matmuls = good GPU utilization, no sleeps)
    input_dim   = 1024
    hidden_dim  = 2048
    num_classes = 1000

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_classes),
    ).to(device)

    # Synthetic data buffers (reused each step; filled in-place to avoid alloc churn)
    batch_size = 16384   # big batch to keep the GPU busy
    X = torch.empty(batch_size, input_dim, device=device, dtype=torch.float32)
    y = torch.empty(batch_size, device=device, dtype=torch.long)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    # Fixed number of real steps; no duration parameter, no sleeps.
    steps = 400

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Starting mock training on CUDA: {steps} steps, batch={batch_size}, dims={input_dim}->{hidden_dim}->{hidden_dim}->{num_classes}")

    model.train()
    for step in range(1, steps + 1):
        # Fill buffers with fresh random data/labels each step (real compute)
        X.normal_()
        y.random_(0, num_classes)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            # Print actual loss to prove we did real work
            # (no time/ETA reporting—per your ask)
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # Ensure kernels complete before exiting
    torch.cuda.synchronize()
    print("Training run finished on CUDA.")

if __name__ == "__main__":
    try:
        import os, threading
        print(f"Current pid/tid: {os.getpid()}, {threading.get_ident()}")
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
