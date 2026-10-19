"""Actual historical QKV + attention kernels overwrite a live normalized input.

PR #3256 fixes exactly this workspace alias. The normalized input is consumed by
the parallel GPT-J MLP after attention, so it must remain intact.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.extension.resolve()))
    op = importlib.import_module(a.module)
    torch.manual_seed(1234)
    h, heads, seq = 256, 4, 7
    op.allocate_workspace_fp32(h, heads, seq, 1, 1, 1, False, 0, 32, 1)
    x = torch.randn(1, seq, h, device="cuda")
    weight = torch.randn(h, 3*h, device="cuda") / h**0.5
    gamma = torch.ones(h, device="cuda")
    beta = torch.zeros(h, device="cuda")
    bias = torch.zeros(3*h, device="cuda")
    qkv, normalized = op.qkv_gemm_fp32(x, weight, torch.empty(1), bias, gamma, beta,
                                      1e-5, False, 1, False, 1, 0, False, False)
    before = normalized.clone()
    attention = op.softmax_context_fp32(qkv, torch.empty(1, device="cuda"),
        0, False, False, heads, (h // heads)**-0.25, True, False, 256, True, 0, 1,
        torch.empty(1))[0].clone()
    after = normalized.clone()
    native = torch.nn.functional.layer_norm(x, (h,), gamma, beta, 1e-5)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"before": before.cpu(), "after": after.cpu(), "attention": attention.cpu(),
                "native": native.cpu()}, a.output)
    print(json.dumps({"normalized_input_preserved": torch.equal(before, after),
          "different": (before != after).sum().item(),
          "max_abs": (before-after).abs().max().item(),
          "before_native_max_abs": (before-native).abs().max().item()}))


if __name__ == "__main__":
    main()
