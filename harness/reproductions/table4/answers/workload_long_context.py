"""DeepSpeed #2243: historical inference kernel with prompt > cache stride 128."""
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
    p.add_argument("--length", type=int, default=192)
    a = p.parse_args()
    sys.path.insert(0, str(a.extension.resolve()))
    op = importlib.import_module(a.module)
    torch.manual_seed(1234)
    h, heads, seq = 768, 12, a.length
    dtype = torch.float16
    x = torch.randn(1, seq, h, device="cuda", dtype=dtype)
    weight = torch.randn(h, 3*h, device="cuda", dtype=dtype) / h**0.5
    gamma = torch.ones(h, device="cuda", dtype=dtype)
    beta = torch.zeros(h, device="cuda", dtype=dtype)
    bias = torch.zeros(3*h, device="cuda", dtype=dtype)
    outputs, inputs = [], []
    for _ in range(3):
        qkv, _ = op.qkv_gemm_fp16(x, weight, bias, gamma, beta, 1e-5, False, 12)
        inputs.append(qkv.clone())
        attention = op.softmax_context_fp16(qkv, torch.empty(1, device="cuda", dtype=dtype),
            0, False, False, heads, (h // heads)**-0.25, True, False, 256, True, 0, 12,
            torch.empty(1))[0].clone()
        outputs.append(attention)
    torch.cuda.synchronize()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"inputs": [x.cpu() for x in inputs], "outputs": [x.cpu() for x in outputs]}, a.output)
    print(json.dumps({"input_equal_repeat": torch.equal(inputs[0], inputs[1]),
                      "output_equal_repeat": torch.equal(outputs[0], outputs[1]),
                      "different": (outputs[0] != outputs[1]).sum().item(),
                      "max_abs_repeat": (outputs[0]-outputs[1]).abs().max().item()}))


if __name__ == "__main__":
    main()
