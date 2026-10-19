"""Evaluate a native attention implementation on repeated inputs."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.extension.resolve()))
    op = importlib.import_module("ds_inference_091")
    torch.manual_seed(1234)
    hidden, heads = 256, 4
    op.allocate_workspace_fp32(hidden, heads, 7, 1, 1, 1, False, 0, 32, 1)
    prompt = torch.randn(1, 7, 3 * hidden, device="cuda")
    token = torch.randn(1, 1, 3 * hidden, device="cuda")
    mask = torch.empty(1, device="cuda")
    alibi = torch.empty(1)
    def attention(qkv):
        return op.softmax_context_fp32(qkv, mask, 0, False, False, heads,
                                       (hidden // heads) ** -0.25, True, False,
                                       256, True, 0, 1, alibi)[0].clone()
    attention(prompt)
    outputs = []
    for _ in range(3):
        outputs.append(attention(token))
    torch.cuda.synchronize()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input": token.cpu(), "outputs": [x.cpu() for x in outputs]}, a.output)
    print(json.dumps({"equal_repeat": torch.equal(outputs[0], outputs[1]),
                      "max_abs_repeat": (outputs[0]-outputs[1]).abs().max().item(),
                      "samples": [x.flatten()[:4].tolist() for x in outputs]}))


if __name__ == "__main__":
    main()
