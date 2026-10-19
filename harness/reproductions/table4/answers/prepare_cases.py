"""Construct small workloads from pinned, unmodified upstream definitions.

The extracted definitions retain their original line numbers. This is a reduced
operator reproduction, not a claim to run the original multi-billion model job.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def extract(source, symbol, prefix):
    text = source.read_text()
    node = next(n for n in ast.parse(text).body if getattr(n, "name", None) == symbol)
    lines = text.splitlines(keepends=True)
    header = prefix.splitlines(keepends=True)
    return "".join(header + ["\n"] * (node.lineno - 1 - len(header)) + lines[node.lineno - 1:node.end_lineno])


def spec(root, case, args=([], []), env=None, ranks=None):
    base_env = {"OPGUARD_CALLSITE_CUPTI": "0", "OPGUARD_CALLSITE_CUDAAPI": "0",
                "OPGUARD_CALLSITE_INCLUDE_CPU": "1",
                "CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1", "OPGUARD_TRACE_DUMP_MODE": "summary"}
    base_env.update(env or {})
    result = {"label": f"table4-bug{case}", "input_id": "seed1234-reduced-upstream",
              "backend": [sys.executable, "-m", "opguard.tools"],
              "local_root": str(root / "staging"), "timeout_seconds": 600,
              "max_stage_attempts": 2}
    for side, argv in zip(("baseline", "candidate"), args):
        result[side] = {"cwd": str(root / f"case{case}" / side), "script": "workload.py",
                        "launcher": [sys.executable], "args": argv, "env": base_env,
                        "ranks": ranks or [0]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "vendor" / "raw"
    for case in (23, 27):
        for side in ("baseline", "candidate"):
            dest = root / f"case{case}" / side
            dest.mkdir(parents=True, exist_ok=True)
            if case == 23:
                code = extract(raw / "tokenizer.py", "_vocab_size_with_padding", "import math\n")
                module = "tokenizer.py"
                workload = '''import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from tokenizer import _vocab_size_with_padding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    config = SimpleNamespace(make_vocab_size_divisible_by=1, tensor_model_parallel_size=args.tp, rank=0)
    padded = _vocab_size_with_padding(50257, config)
    vocab = torch.arange(padded, device="cuda")
    # Identical real-token logits; TP adds a dummy class to the softmax denominator.
    logits = torch.zeros((1, padded), device="cuda")
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([42], device="cuda"))
    torch.save({"vocab": vocab.cpu(), "logits": logits.cpu(), "loss": loss.cpu()}, "observed.pt")
    print(json.dumps({"padded_vocab_size": padded, "loss": loss.item()}))

if __name__ == "__main__":
    main()
'''
            else:
                version = "4.54.1" if side == "baseline" else "4.55.1"
                code = extract(raw / f"smolvlm-{version}.py", "SmolVLMVisionEmbeddings",
                               "from __future__ import annotations\nimport torch\nfrom torch import nn\n")
                module = "modeling_smolvlm.py"
                workload = '''import json
from types import SimpleNamespace
import torch
from modeling_smolvlm import SmolVLMVisionEmbeddings

def main():
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = SimpleNamespace(hidden_size=32, image_size=384, patch_size=14, num_channels=3)
    model = SmolVLMVisionEmbeddings(config).cuda().eval()
    pixels = torch.randn(1, 3, 378, 378, device="cuda")
    mask = torch.ones((1, 27, 27), dtype=torch.bool, device="cuda")
    with torch.no_grad():
        output = model(pixels, mask)
    torch.save(output.cpu(), "observed.pt")
    print(json.dumps({"shape": list(output.shape), "sample": output[0,-1,:10].tolist()}))

if __name__ == "__main__":
    main()
'''
            (dest / module).write_text(code)
            (dest / "workload.py").write_text(workload)
        data = spec(root, case, (["--tp", "1"], ["--tp", "2"]) if case == 23 else ([], []))
        (root / f"case{case}" / "spec.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
