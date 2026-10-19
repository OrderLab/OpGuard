"""Build pinned historical DeepSpeed kernels in a task-local extension cache."""
import argparse
import ast
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("--name", default="ds_inference_091")
p.add_argument("--build", type=Path, required=True)
a = p.parse_args()
a.build.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
os.environ.setdefault("MAX_JOBS", "4")
builder = ast.parse((a.source / "op_builder/transformer_inference.py").read_text())
method = next(n for n in ast.walk(builder) if isinstance(n, ast.FunctionDef) and n.name == "sources")
sources = ast.literal_eval(next(n.value for n in ast.walk(method) if isinstance(n, ast.Return)))
mod = load(name=a.name, sources=[str(a.source.resolve() / s) for s in sources],
           extra_include_paths=[str(a.source.resolve()/s) for s in
                                ("csrc/transformer/inference/includes", "csrc/includes")],
           extra_cflags=["-O2"], extra_cuda_cflags=["-O2", "-U__CUDA_NO_HALF_OPERATORS__",
                   "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
           extra_ldflags=["-lcurand"], build_directory=str(a.build.resolve()), verbose=True)
print("built", mod.__file__)
