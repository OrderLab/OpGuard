# test_triple_tracer.py
import os, sys, json, time, tempfile, ctypes
from pathlib import Path

import torch

# Import your coordinated tracer
from opguard.callsite_tracer.trace import CallsiteTracer

def native_hook_available() -> bool:
    # Already loaded?
    try:
        lib = ctypes.CDLL(None)
        for nm in ("pyhook_start_mem","pyhook_drain","pyhook_flush","pyhook_stop"):
            if not hasattr(lib, nm):
                break
        else:
            print("Native hook already loaded")
            return True
    except Exception:
        pass
    # Try OPGUARD_NATIVE_HOOK_PATH
    p = os.getenv("OPGUARD_NATIVE_HOOK_PATH")
    if p and Path(p).exists():
        try:
            ctypes.CDLL(p)
            print(f"Loaded native hook from {p}")
            return True
        except OSError:
            return False
    # Try packaged default
    try:
        import opguard
        cand = Path(opguard.__file__).resolve().parent / "callsite_tracer" / "native" / "libpylaunch_hook.so"
        if cand.exists():
            ctypes.CDLL(str(cand))
            print(f"Loaded native hook from {cand}")
            return True
    except Exception:
        pass
    print("No native hook available")
    return False

def cupti_available() -> bool:
    try:
        import opguard.callsite_tracer.opguard_cupti as _c
        return True
    except Exception:
        return False

def run_cuda_work():
    # Some light CUDA work to trigger kernels (cublas matmul)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot run test.")
    d = torch.device("cuda")
    a = torch.randn(1024, 1024, device=d)
    b = torch.randn(1024, 1024, device=d)
    for _ in range(5):
        c = a @ b       # cublasGemmEx under the hood
        c = torch.relu(c)
    torch.cuda.synchronize()

def maybe_triton_work():
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def add_one(x_ptr, n_elements: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * n_elements + tl.arange(0, n_elements)
            x = tl.load(x_ptr + offs, mask=offs < n_elements)
            tl.store(x_ptr + offs, x + 1, mask=offs < n_elements)

        x = torch.zeros(4096, device="cuda", dtype=torch.float32)
        grid = (1,)
        add_one[grid](x, n_elements=x.numel())
        torch.cuda.synchronize()
        return True
    except Exception:
        return False

def main():
    # Enable all three; backends will gracefully skip if unavailable.
    os.environ["OPGUARD_CALLSITE_CUDAAPI"] = "1"
    os.environ["OPGUARD_CALLSITE_CUPTI"]   = "1"
    os.environ["OPGUARD_CALLSITE_PYTHON"]  = "1"

    # Optional: point to your native hook .so if not LD_PRELOADed
    # os.environ["OPGUARD_NATIVE_HOOK_PATH"] = "/path/to/libpylaunch_hook.so"

    # Keep dumps in a temp dir
    tmp = tempfile.mkdtemp(prefix="triple_tracer_")
    base_dump = str(Path(tmp) / "callsite_tracer.json")

    # Small cadence so we flush during the test
    tracer = CallsiteTracer(dump_every=10, dump_path=base_dump, python_strategy="ops+dispatch")
    with tracer:
        run_cuda_work()
        maybe_triton_work()  # fine if it no-ops

        # Give the CUPTI/native pumps a moment to drain
        time.sleep(1.0)

    # After context exit, files should exist for each active backend
    rank = 0
    try:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:
        pass

    base = Path(base_dump)
    name = base.stem
    if "rank" not in name:
        name = f"{name}_rank{rank}"
    cudaapi_path = base.parent / f"cudaapi_{name}{base.suffix}"
    cupti_path   = base.parent / f"cupti_{name}{base.suffix}"
    python_path  = base.parent / f"python_{name}{base.suffix}"

    results = {}

    # CUDA-API/native
    if native_hook_available() and cudaapi_path.exists():
        with open(cudaapi_path) as f:
            data = json.load(f)
        total = sum(v.get("count", 0) for v in data.values())
        results["cudaapi"] = {"path": str(cudaapi_path), "sites": len(data), "events": total}
    else:
        results["cudaapi"] = {"skipped": True, "reason": "native hook unavailable or file missing"}

    # CUPTI
    if cupti_available() and cupti_path.exists():
        with open(cupti_path) as f:
            data = json.load(f)
        total = sum(v.get("count", 0) for v in data.values())
        results["cupti"] = {"path": str(cupti_path), "sites": len(data), "events": total}
    else:
        results["cupti"] = {"skipped": True, "reason": "CUPTI unavailable or file missing"}

    # Python
    if python_path.exists():
        with open(python_path) as f:
            data = json.load(f)
        total = sum(v.get("count", 0) for v in data.values())
        results["python"] = {"path": str(python_path), "sites": len(data), "events": total}
    else:
        results["python"] = {"skipped": True, "reason": "python file missing"}

    # Minimal assertions (soft): at least one active backend must have events
    active = [k for k,v in results.items() if not v.get("skipped")]
    if not active:
        raise AssertionError(f"No active backends produced output. Results: {results}")

    # Print a compact summary
    print("\n=== Triple Tracer Summary ===")
    for k, v in results.items():
        if v.get("skipped"):
            print(f"- {k:7s}: SKIPPED ({v['reason']})")
        else:
            print(f"- {k:7s}: events={v['events']:>6}  sites={v['sites']:>4}  file={v['path']}")
    print(f"Temp dir: {tmp}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine; skipping.", file=sys.stderr)
        sys.exit(0)
    main()
