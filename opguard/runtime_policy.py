"""Explicit, recorded changes to a workload's numerical execution policy."""

import json
import os
from pathlib import Path
import sys

_applied_pid = None


def determinism_requested():
    return os.getenv("OPGUARD_DETERMINISTIC", "0").lower() not in ("0", "false", "no", "")


def enable_torch_deterministic_mode(*, reason="explicit_api"):
    """Opt in to deterministic kernels without replacing allocators or fill policy."""
    global _applied_pid
    if _applied_pid == os.getpid():
        return
    import torch
    from opguard._running import get_trace_output_dir
    from opguard.distributed import get_global_rank
    from opguard.local_sync.paths import atomic_output

    def settings():
        return {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
        }

    before = settings()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", os.getenv("OPGUARD_CUBLAS_WORKSPACE", ":4096:8"))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    record = {"schema_version": 1, "reason": reason, "pid": os.getpid(),
              "rank": get_global_rank(), "before": before, "after": settings(),
              "allocator_functions_replaced": False}
    path = Path(get_trace_output_dir()) / "runtime_policy" / f"rank{record['rank']}-pid{record['pid']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(path) as temporary:
        Path(temporary).write_text(json.dumps(record, indent=2) + "\n")
    print("[OpGuard runtime policy] " + json.dumps(record), file=sys.stderr)
    _applied_pid = os.getpid()


def apply_requested_policy():
    if determinism_requested():
        enable_torch_deterministic_mode(reason="OPGUARD_DETERMINISTIC=1")
