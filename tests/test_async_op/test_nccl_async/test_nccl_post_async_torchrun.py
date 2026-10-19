#!/usr/bin/env python3
# test_nccl_post_async_torchrun_clean.py
import os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

# ---- Pick a usable NIC (don’t force loopback) ----
def _pick_nccl_iface():
    try:
        for name in os.listdir("/sys/class/net"):
            if name in ("lo",):              # skip loopback
                continue
            if name.startswith(("docker","veth","virbr","br-","ib","bond")):
                continue
            st = f"/sys/class/net/{name}/operstate"
            if os.path.exists(st):
                with open(st) as f:
                    if f.read().strip() != "up":
                        continue
            return name
    except Exception:
        pass
    return None

iface = _pick_nccl_iface()
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
# Optional but helps avoid watchdog flakiness in CI
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
if iface:
    os.environ.setdefault("NCCL_SOCKET_IFNAME", iface)

from opguard.wrapped_ops import make_wrapper
from opguard.runtime_tracer import flush_trace_buffer
import opguard.tracer.trace_buffer as _tb

# Capture traces while preserving OpGuard’s normal sink
_CAP = []
_ORIG_ADD_TRACE = getattr(_tb, "ADD_TRACE", None)
def _cap_ADD_TRACE(tag, trace):
    _CAP.append((tag, trace))
    if callable(_ORIG_ADD_TRACE):
        return _ORIG_ADD_TRACE(tag, trace)
_tb.ADD_TRACE = _cap_ADD_TRACE  # type: ignore[attr-defined]

def _is_cuda_summary(s):
    return isinstance(s, dict) and s.get("type") == "Tensor" and "cuda" in str(s.get("device", ""))

def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    assert torch.cuda.is_available()
    assert dist.is_nccl_available()

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world,
        rank=rank,
        timeout=timedelta(seconds=90),
    )
    c10d = torch.distributed.distributed_c10d
    group = dist.group.WORLD

    wrapped = make_wrapper(tag_prefix=f"ut_rank{rank}", dump_tensors=False, global_sync=False)
    wrap = wrapped(lineno=4242)

    # ---- 1) all_gather(async) ----
    x = torch.full((4,), float(rank), device="cuda", dtype=torch.float32)
    outs = [torch.empty_like(x) for _ in range(world)]
    w1 = wrap(c10d.all_gather, outs, x, group=group, async_op=True)
    w1.wait()  # outside wrapper → safe; lets post-future fire

    # ---- 2) all_reduce(async) ----
    y = torch.ones(4, device="cuda", dtype=torch.float32) * (rank + 1)
    w2 = wrap(c10d.all_reduce, y, op=dist.ReduceOp.SUM, group=group, async_op=True)
    w2.wait()

    # Let post-async callbacks run; then a global barrier to avoid watchdog during teardown
    torch.cuda.synchronize()
    time.sleep(0.1)
    flush_trace_buffer()
    dist.barrier()

    # Inspect captured traces
    post = [(tag, tr) for (tag, tr) in _CAP if tr.get("meta", {}).get("phase") == "post_async"]
    immed = [(tag, tr) for (tag, tr) in _CAP if tr.get("meta", {}).get("phase") is None]

    # Content checks
    gathered_ok = all(t.item() in (0.0, 1.0) for t in outs[rank].cpu())
    reduced_ok = torch.allclose(y.cpu(), torch.full_like(y.cpu(), fill_value=3.0))  # 1+2 across 2 ranks

    ok_shapes = True
    for _, tr in post:
        out = tr.get("outputs")
        if isinstance(out, list):
            ok_shapes &= all(_is_cuda_summary(s) for s in out)
        else:
            ok_shapes &= _is_cuda_summary(out)

    status = (len(post) >= 2) and gathered_ok and reduced_ok and ok_shapes
    print(f"[RANK {rank}] post={len(post)} immed={len(immed)} "
          f"gather_ok={gathered_ok} reduce_ok={reduced_ok} ok_shapes={ok_shapes} -> "
          f"{'OK' if status else 'FAIL'}",
          flush=True)

    assert len(post) >= 2, "missing post_async traces"
    assert ok_shapes, "post_async outputs are not CUDA summaries"
    assert gathered_ok, "all_gather content wrong"
    assert reduced_ok, "all_reduce content wrong"

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
