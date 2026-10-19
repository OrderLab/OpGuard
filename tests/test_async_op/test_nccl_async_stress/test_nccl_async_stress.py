#!/usr/bin/env python3
import os, sys, time, random
from datetime import timedelta
from typing import List, Tuple
import numpy as np
import torch
import torch.distributed as dist

# ---- NIC selection (don’t force loopback) ----
def _pick_nccl_iface():
    try:
        for name in os.listdir("/sys/class/net"):
            if name in ("lo",): continue
            if name.startswith(("docker","veth","virbr","br-","ib","bond")): continue
            st = f"/sys/class/net/{name}/operstate"
            if os.path.exists(st):
                with open(st) as f:
                    if f.read().strip() != "up": continue
            return name
    except Exception: pass
    return None

iface = _pick_nccl_iface()
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
if iface: os.environ.setdefault("NCCL_SOCKET_IFNAME", iface)

from opguard.wrapped_ops import make_wrapper
from opguard.runtime_tracer import flush_trace_buffer
import opguard.tracer.trace_buffer as _tb
from opguard.tracer.xor_signature import xor_signature

# ---- capture traces while preserving normal sink ----
_CAP = []
_ORIG_ADD_TRACE = getattr(_tb, "ADD_TRACE", None)
def _cap_ADD_TRACE(tag, trace):
    _CAP.append((tag, trace))
    if callable(_ORIG_ADD_TRACE): _ORIG_ADD_TRACE(tag, trace)
_tb.ADD_TRACE = _cap_ADD_TRACE  # type: ignore[attr-defined]

# ---- XOR refs ----
def _cpu_xor_bytes(t: torch.Tensor) -> int:
    b = t.detach().cpu().view(torch.uint8).view(-1).numpy()
    if b.size == 0: return 0
    pad = (-b.size) & 3
    if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    a = b.reshape(-1, 4).astype(np.uint32, copy=False)
    u32 = (a[:,0] | (a[:,1]<<8) | (a[:,2]<<16) | (a[:,3]<<24)).astype(np.uint32, copy=False)
    return int(np.bitwise_xor.reduce(u32).astype(np.uint32))

def _cuda_xor32(t: torch.Tensor) -> int:
    return int(xor_signature(t).to(torch.uint32).cpu().view(torch.int32).item() & 0xFFFFFFFF)

def _check_xor_equal(t: torch.Tensor):
    got = _cuda_xor32(t); ref = _cpu_xor_bytes(t)
    if got != ref:
        raise AssertionError(
            f"XOR mismatch: got=0x{got:08x} ref=0x{ref:08x} "
            f"shape={tuple(t.shape)} dtype={t.dtype} contig={t.is_contiguous()} dev={t.device}"
        )

# ---- deterministic per-iter config (broadcast) ----
_DTYPES = [torch.float32, torch.int64, torch.float16] + ([torch.bfloat16] if hasattr(torch, "bfloat16") else [])
_D2I = {dt:i for i,dt in enumerate(_DTYPES)}; _I2D = {i:dt for dt,i in _D2I.items()}
_OPS = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all_single"]
_O2I = {op:i for i,op in enumerate(_OPS)}; _I2O = {i:op for op,i in _O2I.items()}

def _bcast_cfg(rank, dev):
    cfg = torch.zeros(4, device=dev, dtype=torch.int64)  # op, rows, cols, dtype_id
    if rank == 0:
        op   = random.choice(_OPS)
        rows = random.choice([64,128,256,512,1024,2048])
        cols = random.choice([16,32,64,128,256,512,1024])
        dt   = random.choice(_DTYPES)
        cfg[0], cfg[1], cfg[2], cfg[3] = _O2I[op], rows, cols, _D2I[dt]
    dist.broadcast(cfg, src=0)
    return _I2O[int(cfg[0])], int(cfg[1]), int(cfg[2]), _I2D[int(cfg[3])]

# ---- helpers ----
def _rand(rows, cols, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(rows, cols, device=device, dtype=dtype)
    return torch.randint(-2**15, 2**15, (rows, cols), device=device, dtype=dtype)

def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    assert torch.cuda.is_available() and dist.is_nccl_available()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(2025 + rank); random.seed(2025)

    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=world, rank=rank, timeout=timedelta(seconds=240)
    )
    c10d = torch.distributed.distributed_c10d
    group = dist.group.WORLD

    wrapped_factory = make_wrapper(tag_prefix=f"stress_rank{rank}", dump_tensors=False, global_sync=False)

    handles: List[Tuple[str, object]] = []
    verifiers: List[Tuple[str, object]] = []
    iters, inflight_cap = 40, 12
    dev = torch.device("cuda", local_rank)

    for step in range(iters):
        wrap = wrapped_factory(lineno=10000 + step)
        op, rows, cols, dtype = _bcast_cfg(rank, dev)

        if op == "all_gather":
            x = torch.full((rows, cols), float(rank), device=dev, dtype=dtype)  # contiguous
            outs = [torch.empty_like(x) for _ in range(world)]                  # contiguous
            w = wrap(c10d.all_gather, outs, x, group=group, async_op=True)
            handles.append((op, w)); verifiers.append((op, (x, outs)))

        elif op == "all_reduce":
            y_view = torch.ones(rows, cols, device=dev, dtype=dtype) * (rank + 1)
            # keep a non-contig view for XOR-only coverage; pass contiguous to c10d
            if cols >= 2 and (step % 3 == 0): y_view = y_view[:, ::2]
            y = y_view.contiguous()  # required by your NCCL build
            w = wrap(c10d.all_reduce, y, op=dist.ReduceOp.SUM, group=group, async_op=True)
            handles.append((op, w)); verifiers.append((op, (y_view, y)))  # (noncontig_for_xor, reduced_contig)

        elif op == "reduce_scatter":
            big_rows = rows * world
            base = torch.full((big_rows, cols), float(rank + 1), device=dev, dtype=dtype)  # contiguous
            in_list = list(torch.chunk(base, world, dim=0))  # contiguous chunks
            out = torch.empty(rows, cols, device=dev, dtype=dtype)  # contiguous
            w = wrap(c10d.reduce_scatter, out, in_list, op=dist.ReduceOp.SUM, group=group, async_op=True)
            handles.append((op, w)); verifiers.append((op, out))

        else:  # all_to_all_single
            rows2 = rows - (rows % world);  rows2 = rows2 or world
            inp = torch.arange(rows2, device=dev, dtype=torch.int64).unsqueeze(1).expand(-1, cols).clone()
            inp += (rank * 1_000_000)  # distinct per-rank content
            out = torch.empty_like(inp)  # contiguous
            split = [rows2 // world] * world
            w = wrap(c10d.all_to_all_single, out, inp,
                     output_split_sizes=split, input_split_sizes=split,
                     group=group, async_op=True)
            handles.append((op, w)); verifiers.append((op, (inp, out, split)))

        if len(handles) >= inflight_cap:
            _drain(handles, verifiers, rank, world)

    _drain(handles, verifiers, rank, world)

    torch.cuda.synchronize(); time.sleep(0.2)
    flush_trace_buffer()
    dist.barrier()

    post = [(tag, tr) for (tag, tr) in _CAP if tr.get("meta", {}).get("phase") == "post_async"]
    assert len(post) >= iters, f"Expected >= {iters} post_async traces, got {len(post)}"
    print(f"[RANK {rank}] Stress OK: post={len(post)} (>= {iters})", flush=True)
    dist.destroy_process_group()

# ---- drain & verify (with XOR checks) ----
def _drain(handles, verifiers, rank, world):
    for _, w in handles: w.wait()

    for (name, payload) in verifiers:
        if name == "all_gather":
            x, outs = payload
            _check_xor_equal(x); _check_xor_equal(outs[rank])
            assert torch.equal(outs[rank].cpu(), x.cpu()), "all_gather local slot mismatch"
            for t in outs: _check_xor_equal(t)

        elif name == "all_reduce":
            y_view, y = payload  # y is contiguous reduced result
            # ensure XOR works for the noncontig view (pre-reduce content)
            _check_xor_equal(y_view)
            _check_xor_equal(y)
            target = (world * (world + 1)) // 2
            assert torch.allclose(y.float().cpu(), torch.full_like(y.float().cpu(), float(target))), "all_reduce content wrong"

        elif name == "reduce_scatter":
            out = payload
            _check_xor_equal(out)
            target = (world * (world + 1)) // 2
            assert torch.allclose(out.float().cpu(), torch.full_like(out.float().cpu(), float(target))), "reduce_scatter content wrong"

        elif name == "all_to_all_single":
            inp, out, split = payload
            # Build expected via all_gather of inputs (post-wait, safe)
            inps = [torch.empty_like(inp) for _ in range(world)]
            dist.all_gather(inps, inp)
            chunk = split[0]
            expected = torch.cat([inps[src][rank*chunk:(rank+1)*chunk] for src in range(world)], dim=0)
            _check_xor_equal(out)
            assert torch.equal(out.cpu(), expected.cpu()), "all_to_all_single content wrong"

        else:
            raise AssertionError(f"unknown op {name}")

    handles.clear(); verifiers.clear()

if __name__ == "__main__":
    main()
