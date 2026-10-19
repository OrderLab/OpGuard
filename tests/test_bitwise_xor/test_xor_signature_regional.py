# test_xor_signature.py
import os
import math
import numpy as np
import pytest
import torch

# Import the implementation under test
import opguard.tracer.xor_signature as xs


# ----------------------- Helpers -----------------------

def _pack_and_xor_numpy(byte_arr: np.ndarray) -> np.uint32:
    """Reference: XOR over bytes, packed little-endian into u32 words."""
    if byte_arr.size == 0:
        return np.uint32(0)
    pad = (-byte_arr.size) & 3
    if pad:
        byte_arr = np.concatenate([byte_arr, np.zeros(pad, dtype=np.uint8)])
    a = byte_arr.reshape(-1, 4).astype(np.uint32, copy=False)
    u32 = (a[:, 0] |
           (a[:, 1] << 8) |
           (a[:, 2] << 16) |
           (a[:, 3] << 24)).astype(np.uint32, copy=False)
    return np.bitwise_xor.reduce(u32).astype(np.uint32)


def _regional_reference_cpu(t: torch.Tensor, regional: int) -> torch.Tensor:
    """
    Reference implementation for regional-by-rows XOR signatures on CPU.
    Mirrors xs.xor_signature's row partitioning.
    Returns torch.uint32 of length R (on CPU).
    """
    assert t.dim() >= 1
    if not t.is_contiguous():
        t = t.contiguous()

    rows = t.size(0)
    cols = int(t.numel() // max(rows, 1)) if rows > 0 else 0

    # Cap R to rows
    R = min(int(regional), int(rows)) if rows > 0 else int(regional)
    if rows == 0:
        return torch.zeros(R, dtype=torch.uint32, device="cpu")

    bytes_per_row = cols * t.element_size()
    byte_1d = t.view(torch.uint8).view(-1).cpu().numpy()

    base = rows // R
    rem = rows % R

    out = np.empty(R, dtype=np.uint32)
    start_row = 0
    for i in range(R):
        cnt = base + (1 if i < rem else 0)
        nbytes = cnt * bytes_per_row
        off = start_row * bytes_per_row
        region = byte_1d[off: off + nbytes]
        out[i] = _pack_and_xor_numpy(region)
        start_row += cnt

    return torch.from_numpy(out)


def _make_tensor(shape, dtype, device="cpu", seed=0, noncontig=False):
    import numpy as np
    import torch, math, pytest

    es = torch.tensor([], dtype=dtype).element_size()
    numel = math.prod(shape)
    nbytes = numel * es

    rng = np.random.default_rng(seed)
    # exact number of bytes required for the target shape+dtype
    byte_arr = rng.integers(0, 256, size=nbytes, dtype=np.uint8)

    # reinterpret bytes as target dtype (little-endian assumed; typical on Linux)
    if dtype == torch.uint8:
        arr = byte_arr
    elif dtype == torch.int16 or dtype == torch.float16:
        arr = byte_arr.view(np.int16)  # treat as 16-bit words
        if dtype == torch.float16:
            arr = arr.view(np.float16)
    elif dtype == torch.int32 or dtype == torch.float32:
        arr = byte_arr.view(np.int32)
        if dtype == torch.float32:
            arr = arr.view(np.float32)
    elif dtype == torch.int64 or dtype == torch.float64:
        arr = byte_arr.view(np.int64)
        if dtype == torch.float64:
            arr = arr.view(np.float64)
    else:
        raise ValueError(f"Unhandled dtype {dtype}")

    arr = arr.reshape(*shape)
    t = torch.from_numpy(arr)

    if noncontig and t.dim() >= 2 and t.size(0) >= 2:
        t = t.transpose(0, 1)  # make non-contiguous
        t = t[:, :]            # keep as view

    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = t.to("cuda", non_blocking=True)

    return t


# ----------------------- Tests -----------------------

@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1024, 7), (2048, 3), (513, 11)])
@pytest.mark.parametrize("regional", [4, 16])
def test_regional_correctness_cpu(dtype, shape, regional):
    """Regional signatures match NumPy reference on CPU for various dtypes/shapes."""
    t = _make_tensor(shape, dtype, device="cpu", seed=123)
    # Force regional path by setting threshold <= rows
    rows = t.size(0)
    out = xs.xor_signature(t, regional=regional, threshold=max(1, rows), by="rows")
    ref = _regional_reference_cpu(t, regional)
    assert out.dtype == torch.uint32
    assert torch.equal(out.cpu(), ref), f"Mismatch for dtype={dtype}, shape={shape}, regional={regional}"


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
@pytest.mark.parametrize("shape", [(1500, 5), (4096, 1)])
@pytest.mark.parametrize("regional", [8, 16])
def test_regional_correctness_cuda(dtype, shape, regional):
    """CUDA regional signatures equal CPU reference (if CUDA + extension available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Ensure the extension is importable (xor_signature imports it already)
    try:
        import xor_reduce_cuda  # noqa: F401
    except Exception as e:
        pytest.skip(f"xor_reduce_cuda extension not available: {e}")

    t_cpu = _make_tensor(shape, dtype, device="cpu", seed=7)
    t_cuda = t_cpu.to("cuda", non_blocking=True)

    rows = t_cpu.size(0)
    out_cuda = xs.xor_signature(t_cuda, regional=regional, threshold=max(1, rows), by="rows").cpu()
    ref = _regional_reference_cpu(t_cpu, regional)

    assert out_cuda.dtype == torch.uint32
    assert torch.equal(out_cuda, ref)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.float32])
def test_threshold_fallback_single_signature(dtype):
    """When rows < threshold, even with regional requested, return single signature."""
    t = _make_tensor((128, 17), dtype, device="cpu", seed=99)  # rows=128
    out = xs.xor_signature(t, regional=16, threshold=512, by="rows")
    assert out.shape == (1,), "Should fall back to single-signature output"
    assert out.dtype == torch.uint32


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_exact_threshold_triggers_regional(dtype):
    """When rows == threshold, regional path should be taken."""
    t = _make_tensor((512, 9), dtype, device="cpu", seed=1)  # rows=512
    out = xs.xor_signature(t, regional=16, threshold=512, by="rows")
    assert out.shape == (16,), "Regional should be used at rows == threshold"
    ref = _regional_reference_cpu(t, 16)
    assert torch.equal(out.cpu(), ref)


@pytest.mark.parametrize("noncontig", [False, True])
def test_contiguity_handling(noncontig):
    """Non-contiguous inputs should be handled (made contiguous internally)."""
    t = _make_tensor((1024, 13), torch.int32, device="cpu", seed=5, noncontig=noncontig)
    out = xs.xor_signature(t, regional=4, threshold=1, by="rows")  # force regional
    ref = _regional_reference_cpu(t, 4)
    assert torch.equal(out.cpu(), ref)


def test_default_off_same_as_single():
    """Default call (regional=None) returns single 32-bit signature; equals manual reference."""
    t = _make_tensor((321, 7), torch.float32, device="cpu", seed=2024)
    out = xs.xor_signature(t)  # default off
    assert out.shape == (1,)
    # Manual reference over all bytes
    arr = t.contiguous().view(torch.uint8).view(-1).cpu().numpy()
    ref = _pack_and_xor_numpy(arr)
    assert int(out.item()) == int(ref)


def test_zero_rows_and_empty_tensor():
    """Edge cases: zero rows / empty tensor."""
    t0 = torch.empty((0, 10), dtype=torch.int32)
    out0 = xs.xor_signature(t0)  # default single zero
    assert out0.shape == (1,)
    assert out0.item() == 0

    # Regional requested but rows(=0) < threshold -> fallback to single
    out1 = xs.xor_signature(t0, regional=8, threshold=1)
    assert out1.shape == (1,)
    assert out1.item() == 0

    # Truly empty flat tensor
    t2 = torch.empty((0,), dtype=torch.uint8)
    out2 = xs.xor_signature(t2)
    assert out2.shape == (1,)
    assert out2.item() == 0
