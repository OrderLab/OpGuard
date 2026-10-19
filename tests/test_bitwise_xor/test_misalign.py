# OpGuard/tests/test_bitwise_xor/test_xor_signature.py
import os
import pytest
import torch

from opguard.tracer.xor_signature import xor_signature

CUDA_AVAILABLE = torch.cuda.is_available()

def _cpu_ref_xor(t: torch.Tensor) -> torch.Tensor:
    """Use the CPU path in xor_signature as reference by moving data to CPU."""
    return xor_signature(t.cpu())

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [1, 5, 1025, 12345])
def test_basic_cpu(dtype, n):
    x = torch.randn(n, dtype=dtype, device="cpu")
    y = xor_signature(x)
    assert y.dtype == torch.uint32
    assert y.device.type == "cpu"

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [1, 5, 1025, 12345])
def test_basic_cuda_matches_cpu(dtype, n):
    x = torch.randn(n, dtype=dtype, device="cuda")
    y_cuda = xor_signature(x)
    y_cpu  = _cpu_ref_xor(x)
    assert y_cuda.dtype == torch.uint32
    assert y_cuda.device.type == "cuda"
    # Compare values (move to cpu for comparison)
    assert int(y_cuda.cpu().item()) == int(y_cpu.item())

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_misaligned_bf16_slice_no_raise_and_matches_cpu():
    # Create a known misalignment: bf16 has 2-byte elements; slicing by 1 gives offset % 4 == 2
    t = torch.randn(1025, dtype=torch.bfloat16, device="cuda")
    t_view = t[1:]  # misaligned start
    # Should not raise
    y_cuda = xor_signature(t_view)
    y_cpu  = _cpu_ref_xor(t_view)
    assert y_cuda.dtype == torch.uint32
    assert int(y_cuda.cpu().item()) == int(y_cpu.item())

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tail_bytes_0_to_3(dtype):
    # Check every remainder mod 4 for tail handling without padding copies
    base = 4096
    for rem in [0, 1, 2, 3]:
        n = base + rem
        x = torch.randn(n, dtype=dtype, device="cuda")
        y_cuda = xor_signature(x)
        y_cpu  = _cpu_ref_xor(x)
        assert int(y_cuda.cpu().item()) == int(y_cpu.item())

def test_scalar_cpu():
    s = torch.tensor(3.14, dtype=torch.float32)  # 0-dim
    y = xor_signature(s)
    assert y.dtype == torch.uint32
    assert y.device.type == "cpu"

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_scalar_cuda():
    s = torch.tensor(3.14, dtype=torch.float32, device="cuda")  # 0-dim
    y_cuda = xor_signature(s)
    y_cpu  = _cpu_ref_xor(s)
    assert y_cuda.dtype == torch.uint32
    assert y_cuda.device.type == "cuda"
    assert int(y_cuda.cpu().item()) == int(y_cpu.item())

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_large_random_shapes(dtype):
    # A couple of larger, odd sizes to stress the path
    for n in [65536 + 7, 131072 + 3]:
        x = torch.randn(n, dtype=dtype, device="cuda")
        y_cuda = xor_signature(x)
        y_cpu  = _cpu_ref_xor(x)
        assert int(y_cuda.cpu().item()) == int(y_cpu.item())

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_consistency_across_views_and_clones():
    # Ensure result is invariant to benign view/clone differences
    x = torch.randn(5003, dtype=torch.float32, device="cuda")
    y1 = xor_signature(x)
    y2 = xor_signature(x.clone())  # different storage
    y3 = xor_signature(x[::1])     # view with same bytes
    assert int(y1.cpu().item()) == int(y2.cpu().item()) == int(y3.cpu().item())

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_seeded_determinism():
    torch.manual_seed(1337)
    a = torch.randn(9999, dtype=torch.bfloat16, device="cuda")
    torch.manual_seed(1337)
    b = torch.randn(9999, dtype=torch.bfloat16, device="cuda")
    ya = xor_signature(a)
    yb = xor_signature(b)
    assert int(ya.cpu().item()) == int(yb.cpu().item())
