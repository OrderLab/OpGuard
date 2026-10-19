# tests/test_bitwise_xor/test_div_by4.py

import pytest
import torch
from OpGuard.opguard.tracer.xor_signature import xor_signature

@pytest.mark.parametrize("dtype", [
    torch.uint8,
    torch.int8,
    torch.float32,
    torch.bfloat16
])
@pytest.mark.parametrize("length", [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    15, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129,
    1023, 1024, 1025,
    4095, 4096, 4097
])
def test_divisible_and_nondivisible_lengths(dtype, length):
    # build test tensor
    if dtype in (torch.uint8, torch.int8):
        # random bytes, then reinterpret as int8 if needed
        base = torch.randint(0, 256, (length,), dtype=torch.uint8)
        data = base if dtype is torch.uint8 else base.view(torch.int8)
    else:
        data = torch.randn(length, dtype=dtype)

    # CPU signature (should not error)
    cpu_sig = xor_signature(data)
    assert cpu_sig.shape == (1,)
    assert cpu_sig.dtype == torch.uint32
    assert cpu_sig.device.type == "cpu"

    # GPU signature (should not error)
    gpu_sig = xor_signature(data.cuda())
    assert gpu_sig.shape == (1,)
    assert gpu_sig.dtype == torch.uint32
    assert gpu_sig.device.type == "cuda"

    # They must agree
    assert int(cpu_sig.item()) == int(gpu_sig.item()), (
        f"Mismatch for dtype={dtype} length={length}: "
        f"cpu=0x{cpu_sig.item():08x}, gpu=0x{gpu_sig.item():08x}"
    )

def test_multidim_and_noncontiguous():
    # ensure view(torch.uint8) works for a 2D, non-contiguous
    x = torch.randn(10, 15, dtype=torch.float32)       # 10×15 floats
    x = x[:, ::2].contiguous()                                   # now non-contiguous, shape=(10,8)
    # flatten & run
    cpu_sig = xor_signature(x)
    gpu_sig = xor_signature(x.cuda())
    assert int(cpu_sig.item()) == int(gpu_sig.item())

if __name__ == "__main__":
    pytest.main([__file__])
