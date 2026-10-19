import os
import random
import math
import torch

# Your production API:
from opguard.tracer.xor_signature import xor_signature

torch.manual_seed(0)
random.seed(0)

# ---------------- CPU reference (layout independent) ----------------
import numpy as np

def cpu_xor_sig_bytes_from_tensor(t: torch.Tensor) -> int:
    """
    Layout-independent CPU reference using NumPy (supports uint32 shifts).
    """
    if t.dim() == 0:
        t = t.reshape(1)
    b = t.contiguous().view(torch.uint8).cpu().numpy()  # uint8
    n = b.size
    if n == 0:
        return 0

    pad = (-n) & 3
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)], axis=0)

    a = b.reshape(-1, 4).astype(np.uint32)
    u32 = (a[:, 0]
           | (a[:, 1] << 8)
           | (a[:, 2] << 16)
           | (a[:, 3] << 24)).astype(np.uint32)
    return int(np.bitwise_xor.reduce(u32))


# ---------------- Helpers ----------------
DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

SIZES_TINY = [0, 1, 2, 3, 4, 5, 7, 15]
SIZES_MED  = [17, 33, 65, 127, 255, 257, 1023, 1025]
SIZES_BIG  = [4096, 8191, 8192, 65535, 65536, 262143]  # ~256KB bytes on uint8
# keep big sizes reasonable for runtime; bump if you want more heat

def make_tensor(device, dtype, numel, pattern="zeros"):
    if pattern == "zeros":
        return torch.zeros(numel, dtype=dtype, device=device)
    if pattern == "ones":
        if dtype is torch.bool:
            return torch.ones(numel, dtype=dtype, device=device)
        if dtype.is_floating_point or dtype in (torch.float16, torch.bfloat16):
            return torch.ones(numel, dtype=dtype, device=device)
        # integer ones:
        return torch.ones(numel, dtype=dtype, device=device)
    if pattern == "randn":
        if dtype is torch.bool:
            # fair coin 0/1 as bool
            return (torch.rand(numel, device=device) > 0.5).to(torch.bool)
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            return torch.randn(numel, dtype=dtype, device=device)
        if dtype == torch.uint8:
            return torch.randint(0, 256, (numel,), dtype=dtype, device=device)
        if dtype == torch.int8:
            return torch.randint(-128, 128, (numel,), dtype=dtype, device=device)
        if dtype == torch.int16:
            return torch.randint(-2**15, 2**15, (numel,), dtype=dtype, device=device)
        if dtype == torch.int32:
            return torch.randint(-2**31, 2**31-1, (numel,), dtype=dtype, device=device)
        if dtype == torch.int64:
            # narrower range to avoid long generation time
            return torch.randint(-2**31, 2**31-1, (numel,), dtype=dtype, device=device)
    raise ValueError("Unknown pattern")

def is_fast_path_likely(byte_stream_uint8: torch.Tensor) -> bool:
    # Heuristic: base ptr 4-byte aligned & length multiple of 4.
    # We can strongly encourage alignment by constructing from uint32 base.
    n = byte_stream_uint8.numel()
    if n % 4 != 0:
        return False
    # No portable way to check the raw pointer mod 4 from Python reliably across builds,
    # but tensors created from uint32.view(uint8) start aligned in practice.
    return True

def assert_bool_signature_has_only_low_bytes(sig: int):
    """For a real 0/1-byte bool stream, only bits at positions 0,8,16,24 may be set."""
    mask_allowed = (1 | (1<<8) | (1<<16) | (1<<24))
    unexpected = sig & ~mask_allowed
    assert unexpected == 0, f"Bool signature has unexpected high bits set: 0x{sig:08x}"

# ---------------- Tests ----------------

def test_zeros_all_cases(device="cuda"):
    for dtype in DTYPES:
        for n in (SIZES_TINY + SIZES_MED + SIZES_BIG):
            x = make_tensor(device, dtype, n, "zeros")
            sig_gpu = int(xor_signature(x).item())
            assert sig_gpu == 0, f"zeros case failed dtype={dtype} n={n} sig={sig_gpu}"

def test_random_cpu_parity(device="cuda", trials=200):
    for _ in range(trials):
        dtype = random.choice(DTYPES)
        n = random.choice(SIZES_TINY + SIZES_MED + SIZES_BIG)
        x = make_tensor(device, dtype, n, "randn")
        sig_gpu = int(xor_signature(x).item())
        sig_cpu = cpu_xor_sig_bytes_from_tensor(x)
        assert sig_gpu == sig_cpu, (
            f"cpu parity mismatch dtype={dtype} n={n} "
            f"gpu=0x{sig_gpu:08x} cpu=0x{sig_cpu:08x}"
        )

def test_non_contiguous_and_views(device="cuda"):
    # 2D transpose / strides, then ensure xor_signature equals clone()
    x = torch.randn(129, 257, dtype=torch.float32, device=device)
    y = x.t()           # non-contiguous
    z = y[1:-1, ::2]    # slicing/striding
    sig_nc = int(xor_signature(z).item())
    sig_clone = int(xor_signature(z.clone()).item())
    assert sig_nc == sig_clone, "non-contiguous view changed signature vs clone"

def test_misaligned_byte_offsets(device="cuda"):
    # Build an aligned uint32 base, then misalign at byte level by slicing uint8
    base = torch.randint(0, 2**32-1, (10003,), dtype=torch.int64, device=device)\
            .to(torch.uint32)  # ensures 4-byte alignment base
    b = base.view(torch.uint8)
    for off in (0, 1, 2, 3):
        sub = b[off: b.numel()-((b.numel()-off) % 17)]  # random tail crop
        # CPU reference on the *logical* stream (respecting offset)
        # (wrap back into a tensor via uint8 → view to preserve bytes)
        sub_t = sub.view(torch.uint8)
        sig_gpu = int(xor_signature(sub_t).item())
        # Compose a fake tensor from bytes for CPU ref:
        sig_cpu = cpu_xor_sig_bytes_from_tensor(sub_t)
        assert sig_gpu == sig_cpu, (
            f"misaligned offset={off} mismatch gpu=0x{sig_gpu:08x} cpu=0x{sig_cpu:08x}"
        )

def test_auto_vs_forced_paths(device="cuda"):
    # Force fast path: construct from uint32 → view(uint8) with length %4==0
    u32 = torch.randint(0, 2**32-1, (50007,), dtype=torch.int64, device=device).to(torch.uint32)
    u8_fast = u32.view(torch.uint8)   # aligned, multiple of 4 length
    sig_auto = int(xor_signature(u8_fast).item())

    # Force byte path by misaligning 1 byte (length not multiple of 4)
    u8_byte = u8_fast[1:]  # offset=1 (mod 4)
    sig_auto_byte = int(xor_signature(u8_byte).item())

    # CPU refs
    ref_fast = cpu_xor_sig_bytes_from_tensor(u8_fast)
    ref_byte = cpu_xor_sig_bytes_from_tensor(u8_byte)

    assert sig_auto == ref_fast, "auto fast-path parity failed"
    assert sig_auto_byte == ref_byte, "auto byte-path parity failed"

def test_bool_bit_positions(device="cuda", trials=100):
    # Only bits 0/8/16/24 may appear for *true* bool 0/1 bytes
    for _ in range(trials):
        n = random.randint(0, 5000)
        b = (torch.rand(n, device=device) > 0.5).to(torch.bool)
        sig = int(xor_signature(b).item())
        assert_bool_signature_has_only_low_bytes(sig)

def test_small_edge_cases(device="cuda"):
    # Scalars & tiny tails through dtype zoo
    for dtype in DTYPES:
        # scalar
        x = make_tensor(device, dtype, 1, "zeros")[0]  # 0-dim scalar
        sig = int(xor_signature(x).item())
        assert sig == 0, f"scalar zeros failed dtype={dtype} sig={sig}"

        # 1..3 elements of random bytes (esp. uint8/int8)
        for n in (1, 2, 3):
            x = make_tensor(device, dtype, n, "randn")
            sig_gpu = int(xor_signature(x).item())
            sig_cpu = cpu_xor_sig_bytes_from_tensor(x)
            assert sig_gpu == sig_cpu, f"tiny tail mismatch dtype={dtype} n={n}"

def test_large_random(device="cuda"):
    # A couple of larger byte streams (several MB)
    for bytes_len in [1<<20, (1<<20) + 3, (1<<22) + 1]:  # ~1MB, just-over, ~4MB
        u8 = torch.randint(0, 256, (bytes_len,), dtype=torch.uint8, device=device)
        sig_gpu = int(xor_signature(u8).item())
        sig_cpu = cpu_xor_sig_bytes_from_tensor(u8)
        assert sig_gpu == sig_cpu, f"large random mismatch nbytes={bytes_len}"

def test_multistream_concurrency(device="cuda", iters=500):
    # Try to shake out stream/order bugs by launching work on other streams interleaved
    streams = [torch.cuda.Stream() for _ in range(3)]
    for i in range(iters):
        # Prepare some tensors on aux streams
        for s in streams:
            with torch.cuda.stream(s):
                _ = torch.randn(8192, device=device) * 1.01

        # Meanwhile compute a signature on main stream
        x = torch.zeros(1, dtype=torch.int32, device=device)
        sig = int(xor_signature(x).item())
        if sig != 0:
            raise AssertionError(f"multistream zeros flaked at iter={i} sig={sig}")

    # ensure all streams complete
    torch.cuda.synchronize()

def run_all():
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required for this stress test"

    print("== stress: zeros ==")
    test_zeros_all_cases(device)
    print("== stress: random cpu parity ==")
    test_random_cpu_parity(device)
    print("== stress: non-contiguous views ==")
    test_non_contiguous_and_views(device)
    print("== stress: misaligned byte offsets ==")
    test_misaligned_byte_offsets(device)
    print("== stress: auto vs forced paths ==")
    test_auto_vs_forced_paths(device)
    print("== stress: bool bit positions ==")
    test_bool_bit_positions(device)
    print("== stress: small edge cases ==")
    test_small_edge_cases(device)
    print("== stress: large random ==")
    test_large_random(device)
    print("== stress: multistream concurrency ==")
    test_multistream_concurrency(device)

    print("\n✅ All stress tests passed.")

if __name__ == "__main__":
    run_all()
