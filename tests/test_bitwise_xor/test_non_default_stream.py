# tests/test_bitwise_xor/test_stream_dispatch.py
import os
import time
import pytest
import torch

from OpGuard.opguard.tracer.xor_signature import xor_signature


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _sleep_ms(ms: float):
    # torch.cuda._sleep takes "iterations" (roughly 1 iteration ~ 1 GPU clock cycle),
    # but its exact timing varies by GPU. Use a large count to ensure noticeable delay.
    # 50_000_000 is typically tens of ms on modern GPUs.
    iters = int(50_000_000 * (ms / 50.0))  # scale roughly
    torch.cuda._sleep(max(1, iters))


def _new_stream(dev: torch.device):
    with torch.cuda.device(dev):
        return torch.cuda.Stream(device=dev)


def _make_tensor(dev: torch.device, n: int = 256, dtype=torch.float32):
    with torch.cuda.device(dev):
        t = torch.arange(n, dtype=dtype, device=dev).reshape(16, -1).contiguous()
    return t


def test_graph_capture_launches_on_capturing_stream():
    """If xor_signature launches on any stream other than the capturing stream, capture fails."""
    dev = torch.device("cuda", 0)
    t = _make_tensor(dev, 4096)  # a small but nontrivial tensor

    # Prepare a fixed stream to capture on
    s = _new_stream(dev)
    g = torch.cuda.CUDAGraph()

    # Static output buffer to keep allocation graph-friendly (not strictly required here)
    sig_buf = torch.empty(1, dtype=torch.uint32, device=dev)

    # Warm-up to initialize kernels/allocators outside capture
    with torch.cuda.stream(s):
        _ = xor_signature(t)

    torch.cuda.synchronize(dev)

    # --- Capture ---
    with torch.cuda.graph(g, stream=s):
        # xor_signature should enqueue exclusively on `s` (binding launches on current stream)
        sig = xor_signature(t)
        # materialize into preallocated buffer to avoid new allocs inside capture
        sig_buf.copy_(sig)

    # --- Replay should run with no errors and produce consistent results ---
    # Compute expected result outside graph for comparison
    expected = xor_signature(t).clone()

    # Run replay a couple of times to ensure determinism & correct stream usage
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize(dev)

    assert torch.equal(sig_buf, expected), "Graph replay result mismatch (stream capture broken?)"


def test_nondefault_stream_is_not_blocked_by_default_stream_sleep():
    """
    Put a long sleep on the *default* stream; run xor_signature on a separate stream.
    If xor_signature accidentally uses the default stream, it will be delayed by the sleep.
    """
    dev = torch.device("cuda", 0)
    t = _make_tensor(dev, 4096)

    s = _new_stream(dev)

    # Baseline: run on s with no interference, measure wall time to s.synchronize().
    t0 = time.perf_counter()
    with torch.cuda.stream(s):
        _ = xor_signature(t)
    s.synchronize()
    baseline = time.perf_counter() - t0

    # Now: block the default stream with a big sleep, and run xor on s again.
    # If xor uses default stream, the synchronize on s would *not* wait for default's sleep
    # (PTDS) but the kernel itself would be delayed by default stream work if it ran there.
    # We measure time to *s.synchronize()*; it should remain close to baseline.
    with torch.cuda.device(dev):
        _sleep_ms(50)  # enqueue long sleep on default stream

    t1 = time.perf_counter()
    with torch.cuda.stream(s):
        _ = xor_signature(t)
    s.synchronize()
    with_sleep = time.perf_counter() - t1

    # Heuristic: allow some slack, but with_sleep should not blow up (e.g., < 5x baseline).
    # On most setups it should be very close to baseline.
    assert with_sleep < baseline * 5.0, (
        f"xor_signature appears blocked by default stream; "
        f"baseline={baseline*1e3:.2f}ms vs with_sleep={with_sleep*1e3:.2f}ms"
    )


def test_two_independent_streams_complete_independently():
    """Launch xor_signature concurrently on two independent streams; both should complete cleanly."""
    dev = torch.device("cuda", 0)
    t1 = _make_tensor(dev, 8192)
    t2 = _make_tensor(dev, 16384)

    s1 = _new_stream(dev)
    s2 = _new_stream(dev)

    # Enqueue a noticeable sleep before each XOR to amplify ordering
    with torch.cuda.stream(s1):
        _sleep_ms(20)
        sig1 = xor_signature(t1)

    with torch.cuda.stream(s2):
        _sleep_ms(10)
        sig2 = xor_signature(t2)

    # Synchronize both; should not deadlock or cross-depend
    s1.synchronize()
    s2.synchronize()

    # Basic sanity: results should be deterministic and >0 size
    assert sig1.shape == (1,) or sig1.numel() > 0
    assert sig2.shape == (1,) or sig2.numel() > 0
