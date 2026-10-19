# xor_signature.py

import torch, numpy as np
import sys
from opguard.utils.utils import placeholder_sig_from_meta

_xor_load_error = "extension disabled"
try:
    import xor_reduce_cuda as _xor_reduce_cuda_ext
except Exception as exc:
    _xor_reduce_cuda_ext = None
    _xor_load_error = str(exc)

_fallback_reported = False


def _report_cuda_fallback(reason):
    global _fallback_reported
    if not _fallback_reported:
        print(f"[OpGuard] Native XOR unavailable ({reason}); using tensor operations for GPU fingerprints. "
              "This adds GPU work and temporary memory. Run `opguard doctor` for installation details.", file=sys.stderr)
        _fallback_reported = True

def _is_meta_like(t: torch.Tensor) -> bool:
    if getattr(t, "is_meta", False):  # new PyTorch
        return True
    if t.device.type == "meta":
        return True
    # FakeTensor: avoid hard import; detect by name/attr
    return type(t).__name__ == "FakeTensor" or hasattr(t, "fake_mode")

def _single_sig_cuda(byte_1d_u8: torch.Tensor) -> torch.Tensor:
    if _xor_reduce_cuda_ext is not None:
        try:
            sig_i32 = _xor_reduce_cuda_ext.xor_reduce_cuda(byte_1d_u8)  # int32[1], launches on current stream
            try:
                return sig_i32.view(torch.uint32)
            except RuntimeError:
                return sig_i32.to(torch.uint32)
        except Exception as exc:
            _report_cuda_fallback(str(exc))
    else:
        _report_cuda_fallback(_xor_load_error)

    # Fallback when extension is unavailable: compute XOR reduction on current device/stream.
    n = int(byte_1d_u8.numel())
    if n == 0:
        return torch.zeros(1, dtype=torch.uint32, device=byte_1d_u8.device)
    pad = (-n) & 3
    if pad:
        byte_1d_u8 = torch.cat(
            [byte_1d_u8, torch.zeros(pad, dtype=torch.uint8, device=byte_1d_u8.device)],
            dim=0,
        )
    # CUDA does not implement shifts/reductions for uint32 on supported Torch builds.
    words = byte_1d_u8.view(-1, 4).to(torch.int64)
    acc = words[:, 0] | (words[:, 1] << 8) | (words[:, 2] << 16) | (words[:, 3] << 24)
    while acc.numel() > 1:
        if acc.numel() & 1:
            acc = torch.cat([acc, torch.zeros(1, dtype=acc.dtype, device=acc.device)], dim=0)
        acc = torch.bitwise_xor(acc[0::2], acc[1::2])
    return acc[:1].to(torch.uint32)

def flat_word_layout(nbytes: int, regions: int = 4) -> dict:
    """Fixed logical byte ranges [lo, hi), independent of shape and launch grid.

    Pack little-endian uint32 words, zero-padding only the final partial word.
    Divide words evenly; earlier regions receive the remainder. Empty regions
    remain present, so scalars and empty tensors still have R signature entries.
    """
    if type(regions) is not int or not 1 <= regions <= 64:
        raise ValueError("regions must be an integer between 1 and 64")
    base, rem = divmod((nbytes + 3) // 4, regions)
    ranges, start = [], 0
    for r in range(regions):
        end = start + base + (r < rem)
        ranges.append([min(4 * start, nbytes), min(4 * end, nbytes)])
        start = end
    return {"scheme": "flat-u32-le-v1", "regions": regions,
            "logical_nbytes": nbytes, "byte_ranges": ranges}


def _flat_word_signature(tensor: torch.Tensor, regions: int) -> torch.Tensor:
    layout = flat_word_layout(tensor.numel() * tensor.element_size(), regions)
    if _is_meta_like(tensor):
        return placeholder_sig_from_meta(tensor)[0].expand(regions).clone()
    raw = tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    if not tensor.is_cuda:
        data = raw.cpu().numpy()
        data = np.pad(data, (0, (-data.size) % 4))
        words = data.view('<u4')
        values = [np.bitwise_xor.reduce(words[lo // 4:(hi + 3) // 4], initial=np.uint32(0))
                  if hi > lo else np.uint32(0) for lo, hi in layout['byte_ranges']]
        return torch.from_numpy(np.asarray(values, dtype=np.uint32))

    raw.record_stream(torch.cuda.current_stream(tensor.device))
    if _xor_reduce_cuda_ext is not None and hasattr(_xor_reduce_cuda_ext, 'xor_reduce_regional_cuda'):
        return _xor_reduce_cuda_ext.xor_reduce_regional_cuda(raw, regions).view(torch.uint32)

    _report_cuda_fallback("regional extension missing; install a matching native wheel or rebuild")

    # Correct fallback for absent/older extensions. Reduce all regions together
    # without host reads or scalar .item() calls. Native builds avoid these ops.
    nbytes = raw.numel()
    if nbytes == 0:
        return torch.zeros(regions, dtype=torch.uint32, device=raw.device)
    if nbytes % 4:
        raw = torch.cat((raw, torch.zeros((-nbytes) % 4, dtype=torch.uint8, device=raw.device)))
    packed = raw.reshape(-1, 4).to(torch.int64)
    words = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16) | (packed[:, 3] << 24)
    width = (words.numel() + regions - 1) // regions
    acc = torch.zeros((regions, width), dtype=torch.int64, device=raw.device)
    for r, (lo, hi) in enumerate(layout['byte_ranges']):
        part = words[lo // 4:(hi + 3) // 4] if hi > lo else words[:0]
        acc[r, :part.numel()].copy_(part)
    while acc.shape[1] > 1:
        if acc.shape[1] % 2:
            acc = torch.cat((acc, torch.zeros((regions, 1), dtype=acc.dtype, device=raw.device)), dim=1)
        acc = torch.bitwise_xor(acc[:, 0::2], acc[:, 1::2])
    return acc[:, 0].to(torch.uint32)


def xor_signature(
    tensor: torch.Tensor,
    regional: int | None = None,
    by: str = "rows",
    threshold: int = 512,  # Only do regional when rows >= threshold
) -> torch.Tensor:
    """
    XOR signature(s) over the *bytes* of `tensor`.

    Low-level default (regional=None): uint32[1], retained for compatibility.
    Diagnostic summaries explicitly use by='words', regional=4 by default.
    by='words' uses fixed flat-word regions with no size/row threshold.
    If regional=k and the first dim (rows) >= threshold: returns uint32[k],
      each the XOR over ~1/k of the rows.
    If rows < threshold: falls back to a single signature.

    Args:
      tensor: any shape/dtype; will be made contiguous if needed.
      regional: None or positive int (e.g., 16). Default off.
      by: 'words' (flat little-endian uint32 stream) or legacy 'rows'.
      threshold: minimum number of rows (dim 0) to enable regional splitting.
    """
    if by == "words":
        return _flat_word_signature(tensor, 1 if regional is None else regional)
    from opguard.common_vars import OPGUARD_XOR_REGIONAL_THRESHOLD
    if OPGUARD_XOR_REGIONAL_THRESHOLD > 0: threshold = OPGUARD_XOR_REGIONAL_THRESHOLD

    # Guard meta/fake up front
    if _is_meta_like(tensor):
        # For regional requests, still honor shape-based partition count
        if regional is None:
            return placeholder_sig_from_meta(tensor)
        rows = int(tensor.size(0)) if tensor.dim() > 0 else 1
        if rows < threshold:
            return placeholder_sig_from_meta(tensor)
        R = int(max(1, min(regional or 1, rows)))
        # Emit R identical placeholders derived from full tensor metadata (cheap, deterministic)
        h = placeholder_sig_from_meta(tensor)[0]
        return h.expand(R).clone()

    # scalar/contiguity handling
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Whole-tensor fast path helper
    def _whole_tensor_sig() -> torch.Tensor:
        byte_1d = tensor.view(torch.uint8).view(-1)
        if byte_1d.numel() == 0:
            return torch.zeros(1, dtype=torch.uint32, device=tensor.device)
        if tensor.is_cuda:
            s = torch.cuda.current_stream(tensor.device)
            # lifetime protection only; no context switch
            tensor.record_stream(s)
            return _single_sig_cuda(byte_1d)
        else:
            # CPU path – may fail if Fake/meta slipped through; catch & fallback
            try:
                arr = byte_1d.cpu().numpy()
            except RuntimeError:
                return placeholder_sig_from_meta(tensor).cpu()
            pad = (-arr.size) & 3
            if pad:
                arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
            if arr.size == 0:
                return torch.zeros(1, dtype=torch.uint32, device=tensor.device)
            a = arr.reshape(-1, 4).astype(np.uint32, copy=False)
            u32 = (a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16) | (a[:, 3] << 24)).astype(np.uint32, copy=False)
            return torch.tensor([np.bitwise_xor.reduce(u32).astype(np.uint32)], dtype=torch.uint32, device=tensor.device)

    if regional is None:
        return _whole_tensor_sig()

    # Regional requested: only proceed if by='rows' and rows >= threshold.
    assert by == "rows", "Only by='rows' is supported."
    rows = tensor.size(0)
    if rows < threshold:
        # Too few rows → fallback to single signature
        return _whole_tensor_sig()

    # Reshape to [rows, cols] so rows are contiguous
    cols = int(tensor.numel() // max(rows, 1)) if rows > 0 else 0
    t2d = tensor.view(rows, cols)
    byte_1d = t2d.view(torch.uint8).view(-1)
    if rows == 0 or byte_1d.numel() == 0:
        return torch.zeros(int(regional), dtype=torch.uint32, device=tensor.device)

    R = int(regional or 1)
    if R < 1: raise ValueError("regional must be positive")
    R = min(R, rows)

    bytes_per_row = cols * tensor.element_size()
    base, rem = divmod(rows, R)

    if tensor.is_cuda:
        dev = tensor.device
        s = torch.cuda.current_stream(dev)
        tensor.record_stream(s)
        out = torch.empty(R, dtype=torch.uint32, device=dev)
        start_row = 0
        for i in range(R):
            cnt = base + (1 if i < rem else 0)
            nbytes = cnt * bytes_per_row
            off = start_row * bytes_per_row
            region_bytes = byte_1d.narrow(0, off, nbytes)
            out[i] = _single_sig_cuda(region_bytes)[0]
            start_row += cnt
        return out
    else:
        try:
            arr = byte_1d.cpu().numpy()
        except RuntimeError:
            # CPU meta/fake fallback
            h = placeholder_sig_from_meta(tensor)[0].cpu()
            return h.expand(R).clone()
        out_np = np.empty(R, dtype=np.uint32)
        start_row = 0
        for i in range(R):
            cnt = base + (1 if i < rem else 0)
            nbytes = cnt * bytes_per_row
            off = start_row * bytes_per_row
            slice_u8 = arr[off: off + nbytes]
            pad = (-nbytes) & 3
            if pad:
                slice_u8 = np.concatenate([slice_u8, np.zeros(pad, dtype=np.uint8)])
            if slice_u8.size == 0:
                out_np[i] = np.uint32(0)
            else:
                a = slice_u8.reshape(-1, 4).astype(np.uint32, copy=False)
                u32 = (a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16) | (a[:, 3] << 24)).astype(np.uint32, copy=False)
                out_np[i] = np.bitwise_xor.reduce(u32).astype(np.uint32)
            start_row += cnt
        return torch.from_numpy(out_np)
