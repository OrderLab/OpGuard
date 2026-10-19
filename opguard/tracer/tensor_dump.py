import math
import torch
from collections import defaultdict
import threading
import numpy as np
from opguard.tracer.xor_signature import xor_signature, flat_word_layout
from opguard.common_vars import ENABLE_CHECKSUM_NONCONTIG, OPGUARD_XOR_REGIONAL
from opguard.utils.utils import placeholder_sig_from_meta

def _safe_flatten(t: torch.Tensor) -> torch.Tensor:
    # try:
    #     return t.view(-1)
    # except RuntimeError as e:
    #     if "view size is not compatible" in str(e):
    #         return t.contiguous().view(-1)
    #     raise
    return t.view(-1)

# def safe_mean(t: torch.Tensor) -> torch.Tensor:
#     if t.dtype.is_floating_point or t.dtype.is_complex:
#         return t.mean()
#     return t.sum().to(torch.float32) / t.numel()

def _can_view_as_bytes(t: torch.Tensor) -> bool:
    if t.dim() == 0:
        return True
    
    try:
        # Fast, zero-copy check: only succeeds when last stride == 1 and layout is dense
        _ = t.view(torch.uint8)
        return True
    except RuntimeError as e:
        # Typical message: “stride(-1) must be 1 to view Float as Byte …”
        return False


# --- helpers for summarize_tensor ---

def _is_meta_like(t: torch.Tensor) -> bool:
    # meta device or FakeTensor (from torch.compile / FakeTensorMode)
    if getattr(t, "is_meta", False) or t.device.type == "meta":
        return True
    return type(t).__name__ == "FakeTensor" or hasattr(t, "fake_mode")



def summarize_tensor(t: torch.Tensor, verbosity="low", ignore_cpu=True) -> dict:
    if not isinstance(t, torch.Tensor):
        return {"type": str(type(t))}

    # Skip large CPU tensors if requested
    if t.device.type == "cpu" and ignore_cpu:
        if t.numel() >= 256 * 256:
            return {"type": "cpu"}

    # Meta/Fake tensors: no storage → only metadata + placeholder signature
    if _is_meta_like(t):
        return {
            "type": "meta",
            "shape": list(t.shape),
            "stats": {
                "dtype": str(t.dtype),
                "device": str(t.device),
                "numel": t.numel(),
                "xor_signature": placeholder_sig_from_meta(t),  # uint32[1]
                "xor_sig_mode": "placeholder-meta",
            },
            "requires_grad": bool(getattr(t, "requires_grad", False)),
            "is_leaf": bool(getattr(t, "is_leaf", False)),
        }

    base = {
        "shape": list(t.shape),
        "stats": {
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": t.numel(),
        },
    }

    # If we can't byte-view and noncontig checksums are disabled, return early
    if not _can_view_as_bytes(t) and not ENABLE_CHECKSUM_NONCONTIG:
        base["type"] = "non-contiguous"
        return base

    # XOR signature (robust to meta via try/fallback)
    try:
        regions = max(1, OPGUARD_XOR_REGIONAL)
        base["stats"]["xor_signature"] = xor_signature(t, regional=regions, by="words")
        base["stats"]["xor_sig_mode"] = "bytes"
        base["stats"]["xor_sig_layout"] = flat_word_layout(t.numel() * t.element_size(), regions)
    except RuntimeError as e:
        # e.g., “Cannot copy out of meta tensor” if something upstream passed Fake/Meta
        base["stats"]["xor_signature"] = placeholder_sig_from_meta(t)
        base["stats"]["xor_sig_mode"] = "placeholder-meta"

    if verbosity == "low":
        return base

    # Safe flatten without forcing host copies; keep counts as scalars
    flat = _safe_flatten(t)  # your util; should avoid materializing huge CPU copies
    try:
        nan_c = torch.isnan(flat).sum().item()
    except Exception:
        nan_c = None
    try:
        inf_c = torch.isinf(flat).sum().item()
    except Exception:
        inf_c = None
    try:
        zero_c = (flat == 0).sum().item()
    except Exception:
        zero_c = None
    try:
        neg_c = (flat < 0).sum().item()
    except Exception:
        neg_c = None

    base["stats"].update({
        "nan_count": nan_c,
        "inf_count": inf_c,
        "zero_count": zero_c,
        "neg_count":  neg_c,
    })
    base.update({
        "requires_grad": t.requires_grad,
        "is_leaf": t.is_leaf,
    })

    if verbosity == "medium":
        return base

    # High-verbosity extras (wrap in try to avoid accidental sync/copy failures)
    try:
        base.update({
            "element_size": t.element_size(),
            "memory_bytes": t.numel() * t.element_size(),
            "entropy": tensor_entropy(flat),
            "histogram": tensor_histogram(flat, bins=10),
            "regional_blocks": detect_2d_tile_anomalies(t, tile_h=128, tile_w=128),
            "samples": sample_tensor_regions(t),
        })
    except Exception as e:
        base["high_verbosity_error"] = str(e)

    return base



def finalize_summary(summary):
    stats = summary.get("stats", {})
    finalized = {}
    for k, v in stats.items():
        finalized[k] = (v.cpu().item() if v.numel() == 1 else v.cpu().tolist()) if torch.is_tensor(v) else v
    summary["stats"] = finalized
    return summary

def sample_tensor_regions(t, num=2):
    result = {}
    if t.dim() == 2:
        H, W = t.shape
        center_h = H // 2
        center_w = W // 2

        result["top_rows"] = t[:num, :].tolist()
        result["center_rows"] = t[center_h - num // 2 : center_h + num // 2, :].tolist()
        result["bottom_rows"] = t[-num:, :].tolist()

        result["left_cols"] = t[:, :num].tolist()
        result["center_cols"] = t[:, center_w - num // 2 : center_w + num // 2].tolist()
        result["right_cols"] = t[:, -num:].tolist()
    else:
        flat = t.view(-1).float()
        result["flat_samples"] = flat[torch.randint(0, flat.numel(), (10,))].tolist()
    return result

def detect_2d_tile_anomalies(t, tile_h=128, tile_w=128):
    if t.dim() != 2:
        return []

    H, W = t.shape
    regions = []

    for i in range(0, H, tile_h):
        for j in range(0, W, tile_w):
            tile = t[i:i + tile_h, j:j + tile_w].float().view(-1)
            if tile.numel() == 0:
                continue

            stats = torch.stack([
                tile.mean(), tile.std(), tile.min(), tile.max(),
                torch.isnan(tile).sum(), torch.isinf(tile).sum(),
                (tile == 0).sum()
            ]).cpu()
            mean, std, min_val, max_val, nan_count, inf_count, zero_count = stats.tolist()

            zero_ratio = zero_count / tile.numel()

            summary = {
                "pos": [i, j],
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "nan_count": int(nan_count),
                "inf_count": int(inf_count),
                "zero_ratio": zero_ratio
            }

            # Anomaly detection rules
            if zero_ratio > 0.95:
                summary["anomaly"] = "mostly zero"
            elif nan_count > 0:
                summary["anomaly"] = "contains NaN"
            elif std < 1e-6 and abs(mean) > 1e3:
                summary["anomaly"] = "flat high value"
            elif std > 1e3:
                summary["anomaly"] = "high std"

            if "anomaly" in summary:
                regions.append(summary)

    return regions



def tensor_entropy(flat, bins=100):
    clean_flat = flat[torch.isfinite(flat)]
    if clean_flat.numel() == 0:
        return None  # Can't compute entropy on all-NaN/Inf

    min_val, max_val = float(clean_flat.min()), float(clean_flat.max())
    if min_val == max_val:
        return 0.0  # Perfectly flat distribution

    hist = torch.histc(clean_flat, bins=bins, min=min_val, max=max_val)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = -torch.sum(prob * torch.log2(prob))
    return float(entropy.item())

def tensor_histogram(flat, bins=10):
    clean_flat = flat[torch.isfinite(flat)]
    if clean_flat.numel() == 0:
        return {"bins": [], "edges": []}

    min_val, max_val = float(clean_flat.min()), float(clean_flat.max())
    if min_val == max_val:
        return {
            "bins": [clean_flat.numel()] + [0] * (bins - 1),
            "edges": [min_val] * (bins + 1)
        }

    hist = torch.histc(clean_flat, bins=bins, min=min_val, max=max_val)
    edges = torch.linspace(min_val, max_val, steps=bins + 1)
    return {
        "bins": hist.tolist(),
        "edges": edges.tolist()
    }
