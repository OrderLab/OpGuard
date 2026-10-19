"""Skip instrumenting third-party internals that wrappers cannot safely rewrite.

Wrapping torch.nn.functional.dropout makes @torch.jit.script fail at import.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_PKG_MARKERS = ("site-packages", "dist-packages")
# Rewrite these internals and training can fail at import (torch JIT) or
# change compiled kernel behavior.
_SKIP_PKGS = ("torch", "torchgen")

_TORCH_NN_FUNCTIONAL_ALIASES = {
    "F.dropout",
    "F.dropout_",
}


def instrument_unsafe_allowed() -> bool:
    return os.getenv("OPGUARD_INSTRUMENT_UNSAFE", "0") == "1" or os.getenv("OPGUARD_INSTRUMENT_TORCH", "0") == "1"


def instrument_torch_allowed() -> bool:
    return instrument_unsafe_allowed()


def should_skip_instrument_path(file_path: str) -> bool:
    """True for protected packages under site-packages."""
    if instrument_unsafe_allowed() or not file_path:
        return False
    parts = Path(str(file_path).replace("\\", "/")).parts
    if not any(p in _PKG_MARKERS for p in parts):
        return False
    extra = os.getenv("OPGUARD_SKIP_PACKAGES", "")
    packages = _SKIP_PKGS + tuple(p.strip() for p in extra.split(",") if p.strip())
    return any(p in packages for p in parts)


def should_skip_torch_nn_functional_call(dotted_name: str) -> bool:
    """Protect the known scripted dropout helpers without dropping all F.* calls."""
    if instrument_torch_allowed() or not dotted_name:
        return False
    if dotted_name in _TORCH_NN_FUNCTIONAL_ALIASES:
        return True
    return dotted_name in {"torch.nn.functional.dropout", "torch.nn.functional.dropout_"}


def filter_instrumentation_entries(
    entries: Iterable[Dict],
) -> Tuple[List[Dict], List[str]]:
    kept: List[Dict] = []
    skipped: List[str] = []
    for item in entries:
        fp = item.get("file_path") or ""
        if should_skip_instrument_path(fp):
            skipped.append(fp)
            continue
        kept.append(item)
    return kept, skipped
