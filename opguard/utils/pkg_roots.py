from __future__ import annotations
import importlib.util
import inspect
import pathlib
import traceback
from typing import Iterable, List

# Resolve package roots once (at import time)
def _resolve_pkg_roots(pkg_name: str) -> List[pathlib.Path]:
    roots: set[pathlib.Path] = set()
    spec = importlib.util.find_spec(pkg_name)
    if spec is not None:
        # package (folder) installs
        if spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                p = pathlib.Path(loc).resolve()
                if p.exists():
                    roots.add(p)
        # single-file modules / fallback
        if spec.origin:
            p = pathlib.Path(spec.origin).resolve().parent
            if p.exists():
                roots.add(p)
    try:
        # double-check via import & inspect
        mod = __import__(pkg_name)
        p = pathlib.Path(inspect.getfile(mod)).resolve().parent
        if p.exists():
            roots.add(p)
    except Exception:
        pass
    return sorted(roots)

_OPGUARD_ROOTS = _resolve_pkg_roots("opguard")

def _is_in_any_root(path: pathlib.Path, roots: Iterable[pathlib.Path]) -> bool:
    path = path.resolve()
    for root in roots:
        try:
            # robust “is descendant” check
            if root in path.parents or path == root:
                return True
        except Exception:
            pass
    return False

def filtered_stack_excluding_opguard() -> str:
    raw_stack = traceback.extract_stack()  # returns FrameSummary (filename, lineno, name, ...)
    parts = []
    for fr in raw_stack:
        p = pathlib.Path(fr.filename)
        if not _is_in_any_root(p, _OPGUARD_ROOTS):
            parts.append(f"{fr.filename}:{fr.lineno} in {fr.name}\n")
    return "".join(parts)
