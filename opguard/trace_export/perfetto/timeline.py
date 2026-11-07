from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List

from .types import FilePair

def sort_records(seq: List[dict]) -> List[dict]:
    keyed = []
    for idx, r in enumerate(seq):
        c  = r.get("counter")
        ts = r.get("timestamp", "")
        c_key = (1, 0) if (c is None) else (0, int(c))
        keyed.append((c_key, ts, idx, r))
    keyed.sort(key=lambda x: (x[0], x[1], x[2]))
    return [r for *_, r in keyed]

def _trace_files(base: Path):
    for root, dirs, files in os.walk(base):
        # ops/full contains tensor dumps, not operator event records.
        if Path(root).name == "ops":
            dirs[:] = [name for name in dirs if name != "full"]
        for name in files:
            path = Path(root) / name
            if path.is_file():
                yield path


def list_pairs(base1: Path, base2: Path):
    base1, base2 = base1.resolve(), base2.resolve()
    pairs: List[FilePair] = []; only1: List[Path] = []; only2: List[Path] = []
    b2map: Dict[str, Path] = {}
    n2 = len(str(base2)) + 1
    for q in _trace_files(base2):
        b2map[str(q.resolve())[n2:]] = q.resolve()
    n1 = len(str(base1)) + 1; matched = set()
    for p in _trace_files(base1):
        rel = str(p.resolve())[n1:]
        q = b2map.get(rel)
        if q:
            pairs.append(FilePair(p.resolve(), q, rel)); matched.add(rel)
        else:
            only1.append(p.resolve())
    for rel, q in b2map.items():
        if rel not in matched:
            only2.append(q)
    return pairs, only1, only2
