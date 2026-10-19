from __future__ import annotations
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

def list_pairs(base1: Path, base2: Path):
    base1, base2 = base1.resolve(), base2.resolve()
    pairs: List[FilePair] = []; only1: List[Path] = []; only2: List[Path] = []
    b2map: Dict[str, Path] = {}
    n2 = len(str(base2)) + 1
    for q in base2.rglob("*"):
        if q.is_file(): b2map[str(q.resolve())[n2:]] = q.resolve()
    n1 = len(str(base1)) + 1; matched = set()
    for p in base1.rglob("*"):
        if not p.is_file(): continue
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
