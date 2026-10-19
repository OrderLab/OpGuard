from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class FilePair:
    p1: Path
    p2: Path
    rel: str

@dataclass(frozen=True)
class Key:
    side: str   # "run1" | "run2"
    rel: str
    idx: int

@dataclass
class Item:
    key: Key
    counter: Optional[int]
    rank: int
    matched: bool
    mate: Optional[Key]
    alignment_basis: Optional[str] = None
    unmatched_reason: Optional[str] = None
