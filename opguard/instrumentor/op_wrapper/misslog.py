# opguard/instrumentor/op_wrapper/misslog.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import os
import json

class MissLogger:
    """Collects and summarizes non-wrapped targets with de-dup and throttling.

    NOTE: This logger is now *lineno-agnostic*. We do not use or store line numbers
    to decide whether to log, dedup, or cap entries.

    Env knobs:
      - OPGUARD_MISS_MAX_PER_TARGET: cap total miss rows per (target,reason) bucket (default 1)
    """
    def __init__(self):
        self._missed: List[Dict[str, Any]] = []
        self._hit_keys = set()
        self._dedup = set()
        self._max_per = int(os.getenv("OPGUARD_MISS_MAX_PER_TARGET", "1") or 1)
        self._count_by_bucket = defaultdict(int)

    @staticmethod
    def tkey(t: Dict[str, Any]) -> Tuple:
        return (
            t.get("operation_type"),
            t.get("function"),
            t.get("file_path"),
            t.get("class"),
            tuple(sorted(t.get("tag_names", []) or [])),
        )

    def mark_hit(self, target: Optional[Dict[str, Any]]):
        if target is not None:
            self._hit_keys.add(self.tkey(target))

    # --- base shape ---
    def miss(self, reason: str, target: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
        rec = {
            "reason": reason,
            "target": {
                "operation_type": target.get("operation_type"),
                "function": target.get("function"),
                "file_path": target.get("file_path"),
                "class": target.get("class"),
                # line intentionally omitted (lineno-agnostic)
                "tag_names": target.get("tag_names", []),
            },
            "_file": target.get("file_path"),
        }
        if extra:
            rec["context"] = dict(extra)
        self._missed.append(rec)

    # --- dedup/throttle wrapper ---
    def miss_once(self, reason: str, target: dict, extra: dict | None = None):
        bucket_key = (self.tkey(target), reason)
        if self._max_per > 0 and self._count_by_bucket[bucket_key] >= self._max_per:
            return
        dedup_key = (self.tkey(target), reason, json.dumps(extra, sort_keys=True) if isinstance(extra, dict) else None)
        if dedup_key in self._dedup:
            return
        self._dedup.add(dedup_key)
        self._count_by_bucket[bucket_key] += 1
        self.miss(reason, target, extra)

    @property
    def missed(self) -> List[Dict[str, Any]]:
        return self._missed

    def compute_config_only(self, config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cfg_keys = {self.tkey(t): t for t in (config or [])}
        return [cfg_keys[k] for k in cfg_keys.keys() - self._hit_keys]

    # --- summaries & dumps ---
    def summary(self, config: List[Dict[str, Any]] | None, wrapped: List[Dict[str, Any]] | None) -> Dict[str, Any]:
        wrapped = wrapped or []
        cfg_only = self.compute_config_only(config or []) if config is not None else []
        return {
            "wrapped_count": len(wrapped),
            "missed_count": len(self._missed),
            "config_only_count": len(cfg_only),
            "wrapped": wrapped,
            "missed": list(self._missed),
            "config_only": cfg_only,
        }

    def dump(self, path: str, config: List[Dict[str, Any]] | None = None, wrapped: List[Dict[str, Any]] | None = None):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.summary(config, wrapped), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
