from __future__ import annotations
from bisect import bisect_right
from typing import Dict, List, Tuple

from .types import Item, Key
from .config import BAR_WIDTH_TICKS

def _safe_ctr(v):
    return int(v) if isinstance(v, int) else 10**15

def compute_global_schedule(
    items1: List[Item],
    items2: List[Item],
    map12: Dict[Key, Key],
    *,
    run2_offset: int,
    bar_width: int = BAR_WIDTH_TICKS,
    gap: int = 0,
) -> Tuple[Dict[Key, int], Dict[Key, int]]:
    order1 = sorted(items1, key=lambda it: (_safe_ctr(it.counter), it.key.rel, it.key.idx))
    order2 = sorted(items2, key=lambda it: (_safe_ctr(it.counter), it.key.rel, it.key.idx))

    ts1: Dict[Key, int] = {}; ts2: Dict[Key, int] = {}
    tbar = 0
    for it in order1:
        ts1[it.key] = tbar
        tbar += bar_width + gap
        if it.matched:
            mate = map12[it.key]
            ts2[mate] = ts1[it.key] + run2_offset

    MAX_ITERS = 12
    for _ in range(MAX_ITERS):
        changed = False
        tbar2 = 0
        for it in order2:
            cur = ts2.get(it.key, 0)
            need = max(cur, tbar2)
            if (it.key not in ts2) or (need != cur):
                ts2[it.key] = need
                changed = True
            tbar2 = need + bar_width + gap
            if it.matched and it.mate is not None:
                req1 = ts2[it.key] - run2_offset
                cur1 = ts1.get(it.mate, 0)
                if req1 > cur1: ts1[it.mate] = req1; changed = True

        tbar1 = 0
        for it in order1:
            cur = ts1.get(it.key, 0)
            need = max(cur, tbar1)
            if (it.key not in ts1) or (need != cur):
                ts1[it.key] = need
                changed = True
            tbar1 = need + bar_width + gap
            if it.matched and it.mate is not None:
                mate = map12[it.key]
                req2 = ts1[it.key] + run2_offset
                cur2 = ts2.get(mate, 0)
                if req2 > cur2: ts2[mate] = req2; changed = True
        if not changed: break

    def compact(order: List[Item], ts: Dict[Key, int]) -> None:
        bar = 0
        for it in order:
            cur = ts.get(it.key, 0)
            if cur < bar:
                cur = bar
                ts[it.key] = cur
            bar = cur + bar_width + gap

    compact(order1, ts1); compact(order2, ts2)
    return ts1, ts2

def shrink_global_idle(
    ts1: Dict[Key, int],
    ts2: Dict[Key, int],
    items1: List[Item],
    items2: List[Item],
    *,
    bar_width: int = BAR_WIDTH_TICKS,
    gap: int = 0,
):
    intervals: List[Tuple[int, int]] = []
    w = bar_width + gap
    for it in items1:
        s = ts1[it.key]; intervals.append((s, s + w))
    for it in items2:
        s = ts2[it.key]; intervals.append((s, s + w))
    if not intervals: return ts1, ts2
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e: cur_e = max(cur_e, e)
        else: merged.append((cur_s, cur_e)); cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    gaps: List[Tuple[int, int]] = []
    for (s1, e1), (s2, e2) in zip(merged, merged[1:]):
        if s2 > e1: gaps.append((e1, s2))
    if not gaps: return ts1, ts2

    gap_ends: List[int] = []; cum: List[int] = []; acc = 0
    for a, b in gaps:
        acc += (b - a); gap_ends.append(b); cum.append(acc)

    def _shift(t: int) -> int:
        i = bisect_right(gap_ends, t) - 1
        return t - (cum[i] if i >= 0 else 0)

    ts1c = {k: _shift(v) for k, v in ts1.items()}
    ts2c = {k: _shift(v) for k, v in ts2.items()}
    return ts1c, ts2c
