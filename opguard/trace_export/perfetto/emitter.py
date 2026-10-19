from __future__ import annotations
from typing import List, Dict, Any, Optional

from .config import (
    SCALE_NS_PER_TICK, BAR_WIDTH_TICKS,
    C_OK, C_DIFF, C_ANOM, C_INS, C_FIRST,
)
from .utils import status_tid, _emit_thread_names

class PerfettoEmitter:
    def __init__(self, event_mode: str = "flat"):
        self.event_mode = event_mode

    @staticmethod
    def _ev(name, cat, ph, ts, pid, tid, cname=None, args=None, dur=None):
        e = {"name": name, "cat": cat, "ph": ph, "ts": ts, "pid": pid, "tid": tid}
        if cname: e["cname"] = cname
        if args:  e["args"]  = args
        if ph == "X" and dur is not None:
            e["dur"] = dur
        return e

    def _emit_flat(self, events: List[dict], node: dict, ts_ticks: int, pid: int, tid: str, cname: str, leaf_args: dict):
        ts  = int(ts_ticks * SCALE_NS_PER_TICK)
        dur = int(max(1, BAR_WIDTH_TICKS) * SCALE_NS_PER_TICK)
        events.append(self._ev(node.get("__short", "call"), "opguard", "X", ts, pid, tid, cname, leaf_args, dur=dur))

    def _emit_stacked(self, events: List[dict], node: dict, ts_ticks: int, pid: int, tid: str, cname: str, leaf_args: dict):
        ts = int(ts_ticks * SCALE_NS_PER_TICK)
        te = int((ts_ticks + max(1, BAR_WIDTH_TICKS)) * SCALE_NS_PER_TICK)
        args = leaf_args if node.get("__is_leaf", True) else {"run": leaf_args.get("run"), "rank": leaf_args.get("rank"), "status": leaf_args.get("status")}
        events.append(self._ev(node.get("__short", "call"), "opguard", "B", ts, pid, tid, cname, args))
        events.append(self._ev(node.get("__short", "call"), "opguard", "E", te, pid, tid))

    def emit_call(self, events: List[dict], node: dict, ts_ticks: int, pid: int, tid: str, cname: str, leaf_args: dict):
        if self.event_mode == "flat":
            self._emit_flat(events, node, ts_ticks, pid, tid, cname, leaf_args)
        else:
            self._emit_stacked(events, node, ts_ticks, pid, tid, cname, leaf_args)

    @staticmethod
    def emit_flow(events, ts_src_ticks, pid_src, tid_src, ts_dst_ticks, pid_dst, tid_dst, fid):
        ts_src_start = int(ts_src_ticks * SCALE_NS_PER_TICK)
        ts_dst_start = int(ts_dst_ticks * SCALE_NS_PER_TICK)
        dur_ns       = int(BAR_WIDTH_TICKS * SCALE_NS_PER_TICK)

        ts_src_inside = ts_src_start + max(0, dur_ns - 1)
        ts_dst_inside = ts_dst_start

        events.append({"ph": "s", "ts": ts_src_inside, "pid": pid_src, "tid": tid_src,
                       "id": fid, "cat": "opguard:align", "name": "align", "bp": "e"})
        events.append({"ph": "f", "ts": ts_dst_inside, "pid": pid_dst, "tid": tid_dst,
                       "id": fid, "cat": "opguard:align", "name": "align", "bp": "s"})

def emit_status_slice(events, ts_ticks: int, pid: int, tid: str, status: str):
    ts  = int(ts_ticks * SCALE_NS_PER_TICK)
    dur = int(max(1, BAR_WIDTH_TICKS) * SCALE_NS_PER_TICK)
    events.append({
        "name": {"diff": "CANDIDATE DIFF", "ok": "RECORDED FIELDS AGREE",
                 "unaligned": "UNDETERMINED"}.get(status, status.upper()),
        "cat": "opguard:status",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid
    })
