from __future__ import annotations
import os, re
from typing import Any, Dict, List, Optional, Tuple

from .config import STACK_EXCLUDE_FUNC_GLOBS, STACK_TRACE_MAX_DEPTH, SCALE_NS_PER_TICK, BAR_WIDTH_TICKS
from .utils import _emit_thread_names

_STACK_LINE_RE = re.compile(r"^(.*?):(\d+)\s+in\s+(.*)$")
STACK_CAT = "opguard:stack"

# Source file collector (lazy import to avoid circular dependencies)
_source_collector = None

def _get_source_collector(output_base: Optional[str] = None):
    """Get or create source file collector."""
    global _source_collector
    if _source_collector is None:
        try:
            from opguard.tracer.source_files import _collector
            # Use the global collector if it exists (set during trace export)
            if _collector is not None:
                _source_collector = _collector
            elif output_base:
                # Fallback: try to get collector by output_base
                from opguard.tracer.source_files import get_collector
                _source_collector = get_collector(output_base)
        except Exception:
            pass
    return _source_collector

def _parse_stack_trace_single_thread(raw: Optional[str], output_base: Optional[str] = None) -> List[Tuple[str, int, str, Optional[str], Optional[str]]]:
    """
    Parse stack trace and return frames with file path information.
    
    Returns:
        List of tuples: (short_file, line_num, func_name, original_path, source_file_path)
    """
    if not raw or not isinstance(raw, str): return []
    frames: List[Tuple[str, int, str, Optional[str], Optional[str]]] = []
    collector = _get_source_collector(output_base)
    
    for line in (raw.strip().splitlines() if raw else []):
        s = line.strip()
        if not s: continue
        m = _STACK_LINE_RE.match(s)
        if m:
            path, ln, fn = m.group(1), m.group(2), m.group(3)
            short = os.path.basename(path) or path
            original_path = path if path and path not in ("<unknown>", "?") else None
            source_file_path = None
            
            # Get saved source file path if collector is available
            if collector and original_path:
                source_file_path = collector.get_file_mapping(original_path)
            
            try: 
                line_num = int(ln)
                frames.append((short, line_num, fn, original_path, source_file_path))
            except Exception: 
                frames.append((short, -1, fn, original_path, source_file_path))
        else:
            frames.append(("?", -1, s, None, None))

    def _keep_func(func: str) -> bool:
        from fnmatch import fnmatchcase
        for pat in STACK_EXCLUDE_FUNC_GLOBS:
            if fnmatchcase(func, pat): return False
        return True

    frames = [(f, l, fn, orig, src) for (f, l, fn, orig, src) in frames if _keep_func(str(fn))]
    return frames

def _emit_B(events: List[dict], pid: int, tid: str, ts_ticks: int, file: str, line: int, func: str, depth: int, run_side: str, rank: int, original_path: Optional[str] = None, source_file_path: Optional[str] = None, function_context: Optional[Dict[str, any]] = None):
    args = {
        "file": file, 
        "line": line, 
        "func": func, 
        "depth": depth, 
        "run": run_side, 
        "rank": rank
    }
    
    # Add file path information
    if original_path:
        args["original_path"] = original_path
    if source_file_path:
        args["source_file"] = source_file_path
    
    # Add function context if available
    if function_context:
        args["function_context"] = function_context
    
    events.append({
        "name": f"{func}  ·  {file}:{line}" if line >= 0 else f"{func}  ·  {file}",
        "cat": STACK_CAT, "ph": "B",
        "ts": int(ts_ticks * SCALE_NS_PER_TICK),
        "pid": pid, "tid": tid,
        "args": args,
    })

def _emit_E(events: List[dict], pid: int, tid: str, ts_ticks: int):
    events.append({"ph": "E", "ts": int(ts_ticks * SCALE_NS_PER_TICK), "pid": pid, "tid": tid})

class StackTrackSingleThread:
    def __init__(self, output_base: Optional[str] = None):
        self._open: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self._last_end: Dict[Tuple[str, int], int] = {}
        self._named: set[Tuple[int, str]] = set()
        self.output_base = output_base

    def _ensure_named(self, events: List[dict], pid: int, tid: str, label: str):
        key = (pid, tid)
        if key not in self._named:
            _emit_thread_names(events, pid, tid, label)
            self._named.add(key)

    def _close_to(self, events: List[dict], pid: int, tid: str, key: Tuple[str, int], close_ts: int):
        stack = self._open.get(key)
        if not stack: return
        for _ in range(len(stack)-1, -1, -1):
            _emit_E(events, pid, tid, close_ts)
        stack.clear()

    def feed(self, events: List[dict], pid: int, run_side: str, rank: int, ts_ticks: int, dur_ticks: int, raw_stack: Optional[str]):
        frames = _parse_stack_trace_single_thread(raw_stack, self.output_base)
        lane = (run_side, rank)
        tid = f"gpu_{rank}:{run_side}:stack"
        self._ensure_named(events, pid, tid, f"{run_side} · rank {rank} · stack (nested)")
        start = ts_ticks; end = ts_ticks + max(1, int(dur_ticks))
        last_end = self._last_end.get(lane)
        gap = (last_end is not None and start > last_end)

        cur = self._open.get(lane, [])
        lcp = 0
        while lcp < len(cur) and lcp < len(frames):
            f = cur[lcp]; nf_file, nf_line, nf_func, nf_orig, nf_src = frames[lcp]
            same_frame = (f["file"] == nf_file and f["line"] == nf_line and f["func"] == nf_func)
            contiguous = (f["end"] == start) or (start >= f["end"])
            if not (same_frame and contiguous): break
            f["end"] = end; lcp += 1

        for _ in range(len(cur) - 1, lcp - 1, -1):
            _emit_E(events, pid, tid, start)
            cur.pop()

        # Get source collector for function context
        collector = _get_source_collector(self.output_base)

        for depth in range(lcp, len(frames)):
            file_s, line_s, func_s, orig_path, src_path = frames[depth]
            
            # Get function context if available
            function_context = None
            if collector and orig_path and line_s > 0:
                function_context = collector.get_function_context(orig_path, line_s)
            
            _emit_B(events, pid, tid, start, file_s, line_s, func_s, depth, run_side, rank, 
                   original_path=orig_path, source_file_path=src_path, function_context=function_context)
            cur.append({"file": file_s, "line": line_s, "func": func_s, "start": start, "end": end, 
                       "original_path": orig_path, "source_file": src_path})

        self._open[lane] = cur
        self._last_end[lane] = end

    def flush_all(self, events: List[dict]):
        for (run_side, rank), stack in list(self._open.items()):
            tid = f"gpu_{rank}:{run_side}:stack"
            end_ts = self._last_end.get((run_side, rank), None)
            if end_ts is None: continue
            for _ in range(len(stack)-1, -1, -1):
                _emit_E(events, 0 if run_side == "run1" else 1, tid, end_ts)
        self._open.clear()
