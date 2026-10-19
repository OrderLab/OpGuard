# build.py
from __future__ import annotations
import gzip, json, time, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

@contextmanager
def _timed_step(step_name: str):
    """Context manager for timing a step."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"⏱️  {step_name}: {elapsed:.2f}s")

from .config import (
    DISPLAY_TIME_UNIT, DEFAULT_EVENT_MODE, RUN2_OFFSET_TICKS, BAR_WIDTH_TICKS,
    C_OK, C_DIFF, C_ANOM, C_INS, C_FIRST, DIFF_ATOMS_MAX,
)
from . import config
from .types import Key, Item
from .timeline import list_pairs, sort_records
from .align import align_records
from .schedule import compute_global_schedule, shrink_global_idle
from .scoring import compare_simple, detect_anomaly
from .utils import (
    _get, parse_rank_from_rel, rel_tid, stable_name_with_counter,
    io_summary_with_xor_values, signature_from_trace, status_tid, _emit_thread_names,
)
from .emitter import PerfettoEmitter, emit_status_slice
from .stack_track import StackTrackSingleThread
from opguard.utils.utils import load_traces_any
from opguard.tracer.source_files import load_source_files_from_runs


def _record_id(key: Key) -> str:
    return json.dumps([key.side, key.rel, key.idx], separators=(",", ":"))


def _leaf_args(node: dict, item: Item, status: str) -> dict:
    correspondence = {"state": "paired" if item.matched else "undetermined",
                      "uses_output_values": False, "is_data_dependency": False,
                      "validation": "unreviewed", "kind": "candidate"}
    if item.matched:
        correspondence.update(basis=item.alignment_basis, peer_record_id=_record_id(item.mate))
    else:
        correspondence["reason"] = item.unmatched_reason
    return {
        "run": item.key.side, "rank": item.rank, "status": status,
        "record_id": _record_id(item.key), "correspondence": correspondence,
        "file": node.get("fname"),
        "line": node.get("line_num") or _get(node, "meta", "lineno"),
        "counter": node.get("counter"), "fn": node.get("function"),
        "sig": signature_from_trace(node), "io": io_summary_with_xor_values(node),
        "timing": node.get("timing", {}),
        "comparison_evidence": "recorded_metadata_and_fingerprints",
        "tensor_equality": "not_established",
    }


def _counter(v):
    c = v.get("counter")
    return int(c) if isinstance(c, int) else None


def _operator_records(path, filter):
    # Source indexes and other sidecar metadata are not operator observations.
    return sort_records([record for record in load_traces_any(path, filter)
                         if isinstance(record.get("function"), str) and record["function"]])


def _collect_all_items_and_pairs(matched_sorted, only1, only2, base1, base2, filter=None):
    nodes, items1, items2, map12, map21 = {}, [], [], {}, {}
    for pair in matched_sorted:
        seq1, seq2 = _operator_records(pair.p1, filter), _operator_records(pair.p2, filter)
        alignment = align_records(seq1, seq2)
        left = {i: (j, basis) for i, j, basis in alignment.pairs}
        right = {j: (i, basis) for i, j, basis in alignment.pairs}
        for side, other, seq, mates, unknown, items in (
            ("run1", "run2", seq1, left, alignment.left_unresolved, items1),
            ("run2", "run1", seq2, right, alignment.right_unresolved, items2),
        ):
            for i, node in enumerate(seq):
                key = Key(side, pair.rel, i)
                nodes[key] = node
                match = mates.get(i)
                mate = Key(other, pair.rel, match[0]) if match else None
                items.append(Item(key, _counter(node), node.get("gpu_rank", parse_rank_from_rel(pair.rel)),
                                  mate is not None, mate, match[1] if match else None, unknown.get(i)))
                if mate is not None:
                    (map12 if side == "run1" else map21)[key] = mate
    for side, paths, base, items in (("run1", only1, base1, items1), ("run2", only2, base2, items2)):
        for path in sorted(paths):
            rel = str(path.relative_to(base))
            for i, node in enumerate(_operator_records(path, filter)):
                key = Key(side, rel, i)
                nodes[key] = node
                items.append(Item(key, _counter(node), node.get("gpu_rank", parse_rank_from_rel(rel)),
                                  False, None, unmatched_reason="no_corresponding_trace_file"))
    return nodes, items1, items2, map12, map21


def _safe_counter(v: Optional[int]) -> int:
    return int(v) if isinstance(v, int) else 10**15  # push Nones to the end

def _read_source_files_from_dir(source_files_dir: Path, embedded_source_files: Dict[str, Dict[str, Any]], seen_paths: set):
    """Read source files from a source_files directory."""
    index_file = source_files_dir / "file_index.json"
    
    if not index_file.exists():
        return
    
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            file_map = json.load(f)
        
        for original_path, (saved_path, file_hash) in file_map.items():
            # Skip if we've already loaded this file
            if original_path in seen_paths:
                continue
            
            # Read the actual source file
            saved_file = source_files_dir / Path(saved_path).name
            if saved_file.exists():
                try:
                    with open(saved_file, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    embedded_source_files[original_path] = {
                        "content": content,
                        "saved_path": str(saved_path),
                        "hash": file_hash
                    }
                    seen_paths.add(original_path)
                except Exception as e:
                    logging.warning(f"[OpGuard]: Failed to read source file {saved_file}: {e}")
    except Exception as e:
        logging.warning(f"[OpGuard]: Failed to read source file index from {index_file}: {e}")

def _read_source_files_from_runs(base1: Path, base2: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read source files directly from run directories' source_files/ folders.
    
    Handles two cases:
    1. If run_dir points to a gpu_XXX folder: look for source_files/ inside the gpu_XXX directory
    2. If run_dir points to the trial directory: look for source_files/ in that directory
    
    Returns:
        Dictionary mapping original_path -> {content, saved_path, hash}
    """
    embedded_source_files = {}
    seen_paths = set()
    
    for base_dir in [base1, base2]:
        if not base_dir.exists():
            logging.debug(f"[OpGuard]: Run directory does not exist: {base_dir}")
            continue
        
        # Check if this is a GPU-specific directory (e.g., .../trial_XXX/gpu_001)
        # If so, look for source_files inside the gpu_XXX directory
        if base_dir.name.startswith("gpu_"):
            # This is a GPU folder, look for source_files inside it
            source_files_dir = base_dir / "source_files"
            if source_files_dir.exists():
                logging.info(f"[OpGuard]: Looking up source_files in gpu directory: {source_files_dir}")
                _read_source_files_from_dir(source_files_dir, embedded_source_files, seen_paths)
            else:
                logging.debug(f"[OpGuard]: source_files directory not found: {source_files_dir}")
        else:
            # This is the trial directory, look for source_files here
            source_files_dir = base_dir / "source_files"
            if source_files_dir.exists():
                logging.info(f"[OpGuard]: Looking up source_files in run directory: {source_files_dir}")
                _read_source_files_from_dir(source_files_dir, embedded_source_files, seen_paths)
            else:
                logging.debug(f"[OpGuard]: source_files directory not found: {source_files_dir}")
    
    return embedded_source_files

def build_perfetto(
    dir1: str,
    dir2: str,
    out_path: str,
    ignore_output_only: bool = False,
    filter: str = None,
    event_mode: str = DEFAULT_EVENT_MODE,
    strict_fn: bool = False,
    ignore_fields: Optional[Dict[str, List[str]]] = None,
) -> None:
    # strict_fn is retained for CLI/API compatibility; identity is always required.
    base1 = Path(dir1).resolve(); base2 = Path(dir2).resolve(); outp = Path(out_path).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output base for source file collection
    # Use a subdirectory under the output file's parent to avoid conflicts
    # This ensures source_files are stored in a dedicated location
    # Remove all extensions from filename (e.g., viz5.json.gz -> viz5)
    output_name = outp.name
    while '.' in output_name:
        output_name = Path(output_name).stem
    output_base_dir = outp.parent / f".opguard_{output_name}"
    output_base = str(output_base_dir)

    total_start = time.time()

    # --- pair files
    with _timed_step("Step 1: Pairing trace files"):
        pairs, only1, only2 = list_pairs(base1, base2)
        print(f"🔍 Matched file pairs: {len(pairs)}")
        if only1: print(f"Unpaired run1 trace files: {len(only1)}; correspondence undetermined.")
        if only2: print(f"Unpaired run2 trace files: {len(only2)}; correspondence undetermined.")
    
    # Read source files directly from run directories (no collection needed)
    with _timed_step("Step 2: Reading source files from run directories"):
        print(f"📁 Reading source files from run directories...")
        embedded_source_files = _read_source_files_from_runs(base1, base2)
        if embedded_source_files:
            print(f"✅ Found {len(embedded_source_files)} source files in run directories")
        
        # Initialize source collector and load files from run directories
        # The collector will copy files to its own source_files/ directory for lookup
        # This temporary directory (.opguard_XXX) is used to store source files for function context extraction
        # during trace export. It can be cleaned up after export completes.
        from opguard.tracer.source_files import SourceFileCollector
        print(f"📂 Initializing source collector with output_base: {output_base}")
        print(f"   (Temporary directory for function context lookup during export)")
        source_collector = SourceFileCollector(str(output_base))
        print(f"📂 Loading source files from run directories:")
        print(f"   - run1: {base1}")
        print(f"   - run2: {base2}")
        files_loaded = load_source_files_from_runs(source_collector, str(base1), str(base2))
        if files_loaded > 0:
            print(f"✅ Loaded {files_loaded} unique source files into collector")
            print(f"📂 Collector source_files directory: {source_collector.source_dir}")
            print(f"   (Files copied here for fast lookup during function context extraction)")
        else:
            print(f"⚠️  No source files loaded from run directories")
        
        # Set the collector globally so stack_track can use it
        import opguard.tracer.source_files as source_files_module
        source_files_module._collector = source_collector

    # Establish correspondence once, before inspecting numerical differences.
    with _timed_step("Step 3: Collecting items and building pairs"):
        nodes, items1, items2, map12, map21 = _collect_all_items_and_pairs(
            sorted(pairs, key=lambda pair: pair.rel), only1, only2, base1, base2, filter
        )

    # --- compute global schedule (strict per-run counter order + no overlaps)
    with _timed_step("Step 4: Computing global schedule"):
        ts1, ts2 = compute_global_schedule(
            items1, items2, map12,
            run2_offset=RUN2_OFFSET_TICKS,
            bar_width=BAR_WIDTH_TICKS,
            gap=0,
        )
        # Compact dead air where both runs idle
        ts1, ts2 = shrink_global_idle(ts1, ts2, items1, items2, bar_width=BAR_WIDTH_TICKS, gap=0)

    # --- emit
    emitter = PerfettoEmitter(event_mode=event_mode)
    events: List[dict] = []
    pid1, pid2 = 0, 1
    seen_status_tracks: set[tuple[int, str]] = set()
    first_global: Optional[Tuple[int, int, int, str]] = None
    # Stack tracker will use the global collector that was initialized above
    # Pass output_base so it can find the collector's source_files directory
    stacks = StackTrackSingleThread(output_base=str(output_base))

    # Merge both runs' items and emit in global timestamp order
    with _timed_step("Step 5: Emitting trace events"):
        merged: List[Tuple[int, Item]] = []
        for it in items1:
            merged.append((ts1[it.key], it))
        for it in items2:
            merged.append((ts2[it.key], it))
        
        with _timed_step("  5a: Sorting merged items"):
            merged.sort(key=lambda x: (x[0], _safe_counter(x[1].counter), x[1].key.side, x[1].key.rel, x[1].key.idx))
        
        # Track timing for different operations
        compare_time = 0.0
        stack_time = 0.0
        emit_time = 0.0
        total_items = len(merged)
        
        for idx, (_, it) in enumerate(merged):
            if idx % 10000 == 0 and idx > 0:
                print(f"  Progress: {idx}/{total_items} items processed ({idx*100/total_items:.1f}%)")
            
            # Time compare_simple
            compare_start = time.time()
            node = nodes[it.key]
            rank = it.rank
            tid = rel_tid(rank)
            pid = pid1 if it.key.side == "run1" else pid2
            ts_ticks = ts1[it.key] if it.key.side == "run1" else ts2[it.key]

            mate_key = it.mate

            # Unresolved correspondence is not a numerical difference.
            has_anom, anom = detect_anomaly(node)
            diff_fields: List[str] = []
            details: Dict[str, any] = {}
            atoms: List[str] = []
            event_kind: Optional[str] = None

            if mate_key is not None:
                if it.key.side == "run1":
                    diff_fields, details, atoms = compare_simple(node, nodes[mate_key], ignore_output_only, ignore_fields)
                else:
                    diff_fields, details, atoms = compare_simple(nodes[mate_key], node, ignore_output_only, ignore_fields)
            else:
                event_kind = "unaligned"
                diff_fields, details, atoms = ([], {}, [])
            compare_time += time.time() - compare_start
            
            if event_kind == "unaligned":
                status = "unaligned"
            elif has_anom:
                status = "anomaly"
            elif diff_fields:
                status = "diff"
            else:
                status = "ok"

            node["__short"] = stable_name_with_counter(node)
            node["__is_leaf"] = True

            args = _leaf_args(node, it, status)
            if diff_fields:
                summary = "; ".join(atoms[:DIFF_ATOMS_MAX]) if atoms else ""
                if atoms and len(atoms) > DIFF_ATOMS_MAX:
                    summary += f"  (+{len(atoms)-DIFF_ATOMS_MAX} more)"
                args["diff"] = {"fields": diff_fields, "summary": summary}
                if config.DEBUG_ENABLED and details:
                    args["diff_debug"] = details
            if has_anom:
                args["anomaly"] = anom
            if event_kind:
                args["event_kind"] = event_kind
            args["__search"] = f"run:{it.key.side} rank:{rank} status:{status} fn:{node.get('function','')}"

            if status == "diff" and first_global is None:
                first_global = (pid, ts_ticks, rank, tid)
                args["first_diff"] = True

            cname = (
                C_INS if status == "unaligned"
                else (C_DIFF if status == "diff" else (C_ANOM if status == "anomaly" else C_OK))
            )
            
            # Time emit_call
            emit_start = time.time()
            emitter.emit_call(events, node, ts_ticks, pid, tid, cname, args)
            emit_time += time.time() - emit_start

            # status lane
            st_tid = status_tid(rank)
            key = (pid, st_tid)
            if key not in seen_status_tracks:
                _emit_thread_names(events, pid, st_tid, f"{it.key.side} · rank {rank} · status")
                seen_status_tracks.add(key)
            emit_status_slice(events, ts_ticks, pid, st_tid, status)

            # Time stack processing
            stack_start = time.time()
            stacks.feed(
                events=events,
                pid=pid,
                run_side=it.key.side,
                rank=rank,
                ts_ticks=ts_ticks,
                dur_ticks=BAR_WIDTH_TICKS,
                raw_stack=node.get("stack_trace"),
            )
            stack_time += time.time() - stack_start
        
        print(f"  ⏱️  Breakdown: compare_simple={compare_time:.2f}s, emit_call={emit_time:.2f}s, stacks.feed={stack_time:.2f}s")

    # Emit flows for true matched pairs
    with _timed_step("Step 6: Emitting flows"):
        for fid, it in enumerate(items1, 1):
            if not it.matched:
                continue
            src = it.key
            dst = map12[src]
            rankA = it.rank
            rankB = nodes[dst].get("gpu_rank", parse_rank_from_rel(dst.rel))
            tidA, tidB = rel_tid(rankA), rel_tid(rankB)
            tsA = ts1[src]; tsB = ts2[dst]
            emitter.emit_flow(events, tsA, pid1, tidA, tsB, pid2, tidB, fid)

    # First-diff global marker
    if first_global is not None:
        pid, ts_ticks, rank, tid = first_global
        events.append({
            "name": "first_candidate_diff", "cat": "opguard", "ph": "i", "s": "t",
            "ts": int(ts_ticks * 1_000),  # SCALE_NS_PER_TICK lives in config/emitter; here durations are ticks→ns in emitter
            "pid": pid, "tid": tid, "cname": C_FIRST,
            "args": {"first_diff": True, "rank": rank, "validation": "unreviewed",
                     "scope": "export_presentation_order", "root_cause": "not_determined"}
        })

    # finalize stack tracks (closes open B/E frames)
    with _timed_step("Step 7: Finalizing stack tracks"):
        stacks.flush_all(events)

    # embedded_source_files was already populated from run directories above

    # Write Perfetto trace with embedded source files
    with _timed_step("Step 8: Writing perfetto trace to disk"):
        doc = {
            "traceEvents": events, 
            "displayTimeUnit": DISPLAY_TIME_UNIT,
            "sourceFiles": embedded_source_files,
            "alignment": {
                "method": "callsite_sequence_v1", "uses_output_values": False,
                "unknown_state": "undetermined",
                "semantic_validation": "unreviewed_candidates",
                "assumptions": ["Corresponding execution order within each matched callsite/context.",
                                "Input summaries used as anchors are not proof of tensor equality."],
            }
        }
        with gzip.open(outp, "wb") as f:
            f.write(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
        print(f"✅ Perfetto trace written: {outp}")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"⏱️  Total time: {total_elapsed:.2f}s")
    print(f"{'='*60}")
