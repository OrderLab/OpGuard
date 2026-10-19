#!/usr/bin/env python3
"""
Script to combine callsite data from multiple JSON files (different ranks) into a single JSON file.
This is useful for merging traces from distributed training runs.

Usage Examples:
    # Combine specific files
    python3 opguard/combine_callsites.py --input callsites-rank0.json callsites-rank4.json --output combined.json
    
    # Combine using glob pattern
    python3 opguard/combine_callsites.py --input "callsites-rank*.json" --output combined.json
    
    # Combine using pattern option
    python3 opguard/combine_callsites.py --pattern "callsites-rank*.json" --output combined.json
    
    # Combine with verbose logging
    python3 opguard/combine_callsites.py --input callsites-rank*.json --output combined.json --verbose

    # Simply merge without adding rank information
    python3 opguard/combine_callsites.py --input callsites-rank*.json --output combined.json --no-rank

Features:
    - Merges callsite data from multiple rank files
    - Deduplicates callstacks across ranks
    - Preserves unique callsite entries
    - Provides detailed statistics and logging
    - Supports glob patterns and directory scanning
    - Handles missing files gracefully
"""

import json
import argparse
import os
import glob
from typing import Dict, List, Set
from collections import defaultdict
import re

import re

def extract_rank_from_filename(file_path: str) -> int:
    """Extract the global rank directly from filename/path."""
    m = re.search(r'rank[-_]?(\d+)', file_path)
    if not m:
        # raise ValueError(f"Could not extract rank from filename: {file_path}")
        return -1
    return int(m.group(1))

def remap_stream_ids_to_rank(entry: dict, rank: int) -> dict:
    """Replace local device IDs with the global rank."""
    def replace_dev(lst):
        return [[rank, s] for _, s in (lst or [])]

    def replace_counts(sc):
        return {f"{rank}:{k.split(':',1)[1]}": v for k, v in (sc or {}).items()}

    if "streams" in entry:
        entry["streams"] = replace_dev(entry.get("streams", []))
    if "stream_counts" in entry:
        entry["stream_counts"] = replace_counts(entry.get("stream_counts", {}))

    if "by_op" in entry:
        for meta in entry["by_op"].values():
            if "streams" in meta:
                meta["streams"] = replace_dev(meta.get("streams", []))
            if "stream_counts" in meta:
                meta["stream_counts"] = replace_counts(meta.get("stream_counts", {}))

    return entry


class CallsiteCombiner:
    """Combines callsite data from multiple JSON files."""
    
    def __init__(self):
        self.combined_data = {}
        self.stats = {
            'files_processed': 0,
            'total_entries': 0,
            'unique_callsites': 0,
            'merged_callstacks': 0
        }
    
    def load_callsite_file(self, file_path: str) -> Dict:
        """Load a single callsite JSON file."""
        print(f"[INFO] Loading callsite file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"[INFO] Loaded {len(data)} entries from {file_path}")
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            return {}
    
    def combine_callsite_data(self, callsite_files: List[str], add_rank: bool = True) -> Dict:
        """
        Combine callsite data from multiple files.
        """
        print(f"[INFO] Starting combination of {len(callsite_files)} callsite files")
        
        for file_path in callsite_files:
            if add_rank:
                rank = extract_rank_from_filename(file_path)
                if rank == -1:
                    print(f"[WARNING] Could not extract rank from filename: {file_path}")
                    continue

            data = self.load_callsite_file(file_path)
            if not data:
                continue

            if add_rank:
                for key, entry in data.items():
                    entry = remap_stream_ids_to_rank(entry, rank)
    
            
            self.stats['files_processed'] += 1
            
            for callsite_key, callsite_data in data.items():
                self.stats['total_entries'] += 1

                if callsite_key not in self.combined_data:
                    # shallow copy to avoid aliasing
                    self.combined_data[callsite_key] = json.loads(json.dumps(callsite_data))
                    self.stats['unique_callsites'] += 1
                    if 'callstacks' in callsite_data:
                        # already counted; no merge delta
                        pass
                else:
                    before = len(self.combined_data[callsite_key].get('callstacks', []))
                    self.combined_data[callsite_key] = merge_callsite_entries(self.combined_data[callsite_key], callsite_data)
                    after = len(self.combined_data[callsite_key].get('callstacks', []))
                    if after > before:
                        self.stats['merged_callstacks'] += (after - before)


        
        print(f"[INFO] Combination complete. Statistics:")
        print(f"[INFO]   Files processed: {self.stats['files_processed']}")
        print(f"[INFO]   Total entries processed: {self.stats['total_entries']}")
        print(f"[INFO]   Unique callsites: {self.stats['unique_callsites']}")
        print(f"[INFO]   Merged callstacks: {self.stats['merged_callstacks']}")
        
        return self.combined_data
    
    def save_combined_data(self, output_file: str):
        """Save the combined data to a JSON file."""
        print(f"[INFO] Saving combined data to: {output_file}")
        try:
            with open(output_file, 'w') as f:
                json.dump(self.combined_data, f, indent=2)
            print(f"[INFO] Successfully saved {len(self.combined_data)} entries to {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save {output_file}: {e}")


def find_callsite_files(pattern: str) -> List[str]:
    """
    Find callsite files matching the pattern.
    Supports glob patterns and directory scanning.
    """
    if os.path.isfile(pattern):
        return [pattern]
    
    # Try glob pattern
    files = glob.glob(pattern)
    if files:
        return sorted(files)
    
    # If no files found, try to find files with common callsite patterns
    if os.path.isdir(pattern):
        # Look for files in the directory
        callsite_patterns = [
            "compressed-*.json",
            "callsite-*.json", 
            "trace-*.json",
            "rank-*.json"
        ]
        
        all_files = []
        for pat in callsite_patterns:
            all_files.extend(glob.glob(os.path.join(pattern, pat)))
        
        return sorted(all_files)
    
    return []

def _merge_stream_lists(a, b):
    # streams are stored like [[device, stream], ...]
    sa = {tuple(x) for x in a or []}
    sb = {tuple(x) for x in b or []}
    merged = sorted(sa | sb)
    return [list(x) for x in merged]

def _merge_count_maps(a: dict, b: dict) -> dict:
    # sum counts by key (e.g., {"1:7": 10})
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out

def _merge_by_op(a: dict, b: dict) -> dict:
    """
    by_op:
      {
        "<mangled_op_name>": {
          "count": int,
          "streams": [[d,s], ...],
          "stream_counts": {"d:s": int}
        },
        ...
      }
    """
    out = {k: {
                "count": v.get("count", 0),
                "streams": [lst[:] for lst in v.get("streams", [])],
                "stream_counts": dict(v.get("stream_counts", {})),
             } for k, v in (a or {}).items()}

    for op, meta in (b or {}).items():
        if op not in out:
            out[op] = {
                "count": meta.get("count", 0),
                "streams": [lst[:] for lst in meta.get("streams", [])],
                "stream_counts": dict(meta.get("stream_counts", {})),
            }
        else:
            out[op]["count"] = out[op].get("count", 0) + meta.get("count", 0)
            out[op]["streams"] = _merge_stream_lists(out[op].get("streams", []), meta.get("streams", []))
            out[op]["stream_counts"] = _merge_count_maps(out[op].get("stream_counts", {}), meta.get("stream_counts", {}))
    return out

def _merge_ops_list(a: list, b: list) -> list:
    # simple union while preserving a stable order (a first, then unseen from b)
    seen = set(a or [])
    out = list(a or [])
    for x in (b or []):
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def merge_callstacks(existing_callstacks: List[List[List[str]]], new_callstacks: List[List[List[str]]]) -> List[List[List[str]]]:
    """
    Merge callstacks from different ranks, avoiding duplicates.
    """
    if not existing_callstacks:
        return new_callstacks
    
    if not new_callstacks:
        return existing_callstacks
    
    # Convert callstacks to tuples for deduplication
    existing_set = set()
    for callstack in existing_callstacks:
        # Convert each callstack to a tuple of tuples for hashing
        callstack_tuple = tuple(tuple(call) for call in callstack)
        existing_set.add(callstack_tuple)
    
    merged_callstacks = existing_callstacks.copy()
    
    for callstack in new_callstacks:
        callstack_tuple = tuple(tuple(call) for call in callstack)
        if callstack_tuple not in existing_set:
            merged_callstacks.append(callstack)
            existing_set.add(callstack_tuple)
    
    return merged_callstacks

def merge_callsite_entries(existing_data: dict, incoming: dict) -> dict:
    """
    Merge one callsite's dict (existing_data) with another (incoming), in place and returned.
    Fields handled:
      - count: summed
      - callstacks: dedup via your existing merge_callstacks(...)
      - ops: union
      - streams: union of pairs
      - stream_counts: sum by "d:s"
      - by_op: per-op merge of count/streams/stream_counts
      - stream (single): if both present and differ, keep existing (or choose first non-null)
      - anything else: keep existing unless missing, then copy
    """
    # 1) counts
    if isinstance(incoming.get("count"), (int, float)):
        existing_data["count"] = existing_data.get("count", 0) + incoming["count"]

    # 2) callstacks (you already do this outside, but safe to guard)
    if "callstacks" in incoming:
        existing_data["callstacks"] = merge_callstacks(existing_data.get("callstacks", []),
                                                       incoming.get("callstacks", []))

    # 3) ops (list)
    if "ops" in incoming:
        existing_data["ops"] = _merge_ops_list(existing_data.get("ops", []), incoming.get("ops", []))

    # 4) streams (list of [device, stream])
    if "streams" in incoming:
        existing_data["streams"] = _merge_stream_lists(existing_data.get("streams", []), incoming.get("streams", []))

    # 5) stream_counts (map "d:s" -> count)
    if "stream_counts" in incoming:
        existing_data["stream_counts"] = _merge_count_maps(existing_data.get("stream_counts", {}), incoming.get("stream_counts", {}))

    # 6) by_op (nested aggregation)
    if "by_op" in incoming:
        existing_data["by_op"] = _merge_by_op(existing_data.get("by_op", {}), incoming.get("by_op", {}))

    # 7) optional single-valued fields you might have (first-writer-wins)
    for k, v in incoming.items():
        if k in {"count","callstacks","ops","streams","stream_counts","by_op"}:
            continue
        if k not in existing_data:
            existing_data[k] = v

    return existing_data


def main():
    parser = argparse.ArgumentParser(description="Combine callsite data from multiple JSON files")
    parser.add_argument("--input", "-i", required=True, nargs="+", 
                       help="Input callsite JSON files or patterns (supports glob)")
    parser.add_argument("--output", "-o", required=True,
                       help="Output combined JSON file")
    parser.add_argument("--pattern", "-p", 
                       help="Pattern to find callsite files (alternative to --input)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no-rank", action="store_true",
                       help="Do not add rank information to the output")
    
    args = parser.parse_args()
    
    # Collect input files
    input_files = []
    
    if args.pattern:
        input_files = find_callsite_files(args.pattern)
    else:
        for input_arg in args.input:
            files = find_callsite_files(input_arg)
            input_files.extend(files)
    
    # Remove duplicates and sort
    input_files = sorted(list(set(input_files)))
    
    if not input_files:
        print("[ERROR] No input files found!")
        return
    
    print(f"[INFO] Found {len(input_files)} input files:")
    for f in input_files:
        print(f"[INFO]   {f}")
    
    # Combine the files
    combiner = CallsiteCombiner()
    combined_data = combiner.combine_callsite_data(input_files, not args.no_rank)
    
    # Save the combined data
    combiner.save_combined_data(args.output)
    
    print(f"[INFO] Successfully combined {len(combined_data)} unique callsites into {args.output}")


if __name__ == "__main__":
    main() 