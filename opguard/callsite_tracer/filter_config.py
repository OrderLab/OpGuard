#!/usr/bin/env python3
import sys
import re
import argparse
from typing import Any, Dict, List, Tuple
import yaml

DEFAULT_STREAM_RE = re.compile(r"^\d+:7$")   # only "{rank}:7" is allowed

# A tiny, comprehensible denylist of trivial torch functions to drop in --basic mode.
# Feel free to tweak/extend. These are glue/utility ops unlikely to be insightful for triage.
COMMON_TORCH_FUNCTIONS = {
    "cat", "stack", "reshape", "view", "contiguous", "clone", "detach",
    "squeeze", "unsqueeze", "permute", "transpose", "to", "repeat", "copy",
    "expand", "narrow", "zeros", "zero_", "fill_", "empty", "empty_like", "tensor", "copy_",
    "normal_", "bfloat16", "zeros_", "ones_", "constant_"
}

def is_default_stream_key(k: str) -> bool:
    return DEFAULT_STREAM_RE.match(str(k)) is not None

def summarize_entry(entry: Dict[str, Any]) -> str:
    fp = entry.get("file_path", "?")
    ln = entry.get("line_num", "?")
    fn = entry.get("function", "?")
    return f"{fp}:{ln} :: {fn}"

def should_keep_basic(entry: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Basic mode filter:
      - keep only operation_type == 'function'
      - drop common torch utility functions (cat, stack, reshape, ...)
    Returns (keep, reason_if_dropped)
    """
    op_type = (entry.get("operation_type") or "").strip().lower()
    if op_type not in ["function", "method", "triton_kernel"]:
        return (False, f"non-function/method entry in --basic mode, op_type={op_type}")

    fn_name = (entry.get("function") or entry.get("method") or entry.get("triton_kernel") or "").strip()
    if fn_name in COMMON_TORCH_FUNCTIONS:
        return (False, f"filtered common torch function: {fn_name}")

    return (True, "")

def main():
    ap = argparse.ArgumentParser(
        description="Filter config entries by default CUDA stream ({rank}:7). "
                    "Optionally, '--basic' keeps only operation_type=function and drops common torch functions."
    )
    ap.add_argument("input", help="Input YAML file (use '-' for stdin)")
    ap.add_argument("-o", "--output", help="Output YAML file (default: stdout)", default="-")
    ap.add_argument("--basic", action="store_true",
                    help="Keep only entries with operation_type=function and remove common torch utility functions.")
    args = ap.parse_args()

    # Load YAML
    if args.input == "-":
        raw = yaml.safe_load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

    if raw is None:
        raw = []

    # If we got a dict with "instrumentations", unwrap
    top_is_dict = isinstance(raw, dict) and "instrumentations" in raw
    if top_is_dict:
        data = raw.get("instrumentations", [])
    else:
        data = raw

    if not isinstance(data, list):
        print("ERROR: instrumentations must be a list.", file=sys.stderr)
        sys.exit(2)

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for entry in data:
        # 1) Default-stream rule
        sc: Dict[str, Any] = entry.get("stream_counts") or {}
        non_default_keys = [k for k in sc.keys() if not is_default_stream_key(k)]
        if non_default_keys:
            entry["_non_default_stream_keys"] = sorted(map(str, non_default_keys))
            entry["_drop_reason"] = "non-default stream usage"
            removed.append(entry)
            continue

        # 2) --basic rule (if enabled)
        if args.basic:
            keep, reason = should_keep_basic(entry)
            if not keep:
                entry["_drop_reason"] = reason
                removed.append(entry)
                continue

        # Clear stream_counts when keeping
        if "stream_counts" in entry:
            entry["stream_counts"] = {}

        kept.append(entry)

    # Logging (stderr)
    print("=== Post-processing summary ===", file=sys.stderr)
    print(f"Total entries:   {len(data)}", file=sys.stderr)
    print(f"Kept entries:    {len(kept)}", file=sys.stderr)
    print(f"Removed entries: {len(removed)}", file=sys.stderr)

    if removed:
        print("\nRemoved entries:", file=sys.stderr)
        for e in removed:
            where = summarize_entry(e)
            nd = ", ".join(e.get("_non_default_stream_keys", []))
            ops = e.get("ops") or []
            ops_info = f" ops={len(ops)}" if ops else ""
            reason = e.get("_drop_reason", "filtered")
            extra = f" | non-default streams: {nd}" if nd else ""
            print(f" - {where}{ops_info} | reason: {reason}{extra}", file=sys.stderr)

    # Wrap back if necessary
    if top_is_dict:
        output_data = dict(raw)  # shallow copy
        output_data["instrumentations"] = kept
    else:
        output_data = kept

    # Dump filtered YAML
    if args.output == "-":
        yaml.safe_dump(output_data, sys.stdout, sort_keys=True, allow_unicode=True)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.safe_dump(output_data, f, sort_keys=True, allow_unicode=True)

if __name__ == "__main__":
    main()
