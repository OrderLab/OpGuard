#!/usr/bin/env python3
import sys
import argparse
import yaml
from opguard.common_vars import OPGUARD_STRIP_SYNC_YAML

def is_default_stream(stream_counts: dict, default_stream_id="7") -> bool:
    if not stream_counts:
        return False
    
    for k in stream_counts.keys():
        try:
            stream = str(k).split(":")[1].strip()
            if not (stream == default_stream_id):
                return False
        except Exception:
            continue
    
    return True

def main():
    ap = argparse.ArgumentParser(
        description="Filter config entries by default CUDA stream ({rank}:7). "
                    "Optionally, '--basic' keeps only operation_type=function and drops common torch functions."
    )
    ap.add_argument("input", help="Input YAML file (use '-' for stdin)")
    ap.add_argument("-s", "--default_stream_id", default="7", help="Default stream ID (default: 7)")
    ap.add_argument("-o", "--output", help="Output YAML file (default: stdout)", default="-")
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

    for entry in data:
        stream_counts = entry.get("stream_counts", {})
        if not is_default_stream(stream_counts, args.default_stream_id):
            entry["sync_streams"] = True
        
        if OPGUARD_STRIP_SYNC_YAML:
            entry.pop("ops", None)
            entry.pop("stream_counts", None)
    
    data.sort(key=lambda x: x["file_path"]+x["function"]+str(x["line_num"]))
    
    # Wrap back if necessary
    if top_is_dict:
        output_data = dict(raw)  # shallow copy
        output_data["instrumentations"] = data
    else:
        output_data = data

    # Dump add sync flag YAML
    print("=== Add sync flag summary ===", file=sys.stderr)
    if args.output == "-":
        yaml.safe_dump(output_data, sys.stdout, sort_keys=True, allow_unicode=True)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.safe_dump(output_data, f, sort_keys=True, allow_unicode=True)

if __name__ == "__main__":
    main()
