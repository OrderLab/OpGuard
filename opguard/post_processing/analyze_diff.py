#!/usr/bin/env python3
import sys
import json
import argparse

def main():
    ap = argparse.ArgumentParser(description="Sort JSONL by counter and filter out meta-only diffs.")
    ap.add_argument("input", help="Input JSONL file")
    ap.add_argument("-o", "--output", help="Output file (default: stdout)", default="-")
    args = ap.parse_args()

    # Load all JSON objects
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                sys.stderr.write(f"Skipping invalid JSON line: {line[:80]}...\n")

    # Filter: drop entries where diff_fields == ["meta"]
    filtered = [r for r in records if not (r.get("diff_fields") == ["meta"])]

    # Sort by counter
    filtered.sort(key=lambda x: x.get("counter", 0))

    # Output
    if args.output == "-":
        for r in filtered:
            print(json.dumps(r, ensure_ascii=False))
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in filtered:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    print(f"Input entries: {len(records)}", file=sys.stderr)
    print(f"Filtered entries: {len(filtered)}", file=sys.stderr)

if __name__ == "__main__":
    main()
