# tools/callsite\_to\_instrumentation.py

from __future__ import annotations
import argparse
import os
from opguard.callsite.converter import CallsiteToInstrumentationConverter

def main():
    parser = argparse.ArgumentParser(description="Convert callsite trace data to instrumentation schema (refactored)")
    parser.add_argument("--callsite-file", required=True, help="Path to callsite JSON file")
    parser.add_argument("--output", help="Output YAML file path")
    parser.add_argument("--base-dir", default=os.getcwd(),
                        help="Base directory for relative paths (defaults to current working directory)")
    args = parser.parse_args()

    conv = CallsiteToInstrumentationConverter(base_dir=args.base_dir)
    entries = conv.convert_callsite_data(args.callsite_file)

    print(f"Generated {len(entries)} instrumentation entries")
    print(f"Successful: {len(entries)}, Missing: {len(conv.missing_entries)}, Invalid: {len(conv.invalid_entries)}")
    if conv.missing_entries:
        print(f"Missing entries: {sorted(list(conv.missing_entries))[:10]}{' ...' if len(conv.missing_entries) > 10 else ''}")
    if conv.invalid_entries:
        print(f"Invalid entries: {sorted(list(conv.invalid_entries))[:10]}{' ...' if len(conv.invalid_entries) > 10 else ''}")

    yaml_text = conv.generate_yaml_config(args.output)
    if not args.output:
        print("\nGenerated YAML configuration:\n")
        print(yaml_text)

if __name__ == "__main__":
    main()
