import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import ast  # safe parsing of meta_var dicts
import json

def extract_function_target(path_line: str) -> str:
    match = re.search(r"<function\s+([a-zA-Z0-9_.]+)>", path_line)
    return match.group(1) if match else "<unknown_function>"



def parse_function_stack_pairs_with_meta(
    diff_log_path: str,
    filter_kv: tuple[str, str, callable] = None
) -> list[tuple[str, list[str], int, dict, str, dict]]:
    """
    Parses the JSONL diff log file and extracts:
        (function_name, stack_trace_lines, gpu_rank, meta_var_dict, diff_field, diff_detail_dict)
    If filter_kv is provided, only include entries with comparison:
        filter_op(meta[filter_key], filter_val)
    """
    stack_quints = []
    with open(diff_log_path, "r") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                meta = obj.get("meta", {})

                if filter_kv:
                    key, raw_val, op = filter_kv
                    meta_val = meta.get(key)
                    if meta_val is None:
                        continue
                    try:
                        # Try to cast raw_val to same type as meta_val
                        if isinstance(meta_val, int):
                            cmp_val = int(raw_val)
                        elif isinstance(meta_val, float):
                            cmp_val = float(raw_val)
                        else:
                            cmp_val = raw_val  # treat as str
                    except ValueError:
                        continue  # Skip if casting fails

                    if not op(meta_val, cmp_val):
                        continue

                func_name = obj.get("function", "<unknown>")
                stack_lines = obj.get("stack_tail", "").splitlines()
                diff_fields = obj.get("diff_fields", [])
                rank = extract_rank_from_path(obj.get("file", ""))

                for field in diff_fields:
                    stack_quints.append((
                        func_name,
                        stack_lines,
                        rank,
                        meta,
                        field,
                        obj.get("diff_details", {})
                    ))

            except Exception as e:
                print(f"⚠️ Failed to parse line {i+1}: {e}")
    return stack_quints

def filter_diff_trace(
    diff_log_path: str,
    filter_kv: tuple[str, str, callable] = None
) -> list[dict]:
    """
    Parses the JSONL diff log file and filter entries with comparison:
        filter_op(meta[filter_key], filter_val)
    """
    stack_quints = []
    with open(diff_log_path, "r") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                meta = obj.get("meta", {})

                if filter_kv:
                    key, raw_val, op = filter_kv
                    meta_val = meta.get(key)
                    if meta_val is None:
                        continue
                    try:
                        # Try to cast raw_val to same type as meta_val
                        if isinstance(meta_val, int):
                            cmp_val = int(raw_val)
                        elif isinstance(meta_val, float):
                            cmp_val = float(raw_val)
                        else:
                            cmp_val = raw_val  # treat as str
                    except ValueError:
                        continue  # Skip if casting fails

                    if not op(meta_val, cmp_val):
                        continue
                
                stack_quints.append(obj)

            except Exception as e:
                print(f"⚠️ Failed to parse line {i+1}: {e}")
    return stack_quints


def extract_fine_grained_frame(stack_block: list[str]) -> str:
    """
    Extracts the most relevant frame in the stack: filename, line number, and function.
    Ignores vendor/internal libs.
    """
    for line in reversed(stack_block):
        if "site-packages" in line or "torch" in line:
            continue
        match = re.search(r'([^/\\]+\.py):(\d+)\s+in\s+([^\s]+)', line)
        if match:
            file, lineno, func = match.groups()
            return f"{func} @ {file}:{lineno}"  # More fine-grained!
    return "<unknown frame>"

def aggregate_io_diffs_per_function(
    stack_quints: list[tuple[str, list[str], int, dict, str, dict]]
) -> dict[str, dict[str, int]]:
    """
    Returns:
    {
        "func → frame": {
            "inputs[0]": count,
            "outputs[1]": count,
            ...
        },
        ...
    }
    """
    stats = defaultdict(lambda: defaultdict(int))

    for func_name, stack, _, _, field, diff_detail in stack_quints:
        frame_key = extract_fine_grained_frame(stack)
        func_key = f"{func_name} → {frame_key}"

        if field in {"inputs", "outputs"}:
            v1 = diff_detail.get(field, {}).get("run1")
            v2 = diff_detail.get(field, {}).get("run2")

            if isinstance(v1, list) and isinstance(v2, list):
                for i, (a, b) in enumerate(zip(v1, v2)):
                    if a != b:
                        stats[func_key][f"{field}[{i}]"] += 1
                if len(v1) != len(v2):
                    stats[func_key][f"{field}[size_mismatch]"] += 1

            elif isinstance(v1, dict) and isinstance(v2, dict):
                if v1 != v2:
                    stats[func_key][f"{field}[0]"] += 1  # treat as scalar output

            else:
                stats[func_key][f"{field}[?]"] += 1  # fallback for malformed
    return stats




def aggregate_stack_pairs_by_meta(stack_quads: list[tuple[str, list[str], int, dict]], meta_key: str) -> dict:
    """
    Aggregates stats per (function + stack frame) and breakdown by meta_key value.
    Returns:
        stats = {
            "func → frame": {
                "total": int,
                "per_meta": {meta_value: count}
            },
            ...
        }
    """
    stats = defaultdict(lambda: {"total": 0, "per_meta": defaultdict(int)})

    for func_name, stack_block, rank, meta, _, _ in stack_quads:
        frame_key = extract_fine_grained_frame(stack_block)
        key = f"{func_name} → {frame_key}"
        value = meta.get(meta_key, "<missing>")
        stats[key]["total"] += 1
        stats[key]["per_meta"][value] += 1

    return stats


def aggregate_function_stack_pairs(stack_pairs: list[tuple[str, list[str]]]) -> dict:
    stats = defaultdict(int)
    for func_name, stack_block in stack_pairs:
        frame_key = extract_fine_grained_frame(stack_block)
        key = f"{func_name} → {frame_key}"
        stats[key] += 1
    return stats


def aggregate_diff_fields_per_function(stack_quints: list[tuple[str, list[str], int, dict, str]]) -> dict:
    """
    Returns:
    {
        function_name → {
            "total": int,
            "fields": {
                "inputs": count,
                "outputs": count,
                ...
            }
        }
    }
    """
    stats = defaultdict(lambda: {"total": 0, "fields": defaultdict(int)})

    for func_name, _, _, _, diff_field, _ in stack_quints:
        stats[func_name]["total"] += 1
        stats[func_name]["fields"][diff_field or "<unknown>"] += 1

    return stats



def print_summary(stats: dict):
    print("\n📊 Stack Trace Hotspots Summary:")
    for key, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"{key} — {count} occurrence(s)")

def print_meta_summary(stats: dict, meta_key: str):
    print(f"\n📊 Meta-var Distribution per Stack Trace Hotspot [key: {meta_key}]:")
    for key, val in sorted(stats.items(), key=lambda x: -x[1]["total"]):
        print(f"\n🔹 {key} — {val['total']} total occurrence(s)")
        total = val["total"]
        for mv, count in sorted(val["per_meta"].items(), key=lambda x: -x[1]):
            ratio = count / total * 100
            print(f"   - {meta_key} = {mv}: {count} ({ratio:.1f}%)")

def print_diff_field_summary(stats: dict):
    print("\n📊 Difference Field Breakdown per Function:\n")
    for func, data in sorted(stats.items(), key=lambda x: -x[1]["total"]):
        total = data["total"]
        print(f"🔹 {func} — {total} total diff(s)")
        for field, count in sorted(data["fields"].items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"   - {field}: {count} ({pct:.1f}%)")

def meta_sort_key(entry):
    line_num, func_key, stack, meta, key, a, b = entry
    return (
        meta.get("global_step", -1),
        meta.get("micro_batch_idx", -1),
        meta.get("layer_number", -1),
        line_num,
    )

def print_io_index_per_function(
    stats: dict[str, dict[str, int]], 
    stack_quints: list[tuple[str, list[str], int, dict, str, dict]],
    dump_threshold: int = 50000
):
    """
    Print a per-function breakdown of which inputs/outputs differ most frequently.
    If any specific field (e.g., inputs[1]) has < dump_threshold diffs, show raw entries only for that field.
    """
    print("\n📊 Fine-Grained Input/Output Differences per Function:\n")

    # Collect entries grouped by function and index field
    grouped_entries = defaultdict(lambda: defaultdict(list))
    rank_sorted_entries = defaultdict(list)

    for func_name, stack, rank, meta, field, detail in stack_quints:
        frame_key = extract_fine_grained_frame(stack)
        func_key = f"{func_name} → {frame_key}"

        if field in {"inputs", "outputs"}:
            v1 = detail.get(field, {}).get("run1")
            v2 = detail.get(field, {}).get("run2")

            if isinstance(v1, list) and isinstance(v2, list):
                for i, (a, b) in enumerate(zip(v1, v2)):
                    if a != b:
                        key = f"{field}[{i}]"
                        grouped_entries[func_key][key].append((stack, rank, meta, key, a, b))
                if len(v1) != len(v2):
                    key = f"{field}[size_mismatch]"
                    grouped_entries[func_key][key].append((stack, rank, meta, key, v1, v2))

            elif isinstance(v1, dict) and isinstance(v2, dict):
                if v1 != v2:
                    key = f"{field}[0]"
                    grouped_entries[func_key][key].append((stack, rank, meta, key, v1, v2))

    # Print stats and conditionally dump low-count entries
    for func_key, io_stat in sorted(stats.items(), key=lambda x: -sum(x[1].values())):
        total = sum(io_stat.values())
        print(f"🔹 {func_key} — {total} total diff(s)")
        for field_key, count in sorted(io_stat.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"   - {field_key}: {count} ({pct:.1f}%)")

            if count < dump_threshold and field_key in grouped_entries.get(func_key, {}):
                print(f"     📄 Below threshold — showing raw entries for {field_key}:\n")
                for stack, rank, meta, key, a, b in grouped_entries[func_key][field_key]:
                    print(f"     ➤ GPU Rank: {rank}, Field: {key}")
                    print(f"       Meta: {meta}")
                    print(f"       run1: {a}")
                    print(f"       run2: {b}")
                    print("       Stack Trace Tail:")
                    for line in stack[-3:]:
                        print(f"         {line}")
                    print()

                    # Collect for global rank-wise sorted view
                    try:
                        last_frame = stack[-1]
                        line_part = last_frame.split(":")[-1]
                        line_num = int(line_part.strip().split(" ")[0])
                        
                    except Exception:
                        line_num = -1
                    rank_sorted_entries[rank].append((line_num, func_key, stack, meta, key, a, b))

    print("\n📈 Aggregated View — Sorted by GPU Rank and Line Number:\n")

    for rank in sorted(rank_sorted_entries):
        print(f"🔸 GPU Rank {rank}:")
        for line_num, func_key, stack, meta, key, a, b in sorted(rank_sorted_entries[rank], key=meta_sort_key):
            line_info = f"Line {line_num}" if line_num != -1 else "Unknown line"
            print(f"   • {func_key} — {key} @ {line_info}")
            print(f"     Meta: {meta}")
            print(f"     run1: {a}")
            print(f"     run2: {b}")
            print("     Stack Trace Tail:")
            for line in stack[-3:]:
                print(f"       {line}")
            print()



def plot_top_k(stats: dict, top_k: int = 10, output_path: str = "stacktrace_hotspots.png"):
    sorted_items = sorted(stats.items(), key=lambda x: -x[1])[:top_k]
    labels, counts = zip(*sorted_items)

    plt.figure(figsize=(10, 5))
    plt.barh(labels, counts)
    plt.xlabel("Occurrences")
    plt.title("Top Stack Trace Hotspots")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n📈 Plot saved to {output_path}")


import os
from collections import defaultdict

def extract_rank_from_path(trace_path: str) -> int:
    """
    Extract rank from directory structure like 'gpu_00/' or 'gpu_03/'.
    """
    match = re.search(r"gpu_(\d+)", trace_path)
    return int(match.group(1)) if match else -1


def aggregate_stack_pairs_by_rank(stack_triplets: list[tuple[str, list[str], int]]) -> dict:
    """
    Aggregates stats per (function + stack frame), with GPU rank breakdown.
    Returns:
        stats = {
            "func → frame": {
                "total": int,
                "per_rank": {rank: count}
            },
            ...
        }
    """
    stats = defaultdict(lambda: {"total": 0, "per_rank": defaultdict(int)})

    for func_name, stack_block, rank in stack_triplets:
        frame_key = extract_fine_grained_frame(stack_block)
        key = f"{func_name} → {frame_key}"
        stats[key]["total"] += 1
        stats[key]["per_rank"][rank] += 1

    return stats


def print_rank_summary(stats: dict):
    print("\n📊 Rank Distribution per Stack Trace Hotspot:")
    for key, val in sorted(stats.items(), key=lambda x: -x[1]["total"]):
        print(f"\n🔹 {key} — {val['total']} total occurrence(s)")
        total = val["total"]
        for rank, count in sorted(val["per_rank"].items()):
            ratio = count / total * 100
            print(f"   - Rank {rank}: {count} ({ratio:.1f}%)")

def collect_all_meta_keys(stack_quads: list[tuple[str, list[str], int, dict]]) -> set[str]:
    keys = set()
    for _, _, _, meta, _, _ in stack_quads:
        keys.update(meta.keys())
    return keys

import operator



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze stack trace hotspots from trace diff output.")
    parser.add_argument("diff_log", help="Path to the trace diff output file.")
    parser.add_argument("--plot", action="store_true", help="Generate a bar plot of top hotspots.")
    parser.add_argument("--topk", type=int, default=10, help="Top-K entries to include in the plot.")
    parser.add_argument("--meta_key", type=str, help="Meta_var key to analyze (e.g., 'global_step', or 'all').")
    parser.add_argument("--filter-kv", type=str, help="Filter entries by meta key-value pair (e.g., 'global_step=100')")
    args = parser.parse_args()

    # Parse and interpret filter_kv
    filter_kv = None
    if args.filter_kv:
        ops = {
            "==": operator.eq,
            "=": operator.eq,
            "!=": operator.ne,
            "<=": operator.le,
            ">=": operator.ge,
            "<": operator.lt,
            ">": operator.gt,
        }

    for op_str in sorted(ops.keys(), key=len, reverse=True):  # Match longest first
        if op_str in args.filter_kv:
            key, val = args.filter_kv.split(op_str, 1)
            filter_kv = (key.strip(), val.strip(), ops[op_str])
            break
    else:
        print("❌ Invalid --filter-kv format. Use key<op>val like global_step<=100")
        exit(1)
    stack_quads = parse_function_stack_pairs_with_meta(args.diff_log, filter_kv=filter_kv)

    # Rank-based summary
    stats = aggregate_stack_pairs_by_rank([(f, s, r) for f, s, r, _, _, _ in stack_quads])
    print_summary({k: v["total"] for k, v in stats.items()})
    print_rank_summary(stats)

    # diff-type summary
    diff_stats = aggregate_diff_fields_per_function(stack_quads)
    print_diff_field_summary(diff_stats)

    # Detailed Input/Output summary
    io_index_stats = aggregate_io_diffs_per_function(stack_quads)
    print_io_index_per_function(io_index_stats, stack_quads)

    # Meta-var summary
    if args.meta_key:
        if args.meta_key == "all":
            all_keys = collect_all_meta_keys(stack_quads)
            print("Collect all meta keys:", all_keys)
            for key in sorted(all_keys):
                meta_stats = aggregate_stack_pairs_by_meta(stack_quads, key)
                print_meta_summary(meta_stats, key)
        else:
            meta_stats = aggregate_stack_pairs_by_meta(stack_quads, args.meta_key)
            print_meta_summary(meta_stats, args.meta_key)

    # Optional plot
    if args.plot:
        plot_top_k({k: v["total"] for k, v in stats.items()}, top_k=args.topk)
