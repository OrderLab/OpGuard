#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
import fnmatch
import sys

# Default patterns to filter out (supports wildcard; '<*>' filters any <…> function)
DEFAULT_FILTER_PATTERNS = [
    "<*>",            # filter any <foo> frames
    "_call_impl",
    "__iter__",
    "wrapped",
    "wrapped_*",
    "wrapper",
    "logger",
    "hook",
    "__torch_dispatch__",
    "xor_reduce_cuda",
]

# Default library paths to filter out (enabled by default)
DEFAULT_FILTER_PATHS = [
    # "/usr/lib/python3.",                   # stdlib
    # "/usr/local/lib/python3.",             # pip-installed libs
    # "*site-packages/torch*",                 # Torch
    # "*dist-packages/torch*",                 # Torch
    # "*site-packages/einops*",                 # einops
    # "*dist-packages/einops*",                 # einops
    # "*site-packages/triton*",          # triton
    # "*dist-packages/triton*",          # triton
    "*contextlib.py",                       # Common stdlib pattern
    "*typing_extensions.py",
    "*logger*",
    "*jit*.py",
    "*driver*.py",
    "*/OpGuard/opguard/*",
    "*/OpGuard/setup.py",
    "*/OpGuard/usercustomize.py",
    "*/triton/runtime/*",
    "*/triton/backends/*",
    "*/torch/_*.py",
    "*/torch/_*/*",
    "*/torch/testing/*",
    "*/torch/cuda/_sanitizer.py",
    "*/ctypes/*",
    "*/threading.py",
    "*/torch/autograd/__init__.py",
]

class TrieNode:
    __slots__ = ("frame", "count", "children", "ops", "stream_counts", "self_leaf_count", "self_leaf_ops", "self_leaf_stream_counts")
    def __init__(self, frame=None):
        self.frame    = frame       # tuple(filename, lineno, funcname)
        self.count    = 0
        self.children = {}          # frame_tuple -> TrieNode
        # Aggregates for the subtree that collapses to this leaf
        self.ops = Counter()        # op_name -> weighted count
        self.stream_counts = Counter()  # "device:stream" -> count
        self.self_leaf_count = 0
        self.self_leaf_ops = Counter()
        self.self_leaf_stream_counts = Counter()



def should_filter_frame(frame, filter_patterns, filter_paths):
    """
    Check if a frame should be filtered out based on:
     - function name glob patterns in filter_patterns
     - file path substrings in filter_paths
    """
    funcname = frame[2] if len(frame) > 2 else ""
    filename = frame[0] if len(frame) > 0 else ""

    # wildcard/glob matching for funcname
    match_func = False
    def _matches_name(funcname: str, pat: str) -> bool:
        if any(ch in pat for ch in "*?[]"):
            return fnmatch.fnmatch(funcname, pat)
        return funcname == pat

    match_func = any(_matches_name(funcname, pat) for pat in (filter_patterns or []))

    # wildcard/glob matching for paths (NEW)
    match_path = False
    for pat in (filter_paths or []):
        # if user-specified a true glob, use it; otherwise, do substring match via *...*
        glob = pat if any(ch in pat for ch in "*?[]") else f"*{pat}*"
        if fnmatch.fnmatch(filename, glob):
            match_path = True
            break

    return match_func or match_path

def build_forest(stats, filter_patterns=None, filter_paths=None):
    roots = {}
    for info in stats.values():
        cnt = info.get("count", 0)
        info_ops = info.get("ops", [])   
        info_stream_counts = info.get("stream_counts", {})

        for cs in info.get("callstacks", []):
            frames = [tuple(f) for f in cs]
            frames = [f for f in frames
                      if not should_filter_frame(f, filter_patterns, filter_paths)]
            if not frames:
                continue

            root_key = frames[0]
            if root_key not in roots:
                roots[root_key] = TrieNode(frame=root_key)
            node = roots[root_key]
            node.count += cnt

            for fr in frames[1:]:
                if fr not in node.children:
                    node.children[fr] = TrieNode(frame=fr)
                node = node.children[fr]
                node.count += cnt

            # At the leaf: accumulate CUDA ops and streams
            if info_ops:
                node.ops.update({op: cnt for op in info_ops})
            if info_stream_counts:
                node.stream_counts.update(info_stream_counts)

            node.self_leaf_count += cnt
            if info_ops:
                node.self_leaf_ops.update({op: cnt for op in info_ops})
            if info_stream_counts:
                node.self_leaf_stream_counts.update(info_stream_counts)
    return list(roots.values())



def subtree_has_branch(node):
    if len(node.children) > 1:
        return True
    if not node.children:
        return False
    return subtree_has_branch(next(iter(node.children.values())))

def _merge_aggs(dst: TrieNode, src: TrieNode):
    dst.ops.update(src.ops)
    dst.stream_counts.update(src.stream_counts)

    dst.self_leaf_count += src.self_leaf_count
    dst.self_leaf_ops.update(src.self_leaf_ops)
    dst.self_leaf_stream_counts.update(src.self_leaf_stream_counts)

def compress_node(node):
    if not node.children:
        new = TrieNode(frame=node.frame)
        new.count = node.count
        _merge_aggs(new, node)                      # preserve leaf aggs
        return new

    if len(node.children) > 1:
        new = TrieNode(frame=node.frame)
        new.count = node.count
        # keep children; compress each and attach
        for child in node.children.values():
            comp = compress_node(child)
            if comp:
                new.children[comp.frame] = comp
        # node itself might have aggs if it was a leaf earlier
        _merge_aggs(new, node)
        return new

    # single-child chain
    child = next(iter(node.children.values()))
    if not subtree_has_branch(child):
        # collapse the whole chain to THIS node, but keep child's aggs
        new = TrieNode(frame=node.frame)
        new.count = node.count
        _merge_aggs(new, node)
        comp_child = compress_node(child)
        _merge_aggs(new, comp_child)               # bring leaf aggs up
        return new

    # otherwise, bubble down into the child
    return compress_node(child)



def collect_leaf_distribution(roots):
    file_counter = Counter()
    def dfs(n):
        if not n.children:
            file_counter[n.frame[0]] += 1
        else:
            for c in n.children.values():
                dfs(c)
    for r in roots:
        dfs(r)
    return file_counter

def _count_leaves(n: TrieNode) -> int:
    if not n.children:
        return 1
    return sum(_count_leaves(c) for c in n.children.values())

def prune_small_subtrees(roots, threshold_num):
    def gather_aggs(node):
        acc_ops = Counter()
        acc_stream_counts = Counter()
        def walk(n):
            acc_ops.update(n.ops)
            acc_stream_counts.update(n.stream_counts)
            for ch in n.children.values():
                walk(ch)
        walk(node)
        return acc_ops, acc_stream_counts

    def post(n):
        if not n.children:
            return n, 1

        total = 1
        new_children = {}
        for fr, c in n.children.items():
            nc, size = post(c)
            if nc:
                new_children[fr] = nc
                total += size
        n.children = new_children

        # --- NEW: forbid collapsing multi-leaf subtrees ---
        leaf_cnt = _count_leaves(n)
        is_single_chain = (leaf_cnt == 1)

        if threshold_num > 0 and total < threshold_num and is_single_chain:
            # safe: collapsing won’t merge distinct leaves
            agg_ops, agg_streams = gather_aggs(n)
            n.children = {}
            n.ops.update(agg_ops)
            n.stream_counts.update(agg_streams)
            return n, 1

        return n, total

    out = []
    for r in roots:
        nr, _ = post(r)
        if nr:
            out.append(nr)
    return out

def compress_forest(roots):
    return [c for r in roots if (c := compress_node(r))]

def extract_stats(roots):
    out = {}
    def _emit(node, path, count, ops_ctr, stream_ctr):
        key = f"{node.frame[0]}:{node.frame[1]} in {node.frame[2]}"
        entry = out.setdefault(key, {"count": 0, "callstacks": [], "ops": [], "streams": [], "stream_counts": {}})
        entry["count"] += count
        entry["callstacks"].append([list(f) for f in path])
        # ops
        if ops_ctr:
            merged = Counter(entry.get("ops_counter", {}))
            merged.update(ops_ctr)
            entry["ops_counter"] = dict(merged)
        # streams
        if stream_ctr:
            sc_merged = Counter(entry.get("stream_counts", {}))
            sc_merged.update(stream_ctr)
            entry["stream_counts"] = dict(sc_merged)

    def dfs(n, path):
        path = path + [n.frame]

        # NEW: emit the “self-as-leaf” case even if n has children
        if n.self_leaf_count > 0:
            _emit(n, path, n.self_leaf_count, n.self_leaf_ops, n.self_leaf_stream_counts)

        # OLD behavior: if no children, emit as leaf
        if not n.children:
            _emit(n, path, n.count, n.ops, n.stream_counts)
        else:
            for c in n.children.values():
                dfs(c, path)

    for r in roots:
        dfs(r, [])

    # dedupe and finalize (same as before)
    for entry in out.values():
        seen = { tuple(tuple(f) for f in cs) for cs in entry["callstacks"] }
        entry["callstacks"] = [ [list(f) for f in p] for p in seen ]
        if "ops_counter" in entry:
            oc = Counter(entry.pop("ops_counter"))
            entry["ops"] = [op for op, _ in oc.most_common()]
        if entry.get("stream_counts"):
            uniq = []
            for key in sorted(entry["stream_counts"].keys()):
                try:
                    d, s = key.split(":")
                    uniq.append([int(d), int(s)])
                except Exception:
                    pass
            entry["streams"] = uniq
        else:
            entry["streams"] = []
    return out




def main():
    parser = argparse.ArgumentParser(
        description="Compress callsite JSON by pruning/merging linear chains."
    )
    parser.add_argument("-i","--input", required=True, help="Path to original callsite JSON.")
    parser.add_argument("-o","--output", required=True, help="Path for compressed JSON output.")
    parser.add_argument(
        "--filter-patterns", nargs="+", default=DEFAULT_FILTER_PATTERNS,
        help=f"Glob patterns to filter out function names (default: {DEFAULT_FILTER_PATTERNS})"
    )
    parser.add_argument("--no-filter", action="store_true", help="Disable filtering by function name")
    parser.add_argument(
        "--no-lib-filter", action="store_true",
        help="Disable filtering of internal library function paths"
    )
    parser.add_argument(
        "--threshold-num", type=int, default=0,
        help="Minimum nodes to keep a subtree (default: 0)"
    )
    parser.add_argument(
        "--enable-leaf-pruning", action="store_true",
        help="Enable leaf node pruning (compressing linear chains)"
    )
    parser.add_argument(
        "--disable-subtree-pruning", action="store_true",
        help="Disable small subtree pruning"
    )

    args = parser.parse_args()
    filter_patterns = None if args.no_filter else args.filter_patterns
    filter_paths    = None if args.no_lib_filter else DEFAULT_FILTER_PATHS

    # Check if input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    
    # Check if input file is empty
    if os.path.getsize(args.input) == 0:
        print(f"Warning: Input file is empty: {args.input}", file=sys.stderr)
        print(f"Writing empty JSON object to output: {args.output}", file=sys.stderr)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({}, f, indent=2)
        return

    try:
        with open(args.input) as f:
            stats = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from {args.input}: {e}. The file may be corrupted or not valid JSON.") from e
    
    # Validate that stats is a dictionary
    if not isinstance(stats, dict):
        raise ValueError(f"Expected JSON object (dict) but got {type(stats).__name__} in {args.input}")

    # true total before any compression
    orig_count = sum(info.get("count", 0) * len(info.get("callstacks", [])) for info in stats.values())

    roots = build_forest(stats, filter_patterns, filter_paths)
    # optionally prune linear chains
    if args.enable_leaf_pruning:
        pruned = compress_forest(roots)
    else:
        pruned = roots
    # optionally prune small subtrees
    if not args.disable_subtree_pruning:
        pruned = prune_small_subtrees(pruned, threshold_num=args.threshold_num)

    compressed = extract_stats(pruned)
    compressed_count = sum(e["count"] for e in compressed.values())

    trimmed = orig_count - compressed_count
    pct = 100.0 * trimmed / orig_count if orig_count else 0
    print(f"Trimmed {trimmed} samples: {orig_count} → {compressed_count} ({pct:.1f}% )")

    leaf_dist = collect_leaf_distribution(pruned)
    total_leaves = sum(leaf_dist.values())
    print("\nLeaf function file distribution:")
    for fn, cnt in leaf_dist.most_common():
        pct_fn = 100.0 * cnt / total_leaves if total_leaves else 0
        print(f"  {fn:40s} count: {cnt:6d}  percentage: {pct_fn:5.2f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(compressed, f, indent=2)

    print(f"Wrote compressed JSON to {args.output}")
    if filter_patterns:
        print(f"Filtered out functions matching: {filter_patterns}")
    if filter_paths:
        print(f"Filtered out paths containing: {filter_paths}")

if __name__ == "__main__":
    main()
