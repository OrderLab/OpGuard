#!/usr/bin/env python3
from datetime import datetime
import re, os, json
from collections import defaultdict, OrderedDict
from opguard.analyze import (
    filter_diff_trace,
    extract_fine_grained_frame,
    extract_rank_from_path
)

# ------------------ BLACKLIST FILTER ------------------
# Add function names here to block their diffs from analysis.
BLACKLIST = {
    "<non-function>"
    # example: "<non-function>", "another_func_to_block"
}

def is_blacklisted(func_name):
    return func_name in BLACKLIST

def extract_basic_info(obj, dp_degree):
    counter = obj.get("counter", 0)
    timestamp = obj.get("timestamp", "20000101-000000-000000")
    func = obj.get("function", "<unknown>")
    stack = obj.get("stack_tail", "")
    rank = extract_rank_from_path(obj.get("file", ""))
    dp_group = -1 if dp_degree == 0 else rank // dp_degree
    fname = obj.get("fname")
    meta = obj.get("meta", {})
    diffs = obj["diff_details"].get("io_diff_paths", [])
    return counter, timestamp, func, stack, rank, dp_group, fname, meta, diffs

def extract_meta_info(meta):
    gs = meta.get("global_step") if "global_step" in meta else (meta.get("step") if "step" in meta else "<no_gs>")
    mb = meta.get("micro_batch_idx", "<no_mb>")
    chunk = meta.get("model_chunk_id", -1)
    direction = meta.get("propagation_direction", "<no_direction>")
    return gs, mb, chunk, direction


def build_expanded_events(stack_quints, dp_degree):
    events = []
    for obj in stack_quints:
        counter, timestamp, func, stack, rank, dp_group, fname, meta, diffs = extract_basic_info(obj, dp_degree)
        # stack_detail = extract_fine_grained_frame(stack)

        if is_blacklisted(func):
            continue
           
        events.append({
            "rank": rank,
            "dp_group": dp_group,
            "counter": counter,
            "timestamp": timestamp,
            "func": func,
            "stack": stack,
            "fname": fname,
            "meta": meta,
            "diff_params": diffs,
            "ori_trace1": obj.get("ori_trace1", None),
            "ori_trace2": obj.get("ori_trace2", None),
        })
    
    return events


def group_events(events):
    gs_mb_group = defaultdict(list)
    gs_mb_forward_group = defaultdict(list)
    gs_mb_backward_group = defaultdict(list)
    gs_mb_unknown_group = defaultdict(list)
    gs_mb_set = set()

    for event in events:
        gs, mb, chunk, direction = extract_meta_info(event["meta"])
        gs_mb = (gs, mb)

        gs_mb_set.add(gs_mb)
        
        # TODO: for mul node in dp group, timestamp can not decide the order
        sort_key = (chunk, event["dp_group"], event["timestamp"], event["rank"])
        if direction == "forward":
            gs_mb_forward_group[gs_mb].append((sort_key, event))
        elif direction == "backward":
            gs_mb_backward_group[gs_mb].append((sort_key, event))
        else:
            gs_mb_unknown_group[gs_mb].append((sort_key, event))
        
    
    for key in gs_mb_set:
        gs_mb_forward_group[key].sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3]))
        gs_mb_backward_group[key].sort(key=lambda x: (-x[0][0], -x[0][1], x[0][2], x[0][3]))
        gs_mb_unknown_group[key].sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3]))
        
        gs_mb_group[key] = [x[1] for x in gs_mb_forward_group[key]]
        gs_mb_group[key] += [x[1] for x in gs_mb_backward_group[key]]
        gs_mb_group[key] += [x[1] for x in gs_mb_unknown_group[key]]
    
    return gs_mb_group
    

def get_diff_priority(diff_path:str):
    if diff_path.startswith("inputs.self"):
        return str(0)+diff_path[len("inputs.self"):]
    elif diff_path.startswith("inputs.args"):
        return str(1)+diff_path[len("inputs.args"):]
    elif diff_path.startswith("inputs.kwargs"):
        return str(2)+diff_path[len("inputs.kwargs"):]
    elif diff_path.startswith("outputs"):
        return str(3)+diff_path[len("outputs"):]
    else:
        return str(4)+diff_path

def reset_args_index(path:str):
    if not path.startswith("inputs.args."):
        return path

    pattern = r'(inputs\.args\.)\[(\d+)\]'
    
    match = re.search(pattern, path)
    if not match:
        return path
    
    try:
        prefix = match.group(1)
        num_str = match.group(2)
        
        new_num = int(num_str) - 1
        return re.sub(pattern, f'{prefix}[{new_num}]', path, count=1)
    
    except Exception:
        return path

def formate_diff_params(diff_params, ori_trace1, ori_trace2):
    template = """
  - {diff_key}:
    ↪ run1={run1}
    ↪ run2={run2}"""
    diff_str = ""
    
    for diff in sorted(diff_params, key=lambda x: get_diff_priority(x["path"])):
        diff_str += template.format(diff_key=reset_args_index(diff["path"]), run1=diff["run1"], run2=diff["run2"])

    if ori_trace1 and ori_trace2:
        diff_str += f"""
  - ori_run1: {ori_trace1}
  - ori_run2: {ori_trace2}"""
    return diff_str

def formate_stack(stack):
    if not stack:
        return "<empty>"
    res = ''
    for line in stack.split("\n"):
        res += f"\n   ↪ {line}"
    return res

def ascii_visualize_chain_to_file(chains, gs, mb, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"gs{gs}_mb{mb}.txt"
    path = os.path.join(out_dir, fname)
    template = """• Diff detected at {direction} | Rank {rank}, Chunk {chunk}, DP Group {dp_group} | Time {timestamp} | Func {func} | Fname {fname}
  Meta : {meta}
  Stack: {stack}
  Diff : {diff}

"""
    with open(path, 'w') as f:
        f.write(f"Effect Chain for global_step={gs}, micro_batch={mb}\n")
        for chain in chains:
            f.write(template.format(
                direction=chain["meta"].get("propagation_direction", "unknown"),
                rank=chain["rank"],
                chunk=chain["meta"].get("model_chunk_id", 0),
                dp_group=chain["dp_group"],
                timestamp=datetime.strptime(chain["timestamp"], "%Y%m%d-%H%M%S-%f").strftime("%H:%M:%S.%f"),
                func=chain["func"],
                fname=chain["fname"],
                stack=formate_stack(chain["stack"]),
                meta=chain["meta"],
                diff=formate_diff_params(chain["diff_params"], chain["ori_trace1"], chain["ori_trace2"]),
            ))

    print(f"Wrote {path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build & dump ASCII effect chains per (global_step, micro_batch)."
    )
    parser.add_argument("--dp", default=0, type=int, help="data parallelism degree")
    parser.add_argument("--diff_log", required=True, help="path to combined_diff_log.jsonl")
    parser.add_argument("--out-dir", default=".", help="directory to write per-chain logs")
    args = parser.parse_args()

    # 1. filter diff trace(no use now)
    filtered_trace = filter_diff_trace(args.diff_log)

    # 2. expand events for every diff field, filtering out blacklisted funcs
    events      = build_expanded_events(filtered_trace, int(args.dp))
    
    # 3. group events by (gs, mb) ans sort events
    chains      = group_events(events)

    # 4. format per-(gs,mb) diff
    for (gs, mb), chain in chains.items():
        ascii_visualize_chain_to_file(chain, gs, mb, args.out_dir)
