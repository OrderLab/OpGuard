#!/usr/bin/env python3
import argparse
import json
from collections import Counter
import plotly.graph_objects as go

def build_icicle(data):
    """
    Build prefix_counts for every prefix in each callstack.
    Returns ids, labels, parents, values lists for Icicle chart.
    """
    prefix_counts = Counter()
    for info in data.values():
        count = info.get("count", 0)
        for cs in info.get("callstacks", []):
            path = []
            for filename, lineno, funcname in cs:
                label = f"{funcname} ({filename.split('/')[-1]}:{lineno})"
                path.append(label)
                prefix_counts["/".join(path)] += count

    ids, labels, parents, values = [], [], [], []
    for fullpath, val in prefix_counts.items():
        ids.append(fullpath)
        labels.append(fullpath.split("/")[-1])
        if "/" in fullpath:
            parent = fullpath.rsplit("/", 1)[0]
        else:
            parent = ""
        parents.append(parent)
        values.append(val)

    return ids, labels, parents, values

def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive callsite icicle chart from a JSON dump."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input JSON file (callsite dump)."
    )
    parser.add_argument(
        "-o", "--output", default="callsites_icicle.html",
        help="Path for the output HTML file (default: %(default)s)."
    )
    parser.add_argument(
        "--maxdepth", type=int, default=None,
        help="Maximum depth to display in the icicle chart (default: full depth)."
    )
    args = parser.parse_args()

    # Load JSON data
    with open(args.input, "r") as f:
        data = json.load(f)

    # Build icicle chart data
    ids, labels, parents, values = build_icicle(data)

    # Create Icicle chart config
    icicle_kwargs = dict(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total"
    )
    if args.maxdepth is not None:
        icicle_kwargs["maxdepth"] = args.maxdepth

    # Create the Icicle chart
    fig = go.Figure(go.Icicle(**icicle_kwargs))
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        title="Hierarchical Callsite Forest"
    )

    # Write to HTML
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved interactive visualization to {args.output}")

if __name__ == "__main__":
    main()
