#!/usr/bin/env python3
"""Verify same-launch console/trace provenance and replay the recorded queries."""

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from opguard_harness.tools import dispatch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def main():
    manifest = read_json(ROOT / "manifest.json")
    for item in manifest["files"]:
        data = (ROOT / item["path"]).read_bytes()
        require(len(data) == item["bytes"] and
                hashlib.sha256(data).hexdigest() == item["sha256"],
                f"Changed file: {item['path']}")

    summary = read_json(ROOT / "opguard-output.json")
    metrics = read_json(ROOT / "metrics.json")
    pattern = re.compile(
        r"^rank \[0\] global_step \[(\d+)\] loss \[([^]]+)\] grad_norm \[([^]]+)\]$", re.M)
    with tarfile.open(ROOT / "raw-captures.tar.gz", "r:gz") as archive:
        for link in summary["run_mapping"]:
            label = link["label"]
            provenance = read_json(ROOT / link["provenance"])
            launch = provenance["console_launch"]
            worker = provenance["worker_request"]
            response = provenance["backend_response"]
            require(response["ok"] and launch["returncode"] == 0,
                    f"Unsuccessful capture: {label}")
            capture_id = link["capture_id"]
            require(capture_id == launch["capture_id"] == worker["capture_id"] ==
                    response["result"]["capture_id"], "Console/trace capture IDs differ")
            receipts = provenance["rank_receipts"]
            require(sorted(r["rank"] for r in receipts) == [0, 1], "Missing rank receipt")
            require(all(r["capture_id"] == capture_id and r["phase"] == "trace"
                        for r in receipts), "Rank receipt belongs to another capture")
            side = "baseline" if label == "overlap-off" else "candidate"
            prefix = f"experiment/{side}-trace-1/output/"
            for stream in ("stdout", "stderr"):
                data = (ROOT / link[stream]).read_bytes()
                require(data == archive.extractfile(prefix + stream + ".log").read(),
                        f"Console differs from the original capture archive: {label}/{stream}")
            require(json.load(archive.extractfile(prefix + "worker.json")) == worker,
                    "Archived worker request differs")
            require(json.load(archive.extractfile(prefix + "console-launch.json")) == launch,
                    "Archived console launch differs")
            for receipt in receipts:
                require(json.load(archive.extractfile(
                    prefix + f".opguard-complete/rank-{receipt['rank']}.json")) == receipt,
                    "Archived rank receipt differs")
            rows = [{"step": int(s), "loss": float(loss), "grad_norm": float(norm)}
                    for s, loss, norm in pattern.findall((ROOT / link["stdout"]).read_text())]
            require([row["step"] for row in rows] == list(range(20)),
                    f"Incomplete native training metrics: {label}")
            require(rows == metrics["metrics"][label], "Derived metrics differ from console output")

    for rank in summary["ranks"]:
        trace = json.loads(gzip.decompress((ROOT / rank["artifact"]).read_bytes()))
        for observed in rank["observations"]:
            digest, index = observed["event_ref"].split(":")
            require(digest == rank["sha256"], "Event reference belongs to another artifact")
            event = trace["traceEvents"][int(index)]
            require(event["name"] == observed["operation"] and
                    event["args"]["run"] == observed["run"] and
                    event["args"]["rank"] == rank["rank"] and
                    event["args"]["io"] == observed["io"] and
                    event["args"]["status"] == "diff", "Compact output differs from raw trace")

    queries = read_json(ROOT / "provenance/query-catalog.json")
    for query in queries:
        arguments = {**query["arguments"], "artifact": str(ROOT / query["artifact"])}
        actual = dispatch(ROOT, query["tool"], arguments)
        saved = read_json(ROOT / query["response"])
        require(saved["ok"], "Recorded query failed")
        expected = saved["result"]
        # Only the local filename changes after checkout; artifact SHA stays fixed.
        actual["artifact"]["path"] = expected["artifact"]["path"]
        require(actual == expected, f"Query replay changed: {query['response']}")

    different_steps = [a["step"] for a, b in zip(metrics["metrics"]["overlap-off"],
                                               metrics["metrics"]["overlap-on"]) if a != b]
    require(different_steps == metrics["different_steps"], "Recorded metric differences changed")
    print(json.dumps({"verified": True, "same_launch_console_and_trace": True,
                      "trace_launches": 2, "ranks_per_launch": 2,
                      "native_steps_per_launch": 20, "replayed_queries": len(queries),
                      "different_native_metric_steps": different_steps,
                      "whole_graph_first_diff_claimed": False}, indent=2))


if __name__ == "__main__":
    main()
