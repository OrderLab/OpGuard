#!/usr/bin/env python3
"""Verify recovered console logs and recheck the recorded resume input boundary."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_captures():
    manifest = json.loads((EXAMPLE / "evidence/manifest.json").read_text())
    expected = {item["path"]: item for item in manifest["capture_members"]}
    records = {}
    with tarfile.open(EXAMPLE / "evidence/captures.tar.gz", "r:gz") as archive:
        for member in archive:
            require(member.isfile() and member.name in expected,
                    f"Unexpected capture: {member.name}")
            data = archive.extractfile(member).read()
            require(hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"],
                    f"Changed capture: {member.name}")
            records[member.name] = [json.loads(line) for line in data.splitlines()]
    require(set(records) == set(expected), "Missing capture files")
    return records


def boundary_report():
    records = read_captures()
    catalog = json.loads((ROOT / "row-catalog.json").read_text())["rows"]
    spec = importlib.util.spec_from_file_location(
        "recorded_fingerprints", EXAMPLE / "capture/fingerprints.py")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)

    def select(round_number, step, side):
        ranks = (0, 1) if side == "a" else (0, 2)
        result = []
        for rank in ranks:
            name = f"r{round_number}-{side}.batch-rank{rank}.jsonl"
            rows = [(i + 1, r) for i, r in enumerate(records[name])
                    if r["step"] == step and r.get("train", True)]
            require(len(rows) == (1 if side == "a" else 4),
                    f"Unexpected batch count: {name}, step {step}")
            if side == "a":
                line, record = rows[0]
                result.append([
                    {"rows": record["rows"][micro], "targets": record["targets"][micro],
                     "input": record["inputs"][micro],
                     "capture": name, "line": line, "microbatch": micro}
                    for micro in range(4)])
            else:
                result.append([
                    {"rows": record["rows"], "targets": record["targets"],
                     "input": record["inputs"], "capture": name,
                     "line": line, "microbatch": micro}
                    for micro, (line, record) in enumerate(rows)])
        return result

    def match_window(observations):
        # Match all 16 rows and eight target counts, not one hash that may repeat.
        matches = []
        for start in range(len(catalog) - 15):
            ok = True
            for dp_rank, microbatches in enumerate(observations):
                for micro, observed in enumerate(microbatches):
                    offset = start + micro * 4 + dp_rank * 2
                    pair = catalog[offset:offset + 2]
                    if ([r["sha256_64"] for r in pair] != observed["rows"] or
                            sum(r["causal_targets"] for r in pair) != observed["targets"]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                matches.append(start)
        require(len(matches) == 1, f"Ambiguous row window: {matches}")
        return matches[0]

    observations = []
    for round_number in (1, 2):
        for step in (1, 2, 32, 33, 64):
            a, b = select(round_number, step, "a"), select(round_number, step, "b")
            different = 0
            pairs = []
            for dp_rank in (0, 1):
                for micro in range(4):
                    left, right = a[dp_rank][micro], b[dp_rank][micro]
                    require(left["input"]["shape"] == right["input"]["shape"],
                            "Input shapes differ")
                    result = comparator.compare_regions(left["input"]["stats"],
                                                        right["input"]["stats"])
                    require(result["status"] == "available", "Unavailable comparison")
                    different += bool(result["differing_regions"])
                    if step == 33 and micro == 0:
                        pairs.append({"dp_rank": dp_rank, "a": left, "b": right,
                                      "differing_regions": result["differing_regions"]})
            observations.append({
                "round": round_number, "update": step,
                "a_global_batch_start": match_window(a),
                "b_global_batch_start": match_window(b),
                "compared_microbatches": 8, "differing_microbatches": different,
                "first_microbatch_per_dp_rank": pairs,
            })
    return {
        "kind": "derived_reanalysis_of_existing_probe_records",
        "selected_updates": [1, 2, 32, 33, 64],
        "rank_correspondence": [[0, 0], [1, 2]],
        "observations": observations,
        "limits": [
            "This is a sampled input boundary, not an automatic whole-graph first-diff.",
            "Equal XOR fingerprints do not prove tensor equality.",
            "Round 2 applied several repairs together; it is not a resume-only intervention.",
        ],
    }


def console_report():
    observations = []
    for round_number in (1, 2):
        for side in ("a", "b"):
            for phase in (1, 2):
                path = ROOT / f"standard-logs/round{round_number}/run_{side}/phase_{phase}.log"
                updates = []
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    if side == "a" and line.startswith("{'loss':"):
                        row = ast.literal_eval(line)
                        updates.append({"line": number, "loss": row["loss"],
                                        "epoch": row["epoch"]})
                    elif side == "b":
                        match = re.search(
                            r"iteration\s+(\d+)/\s*64\s*\| consumed samples:\s*(\d+)", line)
                        if match:
                            updates.append({"line": number, "update": int(match[1]),
                                            "consumed_samples": int(match[2])})
                require(len(updates) == 32, f"Expected 32 updates in {path}")
                if side == "b":
                    require([r["update"] for r in updates] ==
                            list(range((phase - 1) * 32 + 1, phase * 32 + 1)),
                            "Megatron update order changed")
                observations.append({"round": round_number, "side": side, "phase": phase,
                                     "file": str(path.relative_to(ROOT)), "updates": 32,
                                     "first": updates[0], "last": updates[-1]})
    return observations


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for item in manifest["files"]:
        data = (ROOT / item["path"]).read_bytes()
        require(len(data) == item["bytes"] and
                hashlib.sha256(data).hexdigest() == item["sha256"],
                f"Changed file: {item['path']}")
    report = boundary_report()
    require(report == json.loads((ROOT / "input-boundary.json").read_text()),
            "Stored input-boundary report differs from capture reanalysis")
    for row in report["observations"]:
        expected_a = ((row["update"] - 1) * 16 if row["round"] == 2 or row["update"] <= 32
                      else 128 + (row["update"] - 33) * 16)
        require(row["a_global_batch_start"] == expected_a and
                row["b_global_batch_start"] == (row["update"] - 1) * 16,
                "Unexpected resumed data progress")
        require(row["differing_microbatches"] ==
                (8 if row["round"] == 1 and row["update"] >= 33 else 0),
                "Unexpected input fingerprint comparison")
    console = console_report()
    a = [r for r in console if r["round"] == 1 and r["side"] == "a"]
    require(a[0]["last"]["epoch"] == 0.5 and a[1]["first"]["epoch"] == 0.140625,
            "Expected epoch regression is missing from the original console output")
    print(json.dumps({"verified": True, "native_log_files": 8,
                      "raw_log_bytes": sum(r["bytes"] for r in manifest["files"]
                                            if r["path"].startswith("standard-logs/")),
                      "training_rerun_performed": False,
                      "streams": "stdout and stderr merged by the original launchers",
                      "standard_logs_already_show_epoch_regression": True,
                      "update33": [r for r in report["observations"] if r["update"] == 33],
                      "console": console}, indent=2))


if __name__ == "__main__":
    main()
