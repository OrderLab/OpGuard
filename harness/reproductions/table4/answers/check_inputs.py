"""Maintainer regression check; expected symptoms and case mapping stay private."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from opguard_harness.input_bundle import sha256, verify_inputs
from opguard_harness.tools import dispatch


def read(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def observations(case, directory):
    result = {"case": case}
    if case in (21, 25):
        ranks = []
        for rank in (0, 1):
            a, b = (read(directory / side / f"observed/rank{rank}.pt") for side in ("baseline", "candidate"))
            assert not equal(a["loss"], b["loss"])
            different = sum(int((x != y).sum()) for x, y in zip(a["weights"], b["weights"]))
            assert different > 0
            ranks.append({"rank": rank, "different_weights": different})
        result["ranks"] = ranks
    elif case in (22, 26):
        actual = read(directory / "candidate/observed.pt")
        reference = read(directory / "reference.pt")
        if case == 22:
            assert equal(actual["input"], reference["input"])
        else:
            assert all(equal(actual["inputs"][0], x) for x in actual["inputs"])
            assert equal(actual["inputs"][0], reference["inputs"][0])
        assert not equal(actual["outputs"][0], actual["outputs"][1])
        assert all(equal(reference["outputs"][0], x) for x in reference["outputs"])
        result["repeat_divergence"] = True
    elif case == 24:
        ranks = []
        for rank in range(4):
            a, b = (read(directory / side / f"observed/rank{rank}.pt") for side in ("baseline", "candidate"))
            assert all(equal(a[k], b[k]) for k in ("input", "routing", "probs", "forward", "counts", "dy"))
            assert not equal(a["dx"], b["dx"])
            ranks.append({"rank": rank, "different_input_gradients": int((a["dx"] != b["dx"]).sum())})
        result["ranks"] = ranks
    else:
        a, b = (read(directory / side / "observed.pt") for side in ("baseline", "candidate"))
        if case == 23:
            assert a["vocab"].shape != b["vocab"].shape and not equal(a["loss"], b["loss"])
        elif case == 27:
            assert not equal(a, b)
        else:
            assert not equal(a["logits"], b["logits"])
        result["divergence"] = True
    result["observations_sha256"] = {str(p.relative_to(directory)): sha256(p)
                                      for p in directory.rglob("*.pt")}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("private_records", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--capture", nargs="*", type=int, default=[])
    args = parser.parse_args()
    private, workdir = args.private_records.resolve(), args.workdir.resolve()
    provenance = json.loads((private / "provenance.json").read_text())
    bundle = Path(provenance["input_bundle"])
    verify_inputs(bundle)
    shutil.copytree(bundle, workdir)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    records = []
    for case in range(21, 29):
        task = provenance["cases"][str(case)]["task"]
        directory = workdir / "tasks" / task
        for capture in ([False, True] if case in args.capture else [False]):
            suffix = "capture" if capture else "direct"
            log = private / f"check-{case}-{suffix}.log"
            command = [sys.executable, str(workdir / "run_task.py"), task]
            if capture:
                command.append("--capture")
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=workdir, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=1800)
            record = {"case": case, "task": task, "mode": suffix, "exit_code": completed.returncode,
                      "log": str(log), "directory": str(directory)}
            records.append(record)
            (private / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
            if completed.returncode:
                raise RuntimeError(f"Workload failed; inspect {log}")
            record["observations"] = observations(case, directory)
            if capture:
                capture_result = json.loads((directory / "capture-results.json").read_text())
                reports = capture_result["comparison"]["reports"]
                queries = []
                for artifact in reports:
                    query = dispatch(workdir / "runs", "first_diff", {"artifact": artifact})
                    assert all(b["first_observed_diff"] is not None for b in query["boundaries"])
                    queries.append(query)
                    for term in ("The wait fix", "--fix", "if a.fix:", "wait_stream(optimizer.reduction_stream)"):
                        found = dispatch(workdir / "runs", "search_source", {"artifact": artifact, "query": term})
                        assert not found["hits"], found
                record["harness_queries"] = queries
                record["embedded_repair_hints"] = False
            record["checked"] = True
            (private / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({"case": case, "mode": suffix, "checked": True}), flush=True)
    verify_inputs(bundle)


if __name__ == "__main__":
    main()
