#!/usr/bin/env python3
"""Recheck the recorded OpGuard observations without executing a model."""

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SELECTED_STEPS = (1, 2, 32, 33, 64)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_evidence():
    manifest = json.loads((ROOT / "evidence/manifest.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        data = path.read_bytes()
        require(len(data) == item["bytes"], f"Size changed: {item['path']}")
        require(hashlib.sha256(data).hexdigest() == item["sha256"],
                f"SHA-256 changed: {item['path']}")
    records = {}
    with tarfile.open(ROOT / "evidence/captures.tar.gz", "r:gz") as archive:
        expected = {item["path"]: item for item in manifest["capture_members"]}
        for member in archive:
            require(member.isfile() and member.name in expected,
                    f"Unexpected capture member: {member.name}")
            require(member.name not in records, f"Duplicate member: {member.name}")
            data = archive.extractfile(member).read()
            item = expected[member.name]
            require(len(data) == item["bytes"] and
                    hashlib.sha256(data).hexdigest() == item["sha256"],
                    f"Capture changed: {member.name}")
            records[member.name] = [json.loads(line) for line in data.splitlines()]
        require(set(records) == set(expected), "Capture files are missing")
    return records


def replay_round(number, records, compare_regions, expected):
    def rows(site, rank):
        return records[f"r{number}-{site}-rank{rank}.jsonl"]

    pairs, differing = 0, 0
    for step in SELECTED_STEPS:
        # This correspondence was selected manually in the investigation:
        # FSDP DP ranks 0/1 correspond to Megatron TP=0 ranks 0/2.
        for a_rank, b_rank in ((0, 0), (1, 2)):
            a = [r for r in rows("a.batch", a_rank) if r["step"] == step]
            b = [r for r in rows("b.batch", b_rank)
                 if r["step"] == step and r["train"]]
            require(len(a) == 1 and len(b) == 4,
                    f"Unexpected microbatch count in round {number}, step {step}")
            require(len(a[0]["inputs"]) == 4, "Expected four A microbatches")
            for left, right in zip(a[0]["inputs"], b):
                right = right["inputs"]
                require(left["shape"] == right["shape"], "Incompatible input tensor shapes")
                result = compare_regions(left["stats"], right["stats"])
                require(result["status"] == "available", "Unsupported fingerprint")
                pairs += 1
                differing += bool(result["differing_regions"])

    evaluation = {}
    for side, ranks in (("a", (0, 1)), ("b", (0, 2))):
        final = [r for rank in ranks
                 for r in [r for r in rows(f"{side}.loss", rank) if not r["train"]][-16:]]
        require(len(final) == 32, "Expected 32 final evaluation microbatches")
        targets = sum(r["targets"] for r in final)
        require(targets == 32704 and all(r["targets"] == 1022 for r in final),
                "Unexpected final evaluation target counts")
        total = sum(r["loss"] * r["targets"] if side == "a" else r["nll_sum"]
                    for r in final)
        evaluation[side] = total / targets
    a_hashes = Counter(h for rank in (0, 1)
                       for r in [r for r in rows("a.loss", rank) if not r["train"]][-16:]
                       for h in r["rows"])
    b_hashes = Counter(h for rank in (0, 2)
                       for r in [r for r in rows("b.batch", rank) if not r["train"]][-16:]
                       for h in r["rows"])
    require(a_hashes == b_hashes and sum(a_hashes.values()) == 64,
            "Final evaluation row-hash multisets differ")

    round_rows = [r for name, values in records.items() if name.startswith(f"r{number}-")
                  for r in values]
    summaries = 0
    for row in round_rows:
        values = row.get("inputs") or row.get("output")
        summaries += len(values) if isinstance(values, list) else int(values is not None)
    require((pairs, differing) == (40, expected["differing_input_pairs"]),
            "Recorded input comparisons changed")
    require(len(round_rows) == 726 and summaries == 120, "Capture counts changed")
    for side in ("a", "b"):
        require(math.isclose(evaluation[side], expected[f"eval_{side}"], abs_tol=1e-12,
                             rel_tol=0), f"Round {number} evaluation mismatch")
    return {
        "round": number, "input_pairs": pairs, "differing_input_pairs": differing,
        "emitted_records": len(round_rows), "opguard_tensor_summaries": summaries,
        "eval_a": evaluation["a"], "eval_b": evaluation["b"],
        "eval_abs_gap": abs(evaluation["a"] - evaluation["b"]),
        # Checkpoints are deliberately not bundled. Do not present this as a
        # fresh comparison of 42 million parameter values.
        "recorded_checkpoint_rms64": expected["model_rms64"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-root", type=Path,
                        help="Use an OpGuard-harness checkout instead of the bundled pinned comparator")
    args = parser.parse_args()
    records = verified_evidence()
    if args.harness_root:
        sys.path.insert(0, str(args.harness_root.resolve()))
        try:
            fingerprints = importlib.import_module("opguard_harness.fingerprints")
        except ModuleNotFoundError as error:
            if error.name not in {"opguard_harness", "opguard_harness.fingerprints"}:
                raise
            parser.error("--harness-root must contain opguard_harness/fingerprints.py")
    else:
        from capture import fingerprints
    result = json.loads((ROOT / "evidence/results.json").read_text())
    rounds = [replay_round(row["round"], records, fingerprints.compare_regions, row)
              for row in result["rounds"]]
    print(json.dumps({
        "verified": True,
        "mode": "offline_reanalysis_of_recorded_native_training_observations",
        "rounds": rounds,
        "checkpoint_values_recomputed": False,
        "fingerprint_equality_is_tensor_equality": False,
        "automatic_semantic_correspondence": False,
        "bitwise_alignment_claimed": False,
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        sys.exit(f"Replay failed: {error}")
