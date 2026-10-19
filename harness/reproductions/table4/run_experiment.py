"""Run an explicit spec through the public harness API, retaining every attempt."""
import argparse
import json
from pathlib import Path

from opguard_harness.tools import dispatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.resume:
        experiment_id = args.resume
    else:
        response = dispatch(args.root, "create_experiment", {"spec": spec})
        experiment_id = response["id"]
    print(json.dumps({"experiment_id": experiment_id}), flush=True)
    (args.spec.parent / (args.spec.stem + ".experiment-id")).write_text(experiment_id + "\n")
    while True:
        response = dispatch(args.root, "experiment_status", {"experiment_id": experiment_id})
        status = response
        print(json.dumps({"status": status["status"], "next": status["next"]}), flush=True)
        if status["status"] == "succeeded":
            break
        response = dispatch(args.root, "advance_experiment", {"experiment_id": experiment_id})
        if response["status"] == "failed":
            print(json.dumps(response, indent=2), flush=True)
            raise SystemExit(1)
    result = dispatch(args.root, "experiment_results", {"experiment_id": experiment_id})
    (args.spec.parent / (args.spec.stem + ".results.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
