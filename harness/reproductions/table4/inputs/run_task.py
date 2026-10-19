"""Run the selected input configuration or create a paired harness capture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    task_dir = (root / "tasks" / args.task).resolve()
    if task_dir.parent != root / "tasks":
        parser.error("Select a task directly inside tasks/")
    task = json.loads((task_dir / "task.json").read_text())
    launches = {}
    for side, workload in task["workloads"].items():
        launch = dict(workload)
        launch["cwd"] = str(task_dir / side)
        launch["launcher"] = [sys.executable]
        if len(launch["ranks"]) > 1:
            launch["launcher"] += ["-m", "torch.distributed.run", "--standalone",
                                   f"--nproc-per-node={len(launch['ranks'])}"]
        launches[side] = launch
    if args.capture:
        if set(launches) != {"baseline", "candidate"}:
            parser.error("This task supplies saved reference tensors, not an executable baseline; use the direct run and inspect its repeated outputs.")
        from opguard_harness.tools import dispatch
        store = root / "runs"
        spec = {"label": args.task, "input_id": args.task, **launches,
                "backend": [sys.executable, "-m", "opguard.tools"],
                "local_root": str(root / "staging"), "timeout_seconds": 600}
        experiment = dispatch(store, "create_experiment", {"spec": spec})
        while experiment["status"] == "pending":
            experiment = dispatch(store, "advance_experiment", {"experiment_id": experiment["id"]})
        result = dispatch(store, "experiment_results", {"experiment_id": experiment["id"]})
        (task_dir / "capture-results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if result["status"] != "succeeded":
            raise SystemExit(1)
    else:
        for side, launch in launches.items():
            env = {k: v for k, v in os.environ.items() if not k.startswith("OPGUARD_")
                   and k != "DEFAULT_CODE_REPO_BASE"}
            env.update(launch["env"])
            env["OPGUARD_AUTOSTART"] = "0"
            subprocess.run([*launch["launcher"], launch["script"], *launch["args"]],
                           cwd=launch["cwd"], env=env, check=True, timeout=600)


if __name__ == "__main__":
    main()
