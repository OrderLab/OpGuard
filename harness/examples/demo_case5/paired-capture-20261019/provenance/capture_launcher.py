"""Block on torchrun and bind untouched stdout/stderr to the OpGuard capture ID."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


arguments = sys.argv[1:]
if arguments[:2] != ["-m", "opguard.tools.worker"]:
    raise SystemExit("Expected the OpGuard worker module and request")
request = json.loads(Path(arguments[2]).read_text())
output = Path(request["output"])
command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc-per-node=2", *arguments]
record = {
    "capture_id": request["capture_id"], "phase": request["phase"],
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "cwd": os.getcwd(), "command": command,
    "heattrace": os.environ.get("HEATTRACE", "enabled by default"),
    "stdout": "stdout.log", "stderr": "stderr.log",
    "streams": "Complete separate streams; no filtering, tee, or postprocessing.",
}
record_path = output / "console-launch.json"
record_path.write_text(json.dumps(record, indent=2) + "\n")
with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
    process = subprocess.run(command, stdout=stdout, stderr=stderr)
record["finished_utc"] = datetime.now(timezone.utc).isoformat()
record["returncode"] = process.returncode
record["files"] = [
    {"path": name, "bytes": (output / name).stat().st_size,
     "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
    for name in ("stdout.log", "stderr.log")
]
record_path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"capture_id": request["capture_id"], "returncode": process.returncode,
                  "console_logs": [str(output / "stdout.log"), str(output / "stderr.log")]}), flush=True)
raise SystemExit(process.returncode)
