"""Verify the installed harness in a uv tool environment without its backend."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import opguard_harness

assert Path(opguard_harness.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert importlib.util.find_spec("torch") is None
assert importlib.util.find_spec("opguard") is None
response = subprocess.run([sys.executable, "-m", "opguard_harness.cli", "tools"],
                          check=True, capture_output=True, text=True)
schemas = json.loads(response.stdout)
assert schemas["protocol_version"] == 1 and schemas["tools"]
print(json.dumps({"isolated_tool": str(sys.prefix), "tool_count": len(schemas["tools"]),
                  "torch_installed": False, "opguard_installed": False, "verified": True}))
