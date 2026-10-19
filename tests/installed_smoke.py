"""Run outside the checkout after installing the built wheel; no GPU is needed."""
import gzip
import argparse
import importlib
from importlib import metadata, resources
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-harness", action="store_true", help="Also query the report through an installed harness")
    args = parser.parse_args()
    import opguard
    package = Path(opguard.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve()), (package, sys.prefix)
    for name in ("tracer.tensor_dump", "instrumentor.op_wrapper.instrumentor",
                 "callsite.model", "tools.operations", "trace_export.perfetto.build",
                 "post_processing.analyze_diff"):
        module = importlib.import_module("opguard." + name)
        assert Path(module.__file__).resolve().is_relative_to(package)
    for path in ("config.yaml", "_running_config.yaml", "collections/default_config.yaml",
                 "bitwise_xor/bitwise_xor_kernel.cu", "callsite_tracer/native/hook.cpp"):
        assert resources.files("opguard").joinpath(path).is_file(), path
    dist = metadata.metadata("opguard")
    assert dist["Requires-Python"] == ">=3.10"
    assert "torch>=2.4" in dist.get_all("Requires-Dist")

    def backend(action, request=None):
        result = subprocess.run([sys.executable, "-m", "opguard.tools", action],
                                input=json.dumps(request) if request else None,
                                text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        response = json.loads(result.stdout)
        assert response["ok"], response
        return response["result"]

    assert backend("capabilities")["numeric_verdict"] is False
    with tempfile.TemporaryDirectory(prefix="opguard-installed-") as temporary:
        root = Path(temporary)
        captures = []
        for side, offset in (("a", "0.0"), ("b", "0.125")):
            source = root / side
            source.mkdir()
            script = source / "train.py"
            script.write_text('''import json, torch
torch.manual_seed(17)
def train_step(x, weight):
    return torch.matmul(x, weight).sum()
x = torch.randn(4, 8) + OFFSET
weight = torch.randn(8, 2, requires_grad=True)
loss = train_step(x, weight)
loss.backward()
print("WORKLOAD_RESULT:" + json.dumps({"loss": float(loss.detach()), "gradient": weight.grad.tolist()}))
'''.replace("OFFSET", offset))
            original = script.read_bytes()
            workload = {"cwd": str(source), "script": str(script), "args": [],
                        "launcher": [sys.executable], "ranks": [0], "env": {
                            "OPGUARD_CALLSITE_INCLUDE_CPU": "1",
                            "OPGUARD_CALLSITE_CUPTI": "0", "OPGUARD_CALLSITE_CUDAAPI": "0"}}
            request = {"protocol_version": 1, "local_root": str(root / "local"),
                       "timeout_seconds": 60, "workload": workload}
            discovery = backend("discover", {**request, "output": str(root / (side + "-discover"))})
            plan = backend("prepare", {**request, "output": str(root / (side + "-plan")), "discovery": discovery})
            capture = backend("trace", {**request, "output": str(root / (side + "-trace")), "plan": plan})
            assert script.read_bytes() == original, "Instrumented source was not restored"
            observations = []
            for execution in (discovery, capture):
                lines = (Path(execution["output"]) / "workload.log").read_text().splitlines()
                observed = [json.loads(line.removeprefix("WORKLOAD_RESULT:"))
                            for line in lines if line.startswith("WORKLOAD_RESULT:")]
                assert len(observed) == 1, "Missing workload result"
                observations.append(observed[0])
            assert observations[0] == observations[1], "Instrumentation changed loss or gradients"
            captures.append(capture)
        result = backend("compare", {"protocol_version": 1, "output": str(root / "comparison"),
                                     "baseline": captures[0], "candidate": captures[1], "ranks": [0]})
        assert result["verdict"] == "not_evaluated"
        with gzip.open(result["reports"][0], "rt") as stream:
            report = json.load(stream)
        operations = [e for e in report["traceEvents"] if e.get("cat") == "opguard" and e.get("ph") == "X"]
        assert operations and report["sourceFiles"], "Empty report from installed package"
        assert any(e["args"].get("first_diff") for e in operations), "Known input difference was not observed"
        if args.with_harness:
            import opguard_harness
            from opguard_harness.tools import TOOLS, dispatch
            assert Path(opguard_harness.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
            assert any(tool["name"] == "first_diff" for tool in TOOLS)
            for name in ("trace_summary", "first_diff"):
                assert dispatch(root / "investigation", name, {"artifact": result["reports"][0]})
        print(json.dumps({"installed_package": str(package), "version": metadata.version("opguard"),
                          "native_workload_launches": 4, "comparison_operations": len(operations),
                          "harness_queries_verified": args.with_harness,
                          "loss_and_gradients_preserved": True,
                          "source_restored": True, "verified": True}))


if __name__ == "__main__":
    main()
