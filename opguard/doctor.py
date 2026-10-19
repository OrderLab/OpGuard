"""Inspect an installation without compiling extensions or starting a capture."""
import argparse
import ctypes
from ctypes.util import find_library
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def _module(name, symbols=()):
    try:
        module = importlib.import_module(name)
        for symbol in symbols:
            getattr(module, symbol)
        return {"available": True, "path": module.__file__}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _library(candidates):
    errors = []
    for name in dict.fromkeys(str(p) for p in candidates if p):
        try:
            ctypes.CDLL(name)
            return {"available": True, "path": name}
        except OSError as exc:
            errors.append(str(exc))
    return {"available": False, "reason": "; ".join(errors) or "not installed"}


def inspect():
    package = Path(__file__).resolve().parent
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "platform": platform.platform(), "package": str(package)}
    try:
        report["version"] = metadata.version("opguard")
    except metadata.PackageNotFoundError:
        report["version"] = "source checkout"
    report["torch"] = _module("torch")
    torch_paths = []
    if report["torch"]["available"]:
        import torch
        report["torch"].update(version=torch.__version__, cuda=torch.version.cuda,
                               cxx11_abi=torch.compiled_with_cxx11_abi())
        try:
            report["torch"]["cuda_available"] = torch.cuda.is_available()
        except Exception as exc:
            report["torch"].update(cuda_available=False, cuda_error=str(exc))
        site = Path(torch.__file__).resolve().parent.parent
        for directory in ("nvidia/cuda_cupti/lib", "nvidia/cu13/lib"):
            torch_paths.extend(sorted((site / directory).glob("libcupti.so*")))
    nvcc = shutil.which("nvcc")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home and nvcc:
        cuda_home = str(Path(nvcc).resolve().parent.parent)
    if not cuda_home and Path("/usr/local/cuda").is_dir():
        cuda_home = "/usr/local/cuda"
    toolkit_cupti, cupti_headers = [], []
    if cuda_home:
        cuda = Path(cuda_home)
        nvcc = nvcc or (str(cuda / "bin/nvcc") if (cuda / "bin/nvcc").is_file() else None)
        for pattern in ("lib64/libcupti.so*", "extras/CUPTI/lib64/libcupti.so*", "targets/*/lib/libcupti.so*"):
            toolkit_cupti.extend(sorted(cuda.glob(pattern)))
        for pattern in ("include/cupti.h", "extras/CUPTI/include/cupti.h", "targets/*/include/cupti.h"):
            cupti_headers.extend(str(p) for p in cuda.glob(pattern))
    report["cuda_toolkit"] = {"home": cuda_home, "nvcc": nvcc}
    report["cupti"] = _library([find_library("cupti"), *torch_paths, *toolkit_cupti])
    report["cupti"]["headers"] = cupti_headers
    report["extensions"] = {
        "xor": _module("xor_reduce_cuda", ("xor_reduce_cuda", "xor_reduce_regional_cuda")),
        "cupti": _module("opguard.callsite_tracer.opguard_cupti", ("start_mem", "drain")),
        "native_hook": _library([package / "callsite_tracer/native/libpylaunch_hook.so"]),
    }
    build_file = package / "_native_build.json"
    report["native_build"] = json.loads(build_file.read_text()) if build_file.is_file() else None
    report["compatibility_errors"] = []
    if report["native_build"] and report["torch"]["available"]:
        build = report["native_build"]
        for key, current in (("torch", torch.__version__.split('+')[0]),
                             ("torch_cuda", torch.version.cuda),
                             ("cxx11_abi", torch.compiled_with_cxx11_abi())):
            expected = build[key].split('+')[0] if key == "torch" else build[key]
            if expected != current:
                report["compatibility_errors"].append(f"{key}: built for {expected}, running {current}")
    report["fingerprints"] = _module("opguard.tracer.xor_signature")
    report["base_ready"] = report["torch"]["available"] and report["fingerprints"]["available"]
    report["native_ready"] = all(item["available"] for item in report["extensions"].values()) and not report["compatibility_errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    parser.add_argument("--require-native", action="store_true", help="Fail if any native extension cannot load or its build is incompatible")
    args = parser.parse_args(argv)
    report = inspect()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"OpGuard {report['version']} | Python {report['python']} | {report['platform']}")
        print(f"Interpreter: {report['executable']}")
        for label, item in (("Torch", report["torch"]), ("CUPTI", report["cupti"]),
                            ("Fingerprints", report["fingerprints"]), *report["extensions"].items()):
            status = "available" if item["available"] else "unavailable"
            print(f"{label}: {status} — {item.get('version', item.get('path', item.get('reason')))}")
        print(f"Torch CUDA: {report['torch'].get('cuda')} | GPU visible: {report['torch'].get('cuda_available', False)}")
        print(f"CUDA toolkit: {report['cuda_toolkit']['home']} | nvcc: {report['cuda_toolkit']['nvcc']}")
        print(f"CUPTI headers: {', '.join(report['cupti']['headers']) or 'not found (needed only for source builds)'}")
        if report["native_build"]:
            print("Native build: " + json.dumps(report["native_build"], sort_keys=True))
        for error in report["compatibility_errors"]:
            print("INCOMPATIBLE: " + error)
        print("Base tracing: " + ("ready (Python callsites and tensor fingerprints)" if report["base_ready"] else "unavailable; check Torch and runtime dependencies"))
        if not report["native_ready"]:
            print("Native acceleration unavailable/incompatible. Install a matching native wheel or build from source; see docs/Installation.md.")
        print("Extension loading does not validate GPU kernels or CUPTI capture permissions.")
    return int(not report["base_ready"] or (args.require_native and not report["native_ready"]))


if __name__ == "__main__":
    sys.exit(main())
