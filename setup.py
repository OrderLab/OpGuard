"""Optional Linux native builds. Pure Python builds never import Torch."""
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path

from setuptools import Extension, setup

xor_enabled = os.environ.get("OPGUARD_TRACING_BUILD", "0") == "1"
callsite_enabled = os.environ.get("OPGUARD_CALLSITE_TRACER", "0") == "1"
ext_modules, cmdclass, options = [], {}, {}

if xor_enabled or callsite_enabled:
    if platform.system() != "Linux":
        raise RuntimeError("OpGuard native extensions currently support Linux only; omit native build flags for the base package.")
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError as exc:
        raise RuntimeError(
            "Native builds require Torch in the build environment. Install the training Torch first, "
            "then build with --no-build-isolation. Omit native build flags for the base package."
        ) from exc
    if not torch.version.cuda or not CUDA_HOME:
        raise RuntimeError("Native builds require CUDA-enabled Torch and the CUDA toolkit; set CUDA_HOME.")
    cuda = Path(CUDA_HOME)
    include_dirs = [p for p in [cuda / "include", *cuda.glob("targets/*/include"), cuda / "extras/CUPTI/include"] if p.is_dir()]
    library_dirs = [p for p in [cuda / "lib64", cuda / "lib", *cuda.glob("targets/*/lib"), cuda / "extras/CUPTI/lib64"] if p.is_dir()]
    # Driver stubs are for linking on build hosts without GPUs, never runtime lookup.
    link_dirs = library_dirs + [p / "stubs" for p in library_dirs if (p / "stubs").is_dir()]
    if not any((p / "cuda.h").is_file() for p in include_dirs):
        raise RuntimeError("Native build requested but cuda.h is missing; install CUDA development headers.")
    nvcc = cuda / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"Native build requested but {nvcc} is missing; install the CUDA toolkit.")
    nvcc_version = subprocess.check_output([str(nvcc), "--version"], text=True)
    toolkit_version = re.search(r"release (\d+\.\d+)", nvcc_version).group(1)
    if toolkit_version.split('.')[0] != torch.version.cuda.split('.')[0]:
        raise RuntimeError(f"CUDA toolkit {toolkit_version} and Torch CUDA {torch.version.cuda} have different major versions.")

    # Relative paths work in pip/uv environments; toolkit paths support source builds.
    def rpaths(site_relative):
        return [f"$ORIGIN/{site_relative}/{p}" for p in
                ("torch/lib", "nvidia/cuda_cupti/lib", "nvidia/cuda_runtime/lib", "nvidia/cu13/lib")] + [str(p) for p in library_dirs]

    if xor_enabled:
        ext_modules.append(CUDAExtension(
            "xor_reduce_cuda",
            ["opguard/bitwise_xor/bitwise_xor_kernel.cu", "opguard/bitwise_xor/binding.cpp"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            runtime_library_dirs=rpaths("."),
        ))
    if callsite_enabled:
        if not any((p / "cupti.h").is_file() for p in include_dirs):
            raise RuntimeError("Native callsite build requested but cupti.h is missing; install CUPTI development headers.")
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if not compiler:
            raise RuntimeError("Native callsite build requires a C++ compiler; set CXX.")
        python_include = sysconfig.get_path("include")
        if not (Path(python_include) / "Python.h").is_file():
            raise RuntimeError("Native callsite build requires Python development headers (Python.h).")
        ext_modules.append(Extension(
            "opguard.callsite_tracer.opguard_cupti",
            ["opguard/callsite_tracer/cupti_tracer/opguard_cupti.cpp"],
            include_dirs=[str(p) for p in include_dirs],
            library_dirs=[str(p) for p in link_dirs],
            libraries=["cupti", "cuda", "pthread", "dl"],
            runtime_library_dirs=rpaths("../.."),
            extra_compile_args=["-O2", "-std=c++17"], language="c++",
        ))

    build_info = {
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_toolkit": toolkit_version,
        "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "auto (build host GPUs)"),
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "python": platform.python_version(), "platform": platform.platform(),
        "glibc": platform.libc_ver()[1],
        "extensions": {"xor": xor_enabled, "callsite": callsite_enabled},
    }
    # An identifying build tag, NOT an automatically enforced Torch/CUDA constraint.
    tag = "1torch{}cu{}x{}c{}".format(
        torch.__version__.split('+')[0].replace('.', ''), torch.version.cuda.replace('.', ''),
        int(xor_enabled), int(callsite_enabled))
    options["bdist_wheel"] = {"build_number": tag}

    class BuildNative(BuildExtension):
        def run(self):
            super().run()
            root = Path(__file__).parent if self.inplace else Path(self.build_lib)
            package = root / "opguard"
            if callsite_enabled:
                output = package / "callsite_tracer/native/libpylaunch_hook.so"
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [compiler, "-std=c++17", "-fPIC", "-shared",
                           "opguard/callsite_tracer/native/hook.cpp", "-o", str(output),
                           f"-I{python_include}"]
                command += [f"-I{p}" for p in include_dirs] + [f"-L{p}" for p in link_dirs]
                command += [f"-Wl,-rpath,{p}" for p in rpaths("../../..")]
                command += ["-lcupti", "-lcuda", "-lcudart", "-ldl", "-lpthread"]
                command += shlex.split(sysconfig.get_config_var("LDFLAGS") or "")
                # Explicitly requested extensions must all succeed.
                subprocess.check_call(command)
            package.mkdir(parents=True, exist_ok=True)
            (package / "_native_build.json").write_text(json.dumps(build_info, indent=2) + "\n")

    cmdclass["build_ext"] = BuildNative

setup(ext_modules=ext_modules, cmdclass=cmdclass, options=options)
