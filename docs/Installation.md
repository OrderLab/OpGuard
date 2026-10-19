# Installation

OpGuard runs inside the training environment; the harness can live in a separate agent environment. Both require Python 3.10+. They share this repository but remain two independently installed packages: `opguard` at the root and `opguard-harness` in [harness/](../harness/). The commands below describe the published packages; until a release is published, install from source or local wheels.

```bash
# Training environment: install the appropriate PyTorch build first.
python -m pip install opguard
# Equivalent alternative in the active virtual environment:
uv pip install opguard
opguard doctor

# Agent environment: isolated tool installation, no Torch dependency.
uv tool install opguard-harness
opguard-harness tools
```

pip and uv install the same distributions. OpGuard declares Torch >=2.4 and NumPy >=1.24 as normal dependencies; an already installed compatible Torch satisfies the requirement. Choose the CPU/CUDA Torch build yourself before installing OpGuard. The minimum versions express API requirements, not validation of every combination. The optional `plot` extra adds Matplotlib for legacy plots.

The harness calls the local backend subprocess specified by `backend: ["/training/venv/bin/python", "-m", "opguard.tools"]`. Training launchers and each rank must use a compatible OpGuard revision. Tool schemas and CLI calls work without a source checkout; optional [agent skills](../harness/skills/) can be read or copied separately. See the [uv tool environment documentation](https://docs.astral.sh/uv/concepts/tools/) and [uv pip installation documentation](https://docs.astral.sh/uv/pip/packages/).

## What the base package includes

| Capability | Default wheel | Native wheel / source build |
| --- | --- | --- |
| Backend protocol, instrumentation, comparison, Perfetto reports | Yes | Yes |
| Python / TorchDispatch callsite capture | Yes | Yes |
| CPU and GPU tensor fingerprints | Tensor-operation fallback | CUDA XOR acceleration on GPU |
| CUPTI capture and native CUDA-API callsites | No | Requires loadable extensions, driver and capture permissions |

The default wheel is pure Python. Neither pip nor uv compiles GPU extensions during a normal installation, even if CUDA happens to be installed. GPU fallback adds tensor operations and temporary allocations; it emits a one-time message when used. Missing native callsite backends are reported when capture starts. Python observations do not replace native kernel visibility.

## Inspect an installation

```bash
opguard doctor
opguard doctor --json
opguard doctor --require-native  # nonzero if any native extension is missing/incompatible
```

The report shows the interpreter, Torch and its CUDA version, GPU visibility, CUDA toolkit/nvcc, CUPTI library and headers, each extension's load status, and the native build metadata. Default exit status is zero when base tracing is available; optional native extensions may be absent. `--require-native` additionally checks all native modules and exact Torch release/CUDA/ABI compatibility. `python -m opguard.doctor` also works without CLI dependencies when diagnosing a damaged installation.

CUPTI headers and nvcc are needed to **build**, not to install a native wheel. A loadable library does not prove kernels run on this GPU or that CUPTI capture permissions are available. Test native wheels with `tests/native_smoke.py` on a GPU before marking a release combination supported.

The CUDA-API interposer must be loaded before the workload resolves CUDA symbols (use `LD_PRELOAD` with the installed `libpylaunch_hook.so`); the native smoke does this in a child process. Loading it late cannot intercept already-bound calls. CUPTI capture does not depend on that interposition mechanism.

## Native wheels

Use the universal wheel on the package index for the default installation. Publish native wheels as explicit release attachments, with separate build tags such as `1torch291cu130x1c1` (Torch 2.9.1, Torch CUDA 13.0, XOR and callsite extensions enabled). They contain the same `opguard` package. Do not put conflicting Torch/CUDA builds on the default index: [wheel compatibility tags](https://packaging.python.org/en/latest/specifications/binary-distribution-format/) encode Python/ABI/platform, not Torch/CUDA, and a build tag is only an identifier.

Select a wheel matching Python, CPU architecture, Torch release, Torch CUDA, C++ ABI, system runtime libraries and the compiled GPU architectures. Install it in the existing training environment:

```bash
python -m pip install /path/to/matching/opguard-*.whl
opguard doctor --require-native
```

The native wheel contains `opguard/_native_build.json`, all three `.so` files, and the same configuration resources as the base wheel. CUDA/CUPTI runtime libraries are supplied by the training environment; the wheel does not bundle the toolkit or driver. Relative library search paths support the NVIDIA libraries installed alongside Torch. Source builds also record toolkit library paths. These are Linux wheels, not a claim of manylinux portability.

Initial release validation scope:

| Build | Validation |
| --- | --- |
| Base wheel: macOS arm64, Python 3.12, Torch 2.9.1 | Built from sdist; installed CPU discovery/tracing/comparison, unchanged loss/gradients, source restoration and Harness queries verified |
| Base wheel: Linux aarch64, Python 3.11, Torch 2.9.1 (CUDA 13.1 build) | Local pip / uv / editable installation, CPU capture workflow and GPU fallback verified |
| Linux, Python 3.10 / 3.12, Torch 2.9.1 CPU | GitHub Actions installed-wheel CPU discovery/tracing/comparison and runtime-policy tests passed |
| Linux aarch64, Python 3.11, Torch 2.9.1+cu130, toolkit 13.1, GB200 (SM 10.0), glibc 2.36 | Local native wheel and GPU smoke verification |
| Linux x86_64, Python 3.10 / 3.12, Torch 2.9.1+cu128, toolkit 12.8, SM 8.0 / 8.6 / 8.9 / 9.0+PTX | Manual build workflow; pending build and GPU validation before release |

The [native-wheel workflow](../.github/workflows/native-wheels.yml) builds candidates from the sdist and uploads wheels plus inspection reports. It does not publish packages or claim GPU validation on a CPU runner. The aarch64 wheel must be built and tested on its target host using the commands below. No native download URL is promised before artifacts are published.

## Source installation

For development and environments needing source changes:

```bash
git clone https://github.com/OrderLab/OpGuard.git
cd OpGuard
python -m pip install -e .
opguard doctor

# Install the agent tool in its own environment from the same checkout.
uv tool install ./harness
```

Editable and ordinary source installs use the same base package. Native acceleration is an explicit Linux build with the target training Python/Torch, CUDA development toolkit, CUPTI headers/libraries, Python headers and C++ compiler:

```bash
python -m pip install 'setuptools>=77.0.3' 'packaging>=24.2' wheel ninja
OPGUARD_TRACING_BUILD=1 OPGUARD_CALLSITE_TRACER=1 \
  TORCH_CUDA_ARCH_LIST='10.0' \
  python -m pip install --no-build-isolation --no-deps .
opguard doctor --require-native
```

Replace `10.0` with the intended GPU architectures. `OPGUARD_TRACING_BUILD` enables XOR; `OPGUARD_CALLSITE_TRACER` at build time enables both native callsite libraries. At runtime the latter enables callsite capture. Set `CUDA_HOME` and `CXX` if needed. Requested extensions must build successfully; missing prerequisites or compiler/linker failures abort the installation instead of silently delivering a partial package.

[pip build isolation](https://pip.pypa.io/en/stable/reference/build-system/) hides the training Torch from the build backend. Base builds need no Torch. Native builds require `--no-build-isolation` so they use your installed Torch; provision the build dependencies first. Native editable installs use the same flags plus `-e .`.

## Validate release artifacts

```bash
python -m pip install build
OPGUARD_TRACING_BUILD=0 OPGUARD_CALLSITE_TRACER=0 python -m build
# Default build creates an sdist, then builds the wheel from that sdist.
python -m pip install dist/opguard-*-py3-none-any.whl
```

Run outside the source checkout, in the environment containing the installed wheel:

```bash
opguard doctor --json
python /path/to/OpGuard/tests/installed_smoke.py
python -m pytest --import-mode=importlib -q /path/to/OpGuard/tests/test_runtime_policy.py /path/to/OpGuard/tests/test_doctor.py
```

The installed smoke verifies module/resource origins, actual CPU discovery, preparation, tracing, comparison, unchanged loss/gradients, source restoration and nonempty reports. Native release validation additionally builds a wheel from the sdist using the native flags and `pip wheel --no-build-isolation --no-deps dist/opguard-*.tar.gz`, installs it, and runs `tests/native_smoke.py` from outside the checkout on the target GPU. The smoke checks native XOR against CPU reference values and starts/collects/stops the native capture backends.

A container can package these demos for repeatability, but is not required for an existing training environment. The [independent training example](../examples/independent-training-alignment/README.md) includes an offline evidence replay; full native training uses separate companion assets.

## Numerical execution policy

Starting tracing does not change deterministic algorithms, TF32, RNG state, allocators, or the caller's uninitialized-memory fill policy. The old `OPGUARD_PATCH_EMPTY` and `OPGUARD_FILL_UNINITIALIZED` interventions have been removed.

For an explicitly controlled deterministic experiment, set `OPGUARD_DETERMINISTIC=1` in the workload environment. This enables deterministic algorithms, disables cuDNN benchmarking and matmul TF32, and sets `CUBLAS_WORKSPACE_CONFIG` if absent. It preserves allocation functions and the existing fill policy. Before/after settings and the reason are recorded in `runtime_policy/rank*-pid*.json` under the capture root and in stderr. It does not seed RNGs or promise agreement across frameworks.

With the harness, use the chosen side's `workload.env`; inherited control-process `OPGUARD_*` settings are cleared. For a script without a startup hook, call `opguard.runtime_policy.apply_requested_policy()` before workload initialization. The compatibility function `opguard.tracer.entry_handler.enable_torch_deterministic_mode()` is also an explicit intervention.
