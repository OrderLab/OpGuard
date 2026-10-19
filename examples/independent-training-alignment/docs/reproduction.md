# Replay and training reproduction

[Back to the case study](../README.md)

The CPU replay is self-contained. Full training reproduction requires companion
archives that are not publicly distributed; the two paths are documented below.

## Replay the recorded evidence on CPU

The [checkpoint-resume case](../checkpoint-resume/README.md) includes the original
merged stdout/stderr logs for rounds 1 and 2, their launch records and hashes, and
a replay of the update-33 input mismatch (A starts at row 128; B at row 512).
The ordinary logs already show A's epoch moving backward across restart; the case
documents that clue alongside the more precise input observations.

From the OpGuard repository root, with Python 3.11 or newer:

```bash
python3 examples/independent-training-alignment/replay.py
```

No GPU, PyTorch, external dataset, or sibling checkout is needed. The replay verifies the packaged hashes, reads the original 36 capture files, runs the bundled snapshot of the actual fingerprint comparator, checks evaluation row-hash correspondence, and recomputes the loss aggregates. It must report `verified: true`, differing input-pair counts `16, 0, 0`, and a final loss gap of approximately `1.899e-7`.

To check the captures against the harness source in this repository instead:

```bash
python3 examples/independent-training-alignment/replay.py \
  --harness-root ./harness
```

Checkpoint RMS figures are explicitly labeled **recorded measurements** in the output. The large trained checkpoints are not bundled, so replay does not recompute those figures or execute training. Original [model comparison](../evidence/model-comparison.json), [loss comparison](../evidence/loss-comparison.json), and [restart-state comparison](../evidence/restart-check.json) outputs are retained separately.

## Restore and launch the full training workload

The companion `runnable-source-inputs.tar.gz` contains the vendored framework sources, fixed data, and initial weights. It is 326,778,170 bytes and is deliberately stored outside git. Its SHA-256, source versions, and the optional pinned OpGuard runtime archive are recorded in [assets.json](../evidence/assets.json). **These companion archives are not included in the public release and have no public download URL. Full training reproduction is therefore not available from this checkout; the CPU evidence replay above is self-contained.** The following instructions are for users who already possess the matching archives.

Given that archive, prepare a new workspace:

```bash
python3 examples/independent-training-alignment/prepare_workload.py \
  --archive /path/to/runnable-source-inputs.tar.gz \
  --destination /scratch/alignment-aligned \
  --stage aligned --capture
```

The preparer verifies the archive and original file hashes, applies the chosen repair stage, installs the six historical observation sites when `--capture` is supplied, adapts the interpreter and probe paths, and writes a new manifest. It performs no training. Available stages are:

| Stage | Applied changes |
| --- | --- |
| `supplied` | Original defective training sources |
| `four_fixes` | Global token weighting, causal target count, resume skip, norm reduction, schedule horizon |
| `gelu_fixed` | Previous changes plus consistent GELU backward |
| `aligned` | Previous changes plus per-parameter Adam step cloning |

Use a separate destination for each stage. The source archive excludes the original observed outputs; it is a runnable reconstruction, not the complete frozen blind package. [all-repairs.patch](../patches/all-repairs.patch) is equivalent to the three sequential repair patches, without instrumentation.

The original native runs completed on four NVIDIA GB200 GPUs with Python 3.11.2, PyTorch 2.9.1+cu130, NumPy 1.26.4, safetensors 0.8.0, and Triton 3.5.1. CUDA/compiler, distributed communication, Python dependencies, and the original namespace environment must be provisioned separately; the archives do not contain the host environment. For captures, make the OpGuard runtime importable in that training environment. Prefer the pinned companion runtime when reproducing the historical experiment.

With a provisioned environment, first inspect the launches:

```bash
export PYTHON_BIN=/path/to/training-venv/bin/python
export PYTHONPATH="/path/to/OpGuard${PYTHONPATH:+:$PYTHONPATH}"
export OPGUARD_XOR_REGIONAL=4
export OPGUARD_RUNTIME_CALLSTACK=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash /scratch/alignment-aligned/launch.sh a outputs/run_a --dry-run
bash /scratch/alignment-aligned/launch.sh b outputs/run_b --dry-run
```

Run the native launches sequentially on those allocated GPUs:

```bash
bash /scratch/alignment-aligned/launch.sh a outputs/run_a
bash /scratch/alignment-aligned/launch.sh b outputs/run_b
```

A uses two visible GPUs; B uses four. Each side permits one execution attempt in that workspace, and each execution includes the two save/resume phases. Results and native checkpoints go under `outputs/`; captures go under `diagnostics/`. The fixed `replay.py` checks the bundled historical evidence, not newly generated training outputs.

The historical experiment executed all repairs with probes installed. Probes were subsequently removed and the resulting sources checked, but training was not rerun after removal. During packaging, archive restoration, all repair/capture patch combinations, Python/shell syntax, and both adapted launchers' dry-runs were checked. **The relocated launchers have not received a new GPU training run**; the recorded successful native executions used the original isolated environment.
