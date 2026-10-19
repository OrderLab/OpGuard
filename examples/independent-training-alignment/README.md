# Iterative alignment of independent training stacks

Two native training implementations continued to disagree after four training bugs were repaired. An OpGuard-assisted investigation combined small tensor fingerprints, scalar observations, source inspection, and checkpoint comparisons to resolve two remaining defects: **an inconsistent GELU backward** and **shared mutable Adam step state after restart**. The final model RMS difference was **1.675e-8**, and the evaluation loss difference was **1.899e-7**.

This example demonstrates a completed, bounded investigation with OpGuard. It supports using runtime evidence to guide and verify iterative alignment. It does **not** establish that OpGuard is necessary to find these defects or that the tool alone localized them. Both repairs still required reasoning about source code and saved state.

This is a retrospective example containing the answers, observations, and patches. It must not be supplied as a blind evaluation task.

## The workload

| Property | Run A | Run B |
| --- | --- | --- |
| Framework | Transformers 4.49.0 Trainer, Accelerate 1.4.0 | Native Megatron-LM pretrain |
| Parallelism | FSDP1 full shard, DP=2 | TP=2, DP=2, distributed optimizer with TorchAdamW fallback |
| Execution choices | Activation checkpointing, sharded parameters | Selective recomputation, communication overlap, bias/dropout/add fusion |
| Training and restart | Native Trainer loop and checkpoints | Native Megatron loop and checkpoints |

Both train a 42,286,080-parameter GPT model: 12 layers, width 512, 8 attention heads, sequence length 512, and vocabulary size 4096. Fixed data and converted initial weights specify the same intended computation; neither implementation is designated the reference. Training uses FP32, dropout zero, TF32 disabled, exact GELU, global batch 16, microbatch 2, and four accumulation steps. Each launch performs 64 updates, with a native save and a new-process resume after update 32.

The training set has 1,024 rows with variable supervised lengths. Evaluation uses the same 64 full-length rows on both sides, totaling 32,704 causal targets. The model, loss, optimizer, and checkpoint paths are independent native implementations. They still share PyTorch primitives and fixed input/initial-weight artifacts.

This revision contains **six primary injected defects and one auxiliary counting issue**, rather than all eight cases from an earlier experiment. It also contains a mathematically equivalent CUDA atomic embedding-gradient accumulation path in A that introduces real nondeterminism.

## The two remaining defects

### Exact forward, approximate backward

A's GELU forward computes exact GELU, but its custom backward asks for the tanh approximation's derivative. A forward-value comparison alone cannot reveal this inconsistency. It changes the training computation, rather than merely choosing a different floating-point reduction order.

After the first repair stage, the selected input fingerprints and token counts agreed, but the model RMS difference was still 2.367e-6 at update 32 and 1.681e-5 at update 64. Source inspection identified the incompatible backward argument. The [GELU repair](patches/02-gelu-backward.patch) changes `approximate="tanh"` to `approximate="none"` in backward, preserving exact forward.

Rerunning A with that change and comparing against the unchanged round-2 B checkpoint reduced the update-32 model RMS difference to **1.249e-8**. At update 64 it remained **1.666e-5**. This focused comparison separated a pre-restart discrepancy from a remaining post-restart discrepancy. A's nondeterminism remained enabled; the comparison does not assign every residual difference to a unique cause.

### Aliased Adam step state on resume

B's optimizer loader assigns one mutable CPU `step` tensor to multiple parameter states. TorchAdamW increments this tensor per parameter, so aliasing corrupts the step used for bias correction after resume. The source line looks like an ordinary state restoration assignment:

```diff
- s["step"] = step
+ s["step"] = step.clone()
```

CPU reads of native checkpoints showed a step of 32 before restart and **2,496** at the end, where 64 was expected. On the examined rank, `2496 = 32 + 77 × 32` matches the shared counter being advanced for 77 parameter states over the remaining 32 updates. That count is rank-specific.

The [Adam repair](patches/03-adam-step.patch) gives each parameter its own step tensor. It was applied before the final B execution. Saved step values then progressed from 32 to 64 on both TP checkpoint ranks. The final comparison across all 149 model tensors / 42,286,080 elements gave RMS **1.6745840398e-8**, relative L2 **7.063e-7**, and maximum absolute difference **3.066e-5**. Reaggregating evaluation over the same targets gave A=5.981992483139038 and B=5.981992673034071.

These are two semantic training defects. Fixing them produced close numerical agreement; **all 149 final model tensors still differed bitwise**. Atomic accumulation and different parallel execution orders remain possible contributors to the residual. The investigation did not isolate their individual contributions or establish full optimizer/RNG-state equality.

## What OpGuard contributed

The investigator selected six observation sites across the two frameworks and called `opguard.tracer.tensor_dump.summarize_tensor` and `finalize_summary`. The captures include regional input fingerprints, loss observations, token counts, gradient norms, and training progress. The investigator manually paired FSDP DP ranks 0/1 with Megatron TP=0 ranks 0/2, then used `opguard_harness.fingerprints.compare_regions` on 40 selected input pairs per round.

| Recorded stage | Different input fingerprints | Model RMS at update 64 | Evaluation loss gap |
| --- | ---: | ---: | ---: |
| Supplied workload | 16 / 40 | 4.021e-4 | 1.026e-2 |
| Four primary repairs plus auxiliary fix | 0 / 40 | 1.681e-5 | 1.707e-3 |
| GELU and Adam repairs also executed | 0 / 40 | 1.675e-8 | 1.899e-7 |

The fingerprints supplied execution evidence for data correspondence, especially across restart, and helped check that the first repairs had addressed that discrepancy. Matching XOR fingerprints are not proof of tensor equality. Correspondence was selected manually; this run did not use automatic graph alignment.

The final two diagnoses came from **source semantics and checkpoint-state/value comparisons**, informed by the remaining differences across iterations. There is no captured GELU operator trace or automatic fingerprint-based attribution of the Adam alias in this record. The result illustrates the combined workflow accurately: obtain bounded runtime evidence, refine hypotheses, repair, and validate again.

Across three A/B rounds, the capture contains 360 OpGuard tensor summaries in 2,178 emitted records. The 36 raw JSONL files total 417,507 bytes. The independently audited upper bound for all additional diagnostic artifacts was **970,808 bytes**, below the 1 MiB limit; elapsed time was 1,019.172 seconds, below 20 minutes. Existing native logs/checkpoints and mandatory unchanged tracing were exempt from the added-data budget. Full native checkpoint values were available for offline analysis.

## What the agent comparison supports

| Investigation | A/B execution pairs | Primary diagnoses | Repairs actually executed |
| --- | ---: | ---: | --- |
| Ordinary attempt 1 | 1 | 4 / 6 | None |
| Ordinary attempt 2 | 1 | 4 / 6 | None |
| Manual iterative attempt, tool available but unused | 3 | 5 / 6 | Four primary repairs; GELU patch written after the last run |
| OpGuard-assisted iterative attempt | 3 | 6 / 6 | All six primary repairs, plus auxiliary fix |

Both ordinary attempts missed the GELU backward and Adam alias. Their four accepted diagnoses concerned token weighting, resume progress, gradient-norm reduction, and the learning-rate schedule. The manual iterative attempt identified GELU but did not execute its repair and missed the Adam alias; its last executed model RMS difference was 1.681e-5. These results are retained alongside the successful tool-assisted attempt.

The ordinary assignments used fresh agents and a general debugging request, without bug locations or prescribed methods. They were allowed source inspection, one execution of each run, and bounded additional observations. The saved assignments and command records were audited; encrypted dispatch plaintext could not be independently attested. The tool-required follow-up was triggered by the manual attempt making no actual tool calls, under a prospectively recorded condition, rather than by its diagnostic score.

The ordinary and iterative tasks had different execution budgets and repair objectives. There is only one manual and one tool-assisted iterative attempt. Consequently, these observations do not establish an equal-budget causal advantage, tool necessity, or a general success rate. The supported conclusion is that **OpGuard-assisted iterative diagnosis successfully aligned this pair under the recorded constraints**, including two defects missed by both ordinary attempts.

The [diagnosis ledger](evidence/diagnoses.json) records all six primary mechanisms and the auxiliary issue, their evidence, and when each patch was executed. One ordinary report also made an incorrect claim about evaluation rows; it was rejected in grading. The auxiliary label-count issue is kept separate from the six-primary denominator.

## Replay the recorded evidence on CPU

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

Checkpoint RMS figures are explicitly labeled **recorded measurements** in the output. The large trained checkpoints are not bundled, so replay does not recompute those figures or execute training. Original [model comparison](evidence/model-comparison.json), [loss comparison](evidence/loss-comparison.json), and [restart-state comparison](evidence/restart-check.json) outputs are retained separately.

## Restore and launch the full training workload

The companion `runnable-source-inputs.tar.gz` contains the vendored framework sources, fixed data, and initial weights. It is 326,778,170 bytes and is deliberately stored outside git. Its SHA-256, source versions, and the optional pinned OpGuard runtime archive are recorded in [assets.json](evidence/assets.json). **These companion archives are not included in the public release and have no public download URL. Full training reproduction is therefore not available from this checkout; the CPU evidence replay above is self-contained.** The following instructions are for users who already possess the matching archives.

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

Use a separate destination for each stage. The source archive excludes the original observed outputs; it is a runnable reconstruction, not the complete frozen blind package. [all-repairs.patch](patches/all-repairs.patch) is equivalent to the three sequential repair patches, without instrumentation.

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

## Evidence provenance

[results.json](evidence/results.json) contains the selected audited results; [provenance.json](evidence/provenance.json) links them to original artifact paths and hashes. [manifest.json](evidence/manifest.json) covers the exported evidence, original patches, probe helper, comparator snapshot, and every raw capture member. Capture files, repair patches, and original comparison outputs are preserved byte for byte. This small export excludes private controls, full trained checkpoints, and model weights.
