# Independent training alignment

Two independent training stacks—Transformers/FSDP and Megatron—train the same
42M-parameter GPT from fixed data and converted initial weights. They disagree.
Over three A/B rounds, an OpGuard-assisted investigation repairs six injected
training defects and one auxiliary counting issue, reducing the final model RMS
difference from **4.021e-4 to 1.675e-8**.

## Takeaways

- **Training alignment includes data progress, backward computation, and restored
  optimizer state.** Matching sampled input fingerprints still left two defects:
  an inconsistent GELU derivative and a shared Adam step counter after restart.
- **Compare before and after restart to narrow the investigation.** Repairing
  GELU brought the pre-restart checkpoints into close agreement; the remaining
  post-restart gap led to inspection of restored optimizer state.
- **Use observations to guide repairs, then rerun to validate them.** OpGuard
  supplied compact runtime evidence; source inspection and checkpoint comparisons
  explained the last two defects. The final result is close numerical agreement,
  with residual bitwise differences.

This is a retrospective case study with known injected defects. The recorded
result demonstrates this workflow on one workload; it does not establish that
OpGuard is necessary or automatically localizes every defect.

## The result

Run A uses Transformers Trainer with FSDP (DP=2); run B uses native Megatron
(TP=2, DP=2). Both train in FP32 for 64 optimizer updates, saving at update 32
and resuming in a new process. Neither stack is designated the reference.

| Recorded stage | Different sampled input fingerprints | Model RMS at update 64 | Evaluation loss gap |
| --- | ---: | ---: | ---: |
| Supplied workload | 16 / 40 | 4.021e-4 | 1.026e-2 |
| Four primary repairs + auxiliary counting fix | 0 / 40 | 1.681e-5 | 1.707e-3 |
| GELU backward + Adam resume repairs added | 0 / 40 | **1.675e-8** | **1.899e-7** |

The final model comparison covers all 149 tensors / 42,286,080 elements.
All 149 tensors still differ bitwise. A's atomic embedding-gradient accumulation
and the stacks' different parallel execution orders remain possible contributors;
their individual effects were not isolated. Full optimizer/RNG-state equality was
not established. See the [recorded results](evidence/results.json).

## How the investigation narrowed the gap

### 1. Align the data and training semantics

At update 33, immediately after restart, A reads the batch starting at row
**128**, while B starts at row **512**. A interpreted 32 completed optimizer
updates as 32 microbatches to skip, overlooking four gradient-accumulation steps
per update. The correct skip is `32 × 4 = 128` microbatches.

The [first repair stage](patches/01-training-semantics.patch) fixes resume progress,
global token weighting, gradient-norm reduction, and the learning-rate schedule,
plus an auxiliary causal-target count. Sampled input fingerprint mismatches fall
from **16/40 to 0/40**, but the model RMS remains **1.681e-5** at update 64.
Input correspondence alone does not establish training equivalence.

The [checkpoint-resume walkthrough](checkpoint-resume/README.md) connects the
original logs, row fingerprints, skip calculation, and repair. Ordinary logs
already show A's epoch moving backward at restart; the input observations make
the resumed batch discrepancy precise.

### 2. Check backward semantics when inputs agree

A computes exact GELU in forward but uses the tanh approximation's derivative
in its custom backward. Forward-value comparisons cannot expose that mismatch.
Source inspection identifies the inconsistency; the
[GELU repair](patches/02-gelu-backward.patch) changes backward's
`approximate="tanh"` to `approximate="none"`.

Rerunning A against the unchanged round-2 B checkpoint reduces model RMS at
update 32 from **2.367e-6 to 1.249e-8**. At update 64, the gap remains
**1.666e-5**. Comparing both checkpoints separates the repaired pre-restart
discrepancy from a remaining post-restart discrepancy. These are
[recorded checkpoint measurements](evidence/restart-check.json), not a captured
GELU operator trace.

### 3. Inspect the state restored at restart

B's optimizer loader assigns one mutable CPU `step` tensor to multiple parameter
states. TorchAdamW increments it per parameter, corrupting bias correction after
resume. Native checkpoint reads show **32 → 2,496**, where **32 → 64** was
expected. On the examined rank, `2496 = 32 + 77 × 32`: the shared counter advances
for 77 parameter states over 32 remaining updates.

The [Adam repair](patches/03-adam-step.patch) gives each parameter its own counter:

```diff
- s["step"] = step
+ s["step"] = step.clone()
```

After rerunning B, saved steps progress from 32 to 64 on both TP checkpoint ranks.
With both final repairs executed, model RMS reaches **1.675e-8** and the evaluation
loss gap reaches **1.899e-7**. The [diagnosis ledger](evidence/diagnoses.json)
records the evidence and executed repair for every defect.

## What OpGuard contributed

Six manually selected observation sites captured input fingerprints, losses,
token counts, gradient norms, and training progress. The investigator paired the
DP/TP ranks manually and compared 40 input pairs per round. Those observations
exposed sampled input disagreement across restart and verified corrected input
progress in later rounds.

The final two diagnoses combined source semantics with checkpoint-state/value
comparisons. This record contains no automatic graph alignment or automatic
fingerprint attribution of the GELU and Adam defects. Matching XOR fingerprints
are also not proof of tensor equality.

The assisted investigation completed within the recorded 20-minute / 1 MiB
additional-diagnostics limits. Native logs and checkpoints were exempt from that
data budget. [Experiment details](docs/experiment.md) retain the observation
protocol, budgets, and agent comparisons, including their unequal execution
budgets and repair objectives.

## Replay the evidence on CPU

From the repository root, with Python 3.11 or newer:

```bash
python3 examples/independent-training-alignment/replay.py
```

No GPU, PyTorch, or external dataset is needed. Expected results:

- `verified: true`
- Different input-pair counts by round: **16, 0, 0** out of 40
- Final evaluation loss gap: approximately **1.899e-7**

The replay verifies artifact hashes, recomputes input fingerprint comparisons,
checks evaluation row-hash correspondence, and reaggregates recorded losses.
Checkpoint RMS values are **recorded measurements**: trained checkpoints are
not bundled, and replay does not execute training.

## Further reading

- [Checkpoint-resume walkthrough](checkpoint-resume/README.md): original native
  logs, the update-33 input mismatch, and a dedicated CPU verifier.
- [Experiment details and evidence](docs/experiment.md): full workload,
  instrumentation, agent comparisons, claim limits, and provenance.
- [Replay and training reproduction](docs/reproduction.md): alternate comparator,
  repair stages, environment, and launch commands. Full training requires
  companion archives with **no public download URL**; only the CPU evidence replay
  is self-contained in this release.
