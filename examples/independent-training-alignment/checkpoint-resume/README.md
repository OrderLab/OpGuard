# Checkpoint resume: optimizer updates mistaken for microbatches

The original native training logs for this case have been recovered from the same
completed investigation as the published OpGuard captures. At update 33, run A
reads the global batch starting at row **128**, while run B reads the batch
starting at row **512**. A's checkpoint loader skipped 32 microbatches for 32
completed optimizer updates, although each update contains four microbatches.

These are historical artifacts from session `b0e8d3b247de` on 2026-10-19, not a
new training run. This is a retrospective example with known, injected defects.
The [parent example](../README.md) describes the other defects and the experiment's
limits.

## Original standard logs

The files below retain the launchers' complete console output with host-specific
paths normalized to `/opt/opguard`. Both launchers used `> phase_N.log 2>&1`, so each file contains **merged
stdout and stderr**, including distributed-worker interleaving. Separate stdout
and stderr streams were not retained. The original redirection is preserved in
[A's launcher](provenance/stack_a-run_training.sh) and
[B's launcher](provenance/stack_b-run_training.sh).

Read phase 1 before phase 2 for each run. All lines, warnings, timestamps and
training metrics remain in their saved order. The launch records show that A
completed both phases before B started in each round.

| Stage | Run | Updates 1–32 | Resumed updates 33–64 | Original launch record |
| --- | --- | --- | --- | --- |
| Supplied defects | A: Transformers / FSDP | [phase 1](standard-logs/round1/run_a/phase_1.log) | [phase 2](standard-logs/round1/run_a/phase_2.log) | [A, round 1](provenance/round1/run_a.json) |
| Supplied defects | B: Megatron | [phase 1](standard-logs/round1/run_b/phase_1.log) | [phase 2](standard-logs/round1/run_b/phase_2.log) | [B, round 1](provenance/round1/run_b.json) |
| First repair stage | A: Transformers / FSDP | [phase 1](standard-logs/round2/run_a/phase_1.log) | [phase 2](standard-logs/round2/run_a/phase_2.log) | [A, round 2](provenance/round2/run_a.json) |
| First repair stage | B: Megatron | [phase 1](standard-logs/round2/run_b/phase_1.log) | [phase 2](standard-logs/round2/run_b/phase_2.log) | [B, round 2](provenance/round2/run_b.json) |

All four launch records report exit code 0. Each phase contains 32 native training
metric records. The standard logs are console output, not renamed or reconstructed
OpGuard traces. OpGuard probes were installed during these same launches and
wrote their records separately; this is not an independently executed,
uninstrumented baseline.

## What the ordinary logs already reveal

Ordinary logging contains a useful resume clue, even without input fingerprints:

- A's last training record before restart reports `epoch: 0.5` (round 1,
  phase 1, line 36). Its first training record after restart reports
  `epoch: 0.140625` (phase 2, line 10). A's update numbers are assigned by the
  32-record phase boundary; they are not printed on these Trainer metric lines.
- B explicitly prints iteration 32 with `consumed samples: 512`, then iteration
  33 with `consumed samples: 528` and global batch size 16. These counts describe
  progress after each completed update, not a logged tensor row ID.

The epoch regression can itself suggest a data-skip or resume-progress error.
A demonstration must retain it and must not promise that a model given only
ordinary logs cannot diagnose the problem. The extra contribution of OpGuard
here is the recorded disagreement at the **resumed data input**, with concrete
row fingerprints and source references.

## Recorded input boundary

The original probes sampled updates 1, 2, 32, 33 and 64. The first captured
post-resume input comparison is update 33; this is not an automatic whole-graph
first-difference result. FSDP ranks 0/1 are paired manually with Megatron TP=0
ranks 0/2, with four microbatches per DP rank.

| Stage | Update | A global batch start | B global batch start | Different input microbatches |
| --- | ---: | ---: | ---: | ---: |
| Supplied defects | 32 | 496 | 496 | 0 / 8 |
| Supplied defects | 33 | 128 | 512 | 8 / 8 |
| First repair stage | 33 | 512 | 512 | 0 / 8 |

[input-boundary.json](input-boundary.json) contains the recomputed comparisons and
references to the original JSONL member, line and microbatch. For DP rank 0's
first microbatch at update 33, A has row IDs `[128, 129]` and regional signatures
`[3740, 1063, 3609, 1077]`; B has row IDs `[512, 513]` and signatures
`[3870, 24, 548, 1279]`. The shape is `[2, 512]`, dtype `int64`.

Row positions come from matching all 16 row hashes and all eight causal-target
counts against the fixed training data's [row catalog](row-catalog.json). A
single row hash can repeat, so it is not used alone to identify a position.
Matching XOR fingerprints are not proof of tensor equality.

The skip units explain both positions:

- Correct skip: 32 optimizer updates × 4 accumulation steps = **128 microbatches**.
- Each distributed microbatch consumes 2 DP ranks × 2 examples = 4 global rows.
- Skipping only 32 microbatches starts at global row 128; skipping 128 starts at
  global row 512.

The first hunk of [the recorded repair](../patches/01-training-semantics.patch)
multiplies `steps_trained_in_current_epoch` by `gradient_accumulation_steps` in
the vendored Transformers trainer. Round 2 executed that change together with
other training repairs. It confirms corrected input progress in that combined
repair stage; it does not isolate the resume fix's effect on the final loss.

## Check provenance and replay

From the repository root, with Python 3.11 or newer:

```bash
python3 examples/independent-training-alignment/checkpoint-resume/verify.py
```

This CPU-only check verifies every recovered file's size and SHA-256, checks the
published capture-member hashes, recomputes the input comparisons and unique row
windows, and reads the native metrics in their original order. It checks all
five sampled updates in rounds 1 and 2 and reports the epoch regression in the
ordinary logs. No training or model inference is performed.

[manifest.json](manifest.json) records provenance paths, path normalization, stream handling, launch
stages and hashes. All 36 published capture members were also compared byte for
byte with the local investigation files when these logs were recovered. The row
catalog was derived from the checksum-verified runnable source/input archive;
its array hashes are included, while the full data and trained checkpoints
remain outside this compact example.
