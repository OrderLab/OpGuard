# Experiment details and evidence

[Back to the case study](../README.md)

This page retains the workload, observation protocol, agent comparisons, and
provenance behind the case study. This is a retrospective example containing
known injected defects, observations, and patches; it must not be supplied as a
blind evaluation task.

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

The [diagnosis ledger](../evidence/diagnoses.json) records all six primary mechanisms and the auxiliary issue, their evidence, and when each patch was executed. One ordinary report also made an incorrect claim about evaluation rows; it was rejected in grading. The auxiliary label-count issue is kept separate from the six-primary denominator.

## Evidence provenance

[results.json](../evidence/results.json) contains the selected audited results; [provenance.json](../evidence/provenance.json) links them to original artifact paths and hashes. [manifest.json](../evidence/manifest.json) covers the exported evidence, original patches, probe helper, comparator snapshot, and every raw capture member. Capture files, repair patches, and original comparison outputs are preserved byte for byte. This small export excludes private controls, full trained checkpoints, and model weights.
