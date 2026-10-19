# Demo 5: console logs and OpGuard output from the same launches

This pair runs the original 20-step workload with communication overlap disabled
(run1) and enabled (run2). Each launch simultaneously records separate, complete
stdout/stderr and OpGuard summaries for GPU ranks 0 and 1. Both launches and all
four rank completion receipts succeeded.

| Run | stdout | stderr | Capture ID |
| --- | --- | --- | --- |
| run1: overlap off | [stdout.log](standard-logs/overlap-off/stdout.log) | [stderr.log](standard-logs/overlap-off/stderr.log) | `1dfec07bd2944f3b992cec2f6e898950` |
| run2: overlap on | [stdout.log](standard-logs/overlap-on/stdout.log) | [stderr.log](standard-logs/overlap-on/stderr.log) | `7f6062dc9f3f444d8ff9ee1d33da84a1` |

Use [opguard-output.json](opguard-output.json) for the compact observations and
artifact-bound event references. The complete exports are [rank 0](opguard/perfetto_gpu_000.json.gz)
and [rank 1](opguard/perfetto_gpu_001.json.gz). [queries/](queries/) retains the
actual harness responses, including first differences, tensor comparisons and
source context.

The native logs still show a numerical discrepancy under instrumentation:

| Native metric | Overlap off | Overlap on |
| --- | ---: | ---: |
| Step 16 grad norm | 0.00197029114 | 0.00204745238 |
| Step 19 loss | 0.12809968 | 0.129150391 |

All 20 metric rows from each launch are retained in [metrics.json](metrics.json)
and in their original stdout order. The four raw log files total 55,637 bytes;
they have not been shortened to fit the website inference budget.

## What OpGuard observed

Both first reported boundaries are inside `get_grad_norm_direct`:

- Rank 0: `all_norms.append(...)`, original operator tag line 1688. The appended
  scalar's regional signatures are `[0, 1077945344, 0, 0]` and
  `[0, 1077804032, 0, 0]`. This observes a norm that has already been calculated;
  it is not a direct capture of the norm operator's input or the racing write.
- Rank 1: `total_norm.pow(...)`, tag line 1699. Its scalar output signatures are
  `[1124147200, 0, 0, 0]` and `[1124478605, 0, 0, 0]`.

On each rank, run2 has **33 unaligned operations before this boundary**; run1 has
zero. Correspondence is an unreviewed callsite-order candidate. These are first
observed differences at captured callsites, not a proven whole-graph first diff
or proof of the precise stream race. Fingerprints are not error magnitudes, and
matching fingerprints do not prove tensor equality.

Operator tags refer to the original source. Instrumentation shifts the embedded
snapshot: the rank-0 append maps to lines 1734/1736 and rank-1 pow to 1745/1747.
The mapped source queries use those captured stack locations.

## Provenance and verification

The [manifest](manifest.json) hashes every delivered file. The console launcher,
worker request, backend result and both rank receipts carry the same capture ID
for each run. [raw-captures.tar.gz](raw-captures.tar.gz) preserves the original
transport bundles, rank receipts, instrumentation plans/restoration records and
separate discovery runs. Discovery logs are not substituted for trace-run logs.

This uses the same application, dependencies and execution settings as the
[previous native rerun](../standard-logs/rerun-20261019/README.md), with OpGuard
capture added. The original source/package files were verified unchanged after
restoration. No optimizer repair, weighted gradient or artificial delay was
added. HEATTRACE remained enabled. Five in-checkout Python files were selected
by discovery on each side; this is not complete kernel or execution coverage.

For a paired demonstration, use the standard logs **in this directory** with
these new OpGuard exports. The earlier uninstrumented logs and historical demo
trace are different executions. This establishes shared execution provenance;
it does not establish that a model given only ordinary logs cannot diagnose the
mechanism, or that the instrumentation left scheduling unchanged.

From the repository root, with Python 3.11 or newer and no GPU required:

```bash
python3 harness/examples/demo_case5/paired-capture-20261019/verify.py
```

The check verifies hashes, links the console files to their archived capture and
rank receipts, checks all native steps and event references, and replays all 14
recorded queries with the bundled harness.
