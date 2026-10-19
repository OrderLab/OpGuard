# Public demo walkthrough: first difference to a source-derived mechanism

This is a **known-answer walkthrough and artifact/tool acceptance test**, not a
blind agent benchmark or a locally reproduced GPU race. The user supplied the
answer; the embedded source also includes a proposed repair. Generic skills and
readers contain no case-specific diagnosis rule.

Complete native [stdout/stderr from a new two-GPU rerun](standard-logs/rerun-20261019/README.md)
are now available for overlap enabled and disabled. These are separately captured
2026-10-19 runs, not the missing original console logs of the historical trace.
Each completed all 20 training steps; source, configuration and log hashes are
recorded with the files. The historical artifact analysis below retains its
original evidence limits.

## Replay the artifact queries

From the harness checkout, use Python 3.10+:

```bash
python examples/fetch_artifacts.py examples/demo_case5/manifest.json /tmp/opguard-demo-case5
printf '%s\n' '{"artifact":"/tmp/opguard-demo-case5/trace.json.gz"}' > /tmp/trace-query.json
python -m opguard_harness call trace_summary --arguments /tmp/trace-query.json
python -m opguard_harness call first_diff --arguments /tmp/trace-query.json
OPGUARD_DEMO_TRACE=/tmp/opguard-demo-case5/trace.json.gz python -m unittest discover -s tests -v
```

Use each returned event reference with `inspect_event`; use the two first-boundary
references with `compare_events` as explicitly selected candidates. Follow source
context with `code_context`, `search_source`, and `source_excerpt`. Paths in the
export are source keys, not files to open on your machine. The downloader verifies
all hashes in `manifest.json`. Artifacts are fetched on demand, not vendored here.

## 1. Observations, before interpreting the known answer

The export has 29,291 records, 5,318 operation records and 23 embedded source files.
For each of run1 and run2, rank 0 has 2,529 exporter-OK operations, 108 differences
and 22 insertions. Only rank 0 is represented; complete multi-rank coverage is not
established.

Both per-run first observed differences select `_linalg.linalg_vector_norm#2144`:

- run1 event index 19005; run2 event index 19327. Full references are the trace
  SHA-256 from `manifest.json`, followed by `:` and the index.
- The recorded input is shape `[1024,1024]`, float64, with XOR signature `0` on
  both sides. This does **not** prove the input tensors equal or all-zero.
- The scalar output XOR signatures differ: 1078077440 and 1077945344. These are
  fingerprints, not measured norm values or a numerical-error magnitude.
- The preceding exporter-OK event is `item#2143`, an overflow-check path. It is
  not evidence that the particular norm input was previously equal.

The source context maps the operation tag line 1626 to the instrumented
`torch/functional.py` target line 1629. Stack transitions at the same timestamp
occur after the operation record in serialization order; consuming only preceding
records incorrectly attributes the norm to the preceding overflow check.

The export's alignment flow endpoints around the pivot land on neighboring
operators. Flow 193749 connects the suspect norm bar to the reference `item` bar
under interval containment. The reader rejects this inconsistent pairing. The two
norms remain explicit comparison candidates based on their per-run first-diff
annotations, function/signature/location/rank and counter; those are supporting
metadata, not independently verified raw alignment. This is an artifact alignment
limitation, not evidence for the training bug.

## 2. Upstream dependencies read from code context

Read the enclosing functions in the embedded `stage_1_and_2.py` snapshot:

1. `get_grad_norm_direct` (1693–1747), target 1729: the norm consumes a converted,
   detached gradient `g`; follow `gradients`, not the norm implementation alone.
2. `scaled_global_norm` (1816–1833), target 1827: supplies
   `averaged_gradients[i]` and partition parameters.
3. `independent_gradient_partition_epilogue` (792–836): builds those gradients
   from partitions after bucket reduction. Its later synchronization cannot undo
   a value already corrupted earlier.
4. `get_flat_partition` (1752–1790) and `copy_grads_in_partition` (1360–1394):
   inspect gradient attributes, views and copied partition storage.
5. `reduce_independent_p_g_buckets_and_remove_grads` (967–1004): launches bucket
   reduction, toggles the double-buffer index, and copies gradients into a view of
   the selected buffer. This identifies a mutable shared storage lifetime.
6. `reduce_ipg_grads` (1396 onward) and `average_tensor` (1077 onward): reduction
   consumes the bucket on a reduction stream when overlap is enabled. The shown
   wait in `average_tensor` orders the reduction after current-stream work; this
   does not by itself order a later current-stream overwrite after earlier reduction.

This gives a concrete **source-derived candidate**: reuse of a bucket before its
previous reduction is finished can corrupt gradients that later feed the norm.
The missing ordering edge is at reuse, rather than a requirement to repair norm.
Each link above is source reasoning; the export does not record the dynamic
producer IDs, buffer versions or stream events to demonstrate the precise overlap.

`debug.py` includes a proposed per-buffer event patch and a `--patch_deepspeed`
branch. Its presence supplies a candidate intervention, not proof that the branch
ran in either capture. The full actual launch arguments and both run configs are
not established by this export.

## 3. Hypothesis validation and known-answer cross-check

The [paper's Table 4, Bug 25](https://www.usenix.org/system/files/osdi26-zhou-ziming.pdf#page=23)
reports a default/reduction-stream race and an `overlap_comm=False` reference.
[DeepSpeed issue #5545](https://github.com/deepspeedai/DeepSpeed/issues/5545)
describes the overlap buffer problem and proposes recording completion per buffer
and waiting before its next reuse. These external reports corroborate the
source-derived mechanism; they are not a local intervention result.

A useful controlled test would compare the original overlap-enabled execution,
overlap-disabled reference, and overlap-enabled execution with the targeted
per-buffer ordering edge. Keep code revision, actual input/state, topology and
comparison method fixed; repeat enough to observe the original failure. Capture
stronger input comparisons around buffer reuse and the norm, together with the
relevant ordering events. Check that restoring overlap alone restores the original
symptom where reproducible. Disabling all overlap changes more than one behavior
and by itself does not isolate the exact edge.

**Outcome:** offline localization and source-context investigation completed;
source-derived explanation agrees with the published cause. **GPU intervention
not run**: this environment has no configured GPU reproduction, and the export
does not establish complete launch provenance. Precise timing/alias identities
and numerical input equality remain unverified. This is not an independently
proven root-cause reproduction.

## Generic lessons incorporated into the library

Code context is the primary upstream path; missing dynamic producer IDs should
not stop source reasoning. Keep fingerprints distinct from actual values, account
for instrumented line mapping, validate exporter alignment hints, and distinguish
source definitions from runtime execution. The tests also cover unrelated
operators, missing/unaligned observations, duplicate flows, stale references and
bounded source queries; no diagnostic rule assumes a stream race.
