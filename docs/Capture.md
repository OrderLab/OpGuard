# Capture storage and comparison evidence

## Local staging for shared-storage jobs

To avoid many ranks writing small files directly to shared storage, configure a local
SSD directory on every node and the same shared target for the run:

```bash
export OPGUARD_TRACE_OUTPUT_BASE=/shared/opguard/experiment-001
export OPGUARD_LOCAL_DUMP_ENABLED=1
export OPGUARD_LOCAL_DUMP_TARGET="$OPGUARD_TRACE_OUTPUT_BASE"
export OPGUARD_LOCAL_DUMP_BASE=/local-ssd/opguard
export OPGUARD_LOCAL_DUMP_SYNC_INTERVAL=60
export OPGUARD_SYNC_BATCH_MB=256
export OPGUARD_SYNC_MBPS=32
```

`OPGUARD_LOCAL_DUMP_BASE` must be node-local and shared by the ranks on that node.
The uploader uses this directory for local locks and durable retry checkpoints.
`OPGUARD_NODE_ID` optionally overrides the hostname used to identify a node; keep it
stable and unique across nodes for a run. All ranks need a global rank, and the node's
metadata uploader needs its launcher-provided local rank.

The upload protocol has three steps: create a complete batch locally, copy it to a
hidden temporary file on shared storage, then atomically rename it to publish it.
Only after publication does the local checkpoint advance. A failed upload retains its
batch for retry. Concurrent sync calls for the same rank serialize locally. A lock timeout
fails the sync rather than starting another upload. Shared storage has no coordination locks.

JSONL/NDJSON files must be append-only between rotations. Only new, newline-terminated
records are uploaded; an incomplete trailing record waits for the next sync. Truncation
or atomic file replacement starts a new version. Tensor and callsite writers publish
complete files by atomic rename. Other changed files are captured as whole versions.
Unchanged files are not re-uploaded. Deleted source files remain in the published history;
this is an additive trace snapshot, not a directory mirror.

Each periodic sync uploads at most one batch (256 MiB of payload by default). A single
non-log file can exceed that soft limit; a single log record exceeding it reports an
error. Each node permits one upload at a time, capped at 32 MiB/s by default; periodic
uploads are staggered. Final sync drains the data visible at its start, using the same
node bandwidth cap. Set the per-node cap approximately to your allowed aggregate upload
bandwidth divided by node count. There is no cluster-wide scheduler or hard aggregate cap.

Shared storage contains immutable batches under `.opguard-bundles/`, not one repeatedly
rewritten `sync.tar.gz`. `opguard compare` restores them to temporary local directories
automatically, while still accepting legacy archives. For other analysis commands:

```bash
python -m opguard.local_sync.materialize /shared/opguard/experiment-001 /local-ssd/restored-run
```

The restore destination must not exist. Restoration verifies checksums and contiguous
batch sequences before publishing the local result. Node metadata stays separate and is
copied into that node's rank directories locally for source-aware comparisons.

Original traces and checkpoints are retained locally; allow space for them plus the
pending batch. Do not remove staging during a run. If a checkpoint is lost for a published
run, restore the staging directory or choose a new run ID. Completed uploads survive a
process crash; an abrupt process or node failure can still lose buffered or not-yet-uploaded
data. Failures enabling local staging never fall back to direct shared-storage writes.
Use a new output directory when migrating from the old snapshot format.

## Correspondence in Perfetto comparisons

`opguard compare` and the tools backend establish call correspondence before
comparing numerical observations. Within corresponding trace files, equal
function/file/line/rank/context sequences pair in execution order. Unequal
sequences require unique, monotone captured-input anchors; equal callsite
sequences between those anchors can then pair in order. Output values and output
shapes never affect pairing, and there are no similarity scores or repair passes.

Each operation carries a stable-within-export `record_id` and `correspondence`:
paired records reference each other and state the pairing basis; unresolved
records have state `undetermined` and a reason. Unresolved operations have status
`unaligned`, without a numerical diff or an insertion/deletion claim. Source-index
metadata is excluded from operation events.

Automatic pairs are **candidates**, marked `validation: unreviewed`; `paired`
does not certify semantic correspondence. `ok` means agreement of the compared
recorded fields, not bitwise tensor equality. Caller/source paths are evidence
for investigation, not a universal cross-framework identity requirement.

Agents can review or replace candidates using an artifact-bound
[alignment map](AlignmentMaps.md). `python -m opguard.tools align` writes a
new Perfetto view from an existing export: it can compare explicitly selected
tensor slots across different files, functions and ranks, or retain a rejected
relation as undetermined. All original observations remain available. This is an
offline report operation; no GPU run, implicit tensor transform or semantic
certification occurs.

Order-based pairing assumes corresponding execution order at the callsite;
captured input summaries are not proof of tensor equality. An observed first diff
after unresolved events may not be the earliest actual divergence. Investigate
caller/callee, upstream writes and downstream consumers to establish context, or
retain the ambiguity. Alignment links do not establish data dependencies.
