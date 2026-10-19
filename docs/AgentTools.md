# Automation protocol v1

OpGuard exposes deterministic backend operations through `python -m opguard.tools`
(or the installed `opguard-tools` entrypoint). The independently installed
[OpGuard Harness](../harness/) package in this repository provides
agent function schemas, durable experiment records and a seven-stage workflow.

Run every backend operation in a fresh process. This API is for trusted local
operators: it executes workload code with the caller's permissions. It is not an
execution sandbox or a network service.

## Envelope

```bash
python -m opguard.tools capabilities
python -m opguard.tools ACTION --request request.json
```

All requests include `"protocol_version": 1`. Success returns
`{"protocol_version":1,"ok":true,"result":{...}}`; errors return
`{"protocol_version":1,"ok":false,"error":{"type":"...","message":"..."}}`
with a nonzero exit code. Only JSON goes to stdout. Workload output goes to
`OUTPUT/workload.log`; diagnostics go to stderr. Argument parser errors use the
usual CLI usage/error channel.

## Workloads and captures

`discover` takes:

```json
{
  "protocol_version": 1,
  "output": "/shared/experiment/baseline-discovery",
  "local_root": "/local-ssd/opguard",
  "timeout_seconds": 1800,
  "workload": {
    "cwd": "/work/baseline",
    "launcher": ["/env/bin/python"],
    "script": "/work/baseline/train.py",
    "args": ["--seed", "123", "--steps", "12"],
    "env": {"OPGUARD_CALLSITE_CUPTI": "0", "OPGUARD_CALLSITE_CUDAAPI": "0"},
    "ranks": [0]
  }
}
```

The launcher receives `-m opguard.tools.worker REQUEST SCRIPT [ARG ...]`.
A Python interpreter or `torchrun --nproc-per-node=...` therefore works without
shell command construction. Shell scripts are not Python entrypoints; a custom
launcher must forward these arguments unchanged and wait for every rank.

The worker configures tracing only after the launcher has started it. It flushes
local staging and publishes one small completion receipt per rank after tracer
exit callbacks. Discovery/tracing only succeed if the blocking launcher exits
successfully, exactly the expected ranks publish receipts, the final batch counts
match, every batch materializes with valid checksums/sequences, and each rank has
nonempty trace files. These checks establish transport/file completeness, **not**
semantic completeness of every expected training step or callsite.

All output directories must be fresh. Automatic launcher restarts into the same
capture directory are unsupported; duplicate rank receipts fail. A retry needs a
new output directory. Workload timeouts terminate the local launcher process group;
external schedulers must propagate termination to their remote workers.

Capture preserves the workload's numerical execution settings by default. An
explicit `OPGUARD_DETERMINISTIC=1` in `workload.env` enables the separately recorded
policy described in [Installation.md](Installation.md#numerical-execution-policy).
It does not replace allocation functions or override the caller's fill policy.

## Preparation and source transactions

`prepare` takes `output`, `workload`, and `discovery` (the discovery result). It
materializes discovery locally, preprocesses calls, and emits a normalized YAML
and a plan with config/source SHA-256 fingerprints. It limits instrumentation to
files in the workload checkout, records excluded files and rejects empty plans.
Coverage is explicitly `observed_callsites_only`.

`trace` takes the same fields as `discover`, plus `plan` (the preparation result).
It checks the plan and source fingerprints before changing source, saves durable
originals under `OUTPUT-instrument/`, instruments, captures, and restores source
in a `finally` block. The source checkout is locked for the entire operation.
Global `entry`/explicit `meta_vars` config sections are not supported by this API.
Existing automatic metadata injection is confined to the selected source files.
Instrumentation diagnostics may remain alongside source files.

Use dedicated workload checkouts; do not run the interactive instrumenter or edit
source during a harness operation. If source changes during capture, the tool
retains those edits and the original copies, fails, and keeps
`.opguard/harness-active.json` to block subsequent harness operations. After an
uncatchable termination, inspect this marker and its referenced `originals.json`,
restore/reconcile the listed source files, verify their hashes against the plan,
then remove the marker. No forced restoration of user edits occurs automatically.

For multiple nodes, v1 requires **the same source checkout and paths visible on
all workers**, or an external system that distributes exactly the instrumented
source before worker execution. Independent node-local source deployment is not
implemented. Requests/output must be visible at the same paths on all workers;
`local_root` must be node-local, shared by that node's ranks, and outside the output
root. Global rank IDs must be unique, supplied by the launcher, and match `ranks`.

## Comparison

`compare` takes `output`, `baseline`, `candidate` (successful trace results), and
`ranks`. It materializes both captures locally, validates requested rank paths,
and invokes the existing Perfetto exporter for every requested rank. Any exporter
failure fails the operation. It returns `reports`, `ranks`,
`coverage: observed_callsites_only`, and **`verdict: not_evaluated`**.

A generated Perfetto report is not a numerical equivalence result or a root-cause
finding. Rank remapping, tolerance-based aggregate verdicts, coverage validation
for training steps, automatic scope refinement, preflight caching and cluster-wide
I/O budgets are not part of protocol v1.

### Diagnostic fingerprints

New captures default to four `uint32` XOR fingerprints per tensor (16 bytes).
`OPGUARD_XOR_REGIONAL` selects 1–64 regions; unset means 4, and 0 is a legacy
alias for 1. This setting affects diagnostic summaries, not explicit low-level
`xor_signature(t)` calls, which retain the historical scalar result. The legacy
row API and `OPGUARD_XOR_REGIONAL_THRESHOLD` do not control the new flat regions.

`stats.xor_sig_layout` (also exported in each Perfetto `io` tensor) records
`scheme: flat-u32-le-v1`, `regions`, `logical_nbytes`, and half-open `byte_ranges`.
In logical contiguous order, pack little-endian 32-bit words, zero-padding only
the final partial word for the checksum. Divide words evenly into R contiguous
regions, giving the first remainder regions one extra word. Empty regions have
zero XOR. This does not modify or zero-fill any workload tensor or its padding.
CPU and CUDA use identical boundaries, independent of shape or launch geometry.

CUDA computes all regions in one first-pass launch and, when needed, one
per-region reduction launch, on the calling stream. There is no per-region host
read or device synchronization. Rebuild `opguard/bitwise_xor` to use the native
path; older/missing extensions use a correct but slower tensor-operation fallback.
Summary materialization remains in the existing background writer; it is not a
new step-end synchronization mechanism.

Different regions identify coarse byte ranges to investigate, not element counts,
ULP errors or proof of padding noise. Matching regions can still collide. Compare
only compatible fingerprint schemes and semantic tensor boundaries; a legacy
single XOR cannot be retrospectively expanded into regional evidence.

The optional deterministic entry handler no longer replaces `torch.empty*` with
zero allocation or overrides `fill_uninitialized_memory`. Its other opt-in
deterministic settings remain; PyTorch uses the caller's fill policy.

## Reviewing an existing comparison

`align` takes `artifact` (an existing flat Perfetto export), `alignment_map`
(artifact-bound review JSON), and `output` (a fresh trace file). It applies
explicit correspondence reviews without rerunning a workload or matching source
paths. Selected tensor slots can belong to different implementations. See the
[map schema and evidence limits](AlignmentMaps.md).

It returns `artifact`, `sha256`, `source_sha256`, `map_sha256`, and
`verdict: not_evaluated`. The source artifact is never overwritten. This changes
the report's declared relationships, not the numerical observations or the
capture backend's rank topology.
