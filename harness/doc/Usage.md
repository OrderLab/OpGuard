# Harness usage

A skills library and diagnostic toolset for coding agents investigating numerical
divergence with [OpGuard](https://github.com/OrderLab/OpGuard). Its focus is the
last mile from **first observed difference to an evidence-supported root cause**.
Apache-2.0 licensed; no model SDK or built-in chat loop.

For diagnostic evaluation, export the [reproduction inputs](../reproductions/table4/README.md)
into a separate workspace. The maintainer checkout also contains answer materials
and historical repair controls; it is not itself an unhinted evaluation input.

## Diagnostic skills

The three skills are siblings, with an investigation entrypoint and two focused
procedures. They are not a mandatory three-stage pipeline or separate agents.

| Skill | Owns | Returns |
| --- | --- | --- |
| [investigate-first-diff](../skills/investigate-first-diff/SKILL.md) | Run comparability, initial difference boundary, routing and evidence synthesis | Root-cause explanation or a clearly bounded inconclusive result |
| [trace-upstream](../skills/trace-upstream/SKILL.md) | Following evidenced producers, mutations and dependencies backward | Narrowed causal boundary, candidate mechanism or missing observation |
| [validate-hypothesis](../skills/validate-hypothesis/SKILL.md) | A discriminating test of one explicit candidate cause | Supported / refuted / inconclusive, with control and evidence |

Start by asking your coding agent to read
`harness/skills/investigate-first-diff/SKILL.md` from the repository root and investigate two identified captures.
Include the comparison criterion and rerun budget when known. The agent reads the
other skills when useful; they can also be invoked independently when a boundary
or hypothesis is already known. Keep all three directories together when copying
the library. Client-specific installation and automatic discovery are separate
from this repository's tool protocol; `pip install` installs the CLI, not skills
into a client's configuration. The skill sources are in the Git checkout.

Example request:

> Use the investigate-first-diff skill to investigate these two captures. Prefer
> existing evidence; use at most two additional workload launches. Explain the
> supported cause, or identify the smallest missing observation.

Budget counts must include actual workload launches: one executor experiment
currently launches discovery and tracing on both sides, not just one workload.

## Responsibilities

- **Skills:** diagnostic method, evidence standards and procedural handoffs.
- **Agent:** source reading, hypotheses, choosing useful queries/experiments, and
  explaining the result. The skills do not instruct agents to spawn subagents.
- **Harness tools:** reliable queries and execution. Currently implemented tools
  cover experiment execution/status, trace/source queries, artifact-scoped
  comparability reviews and rendering explicit alignment maps. Raw tensor comparison, producer-graph
  and targeted-capture APIs remain to be implemented.
- **OpGuard:** instrumentation, trace collection, local staging/uploads and report
  generation through the versioned backend CLI.

The existing resumable executor remains a supporting tool. Its fixed workflow is:

```text
baseline:  discover → prepare → trace (instrument / capture / restore)
candidate: discover → prepare → trace (instrument / capture / restore)
compare:   materialize → generate per-rank Perfetto reports
```

An investigation can start from existing captures without running this workflow.
The current backend generates Perfetto reports, not a numerical verdict:
`verdict: not_evaluated` must never be interpreted as equivalence.

## Install

Harness requires Python 3.10+ on Linux or macOS and has no runtime dependencies.

Install from PyPI without cloning the repository:

```bash
uv tool install opguard-harness
# Or install into an existing agent virtual environment:
python -m pip install opguard-harness
opguard-harness tools
```

uv installs the same wheel as pip into an isolated tool environment; it does not
need a local Torch or OpGuard installation. Invoke its executable from your agent.
If your adapter imports `TOOLS` and `dispatch`, instead use
`python -m pip install opguard-harness` in the adapter's Python environment.

The harness is versioned and released independently of the training package; see
[Publishing](../../docs/Publishing.md).

Editable installation for development:

```bash
git clone https://github.com/OrderLab/OpGuard.git
cd OpGuard/harness
python -m pip install -e .
opguard-harness tools
```

Install the current OpGuard revision separately in the execution environment,
including its tracing dependencies. The required backend API can be checked with:

```bash
/path/to/opguard-env/bin/python -m opguard.tools capabilities
```

Older OpGuard revisions without protocol v1 are unsupported. The controller and
execution environments can use different Python interpreters. For v0.1 the backend
runs as a local subprocess; there is no remote RPC transport. External launchers
can manage workers on multiple nodes.

## Agent integration

For a coding agent with shell access, no model adapter is needed. Install the
harness, give the agent both run entrypoints and the training Python path, and
ask it to run `opguard-harness tools`. It can read the schemas and invoke tools
through `opguard-harness --root ./experiments call TOOL_NAME --arguments args.json`.
The training Python becomes the specification's `backend` and workload launcher.
Include a workload-launch budget when needed; a complete discovery-and-trace
experiment launches each side twice. The agent inspects evidence and decides
what to investigate next.

For agents with a registered-function interface, use the same schemas and
dispatch boundary below.

`opguard-harness tools` returns provider-neutral function schemas. Adapt those
schemas to your agent's tool registration format and forward each function call:

```bash
opguard-harness --root /shared/opguard-experiments call TOOL_NAME --arguments args.json
```

Alternatively send the JSON argument object on stdin. The CLI writes one JSON
response and does not require a model API key. It is not an MCP server; an MCP or
SDK adapter can invoke this same boundary without changing execution logic.

| Tool | Purpose |
| --- | --- |
| `create_experiment` | Validate and persist a spec; no workload execution |
| `advance_experiment` | Execute exactly one next stage |
| `experiment_status` | Read stages, attempts, errors and log locations |
| `experiment_results` | Read final artifacts and explicit verdict status |
| `refine_experiment` | Create a child experiment from a new full spec |
| `recover_experiment` | Reconcile a controller interruption after backend exit |
| `trace_summary` / `first_diff` | Inspect an existing export and its per-run/rank difference boundaries |
| `inspect_event` / `compare_events` | Retrieve recorded fingerprints and explicit comparison candidates |
| `trace_neighbors` | Inspect nearby operations, without asserting dependencies |
| `code_context` | Read the enclosing function and callsite from captured source |
| `search_source` / `source_excerpt` | Follow caller arguments, field writes and definitions in embedded source |
| `difference_candidates` | Page unreviewed differences or revisit previous judgments |
| `review_pair` | Append an evidence-backed comparability judgment or declare a tensor boundary |
| `render_alignment` | Freeze the map and render a new Perfetto view through OpGuard; no workload launch |

Typical interaction:

1. Edit [examples/experiment.json](../examples/experiment.json): backend interpreter,
   two separate source checkouts, workload arguments, launchers, expected ranks,
   and node-local staging root.
2. Call `create_experiment` with that JSON file. Keep the returned ID.
3. Call `advance_experiment` with `{"experiment_id":"RETURNED_ID"}`. Read
   `result.status` and stage details after **every** call. Repeat while pending.
4. On failure, inspect `backend.log`, `response.json`, and `output/workload.log` in
   the attempt directory. A bounded retry uses a fresh output directory. Never
   interpret a tool dispatch success (`ok: true`) as experiment success: the
   returned experiment can have `status: failed`.
5. Read `experiment_results` when all stages succeeded. If a further experiment
   is justified, call `refine_experiment` with the parent ID and a revised spec.

In Python, `opguard_harness.tools.dispatch(root, name, arguments)` exposes the same
boundary. No agent framework is required.

## Investigate existing captures

The read-only tools accept `artifact` as an absolute path to an OpGuard flat
Perfetto JSON/JSON.gz export; no experiment record or GPU is required. Obtain
artifact-bound `event_ref` values from `first_diff`, then use `inspect_event` and
`code_context` to follow the consumer's arguments into caller and write-site code.
New OpGuard exports distinguish paired calls from `unaligned` events whose
correspondence is `undetermined`. Queries preserve the pairing basis or unresolved
reason; `first_diff` also exposes the first unaligned event before the observed
diff. Agents use caller/callee and upstream/downstream source context to investigate
these gaps. Unknown relationships are not tensor mismatches or inserted calls.

Automatic `paired` relations remain unreviewed candidates. A unique peer only
describes the exported relationship; it does not certify semantic correspondence.
Different frameworks and caller paths may produce corresponding data. Use
`difference_candidates` with `cursor: 0`, `limit: 10`, `include_reviewed: false`,
then `review_pair` to record `comparable`, `not_comparable` or `undetermined` with
reason and evidence. Reviews are stored under `ROOT/reviews/ARTIFACT_SHA256.json`;
history is retained and the last judgment for a pair is active. They never become
global function ignore rules or numerical verdicts.

To declare a new relation, select both exported tensor slots, e.g. `["outputs", 0]`
and `["inputs", 1]`; file, function and rank identity are not required. To reject
or leave a whole relation unresolved, use null slots. A genuine difference that
does not explain the final symptom should remain comparable, with the causal
hypothesis refuted in the investigation record.

`render_alignment` accepts `artifact`, a fresh absolute `output` file, and a
`backend` argv such as `["/env/bin/python", "-m", "opguard.tools"]`. It writes a
new Perfetto view with the current map and preserves the original artifact,
observations, review history and backend logs. It does not run a workload,
repartition tensors or certify agent reasoning. Continue reviewing the original
artifact to reuse its history; derived exports have new event references.
See the [evidence/query contract](../skills/investigate-first-diff/references/evidence-contract.md)
and [public demo walkthrough](../examples/demo_case5/README.md).

The demo acceptance run is known-answer, not a blind diagnosis benchmark. It
exercises actual public artifacts and exposes unresolved evidence/alignment gaps.
Read-only tools do not infer numerical equality from fingerprints, interpret
neighbor/flow links as data dependencies, or treat captured source as executed
instructions. They accept flat exports up to 256 MiB encoded/decoded; stacked
operation exports are rejected explicitly.

New OpGuard captures default to four regional XORs. `compare_events` and the peer
evidence in `inspect_event` expose `regional_comparison`: tensor-slot candidates,
exact logical byte ranges and `differing_regions`. Reviewed cross-input/output
slots are honored. Legacy signatures or incompatible layouts remain unavailable
or incompatible rather than receiving invented regions. A tail-only difference
can motivate checking padding, writer bounds and consumer reads; it is not an
automatic noise verdict, and a differing region is not a differing-element count.
Matching regions remain lossy evidence. Keep zero-fill experiments separate from
the original capture; current OpGuard no longer monkey-patches `empty` to zeros.

## Execution and recovery guarantees

Each experiment has its own atomic `experiment.json`, advisory writer lock and
attempt directories. Successful stages are retained; a new attempt never appends
to an old capture. Workload timeout and per-stage attempt budgets are explicit.
The timeout covers workload launch/execution, not the duration of preprocessing,
instrumentation, materialization or report generation.

A killed controller leaves a `running` stage. The harness refuses duplicate
execution. Recovery checks the recorded backend PID, terminal JSON response and
source restoration marker. It conservatively blocks if the situation is ambiguous
(including a reused PID). Inspect/cancel any launcher workers before starting a
replacement experiment. There is no persistent daemon or remote job cancellation
adapter in this version.

The backend uses dedicated source transactions and never uses the interactive
`~/.opguard_state.yaml`. Do not edit workload source or run the interactive
instrumenter concurrently. If source is edited or restoration is interrupted,
originals remain available and further operations on that checkout are blocked;
see the [backend recovery protocol](../../docs/AgentTools.md).

Specs are immutable after creation. Plan fingerprints detect changes to the
instrumented source files; they do not fingerprint the entire Python environment,
data, or non-instrumented files. `input_id` is operator-supplied provenance, not a
data reproducibility guarantee. Pin inputs, seeds, code and dependencies yourself.
Do not put credentials into spec files; specs and workload environments explicitly
provided in them are persisted in plaintext.

## Distributed execution contract

For one node, replace the launcher with, for example:

```json
["/env/bin/torchrun", "--standalone", "--nproc-per-node=8"]
```

Set `ranks` to `[0,1,2,3,4,5,6,7]` on both sides. The backend appends
`-m opguard.tools.worker REQUEST SCRIPT [ARGS]`. Custom/multi-node launchers must
forward these arguments, export global/local rank identities and block until all
workers terminate. They must propagate cancellation; background-only job submission
is unsupported. Disable automatic worker restarts within a capture attempt.

All workers must see request/output paths and the **same instrumented source**.
The first version supports shared source checkouts; independent node-local source
installation/deployment is not automated. `local_root` must be truly node-local,
accessible by all ranks on that node. The output root must be outside it.

Each successful worker uploads local batches before publishing a small completion
receipt. Missing ranks, mismatched batch counts, corrupt archives or absent trace
files prevent the next stage. This validates artifact completeness, not coverage
of every training step or branch. Baseline and candidate must use the same global
rank set; topology remapping is not implemented.

## Scope of v0.1

Implemented: three diagnostic skills, supporting seven-stage execution, agent
schemas/JSON CLI, bounded trace/source-context queries, attempt isolation,
source restoration, per-rank finalization checks, comparability review history,
explicit alignment-map rendering, errors and reports.

The coding agent interprets the skills; there is no built-in chat loop or autonomous
diagnosis engine. Raw tensor comparison, dynamic producer-graph queries and
preflight cache reuse are not implemented, nor are numerical verdicts, capture scope/step selection API,
cluster-wide byte budgets, scheduler adapters, and source deployment to node-local
checkouts. This is an initial implementation, not a claim of validated operation
on a production GPU cluster.

Validation runs the state machine against a subprocess test backend. OpGuard's
backend tests also exercise real CPU worker processes, local batching/uploads,
rank completion, failure and timeout paths. Full PyTorch/CUDA multi-node workflow
validation requires a suitable execution environment.

## Development

From the repository's `harness/` directory:

```bash
python -m unittest discover -s tests -v
```

## License

[Apache License 2.0](../LICENSE).
