---
name: trace-upstream
description: Trace an observed OpGuard tensor or event mismatch back through evidenced producers, mutations and dependencies to narrow its origin. Use when inputs already diverge or the producer of a differing value is unclear.
---

# Trace upstream

Own backward evidence tracing from a known mismatching event. Return a narrowed
causal boundary or a precise missing dependency. Do not independently run a full
investigation or promote a nearby suspicious operation to a proven cause.

## Inputs

Use the baseline/candidate event references, alignment context (rank, step,
invocation), mismatch criterion, available input/output evidence and any known
matching boundary. If the starting events are not demonstrably comparable, mark
their relationship undetermined and investigate correspondence before treating
them as a differing pair. Use caller/callee context, enclosing branches, captured
step/rank and upstream/downstream uses to distinguish repeated invocations or
different API paths. Preserve unresolved alternatives when that evidence is
insufficient; do not force a pair based on similar outputs or a fingerprint alone.

Before interpreting matching fingerprints or a source-only dependency, read
[references/evidence-strength.md](references/evidence-strength.md). Carry the
observation/claim/provenance record from the investigation forward; do not upgrade
inferences to observed edges.

## Follow the code context first

Start from `inspect_event` source-context frames and call
`code_context(artifact, source, line)` for the differing operation and its caller.
Read the argument expressions, local assignments and enclosing function before
requesting more runtime capture. Follow caller arguments into callee definitions,
then follow return values and mutated object fields back to their callers. Trace
upstream assignments and state writes, and downstream consumers far enough to
connect the differing value to the reported symptom. Use `search_source` to find
these sites and `code_context` to read each relevant function, including its
enclosing branch and call context. Distinguish definitions from invocations and
retain source locations for the reasoning chain.

For a proposed correspondence, retain both event references and the source/runtime
evidence that supports it. `compare_events` displays candidates; it does not
certify or change alignment. Source reasoning can support a mapping without a
new capture, while uncertain branch execution may still need a controlled test.

Caller/path equality is not a cross-framework requirement. When source or runtime
evidence supports a relation, `review_pair` can retain it or declare explicitly
selected tensor slots across different operators. Read the sibling investigation
skill's `references/evidence-contract.md` for selectors and rendering. A review
is an agent assertion, not an automatic proof. If the domains require an
unimplemented layout/partition transform, preserve the gap or choose a directly
comparable boundary; do not let the backend guess a transform.

Prefer embedded source when it covers this path. If a callee or caller is missing
from the artifact, inspect the corresponding version of the source checkout with
ordinary source-reading tools; record its revision and file/line references.
The trace reader does not open filesystem source paths. Source inspection is part
of this investigation, not a failure of the skill or a reason to require a dynamic
producer graph. An unverified checkout revision remains an evidence limitation.

This source-level def-use/control-flow reasoning is the primary upstream path;
a dynamic producer graph is not a prerequisite. Track casts, views, indexing,
mutation, state and relevant branch conditions as the code demands. Trace IO
anchors the observed value; source context explains how it can arise. Dynamic
capture is needed when competing executions or mechanisms cannot be separated
from the code and existing evidence.

Do not stop simply because producer IDs or full tensors are absent. First exhaust
the available function, caller and write-site context. Label the resulting edges
as code-derived, reserving claims about the branch actually taken, pointer alias
identity or overlap in time for evidence that establishes them.

## Follow evidenced dependencies

Inspect the mismatching output and corresponding inputs. Classify observations:

- Inputs differ: identify which input differs and inspect its evidenced producer
  on both sides. Continue until the cause narrows or the trace has a gap.
- Input values agree under an explicitly justified comparison, but outputs differ: examine configuration, dtype/layout,
  scalar arguments, mutable state, RNG, implementation and synchronization. This
  narrows the boundary; it does not establish that the operator is defective.
- Inputs or producer links are missing: return the missing observation and the
  smallest region/rank/step that would resolve it.

Use actual dependency IDs, captured values, versioned mutation information and
source semantics to justify edges. A shared pointer, adjacent trace record, or
matching call stack alone does not establish producer identity: addresses can be
reused, views alias storage, and in-place updates change values. Distinguish a
recorded edge from an inferred edge and explain the inference.

For asynchronous or distributed work, consider stream/event ordering and
collective participants where the evidence exists. Never impose a global causal
order using timestamps alone. Retain multiple candidate producer branches when
the observations do not distinguish them.

Inspect bounded summaries/slices before loading full tensors. A matching lossy
summary does not prove equal tensors. Keep the comparison criterion consistent
with the originating investigation.

## Tool boundary and handoff

Use `inspect_event`, `compare_events`, `trace_neighbors`, `code_context`, `search_source` and
`source_excerpt` for bounded inspection, as advertised by `opguard-harness tools`.
Neighbors and alignment flows are not producer edges. There is no producer-graph
query or targeted capture API. After following available source context, if the remaining dependency or
execution condition cannot be resolved, explicitly return that evidence gap; do not
simulate a dependency tool or claim the graph was traversed completely.

Return:

- The upstream path or candidate paths, with evidence references for each edge.
- The last observed matching boundary and first observed differing boundary within
  the inspected region, including the coverage limits.
- A candidate mechanism if one is supported, with competing explanations.
- The smallest missing observation if progress stopped.

Return this to the ongoing investigation. A causal hypothesis requiring an
intervention belongs to `validate-hypothesis`. Further tracing is unnecessary when
the required boundary is already established.
