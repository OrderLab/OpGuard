---
name: investigate-first-diff
description: Investigate numerical divergence between OpGuard runs, starting from an observed first difference and coordinating evidence gathering, upstream tracing and hypothesis validation. Use for an end-to-end explanation of why two executions diverge.
---

# Investigate first diff

Own the investigation question, evidence record and final explanation. A first
observed difference is a boundary of current observations, not necessarily the
operation that caused the fault. Prefer existing artifacts before new captures.

## Establish the boundary

Identify the baseline/candidate artifacts, code revisions, inputs/seed, rank and
step correspondence, comparison criterion and any authorized rerun budget. Use
available evidence to establish these; ask only for missing information that
prevents meaningful comparison. Record unknowns rather than silently assuming
identical inputs or determinism.

Inspect completeness and alignment before explaining numerical differences. A
missing event, unmatched invocation, or incomplete rank is not a tensor mismatch.
Compare corresponding invocations, not merely matching function names or equal
row numbers. Wall-clock order across ranks/streams does not establish causality.

`unaligned` / `correspondence.state: undetermined` is a valid outcome, not a
numerical difference or proof of an inserted/deleted call. Keep the reason and
candidate context. Prefer fewer supported pairs to guessing a complete mapping;
never adjust correspondence to make outputs agree.

Even an exporter `paired`/unique peer is a proposal, not certified semantic
correspondence. Different callers/frameworks may compute corresponding values;
the same helper may operate on different logical domains. Review the intended
boundary and any layout/rank mapping before treating a candidate as comparable.
Different inputs alone do not invalidate alignment: they may carry the bug.

Use `review_pair` to retain a supported comparability judgment and its evidence,
then `difference_candidates` to continue or revisit reviewed pairs. A real value
difference that does not explain the symptom remains a valid observation; refute
its causal hypothesis rather than rejecting the pair. Concrete case judgments
belong in artifact-scoped investigation records, not global ignore rules or
future blind-evaluation inputs.

Use regional fingerprints from `compare_events` / `inspect_event` to narrow the
logical byte ranges worth investigating. A difference confined to the last
region can motivate a padding or incomplete-write hypothesis; it does not
establish either. Inspect allocation, writer bounds, valid extents/masks and
downstream reads before treating a difference as noncausal. Preserve the original
capture: replacing `empty` with zeros is an explicit intervention, not routine
trace cleanup. One differing region is not one differing element or one ULP;
matching regions can still contain XOR collisions.

Record the earliest observed mismatch and the last observed matching boundary
under the chosen criterion. Distinguish exact equality from tolerance-based
agreement. If coverage does not establish a unique first difference, describe the
candidate boundaries instead of selecting one arbitrarily.

## Route the investigation

- If correspondence is undetermined near or before a diff, use `trace-upstream`
  to inspect caller/callee and upstream/downstream source context. Record any
  supported correspondence with its evidence, or retain the unresolved region.
- If the differing operation already receives different inputs, or its producer
  is unclear, use `trace-upstream` with the event references and known boundary.
- If evidence suggests a specific mechanism, use `validate-hypothesis` with that
  hypothesis, its evidence and the remaining execution budget.
- If evidence already resolves the question, report it without forcing another
  skill or rerun. If neither tracing nor testing can advance the investigation,
  state the missing evidence and the smallest useful next observation.

These are procedural handoffs within the current agent's work, not instructions
to spawn other agents. The skills are siblings under `skills/`; read their
`SKILL.md` only when the relevant work is needed.

Maintain a compact investigation record in the user's chosen output location or
conversation: artifact references (path plus event/record identifier), observed
boundary, hypotheses, supporting/contradicting evidence, experiments and remaining
budget. If records lack stable IDs, use an exact artifact path and JSON key or
line locator. Do not invent backend IDs.

## Tool availability and evidence handoff

Read [references/evidence-contract.md](references/evidence-contract.md) when opening
trace artifacts or handing evidence to another procedure. It defines the shared
claim/provenance record, review-map workflow and concrete query sequence. Start with
`trace_summary` and `first_diff`, then retrieve bounded event/source evidence.
Check advertised schemas with `opguard-harness tools`.

Read-only queries do not require a new experiment. Existing experiment tools can
create, advance, inspect and recover captures; `advance_experiment` can incur job
cost. Producer-graph traversal, raw tensor comparison and targeted recapture APIs
are not implemented. Do not claim these capabilities. Perfetto generation returns
`verdict: not_evaluated`, not numerical equivalence.

## Deliver the explanation

State the observed first difference, the supported causal explanation (if any),
source/evidence references, validation outcome and remaining alternatives. Label
an untested explanation as a hypothesis. For an inconclusive investigation, report
the narrowed boundary and evidence gap rather than claiming a root cause.
