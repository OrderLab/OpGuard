---
name: validate-hypothesis
description: Design and evaluate a discriminating test of a concrete numerical-divergence hypothesis using OpGuard evidence and bounded experiments. Use when a candidate cause needs confirmation or rejection, rather than for open-ended anomaly discovery.
---

# Validate hypothesis

Own the test of one explicit candidate mechanism. Return supported, refuted or
inconclusive with the evidence and limitations. Do not confuse a successful run,
report generation, or disappearance of a symptom with a demonstrated root cause.

Read [references/discriminating-tests.md](references/discriminating-tests.md) when
choosing an intervention or interpreting an ambiguous outcome. Retain the shared
evidence record's provenance: externally reported outcomes are not local tests.

## Define a discriminating test

Start with the hypothesis, its supporting evidence, relevant alternative
explanations, matched baseline/candidate inputs, and the user's execution budget.
State what should be observed if the hypothesis is true and what observation
would weaken it. Prefer tests against existing captured evidence when sufficient.

For a rerun, select the smallest intervention that separates the candidate from
its alternatives. Keep unrelated code, inputs, seed, rank mapping, instrumentation
and comparison criterion fixed where possible. Record unavoidable changes and
how they limit interpretation. Establish the original symptom in the control;
otherwise a negative result is usually inconclusive.

If a symptom disappears during capture, first compare the actual launch commands,
source versions, inputs and effective runtime settings with the failing run.
Separate instrumentation changes from environment changes: cross capture on/off
with the suspect setting while keeping the other controls fixed. If available,
compare saved workload outputs as well as trace verdicts; no aligned diff is not
proof that the workload agrees. Only attribute suppression to instrumentation
after a matched control isolates it. Record any remaining timing perturbation.

Examples of interpretation pitfalls:

- Adding synchronization removes a mismatch: timing is implicated, but the exact
  race, producer and missing ordering edge are not yet identified.
- Switching precision removes a mismatch: precision is relevant, but the switch
  can also change kernels or execution order.
- A new run passes once: an intermittent failure may simply not have occurred.

Specify useful repetition and stop conditions according to the observed failure
and available budget. Do not assign an arbitrary confidence percentage.

## Execute through supported interfaces

Read `opguard-harness tools` for installed capabilities. In addition to read-only
trace/source queries, execution tools are
`create_experiment`, `advance_experiment`, `experiment_status`,
`experiment_results`, `refine_experiment`, and `recover_experiment`. There is no
`run_validation` or `capture_window` tool, numerical verdict API, or selective
stage runner. A refined experiment currently repeats the fixed workflow; account
for that cost before choosing it.

Use `refine_experiment` with a revised full spec to preserve prior evidence when
that execution model fits the test. Inspect each stage result; `ok: true` is a
successful tool dispatch, not necessarily a successful experiment. Treat failed
launches, missing captures, incomplete ranks and alignment failures as invalid
tests, not as evidence for or against the numerical hypothesis.

Run experiments and edit workload code only within the user's existing scope and
budget. If necessary authority or resources are missing, finish the concrete test
plan before requesting what is missing. Do not repeatedly request permission for
already authorized runs. Preserve source changes and evidence; do not use forced
restoration to discard unrelated edits.

## Evaluate and hand back

Report the intervention, control, predicted result, actual observations, artifact
references and any confounders. Classify the hypothesis as supported, refuted, or
inconclusive for the tested conditions. Perfetto's `not_evaluated` is not a pass.

For a supported hypothesis, explain the mechanism connecting the intervention to
the first difference and identify alternatives still consistent with the result.
For a refuted or inconclusive hypothesis, return the next discriminating question
to the investigation rather than silently launching another sequence of tests.
