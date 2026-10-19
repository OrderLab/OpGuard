# Choose tests that distinguish mechanisms

Select by evidence; these are examples, not a required checklist or exhaustive
catalog. Separate a diagnostic perturbation from a proposed production fix.

| Candidate mechanism | Useful controlled observation/intervention | Interpretation limit |
| --- | --- | --- |
| Wrong pairing / missing coverage | Recover iteration, rank and invocation identity; capture the missing boundary | Similar names/counters alone do not verify alignment |
| Wrong formula / indexing / shape semantics | Compare a small fully specified input with an independent mathematical reference | A small input may miss size/layout-specific faults |
| Precision / reduction order | Fix input bytes and vary only one dtype, backend or reduction implementation | A flag can also alter scheduling and kernels |
| RNG / initialization / state reuse | Capture actual input and RNG/state identities, restore the same state | The same seed alone does not guarantee the same RNG consumption |
| Mutation / alias / buffer lifetime | Capture versioned values before and after the suspected write; isolate storage where appropriate | Cloning changes memory pressure and timing |
| Async ordering / communication | Establish the write/read dependency; test a targeted ordering edge | Global synchronization suppresses many mechanisms at once |
| Symptom disappears under observation | Match the failing launch and cross capture on/off with any changed runtime setting; check workload outputs independently | Stream connection limits, launch blocking, logging and checksum copies can change scheduling; an unobserved diff may also reflect missing coverage |
| Hardware / environment | Repeat the same fixed workload across controlled placements with error telemetry | A disappearing intermittent symptom does not isolate hardware |

For each selected experiment record: hypothesis, alternatives, unchanged controls,
intervention, predicted observations under each explanation, measurement method,
launch count, actual outcome and confounders. If the baseline symptom does not
reproduce, report inconclusive rather than claiming a repair.

A well-supported causal account connects the suspected mechanism to the observed
divergence and survives a discriminating check. Existing issue reports or papers
can corroborate an explanation but are not a locally executed intervention.

If hardware or capture capabilities are unavailable, deliver an executable test
specification where possible, identify the missing prerequisite, and mark the
experiment `not_run`. Do not report an external author's result as your own run.
