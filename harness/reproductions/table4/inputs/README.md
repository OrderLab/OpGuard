# Diagnostic inputs

Use a new conversation in a workspace containing only this exported bundle and
the installed runtime dependencies. Start with `skills/investigate-first-diff`.
The task identifiers are opaque; source APIs and ordinary launch configuration
are intentionally visible for code-based diagnosis.

Read `tasks/<id>/task.json` for the comparison criterion and run configuration.
Run `python run_task.py <id>` to execute it. Tasks with executable baseline and
candidate configurations also support `--capture`, which uses the public harness
workflow. Tasks with saved reference tensors can be run directly; inspect the
repeated outputs and reference bytes using ordinary tensor-reading code. A saved
reference is an observed result, not an instrumented reference implementation.

Use the pinned source supplied with each workload to inspect callers, callees,
state and upstream/downstream computations. Record hypotheses before proposing
interventions, and retain inconclusive results where evidence is insufficient.
The bundled skills do not replace source inspection or controlled tests.

The inventory in `manifest.json` covers the initial inputs. Verify it before
delivery; run-generated observations and traces are new evidence. Trace generation
does not itself produce a numerical verdict. Pin an isolated Python environment
with PyTorch, the needed model libraries, OpGuard and OpGuard-harness installed
normally; an editable install pointing into a maintainer checkout would expose
that checkout. Do not provide repository history, other experiment directories,
or access to external issue discussions in an artifact-only evaluation.

Directory separation does not enforce access control. Mount only this bundle in
the solver workspace. A previously informed conversation remains a known-answer
walkthrough even with these inputs; a fresh context removes conversation leakage,
but cannot guarantee a model has never encountered the upstream code before.
