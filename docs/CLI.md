# Interactive OpGuard CLI

Use this interface for manual collection and inspection. For agent tool calls, use [OpGuard Harness](../harness/); the machine-readable backend contract is documented in [AgentTools.md](AgentTools.md). Installation and native extension options are in [Installation.md](Installation.md).

The interactive CLI keeps session state in `~/.opguard_state.yaml`. Use a dedicated workload checkout: instrumentation edits Python source. Do not edit those files or run harness instrumentation concurrently in that checkout.

## Command reference

```bash
opguard --help
opguard start --help
opguard run --help
opguard compare --help
```

| Command | Purpose |
| --- | --- |
| `doctor [--json] [--require-native]` | Inspect Torch, CUDA, CUPTI and native extension availability |
| `configure` | Show the current configuration |
| `dumpto DIRECTORY` | Select a capture output root |
| `start [DIRECTORY]` | Emit shell activation commands and enable discovery |
| `stop` | Finish discovery and generate an instrumentation configuration |
| `instrument` | Apply the selected instrumentation configuration |
| `replay` | Rerun the recorded workload with instrumentation |
| `run SCRIPT [SCRIPT ...]` | Run the interactive discovery/capture workflow for scripts |
| `compare RUN_A RUN_B` | Materialize captures and generate Perfetto comparisons |
| `sync` | Request synchronization of locally staged captures |
| `clear` | Restore instrumentation and clean interactive session state |

`replay` and `run` execute workload code. Include discovery and capture executions in the workload budget. Prefer the harness's explicit entrypoint specification when launch provenance and bounded retries matter.

Optional repository helpers are documented in [scripts/README.md](../scripts/README.md).

## Shell discovery and replay

Shell activation commands must be evaluated in the shell that launches the workload:

```bash
eval "$(opguard start /absolute/path/to/new-capture)"
python train.py
eval "$(opguard stop)"
opguard replay
```

Automatic discovery uses Python's `usercustomize` startup hook. An interpreter that disables the user site, including many isolated virtual environments, may not load this hook. Check `python -c 'import sys; print("usercustomize" in sys.modules)'` in that environment. If it is unavailable, use the explicit `opguard.tools` worker through the harness instead; that path does not depend on the startup hook.

`stop` disables autostart for subsequent processes. Ordinary tracing preserves the workload's numerical execution settings. Explicit deterministic interventions are described in [Installation.md](Installation.md#numerical-execution-policy).

For the quick workflow, supply executable Python or shell scripts:

```bash
opguard run run_a.py run_b.py --gpu 0 --disable-cupti --disable-cudaapi
```

The command creates captures and comparisons; inspect its output and workload logs. It does not return a root-cause diagnosis or numerical equivalence verdict.

## Explicit configuration and source restoration

For manual instrumentation, prepare a configuration for your workload and run the low-level instrumenter from a separate working copy:

```bash
python -m opguard.instrument_all_files config.yaml
python train.py
python -m opguard.instrument_all_files --mode revert --dry-run
python -m opguard.instrument_all_files --mode revert
```

Configuration paths resolve relative to the configuration file. Backups and a manifest are maintained under `.opguard/` in the working directory. The regular revert path checks changes; inspect a mismatch before deciding how to reconcile it. Interactive `opguard clear` performs forced restoration, so it is not a substitute for reviewing concurrent user edits. Harness capture uses its separate source transaction and restoration protocol.

To convert an existing callsite artifact into a configuration, the supported module and argument are:

```bash
python -m opguard.tools.callsite_to_instrumentation \
  --callsite-file /path/to/callsites.json \
  --base-dir /path/to/workload \
  --output /path/to/capture.yaml
```

See [Preprocessing.md](Preprocessing.md) for the compression and configuration pipeline.

## Compare captures

```bash
opguard compare /path/to/run-a /path/to/run-b --gpu 0
```

For already materialized rank directories, the exporter can write a selected output directly:

```bash
python -m opguard.trace_export.perfetto \
  --run1 /path/to/run-a/exec_trace/gpu_000 \
  --run2 /path/to/run-b/exec_trace/gpu_000 \
  --out /path/to/comparison.json.gz
```

Open the generated trace in Perfetto. Reported pairs are correspondence candidates, and matching recorded fingerprints do not prove tensor equality. [AlignmentMaps.md](AlignmentMaps.md) describes explicit reviews. [Capture.md](Capture.md) describes local staging, distributed paths, transport completeness and comparison evidence.
