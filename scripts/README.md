# Helper scripts

Run these from the repository root after installing OpGuard.

- `source scripts/opguard_shell_init.sh`: optional Bash/Zsh wrapper that evaluates the shell activation commands emitted by `opguard start`, `stop`, `clear`, and `instrument`. Without it, use the explicit `eval` commands in the [CLI guide](../docs/CLI.md).
- `bash scripts/find_diff.sh RUN_A RUN_B OUTPUT_DIR`: run the low-level inspector and generate an effect chain when differences are found. Optional arguments are `filter_op`, `dp_count`, `worker_count`, and `ignore_config`. For the standard Perfetto workflow, use `opguard compare` as described in the CLI guide.
