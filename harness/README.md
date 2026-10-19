# OpGuard Harness

Give your coding agent [OpGuard](https://github.com/OrderLab/OpGuard) tools to investigate why two training runs differ.

```bash
python -m pip install opguard-harness
opguard-harness tools
```

Then ask your coding agent:

> Use OpGuard to investigate these two runs: `./run-a/train.py` and `./run-b/train.py`.
> My training Python is `/path/to/train-env/bin/python`.
> Run `opguard-harness tools` to discover the tools, and explain the evidence you find.

Requires Python 3.10+. Works with coding agents that can run shell commands. Install OpGuard separately in the training environment; the harness has no runtime dependencies.

[Installation details](https://github.com/OrderLab/OpGuard/blob/main/harness/doc/Usage.md#install).

[Agent setup and tool calls](https://github.com/OrderLab/OpGuard/blob/main/harness/doc/Usage.md#agent-integration) · [Diagnostic skills](https://github.com/OrderLab/OpGuard/blob/main/harness/skills/investigate-first-diff/SKILL.md) · [Example experiment](https://github.com/OrderLab/OpGuard/blob/main/harness/examples/experiment.json) · [Walkthrough](https://github.com/OrderLab/OpGuard/blob/main/harness/examples/demo_case5/README.md)

[Apache-2.0](https://github.com/OrderLab/OpGuard/blob/main/harness/LICENSE).
