# OpGuard Harness

Give your coding agent [OpGuard](../README.md) tools to investigate why two training runs differ.

```bash
git clone https://github.com/OrderLab/OpGuard.git
uv tool install ./OpGuard/harness
```

Then ask your coding agent:

> Use OpGuard to investigate these two runs: `./run-a/train.py` and `./run-b/train.py`.
> My training Python is `/path/to/train-env/bin/python`.
> Run `opguard-harness tools` to discover the tools, and explain the evidence you find.

The harness works with coding agents that can run shell commands. It shares this repository with OpGuard and installs independently. Install OpGuard in the training environment; the harness itself needs no PyTorch.

This is a source release; PyPI packages are not published yet. [Installation details](doc/Usage.md#install).

[Agent setup and tool calls](doc/Usage.md#agent-integration) · [Diagnostic skills](skills/investigate-first-diff/SKILL.md) · [Example experiment](examples/experiment.json) · [Walkthrough](examples/demo_case5/README.md)

[Apache-2.0](LICENSE).
