<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/logo-wordmark-dark.svg">
    <img src="assets/brand/logo-wordmark.svg" alt="OpGuard" width="320">
  </picture>
  <p><strong>Debug training differences with your AI agent.</strong></p>
  <p>
    <a href="https://orderlab.io/OpGuard/">Website</a> ·
    <a href="https://www.usenix.org/conference/osdi26/presentation/zhou-ziming">OSDI ’26 Paper</a> ·
    <a href="https://orderlab.io/OpGuard/#trace">Live Demo</a> ·
    <a href="harness/">Agent Harness</a>
  </p>
</div>

Give your coding agent two training runs. OpGuard helps it find where their computations diverge and inspect the evidence.

<a href="harness/doc/Usage.md#agent-integration">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/quickstart-dark.gif?v=20261019-storage">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/quickstart.gif?v=20261019-storage">
    <img alt="Install OpGuard, ask Codex or Claude Code, and read an evidence-backed root-cause report" src="docs/assets/quickstart.gif?v=20261019-storage" width="960">
  </picture>
</a>

## Find the root cause

<table>
<tr>
<th>Standard logs · <a href="https://orderlab.io/OpGuard/demo/inputs/trace.txt">Expand ↗</a></th>
<th>OpGuard output · <a href="https://orderlab.io/OpGuard/demo/inputs/opguard.txt">Expand ↗</a></th>
</tr>
<tr>
<td><pre>The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
…</pre></td>
<td><pre>First observed difference · rank 0
get_grad_norm_direct · None.append#2246
run1: [0, 1077945344, 0, 0]
run2: [0, 1077804032, 0, 0]
Unaligned before boundary: 0 / 33
…</pre></td>
</tr>
<tr><td>~19k input tokens · 8 calls<br>Budget ≤ $0.010</td><td>~1.5k input tokens · 1 call<br>Budget ≤ $0.0012</td></tr>
<tr><td align="center"><a href="https://orderlab.io/OpGuard/?analyze=trace#rootcause"><kbd>Get rootcause</kbd></a></td><td align="center"><a href="https://orderlab.io/OpGuard/?analyze=opguard#rootcause"><kbd>Get rootcause</kbd></a></td></tr>
</table>

<details>
<summary>Actual root cause</summary>

A gradient buffer is overwritten on the default stream before its reduction on another stream finishes. [Details ↗](harness/examples/demo_case5/README.md#3-hypothesis-validation-and-known-answer-cross-check)

</details>

*Groq · GPT-OSS 20B · Same model and diagnostic prompt; same-launch evidence. Full logs read in eight parts (a few minutes); OpGuard uses first observed differences. Fixed inference budget caps at [Groq list prices](https://console.groq.com/docs/model/openai/gpt-oss-20b), including prompts, reading notes and maximum output; free within quota.*

## Get started

Install from PyPI with Python 3.10+. The training environment needs CUDA-enabled PyTorch and CUDA/CUPTI development tools.

```bash
# Training environment (Linux)
python -m pip install "setuptools>=77.0.3" "packaging>=24.2" wheel ninja
python -m pip install --no-build-isolation opguard

# Agent environment
python -m pip install opguard-harness
```

[Installation details →](docs/Installation.md)

Then ask your coding agent:

> Use OpGuard to investigate why these two training runs differ: `./run-a/train.py` and `./run-b/train.py`.
> My training Python is `/path/to/train-env/bin/python`.
> Read `opguard-harness tools` for the available tools, and explain your findings with evidence.

Works with coding agents that can run shell commands. [Agent setup and tool calls →](harness/doc/Usage.md#agent-integration)

## Impact

Deployed at ByteDance, OpGuard has helped diagnose **20+ production issues**, including kernel races and silent data corruption. The paper reports investigations shortened from **days to minutes**. [Read the results →](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)

**OSDI ’26 · Best Paper Award nominee.** [Project overview →](https://orderlab.io/OpGuard/)

[Installation](docs/Installation.md) · [CLI guide](docs/CLI.md) · [Examples](examples/README.md) · [Apache-2.0](LICENSE)

## Paper

**[OSDI ’26 Paper (PDF)](https://www.usenix.org/system/files/osdi26-zhou-ziming.pdf)** · OpGuard was presented at **[OSDI ’26](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)** (**Best Paper Award nominee**):

> Ziming Zhou, Yinjie Zhao, Hang Zhu, Wenxiao Wang, Zhihao Bai, Yun Zhang, Shuguang Wang, Haibin Lin, Peng Huang.  
> *OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training.*  
> In *Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’26)*, Seattle, WA, USA, July 2026.

<details>
<summary><strong>BibTeX</strong></summary>

```bibtex
@inproceedings{OpGuard2026OSDI,
  author = {Zhou, Ziming and Zhao, Yinjie and Zhu, Hang and Wang, Wenxiao and Bai, Zhihao and Zhang, Yun and Wang, Shuguang and Lin, Haibin and Huang, Peng},
  title = {{OpGuard}: Bitwise Alignment for Precise and General Debugging of Production {LLM} Training},
  booktitle = {Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation},
  series = {OSDI '26},
  month = {July},
  year = {2026},
  address = {Seattle, WA, USA},
  publisher = {USENIX Association},
}
```

</details>

For **SDC machine detection**—identifying GPUs that silently corrupt training computations—see [*SDCs in the Wild: Characterizing and Diagnosing SDC-Defective GPUs in Production LLM Training* (OSDI ’26)](https://www.usenix.org/conference/osdi26/presentation/zheng). The paper introduces **SDCHunter**, which replays the triggering training workload and input data to detect defective GPUs in production clusters.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The website and live demo remain in this repository alongside the Python packages.

## About ByteDance Seed Team

Founded in 2023, the ByteDance Seed Team develops advanced AI foundation models. It aims to advance science and benefit society through world-class research.
