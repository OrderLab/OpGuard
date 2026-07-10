<p align="center">
  <img src="assets/images/brand/logo.svg" alt="OpGuard" width="128" />
</p>

<h1 align="center">OpGuard</h1>

<p align="center">
  <strong>Bitwise Alignment for Precise and General Debugging of Production LLM Training</strong>
</p>

<p align="center">
  <a href="#paper">OSDI ’26</a>
  ·
  <a href="#contact">Contact</a>
</p>

---

OpGuard is a debugging system for large-scale LLM training. When two training runs diverge, it uses **bitwise alignment** to compare them at tensor boundaries and pinpoint the **first divergent operator**—turning vague loss-curve anomalies into precise, actionable evidence.

In production use at ByteDance, OpGuard has reduced root-cause localization from **days** of manual effort to **minutes**.

> **Source code is not open to the public at this time.**  
> Interested in the system, collaboration, or early access? Please **email the authors**.

---

## Paper

OpGuard will appear at **OSDI ’26**:

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

---

## What’s on this site

| Section | Description |
| --- | --- |
| **Bitwise alignment** | Core abstraction for comparing training runs |
| **Live trace demo** | Explore divergence evidence in a Perfetto-style viewer |
| **Workflow** | How OpGuard fits into production debugging |
| **Impact** | Production case studies and timing results |

This repository hosts the **public project website** only — not the OpGuard source tree.

---

## Authors

Ziming Zhou · Yinjie Zhao · Hang Zhu · Wenxiao Wang · Zhihao Bai · Yun Zhang · Shuguang Wang · Haibin Lin · Peng Huang

University of Michigan (OrderLab) · ByteDance Seed

---

## Contact

Source code and internal tooling are **not publicly released**.

For questions about the paper, demos, or access inquiries, please **email the authors** listed above.

---

<p align="center">
  <img src="assets/images/brand/logo.svg" alt="" width="40" />
  <br />
  <sub>OpGuard · OSDI ’26</sub>
</p>
