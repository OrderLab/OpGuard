# OpGuard

**Bitwise Alignment for Precise and General Debugging of Production LLM Training**

OpGuard is a debugging system for large-scale LLM training. When two training runs diverge, it uses **bitwise alignment** to compare them at tensor boundaries and pinpoint the **first divergent operator**—turning vague loss-curve anomalies into precise, actionable evidence.

In production use at ByteDance, OpGuard has reduced root-cause localization from days of manual effort to minutes.

This repository hosts the **public project website** (paper overview, interactive demos, and citation). It is **not** the OpGuard source tree.

> **Source code is not open to the public at this time.**  
> If you are interested in the system, collaboration, or early access, please **email the authors**.

---

## Paper

OpGuard will appear at **OSDI ’26**:

> Ziming Zhou, Yinjie Zhao, Hang Zhu, Wenxiao Wang, Zhihao Bai, Yun Zhang, Shuguang Wang, Haibin Lin, Peng Huang.  
> *OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training.*  
> In *Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’26)*, Seattle, WA, USA, July 2026.

### BibTeX

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

---

## What this site covers

- **Bitwise alignment** — the core abstraction for comparing training runs
- **Live trace demo** — exploring divergence evidence in a Perfetto-style viewer
- **Workflow** — how OpGuard fits into production debugging
- **Impact** — production case studies and timing results

---

## Contact

Source code and internal tooling are **not publicly released**.

For questions about the paper, demos, or access inquiries, please **email the authors** listed above.
