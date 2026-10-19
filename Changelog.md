# Changelog

## Harness 0.1.0 — 2026-10-19

- Publish `opguard-harness` independently to PyPI as a pure-Python wheel and source distribution, using `harness-v<version>` release tags and Trusted Publishing.
- Validate harness package metadata and run installed-package tests before publishing.
- Make pip and uv package installation the default harness setup; reserve source checkouts for development.

## 0.1.0 — 2026-10-19

- Discover and instrument training operations, capture tensor fingerprints, and compare runs to locate their first divergence.
- Inspect aligned operations and supporting evidence in Perfetto traces.
- Use the agent harness and diagnostic skills from Codex, Claude Code, or other coding agents.
- Check runtime and optional native-extension availability with `opguard doctor`.
- Publish `opguard` as a PyPI source distribution with Linux CUDA extensions enabled by default. Explicit CPU-only builds require both native build flags to be disabled.
- Publish GitHub Releases through PyPI Trusted Publishing after package metadata and installed CPU smoke checks pass.
- Avoid waiting for the Python GIL in CUPTI callbacks; preserve nested host-thread dispatch stacks and correct graph callback phases.
- Validate installed wheels with a CPU discovery, capture, and comparison workflow.

At the initial OpGuard 0.1.0 release, `opguard-harness` was available as a separate source installation. Native wheels are distributed separately; the PyPI release does not include prebuilt CUDA binaries.
