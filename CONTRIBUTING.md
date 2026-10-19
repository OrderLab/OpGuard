# Contributing

Report reproducible bugs and propose changes through this repository's GitHub issues
and pull requests. Include Python, Torch and CUDA versions, `opguard doctor --json`
output, and a small synthetic reproducer. Remove credentials, private paths, model
weights and private workload source before sharing traces or reports.

## Development

Use Python 3.10 or newer and install the appropriate Torch build first:

```bash
python -m pip install -e '.[test]' build
python -m pip install -e ./harness
python -m pytest -q tests/test_tools.py tests/test_local_sync.py tests/test_distributed_identity.py tests/test_output_dir.py tests/test_prepare_workload.py
python -m unittest discover -s harness/tests -v
```

The self-contained evidence replay needs Python 3.11 or newer:

```bash
python examples/independent-training-alignment/replay.py --harness-root ./harness
```

See [installation and release validation](docs/Installation.md) for building wheels,
installed-package smoke tests and native GPU validation. A CPU run does not validate
native CUDA extensions. Keep generated traces and local environment files out of git.

The `index.html`, `assets/`, `css/`, and `js/` directories serve the existing project
website. Preserve their URLs when changing the source layout.

Keep third-party attribution and license notices with copied or adapted code.

## Website

The website and live demo are maintained on [`webpage`](https://github.com/OrderLab/OpGuard/tree/webpage). GitHub Pages publishes that branch’s root. Keep website changes on that branch; `main` contains the Python packages, harness, and documentation.
