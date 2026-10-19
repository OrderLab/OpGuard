# Publishing OpGuard

The two packages are versioned and published independently through PyPI Trusted
Publishing (OIDC); no stored PyPI token is needed:

| Package | Version metadata | Release tag | Workflow | PyPI artifacts |
| --- | --- | --- | --- | --- |
| `opguard` | `pyproject.toml` | `v<version>` | `publish-pypi.yml` | Source distribution only |
| `opguard-harness` | `harness/pyproject.toml` | `harness-v<version>` | `publish-harness-pypi.yml` | Pure-Python wheel and source distribution |

Each workflow ignores the other package's release tags. Publishing a harness
release does not rebuild or republish OpGuard.

## One-time setup

1. Verify your PyPI email and enable 2FA, saving your recovery codes privately.
2. In PyPI account **Publishing**, add a pending GitHub publisher for each new
   project, using the package name and workflow filename in the table above:
   - Owner: `OrderLab`
   - Repository: `OpGuard`
   - Environment: `pypi`
3. Commit and push the workflow and the intended release contents to
   `OrderLab/OpGuard`. The workflow must exist in the release tag's commit.
4. Configure the GitHub environment `pypi` with deployment tags restricted to
   `v*` and `harness-v*`. Optional required reviewers add a manual approval before upload.

If a project already exists, its owner must add its publisher in that project's
Publishing settings instead. The existing `opguard` publisher does not authorize
uploads to `opguard-harness`; configure the harness publisher separately. A
pending publisher does not reserve a project name.

## OpGuard releases

1. Update `version` in the root `pyproject.toml`, update the changelog, and commit
   all intended release changes. Complete native GPU validation separately.
2. Create a matching tag, e.g. `v0.1.0`, on that commit and publish a GitHub Release
   for it. Publishing the release starts the workflow.
3. The workflow checks the version, builds the sdist and a temporary CPU wheel,
   checks metadata and runs installed-package smoke tests. Only the sdist is
   uploaded to PyPI. Native wheels remain separate release attachments.
4. Confirm the completed Actions run and the version on PyPI.

## Harness releases

1. Update `version` in `harness/pyproject.toml` and `__version__` in
   `harness/opguard_harness/__init__.py`, update the changelog, and commit all
   intended release changes. The initial harness release is `0.1.0`.
2. Create a matching tag, e.g. `harness-v0.1.0`, on that commit and publish a
   GitHub Release for it. Include the new workflow in that commit. Publishing
   the release starts `publish-harness-pypi.yml`.
3. The workflow checks the tag against harness metadata, builds the sdist and
   then the universal wheel from that sdist, and checks both with Twine. It
   installs the wheel into an isolated uv tool environment and runs CLI smoke
   checks plus the harness unit suite outside the checkout, without Torch or
   OpGuard installed. Both distributions are uploaded only after those checks pass.
4. Confirm the completed Actions run and the version on
   [PyPI](https://pypi.org/project/opguard-harness/). Verify a fresh installation:

   ```bash
   uv tool install opguard-harness
   opguard-harness tools
   # Alternative in a fresh virtual environment:
   python -m pip install opguard-harness
   ```

The harness's pure-Python wheel is its normal installation artifact. The
restriction on publishing CPU-only wheels applies only to `opguard`.

## Failed uploads

For a failed upload, rerun the failed job after resolving the cause. The manual
workflows can also run against their package's matching version tag; branch runs
and mismatched tags are rejected.
Already published files cannot be replaced: code changes need a new version.
CPU smoke tests do not validate CUDA kernels or native wheels.

See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
