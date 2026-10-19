# Publishing OpGuard

The `publish-pypi.yml` workflow publishes the root `opguard` package through PyPI
Trusted Publishing (OIDC); no stored PyPI token is needed. The independently
versioned `harness/` package is not published by this workflow.

## One-time setup

1. Verify your PyPI email and enable 2FA, saving your recovery codes privately.
2. In PyPI account **Publishing**, add a pending GitHub publisher for a new project:
   - PyPI project name: `opguard`
   - Owner: `OrderLab`
   - Repository: `OpGuard`
   - Workflow filename: `publish-pypi.yml`
   - Environment: `pypi`
3. Commit and push the workflow and the intended release contents to
   `OrderLab/OpGuard`. The workflow must exist in the release tag's commit.
4. Configure the GitHub environment `pypi` with deployment tags restricted to
   `v*`. Optional required reviewers add a manual approval before upload.

If `opguard` already exists, its owner must add this publisher in the project's
Publishing settings instead. A pending publisher does not reserve a project name.

## Each release

1. Update `version` in the root `pyproject.toml`, update the changelog, and commit
   all intended release changes. Complete native GPU validation separately.
2. Create a matching tag, e.g. `v0.1.0`, on that commit and publish a GitHub Release
   for it. Publishing the release starts the workflow.
3. The workflow checks the version, builds the sdist and a temporary CPU wheel,
   checks metadata and runs installed-package smoke tests. Only the sdist is
   uploaded to PyPI. Native wheels remain separate release attachments.
4. Confirm the completed Actions run and the version on PyPI.

For a failed upload, rerun the failed job after resolving the cause. The manual
workflow can also run against a matching version tag; branch runs are rejected.
Already published files cannot be replaced: code changes need a new version.
CPU smoke tests do not validate CUDA kernels or native wheels.

See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
