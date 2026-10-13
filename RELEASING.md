# Release setup

The repositories contain the automation, but GitHub, Read the Docs and PyPI need separate account configuration. Ordinary compilation, tests and documentation builds do not need release secrets.

## GitHub Actions

Actions is enabled for `Long-Toolkit/long-toolkit` and `Long-Toolkit/CLI`. Push `main` to start their checks. The Python workflow runs tests on Python 3.10, 3.11 and 3.12 on Ubuntu and macOS; the Go workflow compiles and checks both operating systems.

Local validation currently passes all 14 Python tests on Python 3.10 on Apple Silicon, builds the documentation and Python distributions, and builds the five CLI release archives. Python 3.11 installation fails while compiling `scikit-lexicographical-trees==0.0.5` from its source distribution. Python 3.12 has not been validated. Resolve the upstream build or publish compatible upstream wheels before treating the full Python matrix as release-ready. Python 3.10 remains the validated environment for local use and documentation builds.

For Python 3.10 on macOS, the project constrains SciPy to below 1.15 because the tested SciPy 1.15.3 wheel fails to load its PROPACK extension on this Mac. macOS also needs `brew install libomp`, now included in CI.

## Automatic Python version bumps

The `Bump version` workflow needs a GitHub App installed on the **new** `Long-Toolkit/long-toolkit` repository, with repository **Contents: read and write** permission.

In [the repository's Actions settings](https://github.com/Long-Toolkit/long-toolkit/settings/secrets/actions), configure:

| Setting | Type | Value |
| --- | --- | --- |
| `RELEASE_APP_ID` | Actions variable | The GitHub App ID |
| `RELEASE_APP_PRIVATE_KEY` | Actions secret | The App's PEM private key |

Organisation-level settings can also be used if they grant this repository access. Repository-level settings were empty during the audit; organisation-level settings could not be inspected with the available credentials. If branch rules restrict pushes to `main`, grant the release App the access required by those rules.

Run the workflow from `main` after CI succeeds. It updates the package version, citation and lockfile together, commits them, and pushes the new release tag. The App token allows that tag push to trigger the PyPI workflow; pushes made with the built-in `GITHUB_TOKEN` do not trigger ordinary push workflows.

## PyPI

The publish workflow uses **Trusted Publishing**, so it does not require a `PYPI_TOKEN` secret.

1. Create a GitHub environment named `pypi` in [repository environments](https://github.com/Long-Toolkit/long-toolkit/settings/environments).
2. Log in to PyPI and [add a pending publisher](https://pypi.org/manage/account/publishing/) for the new project, or add a publisher in the project's Publishing settings if the project already exists.
3. Enter these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `long-toolkit` |
| GitHub owner | `Long-Toolkit` |
| GitHub repository | `long-toolkit` |
| Workflow filename | `publish-pypi.yml` |
| GitHub environment | `pypi` |

The public PyPI project endpoint returned 404 during the audit. A pending publisher can create the project on the first successful upload. Keep the version in `pyproject.toml`, `.bumpversion.toml`, `CITATION.cff` and `uv.lock` consistent with the new tag. Use a fresh tag rather than pushing the old tags in bulk.

## Read the Docs

1. Install the Read the Docs GitHub App for `Long-Toolkit/long-toolkit`.
2. Add the repository in the Read the Docs dashboard, with project slug **`long-toolkit`**, the name used by the documentation links.
3. Set `main` as the default branch and use the repository's `.readthedocs.yaml` configuration file.
4. Enable the `latest` version and run the first build. Check the GitHub integration so later pushes trigger rebuilds.

The public Read the Docs project endpoint returned 404 during the audit. If an existing project uses the previous name, update its repository integration or create the new project and configure redirects. The YAML uses Python 3.10 and native `uv sync` installation, which preserves this project's dependency overrides, and exports the Zensical site to Read the Docs' HTML output directory.

## Homebrew, Scoop and GitHub CLI releases

No Homebrew account or submission to Homebrew Core is required for the organisation's own tap. Scoop uses the organisation's own bucket.

1. Push `main` for `homebrew-tap` and `scoop-bucket` before the first CLI release.
2. Create a fine-grained GitHub token owned by an account with write access to those two repositories. Select resource owner **Long-Toolkit**, both repositories, and repository permission **Contents: read and write**. Obtain organisation approval if its token policy requires it.
3. Save it as **`TAP_GITHUB_TOKEN`** in [the CLI repository's Actions secrets](https://github.com/Long-Toolkit/CLI/settings/secrets/actions), or grant CLI access to an organisation secret of that name.
4. After CLI CI succeeds, create and push a fresh CLI release tag, for example `v0.0.9`, updating `CLI/CITATION.cff` to match. The workflow creates the GitHub release and regenerates the tap formula and Scoop manifest with the real archive checksums.

GitHub supplies `GITHUB_TOKEN` automatically for the CLI release itself. Its repository scope cannot update the other two repositories, which is why `TAP_GITHUB_TOKEN` is separate. No secrets are required inside the tap or bucket repositories.

The checked-in formula and manifest still reference the historical `v0.0.2` release, whose renamed archives have not been published in the new organisation. Do not advertise those installers as available until a fresh CLI release has regenerated and published them.

GoReleaser is pinned to the locally validated `v2.18.3`. Its legacy Homebrew formula integration emits a deprecation warning but successfully generated the macOS and Linux formula during the snapshot build. Migrating to Homebrew casks requires a separate review of the supported platforms and installation instructions.

## Installing the Python package in a separate project

Scikit-Longitudinal uses `scikit-lexicographical-trees`, which provides the `sklearn` import namespace. Stock `scikit-learn` must not overwrite it. The repository already contains the upstream-recommended uv override, but dependency overrides do not propagate into a consumer's project.

Before installing with `uv add long-toolkit`, put this in the consuming project's `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = ["scikit-learn ; sys_platform == 'never'"]
```

Use Python 3.10 until the newer-Python dependency build issue is resolved. See [Scikit-Longitudinal's installation guide](https://scikit-longitudinal.readthedocs.io/latest/getting-started/) for the reason for the override.
