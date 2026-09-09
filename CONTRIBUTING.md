# Contributing to Real Simple Stats

Thank you for your interest in contributing! This guide focuses on actionable steps to get you productive quickly: local setup, quality checks, docs, CI/CD, and publishing.

## 1) Getting Started

Since 0.5.0 the numerics live in a Rust extension, so a Rust toolchain is
required to build from source. Install one from [rustup.rs](https://rustup.rs)
if you do not have it.

```bash
pip install --upgrade pip
pip install -e ".[dev]"          # compiles the extension via maturin
```

That first build takes a couple of minutes. After editing any Rust source you
must rebuild before Python sees the change:

```bash
maturin develop --release        # rebuild and reinstall in place
```

Use `--release` rather than a debug build. Measured on this project, a debug
build runs the kernels roughly 3x to 14x slower depending on the operation
(14x for `std_dev` over 200,000 values, 3x for a 2,000-iteration bootstrap),
which is enough to make the parity suite tedious.

- Requires Python 3.12+ and Rust 1.78+.
- Python package code lives in `real_simple_stats/`.
- Rust code lives in `crates/`:
  - `rss-core` — the numerics. Pure Rust, no Python linkage, so it can be
    tested on its own with `cargo test -p rss-core`.
  - `rss-py` — the PyO3 bindings. Only built, never `cargo test`ed, because
    linking the extension needs a Python host.
- Tests live in `tests/`; `tests/parity/` holds the numerical accuracy gates.

### Where to put a change

New numeric work belongs in `rss-core`, exposed through `rss-py`, and wrapped
in Python only for argument handling, validation, and the teaching layer
(docstrings, `explain.py`, the CLI). Keep the explanatory prose in Python — it
is the point of the library, and Rust would only make it harder to edit.

## 2) Quality: Lint, Type-Check, Test

Python:
```bash
ruff format --check .            # formatting
ruff check .                     # lint
mypy real_simple_stats/          # types
pytest -q                        # tests
pytest --cov=real_simple_stats --cov-report=term
```

Rust:
```bash
cargo test -p rss-core --release
cargo fmt --all -- --check
cargo clippy -p rss-core --release -- -D warnings
```

These same checks run in CI via `.github/workflows/ci.yml`.

### Numerical accuracy gates

`tests/parity/` is the safety net for the Rust backend, and it is worth
understanding before changing any kernel.

- `test_special_parity.py` checks the special functions against **mpmath at 60
  digits**, not against SciPy. This matters: SciPy's own `erfc` and `ndtr`
  carry around 1e-13 relative error in the tails, so validating against SciPy
  would both cap our accuracy at SciPy's and flag genuine improvements as
  regressions.
- `test_dist_parity.py` and `test_linalg_parity.py` use SciPy and NumPy, which
  are reliable oracles for those (well-conditioned, evaluated in the bulk).

SciPy, NumPy and mpmath are therefore **development dependencies only** — they
are test oracles, never runtime dependencies. If you add one to
`[project] dependencies` the CI job that installs the wheel into a bare
environment will fail, which is exactly what it is there for.

Points where a limit is genuine rather than a defect are documented in the
tests themselves — for instance, inverting the incomplete beta for
Beta(0.5, 0.5) as p approaches 1 returns the last representable double below 1,
and no round trip can recover p from there. SciPy returns bit-identical values.

## 3) Documentation

- Sphinx docs source: `docs/source/`
- Build locally:
```bash
pip install -r docs/requirements.txt
(cd docs && make clean && make html)
```
- Built HTML outputs to `docs/build/html/`.

### Read the Docs (RTD)
RTD builds are configured via `.readthedocs.yaml`.

Trigger RTD automatically on pushes to `main` using the GitHub Actions job in `.github/workflows/docs.yml`:
1. In RTD: Project → Admin → Integrations → Add a Generic webhook (or view existing)
2. Copy your RTD Webhook URL and Token
3. In your GitHub repo: Settings → Secrets and variables → Actions → New repository secret
   - `RTD_WEBHOOK_URL` = the URL from RTD
   - `RTD_WEBHOOK_TOKEN` = the token from RTD

Alternatively, connect GitHub to RTD directly (RTD GitHub integration/app) so RTD auto-builds on push without the webhook step.

### GitHub Pages (optional)
`docs.yml` can deploy built HTML to GitHub Pages for previews or public hosting.
To enable:
1. Repo Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` (workflow will create/update)

## 4) CI/CD Overview

Workflows live in `.github/workflows/`:
- `ci.yml`: tests, lint, type-check, security, docs build, package build
- `docs.yml`: docs build, PR previews, optional RTD trigger
- `publish.yml`: quality checks, build artifacts, and publish to PyPI/TestPyPI

Badges in `README.md` reflect status of these workflows.

## 5) Publishing to PyPI / TestPyPI

The `publish.yml` workflow supports both manual runs and GitHub Releases.

### Create API tokens
- PyPI: https://pypi.org/manage/account/ → API tokens → Add token (copy value starting with `pypi-`)
- TestPyPI: https://test.pypi.org/manage/account/ → API tokens → Add token

### Add GitHub repository secrets
Repo → Settings → Secrets and variables → Actions → New repository secret:
```
PYPI_API_TOKEN = pypi-********************************
TEST_PYPI_API_TOKEN = pypi-********************************
```

### (Recommended) GitHub Environments
Repo → Settings → Environments:
- Environment `pypi`: add environment secret `PYPI_API_TOKEN`, set protection rules
- Environment `testpypi`: add environment secret `TEST_PYPI_API_TOKEN`

### Release process
1. Update version in `pyproject.toml`
2. Update CHANGELOG (if applicable)
3. Commit and push
4. Create a GitHub Release (tag like `v0.2.1`)
   - This triggers `publish.yml` to build and publish

### Manual publish (optional)
Actions → Publish to PyPI → Run workflow → choose environment (`testpypi` or `pypi`).

## 6) Pull Requests

- Create feature branch from `main`
- Keep commits focused and well-described
- Ensure CI is green (lint, type, tests)
- Add/Update docs for user-facing changes
- Request review when ready

## 7) Troubleshooting

- PyPI publish fails: verify tokens, increment version, check logs
- Tests fail: ensure dependencies and Python version match matrix
- Docs build fails: confirm `docs/requirements.txt`, Sphinx config, and imports

## 8) Helpful Files

- `pyproject.toml`: tooling config (Black, PyTest, MyPy), metadata
- `.readthedocs.yaml`: RTD build config
- `.github/workflows/*.yml`: CI/CD pipelines
- `README.md`: project overview and badges

---

Thank you for helping make Real Simple Stats better!
