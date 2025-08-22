# Contributing to Real Simple Stats

Thank you for your interest in contributing! This guide focuses on actionable steps to get you productive quickly: local setup, quality checks, docs, CI/CD, and publishing.

## 1) Getting Started

- Clone and install (development mode):
```bash
pip install --upgrade pip
pip install -e ".[dev]"
```
- Project requires Python 3.8+.
- Main package code lives in `real_simple_stats/`.
- Tests live in `tests/`.

## 2) Quality: Lint, Type-Check, Test

- Format check (Black):
```bash
black --check --diff real_simple_stats/ tests/
```
- Lint (Flake8):
```bash
flake8 real_simple_stats/ tests/
```
- Type-check (MyPy):
```bash
mypy real_simple_stats/
```
- Run tests (PyTest):
```bash
pytest -v
# with coverage
pytest --cov=real_simple_stats --cov-report=term --cov-report=xml
```

These same checks run in CI via `.github/workflows/ci.yml`.

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
