# Contributing to petrify

Thank you for taking the time to contribute. petrify is an open-source project maintained by [starside.io](https://starside.io). All contributions — bug fixes, new features, documentation improvements, and test coverage — are welcome.

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Getting started](#getting-started)
- [Development setup](#development-setup)
- [Running the tests](#running-the-tests)
- [Code style](#code-style)
- [Opening a pull request](#opening-a-pull-request)
- [Reporting a bug](#reporting-a-bug)
- [Requesting a feature](#requesting-a-feature)
- [License](#license)

---

## Code of conduct

Be kind. Criticism is welcome; personal attacks are not. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## Getting started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-handle>/petrify.git
   cd petrify
   ```

3. **Create a branch** for your change:

   ```bash
   git checkout -b fix/my-bug-description
   # or
   git checkout -b feat/my-feature-name
   ```

---

## Development setup

### System requirements

- Python 3.10 or later
- `ffmpeg` on your `PATH` (see [README installation](README.md#installation))
- `pipx` or `pip` for Python package management

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

This installs `pytest`, `pytest-cov`, `ruff`, `mypy`, and `httpx`.

The web page tests (`tests/test_server.py`) also need FastAPI and uvicorn, from the `web` extra; without them they are skipped:

```bash
pip install -e ".[dev,web]"
```

### Verify the install

```bash
petrify --version
```

---

## Running the tests

```bash
# Run the full test suite
pytest

# With coverage report
pytest --cov=petrify --cov-report=term-missing

# Run a single test file
pytest tests/test_motion.py -v

# Fast tests only (skips the golden parity tests)
pytest -m "not golden"

# Golden parity tests only
pytest -m golden
```

Tests live in the `tests/` directory, one file per module (for example `tests/test_engine.py` covers `petrify/core/engine.py`). If you add a new feature, please add tests.

Tests must never open a real camera, network stream, window, or virtual camera. The live sources and the preview and virtual camera sinks take injected fakes for this (for example `capture_factory`, `camera_factory`, `gui`); use them.

### Golden tests

The golden tests compare the old two-pass pipeline (`petrify.run`), the streaming engine, and the CLI against the fixtures committed in `tests/golden/fixtures/`. They need the demo clips `data/camera.mp4` and `data/camera2.mp4`, which are not in the repository. Without them (or when a clip's checksum does not match the fixture) the golden tests are skipped. A missing fixture is a failure, not a skip.

To record fixtures from the old pipeline, run `python -m tests.golden.regenerate` from the repository root (all cases, or one with `--case <id>`). Existing fixtures are only replaced with `--overwrite`.

---

## Code style

petrify uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy.readthedocs.io) for optional type checking.

```bash
# Lint
ruff check petrify/ tests/

# Auto-fix
ruff check --fix petrify/ tests/

# Format
ruff format petrify/ tests/

# Type check
mypy petrify/
```

CI will reject PRs that fail the ruff lint pass. Run it locally before pushing.

### Style conventions

- **Docstrings**: Use NumPy-style docstrings for all public functions and classes.
- **Type hints**: Required for all new public functions. `from __future__ import annotations` at the top of every file.
- **Line length**: 100 characters (enforced by ruff).
- **No print statements**: Use `logging.getLogger(__name__)` instead.

---

## Opening a pull request

1. Make sure `pytest` and `ruff check` pass locally.
2. Update `CHANGELOG.md` — add a line under `## [Unreleased]` describing your change.
3. Push your branch and open a PR against `main`.
4. Fill in the PR template. Link any related issue with `Closes #<n>`.
5. A maintainer will review within a few days. Please respond to review comments promptly.

### Commit message format

```
<type>: <short summary in present tense>

<optional body>
```

Types: `fix`, `feat`, `docs`, `refactor`, `test`, `chore`.

Examples:

```
fix: lower default bg-diff-area-threshold to 200 for small frames
feat: add --roi-snap-corners flag to disable corner snapping
docs: add Docker example to README
```

---

## Reporting a bug

Open a [GitHub Issue](https://github.com/starside-io/petrify/issues/new?template=bug_report.md) and fill in the bug report template. Please include:

- petrify version (`petrify --version`)
- Python version (`python --version`)
- OS and architecture
- The exact command you ran
- Expected vs. actual behaviour
- A minimal video clip if possible (even 5 seconds is enough)

---

## Requesting a feature

Open a [GitHub Issue](https://github.com/starside-io/petrify/issues/new?template=feature_request.md) and fill in the feature request template. Describe:

- The use case you have in mind
- What the CLI interface might look like
- Any relevant prior art or related tools

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
