# 0002: Python tooling

**Status:** accepted · **Task:** M0-T3

## Decision

| Concern | Choice | Why |
|---|---|---|
| Environment and lock | uv, `uv.lock` committed, `uv run --locked` everywhere | One tool for Python install, venv and lock; `--locked` fails when `pyproject.toml` and the lock disagree, so CI and every machine run the same versions |
| Python | 3.12 only (`>=3.12,<3.13`, `.python-version`) | The versions in roadmap section 4 were verified on 3.12 |
| Build backend | `uv_build` | Ships with uv; a `src/` layout needs nothing more |
| MLX | `mlx==0.32.4` and `mlx-lm==0.32.2`, required only on `darwin`/`arm64` through environment markers | MLX runs only on Apple Silicon. `ullm.core`, `FakeRunner` and the contracts must install and test on Linux CI with no MLX at all |
| Pins | Exact (`==`) for direct dependencies in `pyproject.toml`; the lock pins the rest | Rule 13. A version change is a deliberate, reviewed edit, and goldens record the versions that made them (F4) |
| Lint and format | ruff | Fast, one tool for both |
| Types | mypy, `strict` on `src` and `tests` from the start | The roadmap asks for strict on `core` first. With almost no code yet, strict everywhere costs nothing now and saves a ratchet later. mypy over pyright: it installs from PyPI with no Node runtime in the Python toolchain, and it has a pydantic plugin for the contracts |
| Tests | pytest with `--import-mode=importlib` and strict markers; hypothesis with `dev` and `ci` profiles | Tests need no `__init__.py` files. The `mlx` marker separates model-backed tests from `make check` |

## Enforcement added

- **Rule 3** (`ullm.core` never imports `mlx`): `tests/unit/test_core_is_mlx_free.py` parses every module in `core` for direct imports, and imports all of `core` in a fresh interpreter to catch MLX loaded through another module. Both checks were confirmed to fail on a deliberately planted import.
- **Rule 11** (build the machinery, do not import it): ruff's `banned-api` (TID251) rejects `mlx_lm.generate` (the module, which holds `generate_step`, `stream_generate`, `generate`, `BatchGenerator` and `batch_generate`), the top-level re-exports `mlx_lm.stream_generate` and `mlx_lm.batch_generate`, `mlx_lm.models.cache.BatchKVCache`, `mlx_lm.models.cache.LRUPromptCache`, and `mlx_lm.server`. Paths were checked against the installed mlx-lm 0.32.2 source. `tests/` and `scripts/` are exempt because mlx-lm is the reference they compare against.

## Consequences

- Linux installs have no MLX. Anything that imports it must live in `ullm.runner` or the server's MLX path, never in `core`.
- Upgrading MLX or mlx-lm means editing two pins, re-verifying roadmap section 4, and regenerating the goldens.
