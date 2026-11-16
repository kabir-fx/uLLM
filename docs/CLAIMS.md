# Claims ledger

Every claim uLLM makes, in the README, the write-up or the author's resume, has a row here (roadmap P2). Each row names the evidence and how anyone can reproduce it. **A claim without a row is not made.** Performance numbers appear only once a committed benchmark run produces them; until then there are none.

`make verify` (M2-T12) will reproduce the rows marked *verify* on any Apple Silicon Mac.

| # | Claim | Evidence | Reproduce | Since |
|---|---|---|---|---|
| 1 | The engine core (`ullm.core`) never imports MLX, directly or through another module | `tests/unit/test_core_is_mlx_free.py`; the CI `core` job runs every model-free test on Linux after asserting MLX is not installed | `make check` | M0 |
| 2 | mlx-lm's generation loop, batching and caches cannot be used in package code | ruff `banned-api` rules in `pyproject.toml` (decision 0002) | `uv run ruff check` | M0 |
| 3 | The API contract matches the official OpenAI Python SDK: requests, replies, streams, mid-stream errors and error statuses | `tests/unit/test_api_schemas_sdk.py` (the SDK in process against our models) | `uv run pytest tests/unit/test_api_schemas_sdk.py` | M0 |
| 4 | The terminal's TypeScript types are generated from the Python telemetry models, and CI fails if they drift | `tests/unit/test_telemetry_contract.py::test_committed_schema_is_fresh`; `pnpm check:contracts` in the CI `terminal` job | `make check` | M0 |
| 5 | Telemetry carries no prompt or output text, so recorded runs are safe to publish | `tests/unit/test_telemetry_contract.py::test_no_free_text_fields` pins every string field in the protocol | `make check` | M0 |
| 6 | Synthetic fixtures are labelled as synthetic and are consistent: ordered, gap-free, and every request follows the life cycle, with token counts matching the steps | `tests/unit/test_synthetic_traces.py`; `ullm.telemetry.check` and its tests (decision 0004) | `make check` | M0 |
| 7 | MLX runs on GitHub's hosted macOS runner, on a paravirtualized GPU | The `mlx-probe` artifact on every CI run; decision 0005 | `uv run python scripts/mlx_probe.py` on any runner | M0 |
| 8 | mlx-lm's greedy output reproduces the committed goldens exactly on the machine that made them; on the CI runner's GPU it diverges only at near-tied positions | `tests/equivalence/test_goldens.py`; CI run 43705619599; decision 0006 | `make models && make test-mlx`; full length: `make goldens-check` (*verify*) | M0 |
| 9 | For one request, uLLM's own runner and KV cache give exactly mlx-lm's greedy token IDs, run on the same machine, for every golden prompt | `tests/equivalence/test_runner_equivalence.py`; decision 0008 | `make models && make test-mlx`; full length: `ULLM_GOLDEN_TOKENS=128 uv run pytest -m mlx tests/equivalence/test_runner_equivalence.py` (*verify*) | M1 |

## How to add a row

- One claim per row, worded exactly as it appears wherever it is used.
- Evidence is something a stranger can open: a test name, a `bench/results/` run directory, a CI run, a script.
- Reproduce is a command. If it needs a particular machine or condition, say so in the row.
- When evidence changes (a rerun, a new commit), update the row in the same pull request.
