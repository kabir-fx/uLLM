# uLLM task runner. `make` lists the targets.
# Targets that a later task implements fail loudly and name that task.

SHELL := /bin/bash
.DEFAULT_GOAL := help

define todo
	@echo "make $@: not implemented yet (arrives in $(1), see docs/AGENT_ROADMAP.md section 7)" >&2; exit 2
endef

.PHONY: help
help: ## List targets
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

# --- Fast checks: no model, no GPU, runs anywhere ---------------------------

.PHONY: check check-py check-ts check-go
check: check-py check-ts check-go ## Lint, types, unit and property tests (no model needed)
	@echo "make check: all passed"

UV := uv run --locked

check-py: ## Ruff lint and format check, mypy, unit and property tests
	$(UV) ruff check
	$(UV) ruff format --check
	$(UV) mypy
	$(UV) pytest -m "not mlx" tests/unit tests/property

check-ts: ## Terminal: generated contract types are fresh, typecheck, unit tests
	pnpm -C terminal install --frozen-lockfile --silent
	pnpm -C terminal check:contracts
	pnpm -C terminal typecheck
	pnpm -C terminal test

check-go: ## Go vet and tests for the router module
	cd router && go vet ./... && go test ./...

# --- Full check: one target per CI job; ci.yml calls these -------------------

.PHONY: ci ci-core ci-router ci-terminal ci-mlx
ci: ci-core ci-router ci-terminal ci-mlx ## Full check: every CI job's commands (needs MLX and the model)
	@echo "make ci: all passed"

ci-core: ## CI core job: lint, format, types, every test that needs no model
	$(UV) ruff check
	$(UV) ruff format --check
	$(UV) mypy
	HYPOTHESIS_PROFILE=ci $(UV) pytest -m "not mlx"

ci-router: ## CI router job: gofmt, go vet, tests with the race detector
	test -z "$$(gofmt -l router)" || { gofmt -l router; exit 1; }
	cd router && go vet ./... && go test -race ./...

ci-terminal: check-ts ## CI terminal job: same as check-ts

ci-mlx: equivalence ## CI mlx job: equivalence, then the other tests marked mlx; a missing model fails instead of skipping
	ULLM_REQUIRE_MODEL=1 $(UV) pytest -m mlx --ignore=tests/equivalence

.PHONY: contracts
contracts: ## Regenerate the telemetry JSON Schema and TypeScript types (C2)
	$(UV) python scripts/export_contracts.py
	pnpm -C terminal gen:contracts

.PHONY: fixtures
fixtures: ## Regenerate the synthetic telemetry traces in terminal/fixtures/
	$(UV) python scripts/make_fixtures.py

.PHONY: test
test: ## Every test that runs without the model
	$(UV) pytest -m "not mlx"

# --- Model-backed and system targets -----------------------------------------

.PHONY: models goldens goldens-check
models: ## Download the default model at its pinned revision (others: scripts/fetch_models.py --list)
	$(UV) python scripts/fetch_models.py

goldens: ## Regenerate the golden token IDs from mlx-lm (after a model, prompt or library change)
	$(UV) python scripts/make_goldens.py

goldens-check: ## Regenerate the goldens in memory and compare them in full
	$(UV) python scripts/make_goldens.py --check

.PHONY: bench-baselines
bench-baselines: models ## M1-T7: the c1 and c8 sessions of decision 0016, on an idle Mac on AC power
	$(UV) python -m bench.session bench/groups/m1-c1.json --settle 240 --out bench/results/m1/c1
	$(UV) python -m bench.session bench/groups/m1-c8.json --settle 600 --out bench/results/m1/c8
	$(UV) python -m bench.analysis bench/results/m1/c1 bench/results/m1/c8 --compare \
		> bench/results/m1/comparison.md

.PHONY: test-mlx equivalence bench-smoke sim up verify
test-mlx: ## Tests marked mlx: need MLX (Apple Silicon); the model from M0-T7 on
	$(UV) pytest -m mlx

equivalence: models ## Token-for-token equivalence against mlx-lm (needs the model); a missing model fails
	ULLM_REQUIRE_MODEL=1 $(UV) pytest -m mlx tests/equivalence

bench-smoke: models ## Short benchmark run, for before/after numbers in a pull request (not a B5 run)
	$(UV) python -m bench.run --target ullm --launch "$(UV) ullm serve" \
		--requests 16 --warmup 2 --repeats 1 --cooldown 0

sim: ## Run the real engine on FakeRunner with fake clients, then check the trace (no model)
	$(UV) ullm sim

up: ## Router plus replicas
	$(call todo,M5-T4)

verify: ## Reproduce the claims in docs/CLAIMS.md on any Apple Silicon Mac
	$(call todo,M2-T12)
