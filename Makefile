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

check-ts: ## Terminal: generated contract types are fresh, typecheck, unit tests, local and hosted builds, browser smoke test
	pnpm -C terminal install --frozen-lockfile --silent
	pnpm -C terminal check:contracts
	pnpm -C terminal typecheck
	pnpm -C terminal test
	pnpm -C terminal build
	pnpm -C terminal build:hosted
	pnpm -C terminal exec playwright install --only-shell chromium
	pnpm -C terminal e2e

TERMINAL_PERF := bench/results/m2/terminal-perf

.PHONY: terminal-perf
terminal-perf: ## M2-T10: the terminal's T7 budgets in Chromium (decision 0027), on the author's Mac on AC power
	pnpm -C terminal install --frozen-lockfile --silent
	pnpm -C terminal build
	pnpm -C terminal build:hosted
	pnpm -C terminal exec playwright install --only-shell chromium
	mkdir -p $(TERMINAL_PERF)
	$(UV) python -m bench.conditions $(TERMINAL_PERF)/env.json
	PERF_OUT=$(abspath $(TERMINAL_PERF)) pnpm -C terminal perf

.PHONY: terminal-hosted
terminal-hosted: ## The hosted replay build in terminal/dist-hosted/ (decision 0026); pages.yml publishes it
	pnpm -C terminal install --frozen-lockfile --silent
	pnpm -C terminal build:hosted

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

ci-terminal: ## CI terminal job: Chromium's system libraries, then check-ts
	pnpm -C terminal install --frozen-lockfile --silent
	pnpm -C terminal exec playwright install --with-deps --only-shell chromium
	$(MAKE) check-ts

ci-mlx: equivalence ## CI mlx job: equivalence, then the other tests marked mlx; a missing model fails instead of skipping
	ULLM_REQUIRE_MODEL=1 $(UV) pytest -m mlx --ignore=tests/equivalence

.PHONY: contracts
contracts: ## Regenerate the telemetry (C2) and control JSON Schemas and their TypeScript types
	$(UV) python -m scripts.export_contracts
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

.PHONY: bench-telemetry
bench-telemetry: models ## M2-T4: telemetry on against off at c8 (decision 0023), on an idle Mac on AC power
	$(UV) python -m bench.session bench/groups/m2-telemetry.json --settle 600 --out bench/results/m2/telemetry

SWEEP := bench/results/m2/sweep
SWEEP_SESSIONS := $(SWEEP)/c1 $(SWEEP)/c2 $(SWEEP)/c4 $(SWEEP)/c8 $(SWEEP)/c16

.PHONY: bench-sweep
bench-sweep: models ## M2-T6: the c1 to c16 sessions of decision 0028, on an idle Mac on AC power
	$(UV) python -m bench.session bench/groups/m2-c1.json --settle 240 --out $(SWEEP)/c1
	$(UV) python -m bench.session bench/groups/m2-c2.json --settle 600 --out $(SWEEP)/c2
	$(UV) python -m bench.session bench/groups/m2-c4.json --settle 600 --out $(SWEEP)/c4
	$(UV) python -m bench.session bench/groups/m2-c8.json --settle 600 --out $(SWEEP)/c8
	# A 16 GB Mac may swap at c16 (claims row 33): the verdict is recorded, the sweep goes on.
	-$(UV) python -m bench.session bench/groups/m2-c16.json --settle 600 --out $(SWEEP)/c16
	# Expected to fail: it records whether cached mlx-lm fits at c16 on an idle Mac.
	-$(UV) python -m bench.session bench/groups/m2-c16-cache.json --settle 0 --blocks 1 \
		--out $(SWEEP)/c16-cache
	$(UV) python -m scripts.step_costs --out $(SWEEP)
	$(UV) python -m bench.steps $(SWEEP_SESSIONS)
	$(UV) python -m bench.analysis $(SWEEP_SESSIONS) --sweep > $(SWEEP)/comparison.md
	$(MAKE) readme

.PHONY: readme
readme: ## Regenerate the README's benchmark table from the committed sweep (decision 0030)
	$(UV) python -m bench.readme

COMPOSE := docker compose -f deploy/docker-compose.yml

.PHONY: grafana grafana-down
grafana: ## M2-T11: Prometheus and Grafana in Docker for `ullm serve` on :8000; open http://127.0.0.1:3000/
	$(COMPOSE) up -d

grafana-down: ## Stop the Prometheus and Grafana containers
	$(COMPOSE) down

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
