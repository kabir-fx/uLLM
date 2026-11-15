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

check-ts: ## Terminal: generated contract types are fresh and typecheck
	pnpm -C terminal install --frozen-lockfile --silent
	pnpm -C terminal check:contracts
	pnpm -C terminal typecheck

check-go: ## Go vet and tests for the router module
	cd router && go vet ./... && go test ./...

.PHONY: contracts
contracts: ## Regenerate the telemetry JSON Schema and TypeScript types (C2)
	$(UV) python scripts/export_contracts.py
	pnpm -C terminal gen:contracts

.PHONY: test
test: ## Every test that runs without the model
	$(UV) pytest -m "not mlx"

# --- Model-backed and system targets -----------------------------------------

.PHONY: equivalence bench-smoke sim up verify
equivalence: ## Token-for-token equivalence against mlx-lm (needs the model)
	$(call todo,M1-T5)

bench-smoke: ## Short benchmark run, for before/after numbers in a pull request
	$(call todo,M1-T6)

sim: ## Run the real engine on FakeRunner (no model)
	$(call todo,M1-T3)

up: ## Router plus replicas
	$(call todo,M5-T4)

verify: ## Reproduce the claims in docs/CLAIMS.md on any Apple Silicon Mac
	$(call todo,M2-T12)
