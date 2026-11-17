---
name: run-ullm
description: Run, start, launch, smoke-test or drive uLLM — the `ullm serve` OpenAI-compatible server (fake or real MLX model) and the `ullm sim` engine simulation. Use to see a server or engine change working in the real process, not just in tests.
---

# Run uLLM

uLLM has two runnable surfaces today: `ullm serve`, an OpenAI-compatible HTTP server on 127.0.0.1, and `ullm sim`, the real engine on `FakeRunner` with simulated clients. Drive the server with `smoke.sh`, which launches it, exercises every endpoint with curl, asserts on the replies and stops it. Paths are relative to the repo root.

The router (`make up`, M5-T4) and the terminal UI (M1-T8) have nothing to run yet.

## Prerequisites

Apple Silicon macOS for the real model. `uv`, `curl`. The fake server and `ullm sim` need no model and no MLX.

The real model is downloaded once, at its pinned revision:

```bash
make models
```

## Run the server (agent path)

```bash
.claude/skills/run-ullm/smoke.sh                 # FakeRunner, no model; about 3 s
.claude/skills/run-ullm/smoke.sh --real          # Qwen3-0.6B-bf16 on MLX; about 4 s with the model cached
.claude/skills/run-ullm/smoke.sh --port 8767     # default port is 8765
```

It checks, in order: `/health`, `/v1/models`, a plain completion (200, `finish_reason`), a streamed one (`[DONE]`, usage chunk), an unknown model (404), a `text/plain` body (400), a foreign `Host` header (400), and that `/health` shows `in_flight: 0` after a client drops mid-stream. Last line `smoke: ok`, exit 0. On failure it prints `FAIL: <check>` and the tail of the server log, which is at `$TMPDIR/ullm-smoke-<port>.log`.

To keep the server up and send your own requests:

```bash
.claude/skills/run-ullm/smoke.sh --real --port 8767 --keep
curl -s http://127.0.0.1:8767/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mlx-community/Qwen3-0.6B-bf16","messages":[{"role":"user","content":"Name three primary colours."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
pkill -f "ullm serve --port 8767"
```

The model id is the repo name for the real model and `fake` for `--fake`; `/v1/models` gives it.

## Run the engine simulation

```bash
make sim
uv run --locked ullm sim --clients 20 --seed 1 --trace "$TMPDIR/sim.jsonl.gz"
```

Prints `N clients, S steps, E events`, a count per outcome (`finished (length)`, `left early`), and `trace check: ok`; exit 1 on a trace problem.

## Run (human path)

`uv run --locked ullm serve [--fake]` serves on port 8000 in the foreground until Ctrl-C.

## Gotchas

- Qwen3 thinks by default: replies start with `<think>` and short `max_tokens` runs end inside the thinking. Send `"chat_template_kwargs":{"enable_thinking":false}` for a direct answer.
- The server answers only `Host: 127.0.0.1` or `localhost` and only `content-type: application/json`. A request through a proxy or a tool that sends another Host gets 400.
- `model` must equal the served id exactly; anything else is 404 `model_not_found`.
- Fake output is `" t<id>"` per token, not text.
- `smoke.sh` refuses (exit 2) if the port already answers `/health`; a leftover `--keep` server is the usual cause.
- Under `--keep` the printed pid is `uv`'s; stop the server with `pkill -f "ullm serve --port <port>"`.

## Tests

`make check` (no model) and `make test-mlx` (model) are the suites; see `CLAUDE.md`.
