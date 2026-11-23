"""Offline step costs on the real model, in one process with no server (decision 0026).

At one row it times mlx-lm's ``generate_step``, which queues step ``n + 1`` before it waits on
step ``n``, against the same model call that waits on every step. At each row count it times
uLLM's ``MLXRunner`` decode step, split into the forward pass and the rest, and the repack when
a sequence joins or leaves the batch.

    uv run python -m scripts.step_costs --out bench/results/m2/sweep
"""

import argparse
import itertools
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from bench import conditions
from bench.schema import Frozen, RunEnv, load_workload
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from ullm.models import DEFAULT_MODEL
from ullm.runner.base import DecodePlan, PrefillPlan, SamplingParams
from ullm.runner.mlx_runner import MLXRunner, load_mlx

ROWS = (1, 2, 4, 8, 16)
DECODE_STEPS = 128
ROUNDS = 5
WORKLOAD = Path("bench/workloads/a.json")
JSON = "step-costs.json"
MARKDOWN = "step-costs.md"
GREEDY = SamplingParams(temperature=0, top_p=1, top_k=0, seed=0)


class Single(Frozen):
    """Per-token ms after the first, one median per round."""

    pipelined_ms: tuple[float, ...]
    synchronous_ms: tuple[float, ...]


class Batch(Frozen):
    """One row count, one median per round. ``total_ms`` is what a ``step`` event carries."""

    rows: int
    forward_ms: tuple[float, ...]
    total_ms: tuple[float, ...]
    join_ms: tuple[float, ...]  # the first decode's total minus its forward: the repack
    leave_ms: tuple[float, ...]  # ``release`` of one row of the full batch


class StepCosts(Frozen):
    env: RunEnv
    model: str
    prompt_tokens: int
    decode_steps: int
    rounds: int
    single: Single
    batches: tuple[Batch, ...]


def _gaps_ms(tokens: Iterator[Any], n: int) -> tuple[list[int], float]:
    """The tokens and the median ms between successive ones after the first."""
    out, stamps = [], []
    for token in tokens:
        stamps.append(time.perf_counter())
        out.append(int(token))
        if len(out) == n:
            break
    gaps = [(b - a) * 1000 for a, b in itertools.pairwise(stamps)]
    return out, statistics.median(gaps)


def pipelined(model: Any, prompt: list[int], n: int) -> tuple[list[int], float]:
    steps = generate_step(
        mx.array(prompt), model, max_tokens=n, prompt_cache=make_prompt_cache(model)
    )
    return _gaps_ms((token for token, _ in steps), n)


def synchronous(model: Any, prompt: list[int], n: int) -> tuple[list[int], float]:
    """``generate_step``'s model call and argmax, but each step waits for its token."""
    cache = make_prompt_cache(model)

    def step(x: mx.array) -> mx.array:
        logits = model(x[None], cache=cache)[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        return mx.argmax(logprobs, axis=-1)

    def tokens() -> Iterator[int]:
        model(mx.array(prompt[:-1])[None], cache=cache)
        mx.eval([c.state for c in cache])
        y = step(mx.array(prompt[-1:]))
        while True:
            mx.eval(y)
            token = cast(int, y.item())
            yield token
            y = step(mx.array([token]))

    return _gaps_ms(tokens(), n)


def batch(
    runner: MLXRunner, prompt: list[int], rows: int, n: int
) -> tuple[list[int], dict[str, float]]:
    """``rows`` copies of ``prompt``, prefilled one by one, then ``n`` decode steps of all.
    Returns row 0's tokens and the medians."""
    seqs = tuple(range(rows))
    last = []
    for seq in seqs:
        runner.add(seq, GREEDY)
        last.append(runner.execute(PrefillPlan(seq, tuple(prompt), 0)).tokens[0])
    first = [last[0]]
    forward, total = [], []
    join = 0.0
    for i in range(n - 1):
        result = runner.execute(DecodePlan(seqs, tuple(last)))
        last = list(result.tokens)
        first.append(last[0])
        if i == 0:
            join = result.total_ms - result.forward_ms
        else:
            forward.append(result.forward_ms)
            total.append(result.total_ms)
    began = time.perf_counter()
    runner.release(seqs[-1])
    leave = (time.perf_counter() - began) * 1000
    for seq in seqs[:-1]:
        runner.release(seq)
    medians = {
        "forward_ms": statistics.median(forward),
        "total_ms": statistics.median(total),
        "join_ms": join,
        "leave_ms": leave,
    }
    return first, medians


def measure(
    model: Any, runner: MLXRunner, prompt: list[int], rows: Sequence[int], n: int, rounds: int
) -> tuple[Single, tuple[Batch, ...]]:
    """Every variant once per round, in the same order, so drift in the GPU clock reaches
    each alike. Refuses a round where the one-row loops disagree on a token: they would not
    be doing the same work. Above one row, tokens may differ at near-ties (decision 0021)."""
    single: dict[str, list[float]] = {"pipelined_ms": [], "synchronous_ms": []}
    per_rows: dict[int, dict[str, list[float]]] = {r: {} for r in rows}
    variants: dict[str, Callable[[], tuple[list[int], float]]] = {
        "pipelined_ms": lambda: pipelined(model, prompt, n),
        "synchronous_ms": lambda: synchronous(model, prompt, n),
    }
    for _ in range(rounds):
        seen = []
        for name, run in variants.items():
            tokens, ms = run()
            single[name].append(ms)
            seen.append(tokens)
        for r in rows:
            tokens, medians = batch(runner, prompt, r, n)
            for key, value in medians.items():
                per_rows[r].setdefault(key, []).append(value)
            if r == 1:
                seen.append(tokens)
        if any(tokens != seen[0] for tokens in seen):
            raise RuntimeError("the one-row loops generated different tokens for one prompt")
    return (
        Single(**{k: tuple(v) for k, v in single.items()}),
        tuple(Batch(rows=r, **{k: tuple(v) for k, v in per_rows[r].items()}) for r in rows),
    )


def _cell(values: Sequence[float]) -> str:
    return f"{statistics.median(values):.2f} [{min(values):.2f}, {max(values):.2f}]"


def report(costs: StepCosts) -> str:
    e = costs.env
    lines = [
        "# Step costs, offline",
        "",
        f"- `{costs.model}`, one prompt of {costs.prompt_tokens} tokens from workload A, "
        f"{costs.decode_steps} greedy tokens, {costs.rounds} rounds. Each cell is the median "
        "over rounds of each round's median, [lowest, highest] round.",
        f"- Commit `{e.commit[:12]}`{' (dirty tree)' if e.dirty else ''}. {e.chip}, "
        f"{e.memory_gb:g} GB, macOS {e.macos}. AC {e.power.on_ac}, Low Power Mode "
        f"{e.power.low_power_mode}. Not a B5 session: it explains a gap, it is not one.",
        "",
        "## One row: ms per token after the first",
        "",
        "| Loop | ms |",
        "|---|---|",
        f"| `generate_step`, queues the next step first | {_cell(costs.single.pipelined_ms)} |",
        f"| The same call, waits on each step | {_cell(costs.single.synchronous_ms)} |",
        "",
        "## `MLXRunner` decode step, ms",
        "",
        "| Rows | Forward | Step total | Join repack | Leave repack |",
        "|---|---|---|---|---|",
    ]
    for b in costs.batches:
        lines.append(
            f"| {b.rows} | {_cell(b.forward_ms)} | {_cell(b.total_ms)} | {_cell(b.join_ms)} "
            f"| {_cell(b.leave_ms)} |"
        )
    return "\n".join(lines) + "\n"


def _median_prompt(tokenizer: Any) -> list[int]:
    workload = load_workload(WORKLOAD)
    ordered = sorted(workload.requests, key=lambda r: (r.prompt_tokens, r.id))
    request = ordered[len(ordered) // 2]
    messages = [m.model_dump() for m in request.messages]
    return list(tokenizer.render(messages, workload.chat_template_kwargs))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.step_costs")
    parser.add_argument("--out", type=Path, required=True, help="directory for the two files")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    args = parser.parse_args(argv)
    started = datetime.now(UTC)
    runner, tokenizer = load_mlx(DEFAULT_MODEL)
    prompt = _median_prompt(tokenizer)
    single, batches = measure(runner.model, runner, prompt, ROWS, args.decode_steps, args.rounds)
    costs = StepCosts(
        env=conditions.environment(used_powermetrics=False, started=started),
        model=DEFAULT_MODEL.key,
        prompt_tokens=len(prompt),
        decode_steps=args.decode_steps,
        rounds=args.rounds,
        single=single,
        batches=batches,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON).write_text(costs.model_dump_json(indent=2) + "\n")
    (args.out / MARKDOWN).write_text(report(costs))
    print(f"wrote {args.out / MARKDOWN}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
