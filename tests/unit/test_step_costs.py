"""``scripts.step_costs`` times loops that do the same work: at one row, mlx-lm's pipelined
loop, the same call waiting on each step, and ``MLXRunner`` give the same greedy tokens
(decision 0028)."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

pytestmark = pytest.mark.mlx


def test_each_loop_and_row_count_is_timed_and_reported(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    from bench import conditions
    from scripts import step_costs

    runner, tokenizer = mlx_loaded
    prompt = step_costs._median_prompt(tokenizer)
    single, batches = step_costs.measure(runner.model, runner, prompt, (1, 2), 6, 2)
    assert len(single.pipelined_ms) == len(single.synchronous_ms) == 2
    assert [b.rows for b in batches] == [1, 2]
    assert all(min(b.forward_ms) > 0 and min(b.total_ms) >= min(b.forward_ms) for b in batches)
    costs = step_costs.StepCosts(
        env=conditions.environment(used_powermetrics=False, started=datetime.now(UTC)),
        model="m",
        prompt_tokens=len(prompt),
        decode_steps=6,
        rounds=2,
        single=single,
        batches=batches,
    )
    text = step_costs.report(costs)
    assert f"one prompt of {len(prompt)} tokens" in text
    assert "| `generate_step`, queues the next step first |" in text
    assert "| The same call, waits on each step |" in text
    assert [line.split(" |")[0] for line in text.splitlines() if line[:3] in ("| 1", "| 2")] == [
        "| 1",
        "| 2",
    ]


def test_loops_that_disagree_on_a_token_are_refused(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import step_costs

    runner, tokenizer = mlx_loaded
    real = step_costs.synchronous
    monkeypatch.setattr(
        step_costs, "synchronous", lambda *a: ([t + 1 for t in real(*a)[0]], real(*a)[1])
    )
    prompt = step_costs._median_prompt(tokenizer)
    with pytest.raises(RuntimeError, match="different tokens"):
        step_costs.measure(runner.model, runner, prompt, (1,), 4, 1)


def test_a_repack_is_timed_with_its_copy_evaluated(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", monkeypatch: pytest.MonkeyPatch
) -> None:
    # MLX is lazy: a repack costs nothing until something evaluates the packed KV. A packed
    # layer that carries a costly lazy term shows whether the timing waits for it.
    import mlx.core as mx
    from scripts import step_costs

    from ullm.runner.kv import batch as kv

    real = kv._packed

    def costly(
        rows: list[tuple[kv.LayerKV, int, int, int]], length: int
    ) -> tuple[mx.array | None, mx.array | None]:
        keys, values = real(rows, length)
        assert keys is not None
        x = mx.ones((1024, 1024))
        for _ in range(4):
            x = x @ x * 1e-3
        return keys + 0 * x.sum().astype(keys.dtype), values

    runner, tokenizer = mlx_loaded
    prompt = step_costs._median_prompt(tokenizer)
    _, plain = step_costs.batch(runner, prompt, 2, 6)
    monkeypatch.setattr(kv, "_packed", costly)
    _, slowed = step_costs.batch(runner, prompt, 2, 6)
    layers = len(runner.model.layers)
    assert slowed["join_ms"] - plain["join_ms"] > layers * 0.5
    assert slowed["leave_ms"] - plain["leave_ms"] > layers * 0.5
