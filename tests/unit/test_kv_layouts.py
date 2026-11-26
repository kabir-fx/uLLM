"""``scripts.kv_layouts`` times KV layouts that do the same work: from one prefilled KV, every
candidate decodes the same greedy tokens at one row, and the split attention of candidate C
equals one fused attention over the whole context (E9, M3-T2)."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from scripts import kv_layouts

    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

pytestmark = pytest.mark.mlx


def test_each_candidate_and_cell_is_timed_and_reported(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    from bench import conditions
    from scripts import kv_layouts

    runner, tokenizer = mlx_loaded
    tokens = kv_layouts.long_prompt(tokenizer, 64 + 1)
    source = kv_layouts.prefill(runner.model, tokens)
    cells, joins = kv_layouts.measure(
        runner.model,
        tokens,
        source,
        rows=(1, 2, 3),
        contexts=(40, 64),
        warmup=1,
        steps=17,
        rounds=2,
    )
    assert {(c.rows, c.context, c.candidate) for c in cells} == {
        (r, t, k) for r in (1, 2, 3) for t in (40, 64) for k in kv_layouts.CANDIDATES
    }
    assert all(len(c.mean_ms) == 2 and min(c.mean_ms) > 0 for c in cells)
    assert all(c.same_tokens for c in cells if c.rows == 1)
    assert {(j.rows, j.context, j.candidate) for j in joins} == {
        (r, t, k) for r in (2, 3) for t in (40, 64) for k in ("e7", "b")
    }
    attention = kv_layouts.measure_attention(
        prefixes=(64,), rows=2, tail=16, heads=4, kv_heads=2, dim=8, layers=2, rounds=1
    )
    report = kv_layouts.report(
        kv_layouts.KVLayouts(
            env=conditions.environment(used_powermetrics=False, started=datetime.now(UTC)),
            model="m",
            warmup=1,
            steps=17,
            rounds=2,
            cells=cells,
            joins=joins,
            attention=attention,
        )
    )
    assert "| 3 | 64 |" in report
    assert "## Candidate C" in report


def test_candidates_that_disagree_at_one_row_are_refused(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import kv_layouts

    runner, tokenizer = mlx_loaded
    tokens = kv_layouts.long_prompt(tokenizer, 33)
    source = kv_layouts.prefill(runner.model, tokens)
    real = kv_layouts.decode
    calls: list[None] = []

    def second_is_off_by_one(
        *args: object, **kwargs: object
    ) -> tuple[list[list[int]], list[float]]:
        out, ms = real(*args, **kwargs)  # type: ignore[arg-type]
        calls.append(None)
        shift = 1 if len(calls) == 2 else 0
        return [[t + shift for t in row] for row in out], ms

    monkeypatch.setattr(kv_layouts, "decode", second_is_off_by_one)
    with pytest.raises(RuntimeError, match="different tokens"):
        kv_layouts.measure(
            runner.model, tokens, source, rows=(1,), contexts=(32,), warmup=1, steps=3, rounds=1
        )


@pytest.mark.parametrize("kind", ["a", "a2", "b"])
def test_a_candidate_fetches_the_padded_batch_that_e7_fetches(
    kind: "kv_layouts.Candidate",
) -> None:
    import mlx.core as mx
    from scripts import kv_layouts

    heads, dim, longest = 2, 4, 37
    mx.random.seed(0)
    source = [
        (mx.random.normal((1, heads, longest, dim)), mx.random.normal((1, heads, longest, dim)))
        for _ in range(2)
    ]
    lengths = [37, 20, 16, 1]
    e7 = kv_layouts.build("e7", source, lengths)
    other = kv_layouts.build(kind, source, lengths)
    for step in range(20):
        new_k = mx.random.normal((len(lengths), heads, 1, dim))
        new_v = mx.random.normal((len(lengths), heads, 1, dim))
        for want_layer, got_layer in zip(e7, other, strict=True):
            assert mx.array_equal(want_layer.offset, got_layer.offset)
            want_mask, got_mask = want_layer.make_mask(1), got_layer.make_mask(1)
            assert mx.array_equal(want_mask, got_mask), step
            want = want_layer.update_and_fetch(new_k, new_v)
            got = got_layer.update_and_fetch(new_k, new_v)
            assert mx.array_equal(want[0], got[0]), step
            assert mx.array_equal(want[1], got[1]), step


def test_split_attention_equals_fused_attention_over_the_whole_context() -> None:
    import mlx.core as mx
    from scripts import kv_layouts

    mx.random.seed(1)
    rows, heads, kv_heads, dim, prefix, tail = 3, 4, 2, 8, 50, 7
    q = mx.random.normal((rows, heads, 1, dim)).astype(mx.bfloat16)
    k_prefix, v_prefix = (
        mx.random.normal((1, kv_heads, prefix, dim)).astype(mx.bfloat16) for _ in range(2)
    )
    k_tail, v_tail = (
        mx.random.normal((rows, kv_heads, tail, dim)).astype(mx.bfloat16) for _ in range(2)
    )
    scale = dim**-0.5
    whole = mx.fast.scaled_dot_product_attention(
        q,
        mx.concatenate([mx.broadcast_to(k_prefix, (rows, kv_heads, prefix, dim)), k_tail], 2),
        mx.concatenate([mx.broadcast_to(v_prefix, (rows, kv_heads, prefix, dim)), v_tail], 2),
        scale=scale,
    )
    split = kv_layouts.split_attention(q, k_prefix, v_prefix, k_tail, v_tail, scale)
    assert split.shape == whole.shape
    assert split.dtype == whole.dtype
    assert mx.allclose(split.astype(mx.float32), whole.astype(mx.float32), atol=2e-2).item()


def test_each_round_starts_with_the_next_candidate(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first candidate of a cell runs on a GPU clock the previous cell left; rotating the
    # order spreads that over every candidate.
    from scripts import kv_layouts

    runner, tokenizer = mlx_loaded
    tokens = kv_layouts.long_prompt(tokenizer, 33)
    source = kv_layouts.prefill(runner.model, tokens)
    real_build, order = kv_layouts.build, []

    def recorded(kind: "kv_layouts.Candidate", *args: object) -> object:
        order.append(kind)
        return real_build(kind, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(kv_layouts, "build", recorded)
    kv_layouts.measure(
        runner.model, tokens, source, rows=(1,), contexts=(32,), warmup=1, steps=2, rounds=4
    )
    n = len(kv_layouts.CANDIDATES)
    starts = [order[i * n] for i in range(4)]
    assert starts == list(kv_layouts.CANDIDATES)
    assert all(
        sorted(order[i * n : (i + 1) * n]) == sorted(kv_layouts.CANDIDATES) for i in range(4)
    )
