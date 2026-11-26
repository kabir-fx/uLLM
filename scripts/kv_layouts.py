"""The E9 spike: decode step time and memory of each candidate KV layout on the real model.

Every candidate is a per-layer cache on mlx-lm's plug point (``offset``, ``update_and_fetch``,
``make_mask``, roadmap section 4), built from one prefilled KV, so each decodes the same rows
from the same state:

- ``e7``: ``BatchKV``, the contiguous batch KV, unchanged. The reference.
- ``a``: one array per 16-token block per layer for K and for V; every step writes the new
  token into its row's last block and gathers the left-padded batch by concatenation.
- ``a2``: one preallocated pool per layer for K and for V, a slot per token; every step
  scatters the new tokens into the pool and gathers the batch with ``mx.take``.
- ``b``: E7's working tensor for attention, plus an A block store of record that gets a copy
  of each block as it fills. A row joins by a gather from its blocks.

Candidate C (shared-prefix segments, attention merged across them) is timed as attention
alone, against the fused kernel over the padded batch that E7 holds.

    uv run python -m scripts.kv_layouts --out bench/results/m3/kv-layout
"""

import argparse
import gc
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
from bench import conditions
from bench.schema import Frozen, RunEnv, load_workload
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import make_prompt_cache

from ullm.models import DEFAULT_MODEL
from ullm.runner.kv.batch import BatchKV, LayerKV
from ullm.runner.mlx_runner import load_mlx

Candidate = Literal["e7", "a", "a2", "b"]
CANDIDATES: tuple[Candidate, ...] = ("e7", "a", "a2", "b")
Joiner = Literal["e7", "b"]
JOINERS: tuple[Joiner, ...] = ("e7", "b")
BLOCK = 16
POOL_BLOCKS = 2304  # roadmap section 4: a 4 GB pool of Qwen3-0.6B blocks is about 2,300
ROWS = (1, 2, 4, 8, 16)
CONTEXTS = (512, 1024, 2048)
MAX_SLOTS = 16 * 1024  # skips 16 rows at 2,048: A's transient gather would be about 7 GB
WARMUP = 5
STEPS = 48  # three whole block cycles, so B's block copies weigh in its mean as they recur
ROUNDS = 5
PREFIXES = (1024, 2048, 4096)  # workload B shares a 2k to 4k-token system prompt
ATTENTION_ROWS = 8
TAIL = 256
WORKLOAD = Path("bench/workloads/a.json")
JSON = "kv-layout.json"
MARKDOWN = "kv-layout.md"
MIB = 2**20

LayerKVs = list[tuple[mx.array, mx.array]]  # per layer, K and V of shape [1, H, L, D]


class Cell(Frozen):
    """One candidate at one row count and context. Times are one value per round."""

    rows: int
    context: int  # the longest row's tokens; row i holds ``context - i``
    candidate: Candidate
    mean_ms: tuple[float, ...]
    median_ms: tuple[float, ...]
    held_mib: float  # active memory the candidate's KV holds once built
    peak_mib: float  # the highest active memory above the base while decoding, any round
    same_tokens: bool  # the same greedy tokens as E7 in every round


class Join(Frozen):
    """One row joins a batch of ``rows - 1``; for E7 also one leaves a batch of ``rows``."""

    rows: int
    context: int
    candidate: Joiner
    join_ms: tuple[float, ...]
    leave_ms: tuple[float, ...] = ()  # B leaves as E7 does: its blocks stay, the batch repacks


class Attention(Frozen):
    """Attention alone over every layer of one decode step, one median per round."""

    prefix: int
    rows: int
    tail: int
    fused_ms: tuple[float, ...]  # mx.fast.scaled_dot_product_attention over the padded batch
    manual_ms: tuple[float, ...]  # the same context, one segment, through matmul and logsumexp
    split_ms: tuple[float, ...]  # C: the prefix read once for every row, then each row's tail
    max_abs_diff: float  # split against fused, first layer


class KVLayouts(Frozen):
    env: RunEnv
    model: str
    warmup: int
    steps: int
    rounds: int
    cells: tuple[Cell, ...]
    joins: tuple[Join, ...]
    attention: tuple[Attention, ...]


class BlockLayer:
    """Candidate A: each row's K and V live in 16-token block arrays, gathered every step."""

    def __init__(self, source: tuple[mx.array, mx.array], lengths: Sequence[int]) -> None:
        keys, values = source
        self.tables_k = [_blocks(keys[0, :, :n, :]) for n in lengths]
        self.tables_v = [_blocks(values[0, :, :n, :]) for n in lengths]
        self.lengths = list(lengths)
        longest = max(lengths)
        self.pads = mx.array([longest - n for n in lengths], mx.int32)
        self._padded = longest != min(lengths)
        self.offset = mx.array(self.lengths, mx.int32)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if keys.shape[2] != 1:
            raise ValueError("the spike decodes one token per row")
        for i, n in enumerate(self.lengths):
            block, slot = divmod(n, BLOCK)
            if slot == 0:
                self.tables_k[i].append(mx.zeros((keys.shape[1], BLOCK, keys.shape[3]), keys.dtype))
                self.tables_v[i].append(
                    mx.zeros((values.shape[1], BLOCK, values.shape[3]), values.dtype)
                )
            self.tables_k[i][block][:, slot : slot + 1, :] = keys[i]
            self.tables_v[i][block][:, slot : slot + 1, :] = values[i]
            self.lengths[i] = n + 1
        self.offset = mx.array(self.lengths, mx.int32)
        return _gather(self.tables_k, self.lengths), _gather(self.tables_v, self.lengths)

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None) -> Any:
        return _mask(N, max(self.lengths), self.pads, self._padded)

    def arrays(self) -> list[mx.array]:
        return [b for table in self.tables_k + self.tables_v for b in table]


class PoolLayer:
    """Candidate A2: one slot per token in a preallocated pool, scattered into and gathered
    from every step. Block 0 is never written, so its slots are the padding."""

    def __init__(self, source: tuple[mx.array, mx.array], lengths: Sequence[int]) -> None:
        keys, values = source
        _, heads, _, dim = keys.shape
        self.pool_k = mx.zeros((POOL_BLOCKS * BLOCK, heads, dim), keys.dtype)
        self.pool_v = mx.zeros((POOL_BLOCKS * BLOCK, heads, values.shape[3]), values.dtype)
        self.tables: list[list[int]] = []
        self.next_block = 1
        self.lengths = list(lengths)
        longest = max(lengths)
        rows = []
        for i, n in enumerate(lengths):
            self.tables.append([])
            slots = [self._slot(i, j) for j in range(n)]
            ids = mx.array(slots, mx.int32)
            self.pool_k[ids] = keys[0, :, :n, :].transpose(1, 0, 2)
            self.pool_v[ids] = values[0, :, :n, :].transpose(1, 0, 2)
            rows.append([0] * (longest - n) + slots)
        self.index = mx.array(rows, mx.int32)  # [B, longest]: each row's slots, left-padded
        self.pads = mx.array([longest - n for n in lengths], mx.int32)
        self._padded = longest != min(lengths)
        self.offset = mx.array(self.lengths, mx.int32)

    def _slot(self, row: int, token: int) -> int:
        block, slot = divmod(token, BLOCK)
        if block == len(self.tables[row]):
            if self.next_block == POOL_BLOCKS:
                raise ValueError("the pool is full")
            self.tables[row].append(self.next_block)
            self.next_block += 1
        return self.tables[row][block] * BLOCK + slot

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if keys.shape[2] != 1:
            raise ValueError("the spike decodes one token per row")
        new = mx.array([self._slot(i, n) for i, n in enumerate(self.lengths)], mx.int32)
        self.lengths = [n + 1 for n in self.lengths]
        self.pool_k[new] = keys[:, :, 0, :]
        self.pool_v[new] = values[:, :, 0, :]
        self.index = mx.concatenate([self.index, new[:, None]], axis=1)
        self.offset = mx.array(self.lengths, mx.int32)
        return (
            mx.take(self.pool_k, self.index, axis=0).transpose(0, 2, 1, 3),
            mx.take(self.pool_v, self.index, axis=0).transpose(0, 2, 1, 3),
        )

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None) -> Any:
        return _mask(N, max(self.lengths), self.pads, self._padded)

    def arrays(self) -> list[mx.array]:
        return [self.pool_k, self.pool_v, self.index]


class WriteThroughLayer:
    """Candidate B: E7's ``LayerKV`` does the attention; a block store of record gets a copy
    of each block when it fills. The partial last block lives only in the working tensor."""

    def __init__(
        self, inner: LayerKV, source: tuple[mx.array, mx.array], lengths: Sequence[int]
    ) -> None:
        keys, values = source
        self.inner = inner
        self.tables_k = [_blocks(keys[0, :, : n - n % BLOCK, :]) for n in lengths]
        self.tables_v = [_blocks(values[0, :, : n - n % BLOCK, :]) for n in lengths]
        self.lengths = list(lengths)
        self.pending: list[mx.array] = []  # block copies nothing reads yet, for the step's eval

    @property
    def offset(self) -> mx.array:
        return self.inner.offset

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        fetched = self.inner.update_and_fetch(keys, values)
        end = self.inner.length
        for i, n in enumerate(self.lengths):
            self.lengths[i] = n + 1
            if (n + 1) % BLOCK == 0:
                for table, held in (
                    (self.tables_k, self.inner.keys),
                    (self.tables_v, self.inner.values),
                ):
                    assert held is not None
                    block = mx.contiguous(held[i, :, end - BLOCK : end, :])
                    table[i].append(block)
                    self.pending.append(block)
        return fetched

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None) -> Any:
        return self.inner.make_mask(N, return_array, window_size)

    def arrays(self) -> list[mx.array]:
        held = [b for table in self.tables_k + self.tables_v for b in table]
        return held + _arrays(self.inner)


def _blocks(row: mx.array) -> list[mx.array]:
    """``[H, n, D]`` as 16-token blocks, each its own array; the last is padded with zeros."""
    heads, n, dim = row.shape
    blocks = []
    for start in range(0, n, BLOCK):
        block = row[:, start : start + BLOCK, :]
        if block.shape[1] < BLOCK:
            fill = mx.zeros((heads, BLOCK - block.shape[1], dim), row.dtype)
            block = mx.concatenate([block, fill], axis=1)
        blocks.append(mx.contiguous(block))
    return blocks


def _gather(tables: list[list[mx.array]], lengths: Sequence[int]) -> mx.array:
    """The left-padded ``[B, H, longest, D]`` batch from each row's blocks."""
    longest = max(lengths)
    heads, _, dim = tables[0][0].shape
    rows = []
    for table, n in zip(tables, lengths, strict=True):
        parts = list(table)
        if longest > n:
            parts.insert(0, mx.zeros((heads, longest - n, dim), table[0].dtype))
        rows.append(mx.concatenate(parts, axis=1)[:, :longest, :])
    return mx.stack(rows)


def _mask(N: int, length: int, pads: mx.array, padded: bool) -> Any:
    """``LayerKV.make_mask``'s rule, so every candidate attends as E7 does."""
    if padded:
        return create_causal_mask(N, offset=length, left_padding=pads)
    return None if N == 1 else "causal"


def _arrays(layer: Any) -> list[mx.array]:
    if isinstance(layer, LayerKV):
        return [a for a in (layer.keys, layer.values) if a is not None]
    held: list[mx.array] = layer.arrays()
    return held


def _e7(source: LayerKVs, lengths: Sequence[int]) -> BatchKV:
    rows = []
    for seq, n in enumerate(lengths):
        row = BatchKV.empty(seq, len(source))
        for layer, (keys, values) in zip(row.layers, source, strict=True):
            layer.update_and_fetch(keys[..., :n, :], values[..., :n, :])
        rows.append((row, seq))
    return BatchKV.pack(rows)


def build(kind: Candidate, source: LayerKVs, lengths: Sequence[int]) -> list[Any]:
    """One cache per layer for rows of ``lengths`` tokens, each the prefix of ``source``."""
    if kind == "e7":
        return list(_e7(source, lengths).layers)
    if kind == "a":
        return [BlockLayer(kv, lengths) for kv in source]
    if kind == "a2":
        return [PoolLayer(kv, lengths) for kv in source]
    batch = _e7(source, lengths)
    return [
        WriteThroughLayer(inner, kv, lengths)
        for inner, kv in zip(batch.layers, source, strict=True)
    ]


def _flush(caches: list[Any]) -> list[mx.array]:
    pending = []
    for cache in caches:
        if isinstance(cache, WriteThroughLayer):
            pending += cache.pending
            cache.pending = []
    return pending


def decode(
    model: Any, caches: list[Any], first: Sequence[int], warmup: int, steps: int
) -> tuple[list[list[int]], list[float]]:
    """Greedy decode of every row; each row's tokens, and the ms of each step after warmup."""
    x = mx.array(list(first), mx.int32)[:, None]
    out: list[list[int]] = [[] for _ in first]
    ms = []
    for step in range(warmup + steps):
        start = time.perf_counter()
        y = mx.argmax(model(x, cache=caches)[:, -1, :], axis=-1)
        mx.eval(y, *_flush(caches))
        took = (time.perf_counter() - start) * 1000
        if step >= warmup:
            ms.append(took)
        for row, token in zip(out, cast(list[int], y.tolist()), strict=True):
            row.append(token)
        x = y[:, None]
    return out, ms


def _timed_eval(arrays: list[mx.array], start: float) -> float:
    mx.eval(*arrays)
    return (time.perf_counter() - start) * 1000


def _join_ms(kind: Joiner, source: LayerKVs, lengths: Sequence[int]) -> float:
    """One row of ``lengths[-1]`` tokens joins a batch of the others. E7's row comes from its
    prefill batch, B's from a gather of its blocks."""
    *held, n = lengths
    batch = _e7(source, held)
    joiner = len(held)
    if kind == "e7":
        prefilled = _e7(source, [n])
        row = BatchKV((joiner,), prefilled.pads, prefilled.layers)
        mx.eval(*[a for layer in row.layers for a in _arrays(layer)])
        start = time.perf_counter()
        packed = BatchKV.pack([*((batch, s) for s in batch.seqs), (row, joiner)])
    else:
        tables = [(_blocks(k[0, :, :n, :]), _blocks(v[0, :, :n, :])) for k, v in source]
        mx.eval(*[b for ks, vs in tables for b in ks + vs], *_arrays(batch.layers[0]))
        start = time.perf_counter()
        layers = []
        for ks, vs in tables:
            keys = mx.concatenate(ks, axis=1)[None, :, :n, :]
            values = mx.concatenate(vs, axis=1)[None, :, :n, :]
            layers.append(LayerKV(keys, values, n, mx.zeros((1,), mx.int32), False))
        row = BatchKV((joiner,), (0,), layers)
        packed = BatchKV.pack([*((batch, s) for s in batch.seqs), (row, joiner)])
    return _timed_eval([a for layer in packed.layers for a in _arrays(layer)], start)


def _leave_ms(source: LayerKVs, lengths: Sequence[int]) -> float:
    batch = _e7(source, lengths)
    mx.eval(*[a for layer in batch.layers for a in _arrays(layer)])
    start = time.perf_counter()
    packed = BatchKV.pack([(batch, s) for s in batch.seqs[:-1]])
    return _timed_eval([a for layer in packed.layers for a in _arrays(layer)], start)


def _settle() -> int:
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory()


def measure(
    model: Any,
    tokens: Sequence[int],
    source: LayerKVs,
    rows: Sequence[int],
    contexts: Sequence[int],
    warmup: int,
    steps: int,
    rounds: int,
) -> tuple[tuple[Cell, ...], tuple[Join, ...]]:
    """Every candidate once per round at each cell, each round starting one candidate later,
    so drift in the GPU clock reaches each alike. Row i of a cell holds ``context - i`` tokens
    of ``tokens`` and decodes from token ``context - i``. Refuses a round where the candidates
    disagree on a token at one row: they would not be doing the same work."""
    cells = [(r, t) for t in contexts for r in rows if r * t <= MAX_SLOTS]
    times: dict[tuple[int, int, str], dict[str, list[float]]] = {}
    memory: dict[tuple[int, int, str], tuple[float, float]] = {}
    agree: dict[tuple[int, int, str], bool] = {}
    for round_ in range(rounds):
        k = round_ % len(CANDIDATES)
        order = CANDIDATES[k:] + CANDIDATES[:k]
        for r, t in cells:
            lengths = [t - i for i in range(r)]
            first = [tokens[n] for n in lengths]
            seen: dict[str, list[list[int]]] = {}
            for kind in order:
                base = _settle()
                caches = build(kind, source, lengths)
                mx.eval(*[a for cache in caches for a in _arrays(cache)])
                held = mx.get_active_memory() - base
                mx.reset_peak_memory()
                seen[kind], ms = decode(model, caches, first, warmup, steps)
                peak = mx.get_peak_memory() - base
                del caches
                key = (r, t, kind)
                entry = times.setdefault(key, {"mean_ms": [], "median_ms": []})
                entry["mean_ms"].append(statistics.fmean(ms))
                entry["median_ms"].append(statistics.median(ms))
                _, top = memory.get(key, (0.0, 0.0))
                memory[key] = (held / MIB, max(top, peak / MIB))
            for kind in CANDIDATES:
                agree[(r, t, kind)] = agree.get((r, t, kind), True) and seen[kind] == seen["e7"]
            if r == 1 and any(out != seen["e7"] for out in seen.values()):
                raise RuntimeError("the candidates generated different tokens for one row")
            if r >= 2:
                for joiner in JOINERS:
                    _settle()
                    entry = times.setdefault((r, t, "join-" + joiner), {"join_ms": []})
                    entry["join_ms"].append(_join_ms(joiner, source, lengths))
                _settle()
                times.setdefault((r, t, "leave"), {"leave_ms": []})["leave_ms"].append(
                    _leave_ms(source, lengths)
                )
    out_cells = tuple(
        Cell(
            rows=r,
            context=t,
            candidate=kind,
            mean_ms=tuple(times[(r, t, kind)]["mean_ms"]),
            median_ms=tuple(times[(r, t, kind)]["median_ms"]),
            held_mib=memory[(r, t, kind)][0],
            peak_mib=memory[(r, t, kind)][1],
            same_tokens=agree[(r, t, kind)],
        )
        for r, t in cells
        for kind in CANDIDATES
    )
    joins = tuple(
        Join(
            rows=r,
            context=t,
            candidate=kind,
            join_ms=tuple(times[(r, t, "join-" + kind)]["join_ms"]),
            leave_ms=tuple(times[(r, t, "leave")]["leave_ms"]) if kind == "e7" else (),
        )
        for r, t in cells
        if r >= 2
        for kind in JOINERS
    )
    return out_cells, joins


def _segment(q: mx.array, k: mx.array, v: mx.array, scale: float) -> tuple[mx.array, mx.array]:
    """Attention over one segment: the output and the log-sum-exp of the scores, in float32,
    so segments can be merged. ``mx.fast.scaled_dot_product_attention`` returns no
    log-sum-exp."""
    scores = ((q * scale) @ k.swapaxes(-1, -2)).astype(mx.float32)
    lse = mx.logsumexp(scores, axis=-1, keepdims=True)
    return mx.exp(scores - lse) @ v.astype(mx.float32), lse


def split_attention(
    q: mx.array,
    k_prefix: mx.array,
    v_prefix: mx.array,
    k_tail: mx.array,
    v_tail: mx.array,
    scale: float,
) -> mx.array:
    """Candidate C for one decode token per row: ``q`` is ``[B, Hq, 1, D]``, the shared
    prefix ``[1, Hkv, P, D]``, each row's tail ``[B, Hkv, t, D]``. The prefix is read once for
    every row by folding the rows into its query axis."""
    rows, heads, _, dim = q.shape
    kv_heads = k_prefix.shape[1]
    group = heads // kv_heads
    grouped = q.reshape(rows, kv_heads, group, dim)
    folded = grouped.transpose(1, 0, 2, 3).reshape(kv_heads, rows * group, dim)
    out_p, lse_p = _segment(folded, k_prefix[0], v_prefix[0], scale)
    out_p = out_p.reshape(kv_heads, rows, group, dim).transpose(1, 0, 2, 3)
    lse_p = lse_p.reshape(kv_heads, rows, group, 1).transpose(1, 0, 2, 3)
    out_t, lse_t = _segment(grouped, k_tail, v_tail, scale)
    top = mx.maximum(lse_p, lse_t)
    w_p, w_t = mx.exp(lse_p - top), mx.exp(lse_t - top)
    out = (w_p * out_p + w_t * out_t) / (w_p + w_t)
    return out.reshape(rows, heads, 1, dim).astype(q.dtype)


def _median_ms(run: Any, repeats: int) -> float:
    for _ in range(2):
        mx.eval(run())
    ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(run())
        ms.append((time.perf_counter() - start) * 1000)
    return statistics.median(ms)


def measure_attention(
    prefixes: Sequence[int],
    rows: int,
    tail: int,
    heads: int,
    kv_heads: int,
    dim: int,
    layers: int,
    rounds: int,
    repeats: int = 10,
) -> tuple[Attention, ...]:
    """C against the fused kernel, attention only, random bf16 K and V, one array per layer."""
    out = []
    for prefix in prefixes:
        out.append(_attention(prefix, rows, tail, heads, kv_heads, dim, layers, rounds, repeats))
        _settle()
    return tuple(out)


def _attention(
    prefix: int,
    rows: int,
    tail: int,
    heads: int,
    kv_heads: int,
    dim: int,
    layers: int,
    rounds: int,
    repeats: int,
) -> Attention:
    scale = dim**-0.5
    mx.random.seed(0)
    per_layer = []
    for _ in range(layers):
        q = mx.random.normal((rows, heads, 1, dim)).astype(mx.bfloat16)
        kp, vp = (mx.random.normal((1, kv_heads, prefix, dim)).astype(mx.bfloat16) for _ in "kv")
        kt, vt = (mx.random.normal((rows, kv_heads, tail, dim)).astype(mx.bfloat16) for _ in "kv")
        shape = (rows, kv_heads, prefix, dim)
        k = mx.contiguous(mx.concatenate([mx.broadcast_to(kp, shape), kt], axis=2))
        v = mx.contiguous(mx.concatenate([mx.broadcast_to(vp, shape), vt], axis=2))
        per_layer.append((q, kp, vp, kt, vt, k, v))
    mx.eval(*[a for layer in per_layer for a in layer])

    def fused() -> list[mx.array]:
        return [
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            for q, _, _, _, _, k, v in per_layer
        ]

    def manual() -> list[mx.array]:
        return [
            _segment(q.reshape(rows, kv_heads, heads // kv_heads, dim), k, v, scale)[0]
            for q, _, _, _, _, k, v in per_layer
        ]

    def split() -> list[mx.array]:
        return [split_attention(q, kp, vp, kt, vt, scale) for q, kp, vp, kt, vt, _, _ in per_layer]

    timed: dict[str, list[float]] = {"fused_ms": [], "manual_ms": [], "split_ms": []}
    for _ in range(rounds):
        for name, run in (("fused_ms", fused), ("manual_ms", manual), ("split_ms", split)):
            timed[name].append(_median_ms(run, repeats))
    diff = mx.abs(split()[0].astype(mx.float32) - fused()[0].astype(mx.float32)).max()
    return Attention(
        prefix=prefix,
        rows=rows,
        tail=tail,
        max_abs_diff=cast(float, diff.item()),
        **{k: tuple(v) for k, v in timed.items()},
    )


def _mid(values: Sequence[float]) -> float:
    return statistics.median(values)


def _cell(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    return f"{_mid(values):.2f} [{min(values):.2f}, {max(values):.2f}]"


def report(run: KVLayouts) -> str:
    e = run.env
    by = {(c.rows, c.context, c.candidate): c for c in run.cells}
    contexts = sorted({c.context for c in run.cells})
    lines = [
        "# KV layouts on the real model (E9)",
        "",
        f"- `{run.model}`, greedy decode, {run.warmup} warm-up steps then {run.steps} timed "
        f"steps, {run.rounds} rounds. Row i of a cell holds context - i tokens, so every row "
        "count above one is padded and uses the array mask. Each time is the median over "
        "rounds of each round's value, [lowest, highest] round.",
        f"- Commit `{e.commit[:12]}`{' (dirty tree)' if e.dirty else ''}. {e.chip}, "
        f"{e.memory_gb:g} GB, macOS {e.macos}. AC {e.power.on_ac}, Low Power Mode "
        f"{e.power.low_power_mode}. Not a B5 session: it compares layouts in one process.",
        "",
        "## Decode step, mean ms over the timed steps",
        "",
    ]
    for t in contexts:
        lines += [
            f"### Context {t}",
            "",
            "| Rows | Context | E7 | A | A2 | B | A / E7 | A2 / E7 | B / E7 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted({c.rows for c in run.cells if c.context == t}):
            ref = _mid(by[(r, t, "e7")].mean_ms)
            ratios = [f"{_mid(by[(r, t, k)].mean_ms) / ref:.2f}" for k in ("a", "a2", "b")]
            times = [_cell(by[(r, t, k)].mean_ms) for k in CANDIDATES]
            lines.append(f"| {r} | {t} | " + " | ".join(times + ratios) + " |")
        lines.append("")
    lines += [
        "## Decode step, median ms",
        "",
        "| Rows | Context | E7 | A | A2 | B |",
        "|---|---|---|---|---|---|",
    ]
    for c in run.cells:
        if c.candidate == "e7":
            row = [_cell(by[(c.rows, c.context, k)].median_ms) for k in CANDIDATES]
            lines.append(f"| {c.rows} | {c.context} | " + " | ".join(row) + " |")
    lines += [
        "",
        "## Memory, MiB",
        "",
        "Held: active memory of the KV once built. Peak: the highest active memory above the "
        "base while decoding, which adds activations and each step's gathered batch.",
        "",
        "| Rows | Context | E7 held | A held | A2 held | B held "
        "| E7 peak | A peak | A2 peak | B peak |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in run.cells:
        if c.candidate == "e7":
            cells = [by[(c.rows, c.context, k)] for k in CANDIDATES]
            held = [f"{x.held_mib:.0f}" for x in cells]
            peak = [f"{x.peak_mib:.0f}" for x in cells]
            lines.append(f"| {c.rows} | {c.context} | " + " | ".join(held + peak) + " |")
    lines += [
        "",
        "## Membership changes, ms",
        "",
        "Join: one row of the cell's shortest length joins the other rows. E7 copies it from "
        "its prefill batch, B gathers it from its blocks first. Leave: the batch repacks "
        "without its last row, the same for E7 and B.",
        "",
        "| Rows after join | Context | E7 join | B join | Leave |",
        "|---|---|---|---|---|",
    ]
    joins = {(j.rows, j.context, j.candidate): j for j in run.joins}
    for (r, t, kind), j in joins.items():
        if kind == "e7":
            b = joins[(r, t, "b")]
            lines.append(
                f"| {r} | {t} | {_cell(j.join_ms)} | {_cell(b.join_ms)} | {_cell(j.leave_ms)} |"
            )
    lines += [
        "",
        "## Candidate C: attention alone, ms per decode step",
        "",
        "Every layer's attention for one decode token per row, random bf16 K and V. Fused: "
        "`mx.fast.scaled_dot_product_attention` over the padded batch. Manual: the same "
        "context as one segment through matmul and log-sum-exp. Split: the shared prefix "
        "read once for every row, merged with each row's tail.",
        "",
        "| Prefix | Rows | Tail | Fused | Manual | Split | Split - fused | Max abs diff |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in run.attention:
        saved = _mid(a.split_ms) - _mid(a.fused_ms)
        lines.append(
            f"| {a.prefix} | {a.rows} | {a.tail} | {_cell(a.fused_ms)} | {_cell(a.manual_ms)} "
            f"| {_cell(a.split_ms)} | {saved:+.2f} | {a.max_abs_diff:.4f} |"
        )
    differ = [f"{c.rows} x {c.context} {c.candidate}" for c in run.cells if not c.same_tokens]
    lines += [
        "",
        "## Tokens",
        "",
        "At one row every candidate gave E7's greedy tokens in every round (the script refuses "
        "a round otherwise). Above one row, cells whose tokens differed from E7's: "
        + (", ".join(differ) if differ else "none")
        + ".",
    ]
    return "\n".join(lines) + "\n"


def long_prompt(tokenizer: Any, n: int) -> list[int]:
    """The first ``n`` tokens of workload A's prompts, rendered and laid end to end."""
    workload = load_workload(WORKLOAD)
    tokens: list[int] = []
    for request in sorted(workload.requests, key=lambda r: r.id):
        messages = [m.model_dump() for m in request.messages]
        tokens += tokenizer.render(messages, workload.chat_template_kwargs)
        if len(tokens) >= n:
            return tokens[:n]
    raise ValueError(f"workload A holds fewer than {n} prompt tokens")


def prefill(model: Any, tokens: Sequence[int]) -> LayerKVs:
    """Per layer, the K and V of every token but the last, through mlx-lm's own cache."""
    cache = make_prompt_cache(model)
    model(mx.array(list(tokens[:-1]))[None], cache=cache)
    kv = [c.keys_and_values() for c in cache]
    mx.eval(kv)
    return kv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.kv_layouts")
    parser.add_argument("--out", type=Path, required=True, help="directory for the two files")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--rows", type=int, nargs="+", default=ROWS)
    parser.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    parser.add_argument("--prefixes", type=int, nargs="+", default=PREFIXES)
    args = parser.parse_args(argv)
    started = datetime.now(UTC)
    runner, tokenizer = load_mlx(DEFAULT_MODEL)
    model = runner.model
    tokens = long_prompt(tokenizer, max(args.contexts) + 1)
    source = prefill(model, tokens)
    cells, joins = measure(
        model, tokens, source, args.rows, args.contexts, WARMUP, args.steps, args.rounds
    )
    del source
    _settle()
    shape = model.args
    attention = measure_attention(
        args.prefixes,
        ATTENTION_ROWS,
        TAIL,
        shape.num_attention_heads,
        shape.num_key_value_heads,
        shape.head_dim,
        shape.num_hidden_layers,
        args.rounds,
    )
    run = KVLayouts(
        env=conditions.environment(used_powermetrics=False, started=started),
        model=DEFAULT_MODEL.key,
        warmup=WARMUP,
        steps=args.steps,
        rounds=args.rounds,
        cells=cells,
        joins=joins,
        attention=attention,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON).write_text(run.model_dump_json(indent=2) + "\n")
    (args.out / MARKDOWN).write_text(report(run))
    print(f"wrote {args.out / MARKDOWN}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
