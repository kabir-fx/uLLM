"""Synthetic telemetry traces, so the terminal can be built before the engine.

**These are not measurements and not a model of uLLM's performance.** A small
discrete-event simulation of the scheduling rules in catalog E6 (FCFS
admission, one prefill per step, batched decode) and, when blocks are on, of
E8, E10 and E11 (a block pool, a shared prefix, eviction and preemption),
timed by the toy latency model in ``Latency``. Every trace's ``hello`` says
``source="fixture"`` and ``model="synthetic"``, and the terminal labels it.

Deterministic across platforms: time is integer microseconds and the only
randomness is ``random.Random(seed).random()``, with no libm calls whose last
bits could differ between machines.
"""

import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from random import Random
from typing import Literal

from ullm.telemetry import events as ev

FRAME_US = 1_000_000 // 15  # frames are flushed about 15 times a second


@dataclass(frozen=True)
class Latency:
    """Toy step costs in microseconds. Chosen to look plausible, not measured."""

    prefill_base_us: int = 6_000
    prefill_per_token_us: int = 220
    decode_base_us: int = 11_000
    decode_per_row_us: int = 700
    decode_per_kv_token_ns: int = 600  # attention reads everything held in KV
    jitter: float = 0.04  # +/- uniform
    hiccup_share: float = 0.004  # occasional slow step, so latency charts have wicks
    hiccup_us: tuple[int, int] = (8_000, 30_000)


@dataclass(frozen=True)
class Blocks:
    block_size: int = 16
    num_blocks: int = 1024
    watermark: int = 2  # admission keeps this many blocks spare


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    seed: int
    duration_s: float  # arrivals stop here; the trace runs on until the engine is idle
    clients: int = 0  # closed loop: each client sends again after a think time
    think_ms: tuple[int, int] = (100, 800)
    rate_per_s: float = 0.0  # open loop: Bernoulli arrivals per 10 ms tick
    prompt_tokens: tuple[int, int] = (100, 1000)
    max_tokens: int = 300
    min_output_tokens: int = 50
    length_share: float = 0.25  # requests that run to max_tokens
    shared_prefix_tokens: int = 0
    shared_prefix_share: float = 0.0
    long_prompts: tuple[tuple[float, int], ...] = ()  # (arrival second, prompt tokens)
    abort_share: float = 0.0  # clients that disconnect mid-stream
    max_num_seqs: int = 16
    max_prefill_tokens: int = 8192
    max_model_len: int = 16384
    blocks: Blocks | None = None  # None: contiguous KV, as in M1 and M2
    kv_capacity_tokens: int | None = None
    snapshot_every_ms: int = 250
    latency: Latency = field(default_factory=Latency)


SCENARIOS: dict[str, Scenario] = {
    "steady-chat": Scenario(
        name="steady-chat",
        description=(
            "Workload A shape: 10 closed-loop clients, prompts of 100-1000 tokens, "
            "outputs up to 300, batch cap 8 so a queue forms, contiguous KV, a few aborts."
        ),
        seed=1,
        duration_s=40,
        clients=10,
        max_num_seqs=8,
        abort_share=0.03,
        kv_capacity_tokens=36_000,
    ),
    "prefix-pressure": Scenario(
        name="prefix-pressure",
        description=(
            "Workloads B and C combined: open-loop arrivals, 75% of requests share a "
            "2048-token system prompt, two very long prompts cause prefill stalls, and a "
            "1024-block pool runs short, so blocks are evicted and requests preempted."
        ),
        seed=2,
        duration_s=40,
        rate_per_s=1.8,
        prompt_tokens=(100, 600),
        max_tokens=256,
        min_output_tokens=64,
        shared_prefix_tokens=2048,
        shared_prefix_share=0.75,
        long_prompts=((14.0, 8000), (27.0, 7000)),
        abort_share=0.02,
        blocks=Blocks(),
        kv_capacity_tokens=16 * 1024,
    ),
    "stress": Scenario(
        name="stress",
        description=(
            "The terminal's frame-rate budget (T7): open-loop arrivals keep 64 requests "
            "running on a 2,304-block pool, so the finished tab and the feed reach 200 rows."
        ),
        seed=3,
        duration_s=60,
        rate_per_s=4.0,
        prompt_tokens=(100, 600),
        max_tokens=256,
        min_output_tokens=64,
        max_num_seqs=64,
        blocks=Blocks(num_blocks=2304),
        kv_capacity_tokens=2304 * 16,
    ),
}


_Key = tuple[str, int]


@dataclass
class _Req:
    rid: str
    client: int | None
    arrival_us: int
    prompt: int
    shared: bool  # carries the shared prefix
    max_tokens: int
    target: int  # tokens it will produce before stopping
    abort_after: int | None
    queued_us: int = 0
    generated: int = 0
    kv_len: int = 0
    blocks: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    first_admit_wait_us: int | None = None
    first_token_us: int | None = None
    last_token_us: int = 0


class _Sim:
    def __init__(self, sc: Scenario) -> None:
        self.sc = sc
        self.rng = Random(sc.seed)
        self.now = 0
        self.out: list[tuple[int, int, ev.Event]] = []  # (t_us, order, event)
        self.waiting: deque[_Req] = deque()
        self.running: list[_Req] = []
        self.arrivals: list[tuple[int, int, int | None, int | None]] = []  # t, n, client, prompt
        self.n_arrivals = 0  # scheduled
        self.n_arrived = 0  # processed; numbers the request ids
        self.n_steps = 0
        self.step_start = 0
        self.last_snapshot_us = -(10**12)
        b = sc.blocks
        self.free: deque[int] = deque(range(b.num_blocks)) if b else deque()
        self.ref = [0] * (b.num_blocks if b else 0)
        self.cached: OrderedDict[int, None] = OrderedDict()  # LRU of ref-0 blocks with contents
        # Stand-in for E10's chained hashes: a full block's content is named by
        # ("prefix", i) for the shared system prompt, or (rid, i) for a request's own.
        self.by_key: dict[_Key, int] = {}
        self.key_of: dict[int, _Key] = {}

    # --- helpers -------------------------------------------------------------

    def uniform(self, lo: int, hi: int) -> int:
        return lo + int(self.rng.random() * (hi - lo + 1))

    def emit(self, t_us: int, event: ev.Event) -> None:
        self.out.append((t_us, len(self.out), event))

    def schedule_arrival(self, t_us: int, client: int | None, prompt: int | None = None) -> None:
        heapq.heappush(self.arrivals, (t_us, self.n_arrivals, client, prompt))
        self.n_arrivals += 1

    # --- workload ------------------------------------------------------------

    def seed_arrivals(self) -> None:
        sc, end = self.sc, int(self.sc.duration_s * 1_000_000)
        for c in range(sc.clients):
            self.schedule_arrival(self.uniform(0, 1_000_000), c)
        if sc.rate_per_s:
            for tick in range(0, end, 10_000):
                if self.rng.random() < sc.rate_per_s * 0.01:
                    self.schedule_arrival(tick + self.uniform(0, 9_999), None)
        for second, tokens in sc.long_prompts:
            self.schedule_arrival(int(second * 1_000_000), None, tokens)

    def arrive(self, t_us: int, client: int | None, prompt: int | None) -> None:
        sc = self.sc
        shared = prompt is None and self.rng.random() < sc.shared_prefix_share
        own = prompt if prompt is not None else self.uniform(*sc.prompt_tokens)
        total = own + (sc.shared_prefix_tokens if shared else 0)
        hits_max = self.rng.random() < sc.length_share
        target = (
            sc.max_tokens if hits_max else self.uniform(sc.min_output_tokens, sc.max_tokens - 1)
        )
        abort_after = None
        if self.rng.random() < sc.abort_share:
            abort_after = self.uniform(1, max(1, target - 1))
        req = _Req(
            rid=f"req-{self.n_arrived:04d}",
            client=client,
            arrival_us=t_us,
            prompt=total,
            shared=shared,
            max_tokens=sc.max_tokens,
            target=target,
            abort_after=abort_after,
            queued_us=t_us,
        )
        self.n_arrived += 1
        self.waiting.append(req)
        self.emit(
            t_us,
            ev.ReqArrived(
                t=t_us / 1000, rid=req.rid, prompt_tokens=total, max_tokens=req.max_tokens
            ),
        )

    # --- blocks (E8, E10, E11) -----------------------------------------------

    def blocks_for(self, tokens: int) -> int:
        assert self.sc.blocks
        return -(-tokens // self.sc.blocks.block_size)

    def key(self, req: _Req, i: int) -> _Key:
        assert self.sc.blocks
        if req.shared and i < self.sc.shared_prefix_tokens // self.sc.blocks.block_size:
            return ("prefix", i)
        return (req.rid, i)

    def prefix_hits(self, req: _Req) -> list[int]:
        """Blocks to reuse, in order; matching stops at the first miss (E10)."""
        if not self.sc.blocks:
            return []
        # Always recompute at least the last token, to get next-token logits.
        limit = (req.prompt + req.generated - 1) // self.sc.blocks.block_size
        hits = []
        for i in range(limit):
            block = self.by_key.get(self.key(req, i))
            if block is None:
                break
            hits.append(block)
        return hits

    def take_block(self) -> int | None:
        if self.free:
            return self.free.popleft()
        if self.cached:
            block, _ = self.cached.popitem(last=False)  # evict the least recently used
            del self.by_key[self.key_of.pop(block)]
            return block
        return None

    def register(self, key: _Key, block: int) -> None:
        # Matching stops at the first miss, so a later block can still be
        # registered from an older copy. The newest copy wins; the old block
        # becomes ordinary (freed now if nobody holds it).
        old = self.by_key.get(key)
        if old is not None and old != block:
            del self.key_of[old]
            if self.ref[old] == 0:
                del self.cached[old]
                self.free.append(old)
        self.by_key[key], self.key_of[block] = block, key

    def release(self, req: _Req) -> None:
        """Drop the request's references. Full blocks keep their contents as Cached (E10, E11)."""
        assert self.sc.blocks
        size = self.sc.blocks.block_size
        for i, block in enumerate(req.blocks):
            self.ref[block] -= 1
            if self.ref[block] > 0:
                continue
            if block not in self.key_of and (i + 1) * size <= req.kv_len:
                self.register(self.key(req, i), block)
            if block in self.key_of:
                self.cached[block] = None
            else:
                self.free.append(block)
        req.blocks = []

    def admit_blocks(self, req: _Req, hits: list[int]) -> None:
        for block in hits:
            if self.ref[block] == 0:
                del self.cached[block]
            self.ref[block] += 1
        req.blocks = list(hits)
        for i in range(len(hits), self.blocks_for(req.prompt + req.generated)):
            new = self.take_block()
            assert new is not None, "admission checked that the prompt fits"
            self.ref[new] = 1
            req.blocks.append(new)
            key = self.key(req, i)
            if key[0] == "prefix":
                self.register(key, new)  # shareable as soon as it is computed

    def fits(self, req: _Req, hits: list[int]) -> bool:
        if self.sc.blocks is None:
            return True
        need = self.blocks_for(req.prompt + req.generated) - len(hits)
        spare = len(self.free) + len(self.cached) - sum(1 for b in hits if self.ref[b] == 0)
        return need <= spare - self.sc.blocks.watermark

    def make_room(self) -> None:
        """Every running request writes one token this step; give each a slot (E6 step 2)."""
        assert self.sc.blocks
        size = self.sc.blocks.block_size
        for req in list(self.running):
            if req not in self.running or req.kv_len < len(req.blocks) * size:
                continue
            block = self.take_block()
            while block is None:
                victim = self.running[-1]  # the newest: least work to redo (E11)
                self.preempt(victim)
                if victim is req:
                    break
                block = self.take_block()
            if block is not None:
                self.ref[block] = 1
                req.blocks.append(block)

    def preempt(self, req: _Req) -> None:
        self.running.remove(req)
        freed = len(req.blocks)
        if self.sc.blocks:
            self.release(req)
        req.kv_len, req.queued_us = 0, self.now
        self.waiting.appendleft(req)  # front of the queue
        self.emit(
            self.now,
            ev.ReqPreempted(
                t=self.now / 1000,
                rid=req.rid,
                completion_tokens=req.generated,
                blocks_freed=freed,
            ),
        )

    # --- steps (E6) ----------------------------------------------------------

    def cost_us(self, base: int) -> int:
        lat = self.sc.latency
        cost = int(base * (1 + lat.jitter * (2 * self.rng.random() - 1)))
        if self.rng.random() < lat.hiccup_share:
            cost += self.uniform(*lat.hiccup_us)
        return cost

    def kv_tokens(self) -> int:
        return sum(r.kv_len for r in self.running)

    def prefill(self, req: _Req, hits: list[int]) -> None:
        sc, lat = self.sc, self.sc.latency
        self.waiting.popleft()
        cached = len(hits) * sc.blocks.block_size if sc.blocks else 0
        if req.first_admit_wait_us is None:
            req.first_admit_wait_us = self.now - req.arrival_us
            req.cached_tokens = cached
        self.emit(
            self.now,
            ev.ReqAdmitted(
                t=self.now / 1000,
                rid=req.rid,
                queue_ms=(self.now - req.queued_us) / 1000,
                cached_tokens=cached,
            ),
        )
        if sc.blocks:
            self.admit_blocks(req, hits)
        computed = req.prompt + req.generated - cached
        self.now += self.cost_us(lat.prefill_base_us + lat.prefill_per_token_us * computed)
        req.kv_len = req.prompt + req.generated
        self.running.append(req)
        self.record_step("prefill", [req], computed)
        self.sampled(req)

    def decode(self) -> None:
        lat = self.sc.latency
        rows = list(self.running)
        base = (
            lat.decode_base_us
            + lat.decode_per_row_us * len(rows)
            + lat.decode_per_kv_token_ns * self.kv_tokens() // 1000
        )
        self.now += self.cost_us(base)
        for req in rows:
            req.kv_len += 1
        self.record_step("decode", rows, len(rows))
        for req in rows:
            self.sampled(req)

    def record_step(
        self, kind: Literal["prefill", "decode"], rows: list[_Req], tokens: int
    ) -> None:
        self.drain_arrivals()  # so `waiting` counts everyone who arrived during the step
        self.emit(
            self.now,
            ev.Step(
                t=self.now / 1000,
                n=self.n_steps,
                kind=kind,
                rids=[r.rid for r in rows],
                tokens=tokens,
                ms=(self.now - self.step_start) / 1000,
                waiting=len(self.waiting),
                running=len(self.running),
                kv_tokens=self.kv_tokens(),
            ),
        )
        self.n_steps += 1

    def sampled(self, req: _Req) -> None:
        req.generated += 1
        req.last_token_us = self.now
        if req.first_token_us is None:
            req.first_token_us = self.now
            self.emit(
                self.now,
                ev.ReqFirstToken(
                    t=self.now / 1000, rid=req.rid, ttft_ms=(self.now - req.arrival_us) / 1000
                ),
            )

    def end_of_step(self) -> None:
        for req in list(self.running):
            if req.abort_after is not None and req.generated >= req.abort_after:
                self.finish(req, aborted=True)
            elif req.generated >= req.target:
                self.finish(req, aborted=False)

    def finish(self, req: _Req, *, aborted: bool) -> None:
        self.running.remove(req)
        if self.sc.blocks:
            self.release(req)
        t = self.now / 1000
        if aborted:
            self.emit(
                self.now,
                ev.ReqAborted(
                    t=t, rid=req.rid, reason="client_disconnect", completion_tokens=req.generated
                ),
            )
        else:
            assert req.first_token_us is not None
            assert req.first_admit_wait_us is not None
            tpot = None
            if req.generated > 1:
                tpot = (req.last_token_us - req.first_token_us) / 1000 / (req.generated - 1)
            self.emit(
                self.now,
                ev.ReqFinished(
                    t=t,
                    rid=req.rid,
                    reason="length" if req.generated >= req.max_tokens else "stop",
                    usage=ev.TokenUsage(
                        prompt_tokens=req.prompt,
                        completion_tokens=req.generated,
                        cached_tokens=req.cached_tokens,
                    ),
                    timings=ev.Timings(
                        queue_ms=req.first_admit_wait_us / 1000,
                        ttft_ms=(req.first_token_us - req.arrival_us) / 1000,
                        tpot_ms=tpot,
                        e2e_ms=(self.now - req.arrival_us) / 1000,
                    ),
                ),
            )
        end = int(self.sc.duration_s * 1_000_000)
        if req.client is not None and self.now < end:
            self.schedule_arrival(self.now + 1000 * self.uniform(*self.sc.think_ms), req.client)

    def snapshot(self) -> None:
        if (
            not self.sc.blocks
            or self.now - self.last_snapshot_us < self.sc.snapshot_every_ms * 1000
        ):
            return
        self.last_snapshot_us = self.now
        holders: dict[int, list[str]] = {}
        for req in self.running:
            for block in req.blocks:
                holders.setdefault(block, []).append(req.rid)
        owners: list[str] = []
        states = bytearray(self.sc.blocks.num_blocks)
        for block in range(self.sc.blocks.num_blocks):
            held = holders.get(block, [])
            assert len(held) == self.ref[block], "simulation lost track of a ref count"
            if len(held) > 1:
                states[block] = ev.BLOCK_SHARED
            elif held:
                if held[0] not in owners:
                    owners.append(held[0])
                states[block] = ev.BLOCK_OWNED_BASE + owners.index(held[0])
            elif block in self.cached:
                states[block] = ev.BLOCK_CACHED
        self.emit(self.now, ev.KvSnapshot.build(self.now / 1000, owners, bytes(states)))

    # --- main loop -----------------------------------------------------------

    def drain_arrivals(self) -> None:
        while self.arrivals and self.arrivals[0][0] <= self.now:
            t_us, _, client, prompt = heapq.heappop(self.arrivals)
            self.arrive(t_us, client, prompt)

    def run(self) -> list[ev.Frame]:
        self.emit(0, self.hello())
        self.seed_arrivals()
        while True:
            self.drain_arrivals()
            if not self.running and not self.waiting:
                if not self.arrivals:
                    break
                self.now = self.arrivals[0][0]
                continue
            self.step_start = self.now
            head = self.waiting[0] if self.waiting else None
            hits = self.prefix_hits(head) if head else []
            if head and len(self.running) < self.sc.max_num_seqs and self.fits(head, hits):
                self.prefill(head, hits)
            elif self.running:
                if self.sc.blocks:
                    self.make_room()
                if not self.running:
                    continue
                self.decode()
            else:
                raise AssertionError(f"{head.rid if head else '?'} can never fit the pool")
            self.end_of_step()
            self.snapshot()
        return self.frames()

    def hello(self) -> ev.Hello:
        sc = self.sc
        return ev.Hello(
            t=0,
            replica="r1",
            model="synthetic",
            source="fixture",
            version=f"synthetic/{sc.name}",
            commit=None,
            epoch_unix_ms=0,
            limits=ev.Limits(
                max_num_seqs=sc.max_num_seqs,
                max_prefill_tokens=sc.max_prefill_tokens,
                max_model_len=sc.max_model_len,
            ),
            kv=ev.KvInfo(
                block_size=sc.blocks.block_size if sc.blocks else None,
                num_blocks=sc.blocks.num_blocks if sc.blocks else None,
                capacity_tokens=sc.kv_capacity_tokens,
            ),
            machine=None,
        )

    def frames(self) -> list[ev.Frame]:
        ordered = sorted(self.out, key=lambda item: (item[0], item[1]))
        frames: list[ev.Frame] = []
        batch: list[ev.Event] = []
        window = 0
        for t_us, _, event in ordered:
            w = max(0, (t_us - 1) // FRAME_US)
            if batch and w != window:
                frames.append(self.frame(len(frames), window, batch))
                batch = []
            window = w
            batch.append(event)
        if batch:
            frames.append(self.frame(len(frames), window, batch))
        return frames

    def frame(self, seq: int, window: int, events: list[ev.Event]) -> ev.Frame:
        return ev.Frame(replica="r1", seq=seq, t=(window + 1) * FRAME_US / 1000, events=events)


def generate(scenario: Scenario) -> list[ev.Frame]:
    return _Sim(scenario).run()
