"""O1: the hub fans the bus out to subscriptions, each a bounded reader (decision 0023)."""

import asyncio
from collections.abc import Awaitable, Callable

from ullm.telemetry import events as ev
from ullm.telemetry.bus import Bus, Hub, TelemetryClock

HELLO = ev.Hello(
    t=0.0,
    replica="r1",
    model="fake",
    source="sim",
    version="0",
    commit=None,
    epoch_unix_ms=0,
    limits=ev.Limits(max_num_seqs=1, max_prefill_tokens=8, max_model_len=8),
    kv=ev.KvInfo(),
    machine=None,
)


def _arrived(i: int) -> ev.ReqArrived:
    return ev.ReqArrived(t=float(i), rid=f"r{i}", prompt_tokens=1, max_tokens=1)


class _Setup:
    def __init__(self, bus_capacity: int = 64, capacity: int = 64) -> None:
        self.now = [0.0]
        self.clock = TelemetryClock(epoch=0.0, epoch_unix_ms=0, monotonic=lambda: self.now[0])
        self.bus = Bus(bus_capacity)
        self.hub = Hub(self.bus, self.clock, "r1", capacity=capacity)


def _run(test: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(asyncio.wait_for(test(), timeout=5))


def _types(frame: ev.Frame) -> list[str]:
    return [e.type for e in frame.events]


def test_a_reader_gets_frames_numbered_from_zero_timed_on_the_clock() -> None:
    async def test() -> None:
        s = _Setup()
        sub = s.hub.subscribe()
        s.bus.emit(HELLO)
        s.bus.emit(_arrived(1))
        s.now[0] = 2.0
        s.hub.flush()
        first = await anext(sub)
        s.bus.emit(_arrived(3))
        s.now[0] = 4.0
        s.hub.flush()
        second = await anext(sub)
        assert (first.seq, first.t, first.replica, _types(first)) == (
            0,
            2000.0,
            "r1",
            ["hello", "req.arrived"],
        )
        assert (second.seq, second.t, second.events) == (1, 4000.0, [_arrived(3)])

    _run(test)


def test_a_late_reader_starts_with_hello_then_only_new_events() -> None:
    async def test() -> None:
        s = _Setup()
        s.bus.emit(HELLO)
        s.bus.emit(_arrived(1))
        s.hub.flush()
        late = s.hub.subscribe()
        s.bus.emit(_arrived(2))
        s.hub.flush()
        frame = await anext(late)
        assert (frame.seq, frame.events, frame.dropped) == (0, [HELLO, _arrived(2)], 0)

    _run(test)


def test_a_slow_reader_loses_its_own_oldest_events_and_counts_them() -> None:
    async def test() -> None:
        s = _Setup(capacity=2)
        slow = s.hub.subscribe()
        for i in range(5):
            s.bus.emit(_arrived(i))
        s.hub.flush()
        frame = await anext(slow)
        assert (frame.events, frame.dropped) == ([_arrived(3), _arrived(4)], 3)
        s.bus.emit(_arrived(5))
        s.hub.flush()
        assert (await anext(slow)).dropped == 0

    _run(test)


def test_one_slow_reader_does_not_cost_another_reader_events() -> None:
    async def test() -> None:
        s = _Setup(capacity=2)
        slow, fast = s.hub.subscribe(), s.hub.subscribe()
        got: list[ev.Event] = []
        for i in range(5):
            s.bus.emit(_arrived(i))
            s.hub.flush()
            got += (await anext(fast)).events
        assert got == [_arrived(i) for i in range(5)]
        assert (await anext(slow)).dropped == 3

    _run(test)


def test_events_the_bus_dropped_count_in_every_readers_frame() -> None:
    async def test() -> None:
        s = _Setup(bus_capacity=2)
        a, b = s.hub.subscribe(), s.hub.subscribe()
        for i in range(5):
            s.bus.emit(_arrived(i))
        s.hub.flush()
        for sub in (a, b):
            frame = await anext(sub)
            assert (frame.events, frame.dropped) == ([_arrived(3), _arrived(4)], 3)

    _run(test)


def test_a_reader_waits_for_events_instead_of_returning_empty_frames() -> None:
    async def test() -> None:
        s = _Setup()
        sub = s.hub.subscribe()
        s.hub.flush()
        waiting = asyncio.ensure_future(anext(sub))
        await asyncio.sleep(0.01)
        assert not waiting.done()
        s.bus.emit(_arrived(1))
        s.hub.flush()
        assert (await waiting).events == [_arrived(1)]

    _run(test)


def test_close_delivers_the_last_events_then_ends_every_reader() -> None:
    async def test() -> None:
        s = _Setup()
        sub = s.hub.subscribe()
        s.bus.emit(_arrived(1))
        s.hub.close()
        assert [e for f in [f async for f in sub] for e in f.events] == [_arrived(1)]

    _run(test)


def test_run_flushes_on_its_own_until_close() -> None:
    async def test() -> None:
        s = _Setup()
        s.hub = Hub(s.bus, TelemetryClock.start(), "r1", frame_ms=1.0)
        sub = s.hub.subscribe()
        running = asyncio.create_task(s.hub.run())
        s.bus.emit(_arrived(1))
        assert (await anext(sub)).events == [_arrived(1)]
        s.hub.close()
        await running

    _run(test)


def test_a_reader_that_leaves_is_no_longer_fed() -> None:
    async def test() -> None:
        s = _Setup()
        async with s.hub.subscribe():
            assert s.hub.readers == 1
        assert s.hub.readers == 0

    _run(test)


def test_the_observer_gets_every_drain_whatever_the_readers_keep() -> None:
    seen: list[tuple[list[ev.Event], int]] = []
    bus = Bus(2)
    clock = TelemetryClock(epoch=0.0, epoch_unix_ms=0, monotonic=lambda: 0.0)
    hub = Hub(bus, clock, "r1", capacity=1, observe=lambda events, n: seen.append((events, n)))
    hub.subscribe()  # never read, and holds one event
    for i in range(5):
        bus.emit(_arrived(i))
    hub.flush()
    hub.flush()  # an empty drain tells the observer nothing
    bus.emit(_arrived(5))
    hub.close()
    assert seen == [([_arrived(3), _arrived(4)], 3), ([_arrived(5)], 0)]
