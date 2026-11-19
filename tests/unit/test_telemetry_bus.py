"""O1: the engine thread's bounded telemetry buffer (decision 0023)."""

import threading

from ullm.telemetry import events as ev
from ullm.telemetry.bus import Bus, TelemetryClock


def _arrived(i: int) -> ev.ReqArrived:
    return ev.ReqArrived(t=float(i), rid=f"r{i}", prompt_tokens=1, max_tokens=1)


def _rids(events: list[ev.Event]) -> list[str]:
    return [e.rid for e in events if isinstance(e, ev.ReqArrived)]


def test_drain_returns_the_events_in_emit_order_and_empties_the_bus() -> None:
    bus = Bus(capacity=8)
    for i in range(3):
        bus.emit(_arrived(i))
    events, dropped = bus.drain()
    assert (_rids(events), dropped) == (["r0", "r1", "r2"], 0)
    assert bus.drain() == ([], 0)


def test_a_full_bus_drops_the_oldest_and_counts_each_drop() -> None:
    bus = Bus(capacity=3)
    for i in range(5):
        bus.emit(_arrived(i))
    events, dropped = bus.drain()
    assert (_rids(events), dropped) == (["r2", "r3", "r4"], 2)


def test_drops_are_counted_since_the_previous_drain_only() -> None:
    bus = Bus(capacity=3)
    bus.emit(_arrived(0))
    assert bus.drain()[1] == 0
    for i in range(1, 6):
        bus.emit(_arrived(i))
    events, dropped = bus.drain()
    assert (_rids(events), dropped) == (["r3", "r4", "r5"], 2)
    bus.emit(_arrived(6))
    assert bus.drain() == ([_arrived(6)], 0)


def test_with_a_concurrent_producer_every_event_is_delivered_in_order_or_counted_dropped() -> None:
    total = 20_000
    bus = Bus(capacity=64)
    events = [_arrived(i) for i in range(total)]

    def produce() -> None:
        for e in events:
            bus.emit(e)

    producer = threading.Thread(target=produce)
    got: list[int] = []
    dropped = 0
    producer.start()
    while producer.is_alive() or got[-1:] != [total - 1]:
        batch, lost = bus.drain()
        got += [int(e.t) for e in batch]
        dropped += lost
    producer.join()
    assert got == sorted(set(got)), "events keep emit order, none twice"
    assert len(got) + dropped == total


def test_the_clock_counts_milliseconds_from_its_epoch() -> None:
    now = [100.0]
    clock = TelemetryClock.start(monotonic=lambda: now[0])
    now[0] = 100.25
    assert clock.now_ms() == 250.0
