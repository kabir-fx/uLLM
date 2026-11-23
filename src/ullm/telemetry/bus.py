"""O1: the path from the engine thread's events to readers (decision 0023).

``Bus`` is the one structure the two threads share. The engine thread is its only
producer and the hub its only consumer; ``deque`` appends and pops are thread-safe,
so neither side takes a lock or waits (rules 2 and 5).
"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

from ullm.telemetry.events import Event, Frame, Hello
from ullm.telemetry.trace import FRAME_MS

CAPACITY = 4096  # a guess, not a measurement; `dropped` shows when it is too small


@dataclass(frozen=True, slots=True)
class TelemetryClock:
    """The replica's telemetry epoch (C2): event and frame times are ms since it."""

    epoch: float
    epoch_unix_ms: int
    monotonic: Callable[[], float] = time.monotonic

    @classmethod
    def start(cls, monotonic: Callable[[], float] = time.monotonic) -> Self:
        return cls(monotonic(), int(time.time() * 1000), monotonic)

    def now_ms(self) -> float:
        return (self.monotonic() - self.epoch) * 1000


class Bus:
    """Bounded: a full bus drops its oldest event. Each event carries its emit index, so the
    consumer counts drops exactly from the gaps."""

    def __init__(self, capacity: int = CAPACITY) -> None:
        self._events: deque[tuple[int, Event]] = deque(maxlen=capacity)
        self._emitted = 0  # written by the producer only
        self._expected = 0  # written by the consumer only

    def emit(self, event: Event) -> None:
        """The engine thread only. Never blocks."""
        self._events.append((self._emitted, event))
        self._emitted += 1

    def drain(self) -> tuple[list[Event], int]:
        """The consumer only: the events present, in emit order, and how many were dropped
        since the previous drain."""
        events: list[Event] = []
        dropped = 0
        while True:
            try:
                index, event = self._events.popleft()
            except IndexError:
                return events, dropped
            dropped += index - self._expected
            self._expected = index + 1
            events.append(event)


class Hub:
    """Moves events from the bus to ``observe`` and every subscription. Lives on the event
    loop's thread: every method runs there, so the subscriptions need no lock."""

    def __init__(
        self,
        bus: Bus,
        clock: TelemetryClock,
        replica: str,
        capacity: int = CAPACITY,
        frame_ms: float = FRAME_MS,
        observe: Callable[[list[Event], int], None] | None = None,
    ) -> None:
        self._bus = bus
        self._observe = observe
        self._clock = clock
        self._replica = replica
        self._capacity = capacity
        self._frame_s = frame_ms / 1000
        self._subscriptions: set[Subscription] = set()
        self._hello: Hello | None = None
        self._closed = asyncio.Event()

    @property
    def readers(self) -> int:
        return len(self._subscriptions)

    def subscribe(self) -> "Subscription":
        """A new reader. Its first frame starts with ``hello`` once the engine has sent it."""
        sub = Subscription(self, self._capacity)
        if self._hello is not None:
            sub._put([self._hello], 0)
        if self._closed.is_set():
            sub._end()
        else:
            self._subscriptions.add(sub)
        return sub

    async def run(self) -> None:
        """Flushes about 15 times a second until ``close``."""
        while not self._closed.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._closed.wait(), self._frame_s)
            self.flush()

    def flush(self) -> None:
        events, dropped = self._bus.drain()
        for event in events:
            if isinstance(event, Hello):
                self._hello = event
        if events:  # the bus counts a drop only when it pops a later event
            if self._observe is not None:
                self._observe(events, dropped)
            for sub in self._subscriptions:
                sub._put(events, dropped)

    def close(self) -> None:
        """After the engine has exited: the last events reach every reader, then each ends."""
        self.flush()
        self._closed.set()
        for sub in self._subscriptions:
            sub._end()
        self._subscriptions.clear()

    def _leave(self, sub: "Subscription") -> None:
        self._subscriptions.discard(sub)

    def _frame(self, seq: int, dropped: int, events: list[Event]) -> Frame:
        return Frame(
            replica=self._replica, seq=seq, t=self._clock.now_ms(), dropped=dropped, events=events
        )


class Subscription:
    """One reader's frames, numbered from 0. Bounded: a slow reader loses its own oldest
    events, counted in its next frame's ``dropped``. Leave it with ``async with`` or
    ``leave``; until then the hub keeps feeding it."""

    def __init__(self, hub: Hub, capacity: int) -> None:
        self._hub = hub
        self._events: deque[Event] = deque(maxlen=capacity)
        self._dropped = 0
        self._seq = 0
        self._ended = False
        self._ready = asyncio.Event()

    def _put(self, events: list[Event], dropped: int) -> None:
        before = len(self._events)
        self._events.extend(events)
        self._dropped += dropped + before + len(events) - len(self._events)
        self._ready.set()

    def _end(self) -> None:
        self._ended = True
        self._ready.set()

    def leave(self) -> None:
        self._hub._leave(self)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> Frame:
        while not (self._events or self._dropped):
            if self._ended:
                raise StopAsyncIteration
            self._ready.clear()
            await self._ready.wait()
        frame = self._hub._frame(self._seq, self._dropped, list(self._events))
        self._seq += 1
        self._dropped = 0
        self._events.clear()
        return frame

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.leave()
