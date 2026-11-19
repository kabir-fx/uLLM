"""O1: the path from the engine thread's events to readers (decision 0023).

``Bus`` is the one structure the two threads share. The engine thread is its only
producer and the hub its only consumer; ``deque`` appends and pops are thread-safe,
so neither side takes a lock or waits (rules 2 and 5).
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

from ullm.telemetry.events import Event

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
