"""Which requests run next (E6, decision 0010). Pure decisions; it never calls the runner.

E6's first step, removing finished requests, happens at the end of the step
that finished them, so a reply's last chunk is not delayed a step.
"""

from collections import deque
from dataclasses import dataclass

from ullm.core.request import Request
from ullm.runner.base import DecodePlan, StepPlan


@dataclass(frozen=True, slots=True)
class Scheduled:
    plan: StepPlan
    rows: tuple[Request, ...]  # one per sampled row, in plan order
    admitted: Request | None  # admitted this step: the engine calls runner.add for it


class Scheduler:
    def __init__(self, max_num_seqs: int, max_kv_slots: int) -> None:
        self._max_num_seqs = max_num_seqs
        self._max_kv_slots = max_kv_slots
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []  # admitted: prefilling or running

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    @property
    def running(self) -> int:
        return len(self._running)

    def add(self, req: Request) -> None:
        self._waiting.append(req)

    def remove(self, req: Request) -> None:
        if req in self._running:
            self._running.remove(req)
        elif req in self._waiting:
            self._waiting.remove(req)
        else:
            raise ValueError(f"{req.rid} is not scheduled")

    def schedule(self, t: float) -> Scheduled | None:
        """One prefill for the oldest waiting request if it fits, else one decode (DEC-7).

        A head that does not fit admits nobody, so no request skips ahead of it.
        """
        if self._waiting and self._fits(self._waiting[0]):
            req = self._waiting.popleft()
            req.admit(t)
            self._running.append(req)
            return Scheduled(plan=req.prefill_plan(), rows=(req,), admitted=req)
        if self._running:
            rows = tuple(self._running)
            plan = DecodePlan(
                seqs=tuple(r.seq for r in rows), token_ids=tuple(r.decode_input() for r in rows)
            )
            return Scheduled(plan=plan, rows=rows, admitted=None)
        return None

    def _fits(self, req: Request) -> bool:
        """Every row is padded to the longest, and nothing preempts a row until M3, so the
        batch is held to rows x the longest reservation (decision 0020)."""
        rows = len(self._running) + 1
        longest = max(r.kv_reservation for r in (*self._running, req))
        return rows <= self._max_num_seqs and rows * longest <= self._max_kv_slots
