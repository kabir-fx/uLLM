"""Which requests run next (E6, decision 0010). Pure decisions; it never calls the runner.

E6's first step, removing finished requests, happens at the end of the step
that finished them, so a reply's last chunk is not delayed a step.
"""

from collections import deque
from dataclasses import dataclass

from ullm.core.block_manager import BlockManager
from ullm.core.request import Request
from ullm.runner.base import BlockId, DecodePlan, StepPlan


@dataclass(frozen=True, slots=True)
class Scheduled:
    plan: StepPlan
    rows: tuple[Request, ...]  # one per sampled row, in plan order
    admitted: Request | None  # admitted this step: the engine calls runner.add for it


class Scheduler:
    def __init__(self, max_num_seqs: int, max_kv_slots: int, block_size: int) -> None:
        self._max_num_seqs = max_num_seqs
        self._max_kv_slots = max_kv_slots
        # Admission holds rows x the longest reservation to max_kv_slots, and each row wastes
        # under one block, so this pool never runs out (decision 0034). M3-T4 admits by blocks.
        num_blocks = -(-max_kv_slots // block_size) + max_num_seqs
        self._blocks = BlockManager(num_blocks, block_size)
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

    def remove(self, req: Request) -> tuple[BlockId, ...]:
        """The blocks that became Free: none for a request that was still waiting."""
        if req in self._running:
            self._running.remove(req)
            return self._blocks.free(req.seq)
        if req in self._waiting:
            self._waiting.remove(req)
            return ()
        raise ValueError(f"{req.rid} is not scheduled")

    def schedule(self, t: float) -> Scheduled | None:
        """One prefill for the oldest waiting request if it fits, else one decode (DEC-7).

        A head that does not fit admits nobody, so no request skips ahead of it.
        """
        if self._waiting and self._fits(self._waiting[0]):
            req = self._waiting.popleft()
            req.admit(t)
            self._running.append(req)
            self._blocks.allocate(req.seq, len(req.prompt))
            prefill = req.prefill_plan(self._blocks.table(req.seq))
            return Scheduled(plan=prefill, rows=(req,), admitted=req)
        if self._running:
            rows = tuple(self._running)
            for r in rows:
                self._blocks.append_slot(r.seq)
            plan = DecodePlan(
                seqs=tuple(r.seq for r in rows),
                token_ids=tuple(r.decode_input() for r in rows),
                blocks=tuple(self._blocks.table(r.seq) for r in rows),
            )
            return Scheduled(plan=plan, rows=rows, admitted=None)
        return None

    def _fits(self, req: Request) -> bool:
        """Every row is padded to the longest, and nothing preempts a row until M3, so the
        batch is held to rows x the longest reservation (decision 0020)."""
        rows = len(self._running) + 1
        longest = max(r.kv_reservation for r in (*self._running, req))
        return rows <= self._max_num_seqs and rows * longest <= self._max_kv_slots
