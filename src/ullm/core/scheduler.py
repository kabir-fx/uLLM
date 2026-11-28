"""Which requests run next (E6, decision 0010). Pure decisions; it never calls the runner.

E6's first step, removing finished requests, happens at the end of the step
that finished them, so a reply's last chunk is not delayed a step.
"""

from collections import deque
from dataclasses import dataclass

from ullm.core.block_manager import BlockManager, KvUsage
from ullm.core.request import Request
from ullm.runner.base import BlockId, DecodePlan, StepPlan


@dataclass(frozen=True, slots=True)
class Scheduled:
    plan: StepPlan
    rows: tuple[Request, ...]  # one per sampled row, in plan order
    admitted: Request | None  # admitted this step: the engine calls runner.add for it


class Scheduler:
    def __init__(
        self, max_num_seqs: int, max_kv_slots: int, block_size: int, num_blocks: int
    ) -> None:
        self._max_num_seqs = max_num_seqs
        self._max_kv_slots = max_kv_slots
        self._blocks = BlockManager(num_blocks, block_size)
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []  # admitted: prefilling or running

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    @property
    def running(self) -> int:
        return len(self._running)

    def kv_usage(self) -> KvUsage:
        return self._blocks.usage()

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
        """Nothing preempts a row until M3-T6, so each row reserves the most it can hold.
        The working tensor pads every row to the longest (decision 0020), and the pool holds
        each row's blocks (decision 0035): a row that is admitted never runs either out."""
        batch = (*self._running, req)
        longest = max(r.kv_reservation for r in batch)
        blocks = sum(-(-r.kv_reservation // self._blocks.block_size) for r in batch)
        return (
            len(batch) <= self._max_num_seqs
            and len(batch) * longest <= self._max_kv_slots
            and blocks <= self._blocks.num_blocks
        )
