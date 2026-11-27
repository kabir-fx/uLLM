"""C3 v2 plans for runner tests, with the block tables the core's block manager would give."""

from ullm.runner.base import BlockId, DecodePlan, PrefillPlan, SeqId

SPAN = 1 << 16  # block ids of sequence s are s * SPAN + index, so no two sequences share one


class Plans:
    """Each sequence's table follows the tokens its plans name: a prefill sets the count,
    a decode adds one. A test that goes on after the runner rejects a plan calls ``undo``."""

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = block_size
        self._tokens: dict[SeqId, int] = {}
        self._before: dict[SeqId, int] = {}

    def prefill(
        self, seq: SeqId, token_ids: tuple[int, ...], start: int = 0, sample: bool = True
    ) -> PrefillPlan:
        self._before = dict(self._tokens)
        self._tokens[seq] = start + len(token_ids)
        return PrefillPlan(seq, token_ids, start, self.table(seq), sample)

    def decode(self, seqs: tuple[SeqId, ...], token_ids: tuple[int, ...]) -> DecodePlan:
        self._before = dict(self._tokens)
        for seq in seqs:
            self._tokens[seq] = self._tokens.get(seq, 0) + 1
        return DecodePlan(seqs, token_ids, tuple(self.table(seq) for seq in seqs))

    def undo(self) -> None:
        """Forget the last plan: the runner rejected it."""
        self._tokens = self._before

    def release(self, seq: SeqId) -> tuple[BlockId, ...]:
        """The table, as the blocks that became Free: nothing is shared in these tests."""
        freed = self.table(seq)
        self._tokens.pop(seq, None)
        return freed

    def table(self, seq: SeqId) -> tuple[BlockId, ...]:
        blocks = -(-self._tokens.get(seq, 0) // self.block_size)
        return tuple(seq * SPAN + i for i in range(blocks))
