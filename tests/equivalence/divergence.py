"""Where uLLM's greedy tokens first leave mlx-lm's, and how near a tie mlx-lm saw there (Q3)."""

from collections.abc import Sequence


def first_difference(got: Sequence[int], want: Sequence[int]) -> int:
    """The first position that differs; the shorter length when one is a prefix of the other."""
    return next(
        (n for n, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
        min(len(got), len(want)),
    )


def divergence(
    case_id: str, got: Sequence[int], want: Sequence[int], margins: Sequence[float]
) -> str:
    i = first_difference(got, want)
    margin = margins[i] if i < len(margins) else "-"
    return f"{case_id}: first difference at token {i}, mlx-lm top-2 margin {margin}"
