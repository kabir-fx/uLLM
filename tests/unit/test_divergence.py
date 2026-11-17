"""Q3's mismatch report: the first differing token and mlx-lm's margin there (decision 0012)."""

from tests.equivalence.divergence import divergence, first_difference


def test_the_first_difference_is_the_first_unequal_position() -> None:
    assert first_difference([5, 6, 7, 8], [5, 6, 9, 8]) == 2


def test_a_prefix_differs_at_the_shorter_length() -> None:
    assert first_difference([5, 6], [5, 6, 7]) == 2
    assert first_difference([5, 6, 7], [5, 6]) == 2


def test_the_report_gives_the_margin_at_the_first_difference() -> None:
    report = divergence("case", [1, 2, 3], [1, 9, 3], [0.5, 0.125, 2.0])
    assert report == "case: first difference at token 1, mlx-lm top-2 margin 0.125"


def test_a_difference_past_the_margins_has_no_margin() -> None:
    assert divergence("case", [1, 2], [1], [0.5]).endswith("margin -")
