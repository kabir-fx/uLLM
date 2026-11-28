"""Q5: perplexity scores whole windows of the text and drops its short tail (decision 0036)."""

from evals.perplexity import windows


def test_windows_do_not_overlap_and_drop_the_short_tail() -> None:
    assert windows(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_text_shorter_than_one_window_gives_none() -> None:
    assert windows([1, 2, 3], 4) == []


def test_an_exact_multiple_keeps_every_token() -> None:
    assert windows(list(range(8)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]
