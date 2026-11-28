"""Q5: how an exact-match answer is read from a reply and from GSM8K's gold (decision 0036)."""

from decimal import Decimal

import pytest
from evals.score import answer, gold


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("so she makes 9 * 2 = 18.\n\n\\boxed{18}", Decimal(18)),
        ("\\boxed{\\$18}", Decimal(18)),
        ("\\boxed{18 \\text{ dollars}}", Decimal(18)),
        ("\\boxed{1,000}", Decimal(1000)),
        ("\\boxed{-5}", Decimal(-5)),
        ("\\boxed{3.50}", Decimal("3.5")),
        ("\\boxed{50\\%}", Decimal(50)),
        ("\\boxed{5 \\times 3 = 15}", Decimal(15)),
        ("first \\boxed{4}, then on reflection \\boxed{7}", Decimal(7)),
        ("$$\n\\boxed{70000}\n$$", Decimal(70000)),
    ],
)
def test_the_answer_is_the_last_number_in_the_last_box(reply: str, expected: Decimal) -> None:
    assert answer(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "The answer is 18.",
        "\\boxed{}",
        "\\boxed{eighteen}",
        "\\boxed{18",
        "",
    ],
)
def test_a_reply_without_a_number_in_a_closed_box_has_no_answer(reply: str) -> None:
    assert answer(reply) is None


def test_equal_numbers_match_whatever_their_spelling() -> None:
    assert answer("\\boxed{18.00}") == gold("... #### 18")


def test_the_gold_answer_follows_the_four_hashes() -> None:
    assert gold("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs.\n#### 18") == Decimal(18)
    assert gold("#### -3") == Decimal(-3)
    assert gold("#### 1,200") == Decimal(1200)


@pytest.mark.parametrize("solution", ["no hashes, 18", "#### many", "#### 1\n#### 2"])
def test_a_gold_answer_that_is_not_one_number_is_refused(solution: str) -> None:
    with pytest.raises(ValueError, match="gold"):
        gold(solution)


def test_an_unclosed_last_box_hides_an_earlier_closed_one() -> None:
    assert answer("\\boxed{4} then, cut off: \\boxed{7") is None
