"""E3: the stop check (decision 0009). MLX-free, on a detokenizer that maps each
token to a fixed piece of text, so any text can arrive split at any point."""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ullm.core.stop import Chunk, StopCheck

EOS = 1_000


class Pieces:
    """Token ``i`` is ``pieces[i]``; flush returns what a test sets as held back."""

    def __init__(self, pieces: list[str], held: str = "") -> None:
        self.pieces = pieces
        self.held = held
        self.added: list[int] = []

    def add(self, token: int) -> str:
        self.added.append(token)
        return self.pieces[token]

    def flush(self) -> str:
        return self.held


def _check(
    pieces: list[str], stop: tuple[str, ...] = (), max_tokens: int = 10_000, held: str = ""
) -> tuple[StopCheck, Pieces]:
    detokenizer = Pieces(pieces, held)
    check = StopCheck(detokenizer, frozenset({EOS}), max_tokens=max_tokens, stop=stop)
    return check, detokenizer


def _run(check: StopCheck, tokens: list[int]) -> tuple[str, str | None]:
    text = ""
    for token in tokens:
        chunk = check.add(token)
        text += chunk.text
        if chunk.finish_reason is not None:
            return text, chunk.finish_reason
    return text, None


def _cut_at_first_stop(text: str, stop: tuple[str, ...]) -> str | None:
    """The text before the stop string that ends first; the longer one on a tie."""
    hits = [(text.find(s) + len(s), text.find(s)) for s in stop if s in text]
    return text[: min(hits)[1]] if hits else None


short_text = st.text(alphabet="ab\n", max_size=4)


@given(st.lists(short_text, max_size=12), st.lists(st.text("ab\n", min_size=1, max_size=3)))
def test_the_reply_ends_before_the_first_stop_string_however_it_is_split(
    pieces: list[str], stop_list: list[str]
) -> None:
    stop = tuple(stop_list)
    text, finish = _run(_check(pieces, stop)[0], list(range(len(pieces))))
    cut = _cut_at_first_stop("".join(pieces), stop)
    if cut is None:
        assert finish is None
        assert "".join(pieces).startswith(text)
    else:
        assert (text, finish) == (cut, "stop")


@given(st.lists(short_text, max_size=12), st.lists(st.text("ab\n", min_size=1, max_size=3)))
def test_no_emitted_text_contains_a_stop_string(pieces: list[str], stop_list: list[str]) -> None:
    text, _ = _run(_check(pieces, tuple(stop_list))[0], list(range(len(pieces))))
    assert not any(s in text for s in stop_list)


def test_text_that_could_start_a_stop_string_is_held_until_it_cannot() -> None:
    check, _ = _check(["Hel", "p", "Hel", "lo"], stop=("Hello",))
    assert check.add(0) == Chunk("", None)
    assert check.add(1) == Chunk("Help", None)
    assert check.add(2) == Chunk("", None)
    assert check.add(3) == Chunk("", "stop")


def test_a_stop_string_split_across_tokens_is_found() -> None:
    check, _ = _check(["x", "<|e", "nd|", ">tail"], stop=("<|end|>",))
    assert _run(check, [0, 1, 2, 3]) == ("x", "stop")


def test_eos_stops_without_its_text_and_releases_held_text() -> None:
    check, detokenizer = _check(["ab", "Hel"], stop=("Hello",), held="!")
    assert check.add(0) == Chunk("ab", None)
    assert check.add(1) == Chunk("", None)
    assert check.add(EOS) == Chunk("Hel!", "stop")
    assert detokenizer.added == [0, 1]


def test_max_tokens_ends_with_length_and_flushes() -> None:
    check, _ = _check(["a", "b", "Hel"], stop=("Hello",), max_tokens=3, held="?")
    assert _run(check, [0, 1, 2]) == ("abHel?", "length")


def test_a_stop_string_in_the_last_token_wins_over_length() -> None:
    check, _ = _check(["a", "b!"], stop=("!",), max_tokens=2)
    assert _run(check, [0, 1]) == ("ab", "stop")


def test_eos_as_the_last_allowed_token_is_a_stop() -> None:
    check, _ = _check(["a"], max_tokens=2)
    assert _run(check, [0, EOS]) == ("a", "stop")


def test_nothing_is_accepted_after_the_finish() -> None:
    check, _ = _check(["a"], max_tokens=1)
    assert check.add(0).finish_reason == "length"
    with pytest.raises(ValueError, match="finished"):
        check.add(0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"stop": ("",)}, "stop"),
        ({"stop": ["a"]}, "stop"),
    ],
)
def test_settings_are_checked(changes: dict[str, Any], match: str) -> None:
    settings: dict[str, Any] = {"max_tokens": 4, "stop": ()} | changes
    with pytest.raises(ValueError, match=match):
        StopCheck(Pieces([]), frozenset({EOS}), **settings)


def test_text_released_at_the_end_token_can_complete_a_stop_string() -> None:
    check, _ = _check(["Hel"], stop=("Hello",), held="lo")
    assert check.add(0) == Chunk("", None)
    assert check.add(EOS) == Chunk("", "stop")
