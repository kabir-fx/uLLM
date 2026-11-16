"""E3: the streaming detokenizer, on a byte-level tokenizer like Qwen3's (decision 0009).

Each fake token is a few bytes, so one character can span tokens, as with emoji
and CJK text in a real byte-level BPE vocabulary.
"""

from collections.abc import Sequence
from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from ullm.runner.detokenizer import IncrementalDetokenizer


class ByteTokens:
    def __init__(self) -> None:
        self.pieces: list[bytes] = []

    def encode(self, data: bytes, cuts: Sequence[int]) -> list[int]:
        bounds = sorted({0, len(data), *(c % (len(data) + 1) for c in cuts)})
        ids = []
        for start, end in pairwise(bounds):
            self.pieces.append(data[start:end])
            ids.append(len(self.pieces) - 1)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", "replace")


def _stream(tokens: ByteTokens, ids: list[int]) -> tuple[list[str], str]:
    detokenizer = IncrementalDetokenizer(tokens.decode)
    return [detokenizer.add(i) for i in ids], detokenizer.flush()


@given(st.text(), st.lists(st.integers(min_value=0)))
def test_streamed_text_is_the_decoded_text(text: str, cuts: list[int]) -> None:
    tokens = ByteTokens()
    ids = tokens.encode(text.encode(), cuts)
    chunks, rest = _stream(tokens, ids)
    assert "".join(chunks) + rest == text


@given(st.text().filter(lambda t: "\ufffd" not in t), st.lists(st.integers(min_value=0)))
def test_no_chunk_holds_part_of_a_character(text: str, cuts: list[int]) -> None:
    tokens = ByteTokens()
    chunks, rest = _stream(tokens, tokens.encode(text.encode(), cuts))
    assert all("\ufffd" not in chunk for chunk in chunks)
    assert rest == ""


@given(st.binary(), st.lists(st.integers(min_value=0)))
def test_invalid_bytes_stream_as_their_replacement(data: bytes, cuts: list[int]) -> None:
    tokens = ByteTokens()
    ids = tokens.encode(data, cuts)
    chunks, rest = _stream(tokens, ids)
    assert "".join(chunks) + rest == tokens.decode(ids)


def test_a_character_split_across_tokens_waits_for_its_last_byte() -> None:
    tokens = ByteTokens()
    party = "🎉".encode()  # four bytes
    ids = tokens.encode(b"a" + party + b"b", [1, 3, 5])  # a | 2 bytes | 2 bytes | b
    assert _stream(tokens, ids) == (["a", "", "🎉", "b"], "")


def test_flush_gives_up_on_an_unfinished_character() -> None:
    tokens = ByteTokens()
    ids = tokens.encode("🎉".encode()[:2], [])
    assert _stream(tokens, ids) == ([""], "\ufffd")


def test_a_word_keeps_its_space_when_the_tokenizer_drops_a_leading_one() -> None:
    # SentencePiece-style: decoding drops the space that starts the text, so a
    # token decoded alone loses the space that joins it to the word before.
    pieces = [" Hello", " world", "!"]

    def decode(ids: Sequence[int]) -> str:
        return "".join(pieces[i] for i in ids).removeprefix(" ")

    detokenizer = IncrementalDetokenizer(decode)
    chunks = [detokenizer.add(i) for i in range(3)] + [detokenizer.flush()]
    assert "".join(chunks) == decode([0, 1, 2]) == "Hello world!"
