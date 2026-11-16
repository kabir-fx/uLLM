"""The streaming detokenizer (E3, decision 0009). MLX-free.

A token's text can depend on its neighbours: a byte-level token may hold part
of a character, and some tokenizers drop a leading space when a token is decoded
alone. So each call decodes a short window of tokens, starting at the last point
where the text was complete, and emits what the newest tokens add to it.
"""

from collections.abc import Callable, Sequence


class IncrementalDetokenizer:
    def __init__(self, decode: Callable[[Sequence[int]], str]) -> None:
        self._decode = decode
        self._tokens: list[int] = []
        self._context = 0  # first token of the window: the start of the last chunk emitted
        self._unread = 0  # first token whose text is not emitted yet

    def add(self, token: int) -> str:
        """The text this token completes; empty while a character is still incomplete."""
        self._tokens.append(token)
        return self._advance(final=False)

    def flush(self) -> str:
        """The text held back, with any unfinished character as U+FFFD, at the end."""
        return self._advance(final=True)

    def _advance(self, final: bool) -> str:
        emitted = self._decode(self._tokens[self._context : self._unread])
        text = self._decode(self._tokens[self._context :])
        if text.endswith("\ufffd") and not final:
            return ""
        self._context, self._unread = self._unread, len(self._tokens)
        return text[len(emitted) :]
