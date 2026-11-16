"""The streaming detokenizer (E3, decision 0009). MLX-free.

A byte-level vocabulary spells every token as bytes, and one character can span
tokens, or end one token and start the next. So the detokenizer feeds each
token's bytes to an incremental UTF-8 decoder, which holds back only an
unfinished character.
"""

import codecs
from collections.abc import Sequence


class ByteDetokenizer:
    def __init__(self, token_bytes: Sequence[bytes]) -> None:
        self._token_bytes = token_bytes  # indexed by token id
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def add(self, token: int) -> str:
        """The text this token completes; empty while a character is still incomplete."""
        return self._utf8.decode(self._token_bytes[token])

    def flush(self) -> str:
        """The text held back, with any unfinished character as U+FFFD, at the end."""
        return self._utf8.decode(b"", final=True)
