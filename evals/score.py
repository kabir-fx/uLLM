"""Exact match on GSM8K (Q5, decision 0036): the gold answer follows ``####``, and the model's
answer is the last number in its last ``\\boxed{}``. Two answers match as numbers."""

import re
from decimal import Decimal

_NUMBER = re.compile(r"-?\d(?:[\d,]*\d)?(?:\.\d+)?")
_BOX = "\\boxed{"


def _last_number(text: str) -> Decimal | None:
    found = _NUMBER.findall(text)
    return Decimal(found[-1].replace(",", "")) if found else None


def _last_box(reply: str) -> str | None:
    start = reply.rfind(_BOX)
    if start < 0:
        return None
    depth = 1
    for i in range(start + len(_BOX), len(reply)):
        depth += {"{": 1, "}": -1}.get(reply[i], 0)
        if depth == 0:
            return reply[start + len(_BOX) : i]
    return None


def answer(reply: str) -> Decimal | None:
    box = _last_box(reply)
    return None if box is None else _last_number(box)


def gold(solution: str) -> Decimal:
    parts = solution.split("####")
    if len(parts) != 2 or not _NUMBER.fullmatch(parts[1].strip()):
        raise ValueError(f"no single gold number after ####: {solution[-80:]!r}")
    return Decimal(parts[1].strip().replace(",", ""))
