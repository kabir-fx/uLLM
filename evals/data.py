"""The committed eval datasets (Q5, decision 0036). Each loads only while its bytes have the
sha256 recorded here, so an edited copy cannot change a score unnoticed. The sources and
licenses are in each folder's README.md."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from evals.score import gold

DATASETS = Path(__file__).parent / "datasets"
GSM8K = "gsm8k/test-200.jsonl"
WIKITEXT = "wikitext-2/wiki.test.raw"
SHA256 = {
    GSM8K: "bd70035c7acaf107b4e0d077c605a23c3d3a0acb4342e5bc60099e6ad9ff4284",
    WIKITEXT: "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
}


class DataChanged(Exception):
    """A dataset whose bytes differ from the recorded ones."""


@dataclass(frozen=True, slots=True)
class Item:
    index: int  # the line in GSM8K's test.jsonl, from 0
    question: str
    gold: Decimal


def _read(name: str) -> bytes:
    raw = (DATASETS / name).read_bytes()
    if (digest := hashlib.sha256(raw).hexdigest()) != SHA256[name]:
        raise DataChanged(f"{name}: sha256 {digest}, recorded {SHA256[name]}")
    return raw


def gsm8k() -> tuple[Item, ...]:
    rows = [json.loads(line) for line in _read(GSM8K).decode().splitlines()]
    return tuple(Item(i, row["question"], gold(row["answer"])) for i, row in enumerate(rows))


def wikitext() -> str:
    return _read(WIKITEXT).decode()
