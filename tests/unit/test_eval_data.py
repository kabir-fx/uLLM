"""Q5: the committed datasets load, and a changed copy is refused (decision 0035)."""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from evals import data


def test_gsm8k_has_the_first_200_test_items_in_order() -> None:
    items = data.gsm8k()
    assert [item.index for item in items] == list(range(200))
    assert items[0].question.startswith("Janet\u2019s ducks lay 16 eggs per day.")
    assert items[0].gold == 18
    assert len({item.question for item in items}) == 200


def test_wikitext_is_the_raw_test_text() -> None:
    text = data.wikitext()
    assert text.startswith(" \n = Robert Boulter = \n")
    assert len(text.encode()) == 1_290_590


@pytest.mark.parametrize(
    ("name", "load"),
    [("gsm8k/test-200.jsonl", data.gsm8k), ("wikitext-2/wiki.test.raw", data.wikitext)],
)
def test_a_copy_with_a_changed_byte_is_refused(
    name: str, load: Callable[[], object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / name
    copy.parent.mkdir()
    shutil.copy(data.DATASETS / name, copy)
    with copy.open("ab") as f:
        f.write(b" ")
    monkeypatch.setattr(data, "DATASETS", tmp_path)
    with pytest.raises(data.DataChanged, match=name):
        load()
