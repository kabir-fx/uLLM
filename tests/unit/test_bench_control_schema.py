"""The load control schema the terminal's types are generated from (decision 0025)."""

from bench.control import SCHEMA_PATH, render_schema


def test_the_committed_schema_is_fresh() -> None:
    assert SCHEMA_PATH.read_text() == render_schema(), "run `make contracts`"
