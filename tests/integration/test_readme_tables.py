"""The README's benchmark table is the one `bench.readme` makes from the committed sweep
sessions (P1, rule 8, decision 0030). It loads four sessions, about 30 s, so it runs in
`make ci-core`, not `make check`."""

from pathlib import Path

import pytest
from bench.analysis import load_session
from bench.readme import BEGIN, END, SESSIONS, splice, table

ROOT = Path(__file__).resolve().parents[2]


def test_the_readme_benchmark_table_is_generated_from_the_committed_sweep() -> None:
    readme = (ROOT / "README.md").read_text()
    fresh = table([load_session(ROOT / p) for p in SESSIONS])
    assert splice(readme, fresh) == readme, "README.md is stale: run `make readme`"


def test_splice_replaces_only_the_text_between_the_markers() -> None:
    readme = f"top\n{BEGIN}\nold\n{END}\nbottom\n"
    assert splice(readme, "new\n") == f"top\n{BEGIN}\nnew\n{END}\nbottom\n"


@pytest.mark.parametrize(
    "text", ["no markers\n", f"{END}\n{BEGIN}\n", f"{BEGIN}\n{BEGIN}\n{END}\n"]
)
def test_splice_refuses_a_readme_without_one_ordered_pair_of_markers(text: str) -> None:
    with pytest.raises(SystemExit, match="markers"):
        splice(text, "new\n")
