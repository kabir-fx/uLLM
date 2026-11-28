"""Q5: the exact-match part starts a server, asks it every item one at a time, reads the
server's peak memory, and stops it (decision 0035). ``ullm serve --fake`` stands in for the
model, so every answer is wrong; the plumbing is what this checks."""

import socket
import sys
from pathlib import Path

from evals import data
from evals.exact_match import run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_every_item_gets_one_reply_from_a_server_the_run_stops(tmp_path: Path) -> None:
    items = data.gsm8k()[:5]
    port = _free_port()
    log = tmp_path / "server.log"

    summary, results = run(
        f"{sys.executable} -m ullm.cli serve --fake --port {port}", port, items, log
    )

    assert [r.index for r in results] == [0, 1, 2, 3, 4]
    assert all(r.completion_tokens > 0 and r.reply for r in results)
    assert summary.items == 5
    assert summary.output_tokens == sum(r.completion_tokens for r in results)
    if sys.platform == "darwin":
        assert summary.server_peak_footprint_bytes is not None
        assert summary.server_peak_footprint_bytes > 0
    else:  # footprint(1) is a macOS tool
        assert summary.server_peak_footprint_bytes is None
    assert log.stat().st_size > 0
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0, "the server still listens"
