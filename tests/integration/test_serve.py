"""``ullm serve --fake``: the server as a process, on 127.0.0.1, with no model."""

import contextlib
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx2

from ullm.server.app import GRACEFUL_SHUTDOWN_S
from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace
from ullm.telemetry.trace import iter_frames


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _serve(*args: str) -> tuple[subprocess.Popen[str], str]:
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "ullm.cli", "serve", "--fake", "--port", str(port), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while True:
        try:
            httpx2.get(f"{base}/health")
            return server, base
        except httpx2.ConnectError:
            assert server.poll() is None, server.communicate()[0]
            assert time.monotonic() < deadline, "the server did not start"
            time.sleep(0.05)


def _chat(base: str, max_tokens: int) -> httpx2.Response:
    return httpx2.post(
        f"{base}/v1/chat/completions",
        json={
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": max_tokens,
        },
    )


def test_ullm_serve_fake_answers_then_stops_on_interrupt() -> None:
    server, base = _serve()
    try:
        health = httpx2.get(f"{base}/health")
        assert health.json()["alive"] is True
        models = httpx2.get(f"{base}/v1/models").json()
        assert [m["id"] for m in models["data"]] == ["fake"]
        reply = _chat(base, 3).json()
        assert reply["usage"]["completion_tokens"] == 3
    finally:
        server.send_signal(signal.SIGINT)
        out = server.communicate(timeout=15)[0]
    assert server.returncode == 0, out


def test_ullm_serve_passes_max_kv_slots_to_the_engine() -> None:
    # The fake model's max_model_len is 4096, so the engine refuses this budget at startup.
    limits = ["--max-num-seqs", "4", "--max-kv-slots", "4095"]
    run = subprocess.run(
        [sys.executable, "-m", "ullm.cli", "serve", "--fake", "--port", str(_free_port()), *limits],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode != 0
    assert "max_kv_slots 4095 is below max_model_len 4096" in run.stdout + run.stderr


def test_ullm_serve_passes_max_num_seqs_to_the_engine() -> None:
    run = subprocess.run(
        [sys.executable, "-m", "ullm.cli", "serve", "--fake", "--max-num-seqs", "0"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode != 0
    assert "max_num_seqs must be an int >= 1, got 0" in run.stdout + run.stderr


def test_an_open_tab_does_not_keep_ullm_serve_from_stopping_and_closing_its_recording(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl.gz"
    server, base = _serve("--record", str(path))
    first_frame = threading.Event()

    def tab() -> None:
        # The server cuts the stream when its graceful timeout ends.
        with (
            contextlib.suppress(httpx2.RemoteProtocolError),
            httpx2.stream("GET", f"{base}/telemetry/stream", timeout=None) as response,
        ):
            for line in response.iter_lines():
                if line.startswith("data: "):
                    first_frame.set()

    threading.Thread(target=tab, daemon=True).start()
    try:
        assert first_frame.wait(10)
        assert _chat(base, 3).status_code == 200
    finally:
        server.send_signal(signal.SIGINT)
        out = server.communicate(timeout=GRACEFUL_SHUTDOWN_S + 10)[0]
    assert server.returncode == 0, out
    frames = list(iter_frames(path))
    assert check_trace(frames).problems == []
    assert [e.type for f in frames for e in f.events if isinstance(e, ev.ReqFinished)] == [
        "req.finished"
    ]


def test_ullm_serve_with_telemetry_off_has_no_stream() -> None:
    server, base = _serve("--telemetry", "off")
    try:
        assert httpx2.get(f"{base}/telemetry/stream").status_code == 404
    finally:
        server.send_signal(signal.SIGINT)
        server.communicate(timeout=15)


def test_ullm_serve_refuses_to_start_when_it_cannot_open_the_recording(tmp_path: Path) -> None:
    run = subprocess.run(
        [sys.executable, "-m", "ullm.cli", "serve", "--fake", "--port", str(_free_port())]
        + ["--record", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode != 0
    assert "Is a directory" in run.stdout + run.stderr
