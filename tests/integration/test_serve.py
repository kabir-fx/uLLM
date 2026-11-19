"""``ullm serve --fake``: the server as a process, on 127.0.0.1, with no model."""

import signal
import socket
import subprocess
import sys
import time

import httpx2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def test_ullm_serve_fake_answers_then_stops_on_interrupt() -> None:
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "ullm.cli", "serve", "--fake", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 15
        while True:
            try:
                health = httpx2.get(f"{base}/health")
                break
            except httpx2.ConnectError:
                assert server.poll() is None, server.communicate()[0]
                assert time.monotonic() < deadline, "the server did not start"
                time.sleep(0.05)
        assert health.json()["alive"] is True
        models = httpx2.get(f"{base}/v1/models").json()
        assert [m["id"] for m in models["data"]] == ["fake"]
        reply = httpx2.post(
            f"{base}/v1/chat/completions",
            json={
                "model": "fake",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
            },
        ).json()
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
