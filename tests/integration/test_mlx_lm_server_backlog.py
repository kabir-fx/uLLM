"""The ``mlx_lm.server`` the benchmarks launch queues a burst of connections while it cannot
accept them. With ``socketserver``'s queue of 5, macOS refused the rest: 4 requests of the first
``mlx-lm-batched-c8`` run failed with ``ConnectError``."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import pytest

pytestmark = pytest.mark.mlx

ROOT = Path(__file__).resolve().parents[2]
# B3's largest closed-loop concurrency. A queue of 5 refuses a burst of 8 only sometimes, and a
# burst of 16 every time.
BURST = 16


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_healthy(port: int, server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        assert server.poll() is None, "the server exited"
        try:
            if httpx2.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                return
        except httpx2.TransportError:
            pass
        time.sleep(0.2)
    pytest.fail("the server did not report healthy")


async def _connected(port: int) -> int:
    async def connect() -> asyncio.StreamWriter | None:
        try:
            return (await asyncio.open_connection("127.0.0.1", port))[1]
        except OSError:  # macOS refuses a full queue as ECONNRESET, sometimes EINVAL
            return None

    writers = await asyncio.gather(*(connect() for _ in range(BURST)))
    for writer in writers:
        if writer is not None:
            writer.close()
    return sum(writer is not None for writer in writers)


def test_a_stalled_server_queues_a_burst_of_connections() -> None:
    pytest.importorskip("mlx_lm")
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "scripts/mlx_lm_server.py", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_healthy(port, server)
        os.kill(server.pid, signal.SIGSTOP)
        assert asyncio.run(_connected(port)) == BURST
    finally:
        if server.poll() is None:
            os.kill(server.pid, signal.SIGCONT)
            server.terminate()
        server.wait(timeout=10)
