"""The server under test: started from a command line, stopped as a process group, and asked
which model it serves (decisions 0013 and 0016)."""

import asyncio
import os
import shlex
import signal
import socket
import subprocess
import time
from typing import IO

import httpx2


class Target:
    """The server under test, started from a command line in its own process group."""

    def __init__(self, command: str, log: IO[bytes]) -> None:
        self.proc = subprocess.Popen(
            shlex.split(command), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()


def port_in_use(base_url: str) -> bool:
    url = httpx2.URL(base_url)
    try:
        with socket.create_connection((url.host, url.port or 80), timeout=1):
            return True
    except OSError:
        return False


async def ready(
    client: httpx2.AsyncClient, target: Target | None, timeout_s: float, model: str | None
) -> str:
    """Wait for ``GET /health``, then return the model id to send: ``model`` when
    ``GET /v1/models`` lists it, else the first id it lists."""
    deadline = time.monotonic() + timeout_s
    while True:
        if target is not None and target.proc.poll() is not None:
            raise SystemExit(f"the target exited with {target.proc.returncode} before it was ready")
        try:
            healthy = (await client.get("/health")).status_code == 200
        except httpx2.TransportError:
            healthy = False
        if healthy:
            break
        if time.monotonic() > deadline:
            raise SystemExit(f"the target was not healthy after {timeout_s:g} s")
        await asyncio.sleep(0.5)
    served = [str(m["id"]) for m in (await client.get("/v1/models")).json()["data"]]
    if model is None:
        return served[0]
    if model not in served:
        raise SystemExit(f"the target does not serve {model!r}; it lists {served}")
    return model
