"""B5 run conditions: observe them, refuse a full run that breaks them, log powermetrics, and
record the machine in env.json (decision 0013)."""

import platform
import re
import signal
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bench.schema import Power, RunEnv

POWERMETRICS = "/usr/bin/powermetrics"
LIBRARIES = ("mlx", "mlx-lm", "transformers", "tokenizers", "fastapi", "uvicorn", "httpx2")


@dataclass(frozen=True)
class Plan:
    repeats: int
    warmup: int
    cooldown_s: float


def _run(*argv: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return None


def _out(*argv: str) -> str:
    done = _run(*argv)
    return done.stdout if done and done.returncode == 0 else ""


def parse_power(batt: str, settings: str) -> Power:
    """Read ``pmset -g batt`` and ``pmset -g``."""
    source = re.search(r"drawing from '([^']+)'", batt)
    percent = re.search(r"\t(\d+)%;", batt)
    low = re.search(r"^\s*lowpowermode\s+(\d)", settings, re.MULTILINE)
    return Power(
        on_ac=None if source is None else source.group(1) == "AC Power",
        battery_percent=None if percent is None else int(percent.group(1)),
        low_power_mode=None if low is None else low.group(1) == "1",
    )


def power() -> Power:
    return parse_power(_out("pmset", "-g", "batt"), _out("pmset", "-g"))


def can_sudo() -> bool:
    """True when powermetrics runs as root without a password prompt: after a fresh ``sudo -v``,
    or under a NOPASSWD sudoers rule for unattended matrices. ``sudo -n -l`` is not enough: any
    unrelated NOPASSWD rule lets it pass (man sudoers)."""
    done = _run("sudo", "-n", POWERMETRICS, "--samplers", "thermal", "-n", "1", "-i", "1")
    return done is not None and done.returncode == 0


def containers() -> int | None:
    """Running Docker containers, or None when no daemon answers."""
    done = _run("docker", "ps", "-q")
    if done is None or done.returncode != 0:
        return None
    return len(done.stdout.split())


def violations(power: Power, *, sudo: bool, containers: int | None, plan: Plan) -> list[str]:
    """Every B5 condition a full run would break, in words. An unknown power state counts as
    broken; no Docker daemon means no containers."""
    found = []
    if power.on_ac is not True:
        found.append("not on AC power: plug in")
    if power.low_power_mode is not False:
        found.append("Low Power Mode is on or unknown: turn it off")
    if not sudo:
        found.append("powermetrics needs root: run `sudo -v` first, or allow it without a password")
    if containers:
        found.append(f"{containers} Docker containers are running: stop the compose stack")
    if plan.repeats < 3:
        found.append(f"{plan.repeats} repeats: B5 takes the median of at least 3")
    if plan.warmup < 1:
        found.append("no warm-up: B5 discards at least one warm-up request")
    if plan.cooldown_s <= 0:
        found.append("no cooldown between repeats")
    return found


class PowerMetrics:
    """``powermetrics`` writing GPU and CPU power, frequency and thermal pressure to a plist."""

    def __init__(self, path: Path) -> None:
        self.proc = subprocess.Popen(
            [
                "sudo",
                "-n",
                POWERMETRICS,
                "--samplers",
                "gpu_power,cpu_power,thermal",
                "--sample-rate",
                "1000",
                "--format",
                "plist",
                "--output-file",
                str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        # sudo relays SIGINT to powermetrics, which flushes its last sample and exits.
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def _library(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment(*, used_powermetrics: bool, started: datetime) -> RunEnv:
    sysctl = {
        key: _out("sysctl", "-n", key).strip()
        for key in ("machdep.cpu.brand_string", "hw.ncpu", "hw.memsize")
    }
    top = _out("ps", "-Ao", "pcpu,comm", "-r").splitlines()[1:11]
    libraries = {name: v for name in LIBRARIES if (v := _library(name)) is not None}
    return RunEnv(
        commit=_out("git", "rev-parse", "HEAD").strip(),
        dirty=bool(_out("git", "status", "--porcelain").strip()),
        chip=sysctl["machdep.cpu.brand_string"],
        cores=int(sysctl["hw.ncpu"] or 0),
        memory_gb=round(int(sysctl["hw.memsize"] or 0) / 2**30, 1),
        macos=_out("sw_vers", "-productVersion").strip(),
        python=platform.python_version(),
        libraries=libraries,
        power=power(),
        powermetrics=used_powermetrics,
        containers=containers(),
        top_processes=tuple(line.strip() for line in top),
        started=started.isoformat(timespec="seconds"),
        ended=datetime.now(UTC).isoformat(timespec="seconds"),
    )
