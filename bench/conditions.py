"""B5 run conditions: observe them, refuse a session that breaks them, log powermetrics, and
record the machine in env.json (decisions 0013 and 0016)."""

import argparse
import platform
import re
import signal
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bench.schema import Power, RunEnv

POWERMETRICS = "/usr/bin/powermetrics"
LIBRARIES = ("mlx", "mlx-lm", "transformers", "tokenizers", "fastapi", "uvicorn", "httpx2")


def _run(*argv: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return None


def _out(*argv: str) -> str:
    done = _run(*argv)
    return done.stdout if done and done.returncode == 0 else ""


def parse_power(batt: str, settings: str, clamshell: str) -> Power:
    """Read ``pmset -g batt``, ``pmset -g`` and ``ioreg -r -k AppleClamshellState -d 1``."""
    source = re.search(r"drawing from '([^']+)'", batt)
    percent = re.search(r"\t(\d+)%; ([^;]+);", batt)
    low = re.search(r"^\s*lowpowermode\s+(\d)", settings, re.MULTILINE)
    closed = re.search(r'"AppleClamshellState" = (Yes|No)', clamshell)
    return Power(
        on_ac=None if source is None else source.group(1) == "AC Power",
        battery_percent=None if percent is None else int(percent.group(1)),
        low_power_mode=None if low is None else low.group(1) == "1",
        charging=None if percent is None else percent.group(2) in ("charging", "finishing charge"),
        lid_open=None if closed is None else closed.group(1) == "No",
    )


def power() -> Power:
    return parse_power(
        _out("pmset", "-g", "batt"),
        _out("pmset", "-g"),
        _out("ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"),
    )


def parse_swap(swapusage: str) -> int | None:
    """Read ``sysctl -n vm.swapusage``: the swap in use, in bytes."""
    used = re.search(r"used = ([\d.]+)M", swapusage)
    return None if used is None else int(float(used.group(1)) * 2**20)


def swap_used() -> int:
    used = parse_swap(_out("sysctl", "-n", "vm.swapusage"))
    if used is None:
        raise RuntimeError("sysctl vm.swapusage gave no swap figure")
    return used


def parse_footprint(text: str) -> tuple[int, int] | None:
    """Read ``footprint -f bytes --noCategories``: phys_footprint and its peak."""
    now = re.search(r"phys_footprint: (\d+) B", text)
    peak = re.search(r"phys_footprint_peak: (\d+) B", text)
    return None if now is None or peak is None else (int(now.group(1)), int(peak.group(1)))


def footprint(pgid: int) -> tuple[int, int] | None:
    """phys_footprint and its peak summed over a process group, which counts compressed and
    swapped pages; None when no process of the group can be read."""
    found = [
        parse_footprint(_out("footprint", "-f", "bytes", "--noCategories", "-p", pid))
        for pid in _out("pgrep", "-g", str(pgid)).split()
    ]
    read = [f for f in found if f is not None]
    if not read:
        return None
    return sum(f[0] for f in read), sum(f[1] for f in read)


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


def powermetrics_running() -> bool:
    done = _run("pgrep", "-x", "powermetrics")
    return done is not None and done.returncode == 0


def violations(
    power: Power, *, sudo: bool, containers: int | None, powermetrics_running: bool
) -> list[str]:
    """Every B5 condition a session would break, in words. An unknown power state counts as
    broken; no Docker daemon means no containers. The lid and charging are recorded, not
    refused (decision 0016)."""
    found = []
    if power.on_ac is not True:
        found.append("not on AC power: plug in")
    if power.low_power_mode is not False:
        found.append("Low Power Mode is on or unknown: turn it off")
    if not sudo:
        found.append("powermetrics needs root: run `sudo -v` first, or allow it without a password")
    if containers:
        found.append(f"{containers} Docker containers are running: stop the compose stack")
    if powermetrics_running:
        found.append(
            "another powermetrics is running, perhaps left by a killed session: "
            "`sudo pkill -TERM -x powermetrics`"
        )
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
        # sudo relays the signal. Under SIGINT a root powermetrics outlived the spike's harness;
        # SIGTERM stopped it (decision 0016).
        self.proc.send_signal(signal.SIGTERM)
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


def main(argv: list[str] | None = None) -> None:
    """Write env.json for a run measured by another tool, such as `make terminal-perf`."""
    parser = argparse.ArgumentParser(description="Record the machine and run conditions.")
    parser.add_argument("out", type=Path, help="where to write env.json")
    args = parser.parse_args(argv)
    env = environment(used_powermetrics=False, started=datetime.now(UTC))
    args.out.write_text(env.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
