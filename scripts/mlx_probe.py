"""Report what MLX can use on this machine, as one JSON object.

Answers F2's first question for any runner: does MLX run here, on the GPU or
only the CPU, and do the two agree on a small computation? CI attaches the
output to the job summary so the answer is checkable.

    uv run python scripts/mlx_probe.py
"""

import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import mlx.core as mx


def _sh(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _matmul(device: mx.Device) -> dict[str, Any]:
    """Time one bf16 1024x1024 matmul on a device, after a warm-up."""
    try:
        with mx.stream(device):
            key_a, key_b = mx.random.split(mx.random.key(0))
            a = mx.random.normal((1024, 1024), key=key_a).astype(mx.bfloat16)
            b = mx.random.normal((1024, 1024), key=key_b).astype(mx.bfloat16)
            mx.eval(a @ b)  # warm-up: compiles kernels, allocates
            start = time.perf_counter()
            c = a @ b
            mx.eval(c)
            elapsed_ms = (time.perf_counter() - start) * 1000
            checksum = float(c.astype(mx.float32).abs().sum())
        return {"ok": True, "ms": round(elapsed_ms, 3), "abs_sum": checksum}
    except Exception as exc:  # the point is to report any failure, whatever it is
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def probe() -> dict[str, Any]:
    gpu_available = mx.is_available(mx.gpu)
    report: dict[str, Any] = {
        "mlx": mx.__version__,
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "cpu_brand": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_count": os.cpu_count(),
        "memory_bytes": int(_sh("sysctl", "-n", "hw.memsize") or 0),
        # GitHub's hosted runners set these mixed-case names.
        "runner_image": os.environ.get("ImageOS"),  # noqa: SIM112
        "runner_image_version": os.environ.get("ImageVersion"),  # noqa: SIM112
        "default_device": str(mx.default_device()),
        "gpu_available": gpu_available,
        "metal_available": mx.metal.is_available(),
    }
    try:
        report["device_info"] = mx.device_info()
    except Exception as exc:
        report["device_info"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["cpu_matmul"] = _matmul(mx.cpu)
    report["gpu_matmul"] = _matmul(mx.gpu) if gpu_available else {"ok": False, "error": "no GPU"}
    cpu, gpu = report["cpu_matmul"], report["gpu_matmul"]
    if cpu["ok"] and gpu["ok"]:
        report["gpu_cpu_rel_diff"] = abs(cpu["abs_sum"] - gpu["abs_sum"]) / cpu["abs_sum"]
    return report


if __name__ == "__main__":
    result = probe()
    json.dump(result, sys.stdout, indent=2, default=str)
    print()
    sys.exit(0 if result["cpu_matmul"]["ok"] else 1)
