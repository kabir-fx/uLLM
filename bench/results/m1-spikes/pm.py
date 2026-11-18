"""Parse a powermetrics plist log (gzipped or not) into one small dict per sample.

    python pm.py <powermetrics.plist[.gz]> <out.json>

Each sample: t (unix seconds, sample end), el (seconds covered), pressure, freq (mean active MHz),
idle (GPU idle ratio), top (share of the sample at the 1470 MHz step), steps ({MHz: share}),
gpu_w, cpu_w, p_mhz, e_mhz, p_idle.
"""

import calendar
import gzip
import json
import plistlib
import sys
from pathlib import Path

TOP_STEP = 1470


def parse(path: Path) -> list[dict]:
    data = gzip.open(path).read() if path.suffix == ".gz" else path.read_bytes()
    out: list[dict] = []
    for part in data.split(b"\x00"):
        if not part.strip():
            continue
        try:
            s = plistlib.loads(part)
        except Exception:
            continue  # a sample cut off when powermetrics stopped
        g, p = s["gpu"], s["processor"]
        steps = {int(d["freq"]): d["used_ratio"] for d in g["dvfm_states"] if d["used_ratio"] > 0}
        # Mean requested and applied state index (P1 = 1). Applied below requested = capped.
        req = sum(int(d["sw_req_state"][1:]) * d["used_ratio"] for d in g.get("sw_requested_state", []))
        app = sum(int(d["sw_state"][4:]) * d["used_ratio"] for d in g.get("sw_state", []))
        req_n = sum(d["used_ratio"] for d in g.get("sw_requested_state", []))
        app_n = sum(d["used_ratio"] for d in g.get("sw_state", []))
        clusters = {c["name"]: c for c in p.get("clusters", [])}
        pc, ec = clusters.get("P-Cluster", {}), clusters.get("E-Cluster", {})
        out.append(
            {
                "t": calendar.timegm(s["timestamp"].timetuple()),
                "el": s["elapsed_ns"] / 1e9,
                "pressure": s["thermal_pressure"],
                "freq": g["freq_hz"],
                "idle": g["idle_ratio"],
                "top": steps.get(TOP_STEP, 0.0),
                "steps": steps,
                "req": round(req / req_n, 3) if req_n > 0.05 else None,
                "app": round(app / app_n, 3) if app_n > 0.05 else None,
                "gpu_w": p["gpu_power"] / 1000,
                "cpu_w": p["cpu_power"] / 1000,
                "p_mhz": pc.get("freq_hz", 0) / 1e6,
                "e_mhz": ec.get("freq_hz", 0) / 1e6,
                "p_idle": pc.get("idle_ratio"),
            }
        )
    # Timestamps have 1 s resolution; rebuild a smooth clock from the elapsed times.
    if out:
        t = float(out[0]["t"])
        for i, s in enumerate(out):
            if i:
                t += s["el"]
            s["t"] = round(t, 3)
    return out


if __name__ == "__main__":
    samples = parse(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(samples))
    print(f"{len(samples)} samples -> {sys.argv[2]}")
