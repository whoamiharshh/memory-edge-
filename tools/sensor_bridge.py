"""Bridge a real sensor to a device: reads samples from a serial port (e.g. an ESP32 + ADXL345 / MPU-6050 printing
"ax,ay,az" lines), a CSV file or stdin, and posts them in chunks to the device's /api/ingest/signal.

  .venv\\Scripts\\python.exe tools\\sensor_bridge.py --device https://127.0.0.1:8101 --token <operator> \\
        --serial COM5 --baud 921600 --fs 1600 --axes 3          (needs: uv pip install pyserial)
  .venv\\Scripts\\python.exe tools\\sensor_bridge.py --device ... --csv recording.csv --fs 1600

Why: a phone's motion sensor is capped at ~60 Hz in browsers; a ~$5 MEMS accelerometer on a microcontroller gives
0.8-3.2 kHz, enough for imbalance, misalignment and (with >= 2 kHz) early bearing-defect envelopes. Use the profile
that matches the rate: lowrate-accel (< 1 kHz, m/s^2) or rotating-hf (>= 2 kHz, g).
STATUS: the parser and the chunk/post logic are unit-tested (tests/unit/test_sensor_bridge.py); it has NOT been run
against physical hardware in this project.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable, Iterator

import httpx


def parse_line(line: str, axes: int) -> list[float] | None:
    """'0.01, -0.02, 9.81' / tab / space separated -> floats; None for headers, blanks or malformed lines."""
    parts = [p for p in line.replace("\t", ",").replace(";", ",").replace(" ", ",").split(",") if p]
    if len(parts) < axes:
        return None
    try:
        vals = [float(p) for p in parts[:axes]]
    except ValueError:
        return None
    return vals if all(abs(v) < 1e6 for v in vals) else None


def chunks(lines: Iterable[str], axes: int, size: int) -> Iterator[list[list[float]]]:
    buf: list[list[float]] = []
    for line in lines:
        v = parse_line(line, axes)
        if v is None:
            continue
        buf.append(v)
        if len(buf) >= size:
            yield buf
            buf = []


def body(chunk: list[list[float]], fs: float, rpm: float | None, source: str) -> dict:
    b = {"fs": fs, "source": source}
    if rpm:
        b["rpm"] = rpm
    if len(chunk[0]) == 1:
        b["samples"] = [r[0] for r in chunk]
    else:
        b["axes"] = chunk
    return b


def serial_lines(port: str, baud: int) -> Iterator[str]:
    try:
        import serial  # pyserial (optional)
    except ImportError:
        raise SystemExit("serial input needs pyserial: $env:VIRTUAL_ENV='.venv'; uv pip install pyserial")
    with serial.Serial(port, baud, timeout=2) as s:
        while True:
            yield s.readline().decode("ascii", errors="ignore")


def main(argv: list[str] | None = None) -> None:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--device", required=True)
    a.add_argument("--token", required=True, help="the device's operator token")
    a.add_argument("--fs", type=float, required=True, help="the sensor's sampling rate in Hz")
    a.add_argument("--axes", type=int, default=3)
    a.add_argument("--rpm", type=float, default=None)
    a.add_argument("--seconds", type=float, default=1.0, help="seconds of signal per request")
    a.add_argument("--ca", default=None, help="CA file for an https device (runtime/tls/ca.pem)")
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--serial")
    src.add_argument("--csv")
    src.add_argument("--stdin", action="store_true")
    a.add_argument("--baud", type=int, default=921600)
    p = a.parse_args(argv)
    lines = (serial_lines(p.serial, p.baud) if p.serial else open(p.csv, encoding="utf-8") if p.csv else sys.stdin)
    size = max(16, int(p.fs * p.seconds))
    with httpx.Client(base_url=p.device, verify=p.ca or True, timeout=30,
                      headers={"X-Operator-Token": p.token}) as c:
        for ch in chunks(lines, p.axes, size):
            t = time.time()
            r = c.post("/api/ingest/signal", json=body(ch, p.fs, p.rpm, f"bridge:{p.serial or p.csv or 'stdin'}"))
            last = (r.json() or {}).get("last") or {} if r.status_code == 200 else {}
            print(f"{len(ch)} samples -> HTTP {r.status_code} {last.get('state', r.text[:120])} "
                  f"({(time.time() - t) * 1000:.0f} ms)", flush=True)


if __name__ == "__main__":
    main()
