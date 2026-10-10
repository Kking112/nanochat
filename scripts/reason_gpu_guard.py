"""Run one owned process group while enforcing a total-device VRAM ceiling."""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence


def vram_fraction(gpu_index: int) -> float:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    used, total = (float(value.strip()) for value in result.stdout.strip().split(","))
    if not math.isfinite(used) or not math.isfinite(total) or not 0 <= used <= total or total <= 0:
        raise ValueError("Invalid GPU memory reading")
    return used / total


def stop_owned_process(process: subprocess.Popen) -> None:
    """The child starts a new session; never signal an unrelated training PID."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return  # The child finished between the poll and signal.
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_guarded(command: Sequence[str], limit_percent: float = 80.0,
                gpu_index: int = 0, poll_seconds: float = 1.0) -> int:
    if not command or not 0 < limit_percent <= 80 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("Command required; VRAM limit must be in (0,80] and polling interval positive")
    limit = limit_percent / 100
    if vram_fraction(gpu_index) > limit:
        print("GPU VRAM is above the limit; no project process was started.", file=sys.stderr)
        return 75
    process = subprocess.Popen(list(command), start_new_session=True)
    try:
        while process.poll() is None:
            if vram_fraction(gpu_index) > limit:
                print(f"Total GPU VRAM exceeded {limit_percent:g}%; stopping this project's process only.", file=sys.stderr)
                stop_owned_process(process)
                return 75
            time.sleep(poll_seconds)
        return process.returncode
    finally:
        # Also fail closed if monitoring fails or the wrapper is interrupted.
        stop_owned_process(process)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-percent", type=float, default=80)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_guarded(command, args.limit_percent, args.gpu_index, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
