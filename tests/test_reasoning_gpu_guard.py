import signal
import subprocess
from typing import Any

import pytest

from scripts import reason_gpu_guard as guard


class OwnedProcess:
    pid = 12345
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = -signal.SIGTERM
        return self.returncode


def test_above_limit_never_starts_a_process(monkeypatch: Any) -> None:
    monkeypatch.setattr(guard, "vram_fraction", lambda _: 0.81)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    assert guard.run_guarded(["fixture"]) == 75


def test_limit_stops_only_the_owned_new_process_group(monkeypatch: Any) -> None:
    readings = iter([0.4, 0.8, 0.81])
    monkeypatch.setattr(guard, "vram_fraction", lambda _: next(readings))
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    process = OwnedProcess()
    signals: list[tuple[int, int]] = []

    def launch(command: list[str], start_new_session: bool) -> OwnedProcess:
        assert command == ["fixture"] and start_new_session
        return process

    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    monkeypatch.setattr(guard.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert guard.run_guarded(["fixture"]) == 75
    assert signals == [(process.pid, signal.SIGTERM)]


def test_monitor_failure_stops_owned_child(monkeypatch: Any) -> None:
    calls = 0

    def query(_: int) -> float:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise subprocess.TimeoutExpired("nvidia-smi", 10)
        return 0.2

    process = OwnedProcess()
    signals: list[int] = []
    monkeypatch.setattr(guard, "vram_fraction", query)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(guard.os, "killpg", lambda pid, sig: signals.append(pid))
    with pytest.raises(subprocess.TimeoutExpired):
        guard.run_guarded(["fixture"])
    assert signals == [process.pid]
