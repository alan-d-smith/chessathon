import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

from harness.rules import STDOUT_CAP, WATCHDOG_GRACE_MS

RUNNER = Path(__file__).resolve().parent / "runner.py"


class AgentFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def local(directory: Path) -> "Agent":
    """Run an agent as a process on this machine, through the platform's runner."""
    return Agent([sys.executable, str(RUNNER), str(directory.resolve())])


class Agent:
    """One agent process, spoken to exactly as the platform speaks to a container."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.stderr_tail = ""
        self._process: subprocess.Popen[bytes] | None = None
        self._selector = selectors.DefaultSelector()
        self._buffer = b""
        self._tail = b""

    def start(self, init_budget_s: float) -> None:
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._process = process
        self._selector.register(_pipe(process.stdout), selectors.EVENT_READ, "stdout")
        self._selector.register(_pipe(process.stderr), selectors.EVENT_READ, "stderr")
        ready = self._await_line(time.monotonic() + init_budget_s)
        if ready is None:
            raise AgentFailure("init" if process.poll() is None else "crash")
        if not _is_ready(ready):
            raise AgentFailure("init")

    def move(self, fen: str, time_left_ms: int) -> str:
        if self._process is None:
            raise RuntimeError("agent moved before start")
        request = json.dumps({"fen": fen, "time_left_ms": time_left_ms}).encode()
        try:
            _pipe(self._process.stdin).write(request + b"\n")
        except BrokenPipeError:
            raise AgentFailure("crash") from None
        line = self._await_line(time.monotonic() + (time_left_ms + WATCHDOG_GRACE_MS) / 1000.0)
        if line is None:
            raise AgentFailure("flag")
        return _parse_move(line)

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.kill()
        self._drain()
        self.stderr_tail = self._tail.decode("utf-8", "replace")
        self._selector.close()
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()
        self._process.wait()
        self._process = None

    def _await_line(self, deadline: float) -> bytes | None:
        while b"\n" not in self._buffer:
            if len(self._buffer) >= STDOUT_CAP:
                raise AgentFailure("illegal")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            for key, _ in self._selector.select(remaining):
                chunk = os.read(key.fd, STDOUT_CAP)
                if key.data == "stderr":
                    self._keep(key, chunk)
                elif not chunk:
                    raise AgentFailure("crash")
                else:
                    self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    # the writer is dead by now, so the pipes hold a bounded amount and this terminates
    def _drain(self) -> None:
        while self._selector.get_map():
            events = self._selector.select(0)
            if not events:
                return
            for key, _ in events:
                chunk = os.read(key.fd, STDOUT_CAP)
                if key.data == "stderr":
                    self._keep(key, chunk)
                elif not chunk:
                    self._selector.unregister(key.fileobj)

    def _keep(self, key: selectors.SelectorKey, chunk: bytes) -> None:
        if not chunk:
            self._selector.unregister(key.fileobj)
            return
        self._tail += chunk


def _pipe(stream: IO[bytes] | None) -> IO[bytes]:
    if stream is None:
        raise RuntimeError("the agent process exposed no pipe")
    return stream


def _is_ready(line: bytes) -> bool:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("ready") is True


def _parse_move(line: bytes) -> str:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AgentFailure("illegal") from None
    move = payload.get("move") if isinstance(payload, dict) else None
    if not isinstance(move, str):
        raise AgentFailure("illegal")
    return move
