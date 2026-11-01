"""Lifecycle of local HTTP servers that run as subprocesses."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Self

import requests

STARTUP_TIMEOUT_S = 300.0
HEALTH_POLL_S = 0.25
SHUTDOWN_TIMEOUT_S = 5.0


class ServerError(Exception):
    """A local server failed to start."""


class LocalServer:
    """Runs a server for the duration of a with block, once its /health answers."""

    def __init__(self, command: Sequence[str], port: int) -> None:
        self.url = f"http://127.0.0.1:{port}"
        self._command = list(command)
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        self._process = subprocess.Popen(
            self._command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            self._wait_until_healthy()
        except ServerError:
            self._stop()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop()

    def _stop(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._process.kill()

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise ServerError(f"{self._command[0]} exited: {self._process.returncode}")
            try:
                if requests.get(f"{self.url}/health", timeout=1).ok:
                    return
            except requests.RequestException:
                time.sleep(HEALTH_POLL_S)
        raise ServerError(f"{self._command[0]} did not become healthy in time")
