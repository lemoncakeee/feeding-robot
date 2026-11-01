"""Speech recognition backends."""

from __future__ import annotations

import io
import shutil
import time
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import requests

from feeding_robot.speech.server import LocalServer

SAMPLE_RATE = 16_000
DEFAULT_ASR_MODEL = "haas-ja-v1.0"
SPEECH_PORT = 18_080
LANGUAGE = "ja"
REQUEST_TIMEOUT_S = 10.0
INT16_MAX = 32_767


class ASRError(Exception):
    """Speech recognition failed."""


class ASRBackend(Protocol):
    def transcribe(self, audio: np.ndarray) -> tuple[str, float]: ...


def encode_wav(audio: np.ndarray) -> bytes:
    """Encode float samples in [-1, 1] as 16 kHz mono 16-bit WAV."""
    pcm = (np.clip(audio, -1.0, 1.0) * INT16_MAX).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def find_speech_binary() -> str:
    binary = shutil.which("speech") or Path.home() / ".local/bin/speech"
    if not Path(binary).exists():
        raise ASRError("speech.cpp is not installed")
    return str(binary)


class SpeechCppServer(LocalServer):
    """A speech.cpp server holding the given models."""

    def __init__(self, models: Sequence[str], port: int = SPEECH_PORT) -> None:
        super().__init__([find_speech_binary(), "serve", *models, "--port", str(port)], port)


class SpeechCppBackend:
    """Transcribes through a running speech.cpp server."""

    name = "speechcpp"

    def __init__(self, url: str, model: str = DEFAULT_ASR_MODEL) -> None:
        self.model = model
        self._url = f"{url}/v1/audio/transcriptions"

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        start = time.perf_counter()
        try:
            response = requests.post(
                self._url,
                files={"file": ("utterance.wav", encode_wav(audio), "audio/wav")},
                data={"language": LANGUAGE},
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            text = response.json()["text"]
        except (requests.RequestException, ValueError, KeyError) as error:
            raise ASRError(str(error)) from error
        return text.strip(), (time.perf_counter() - start) * 1000
