"""Speech output. Playback runs on its own thread and mutes speech detection."""

from __future__ import annotations

import base64
import hashlib
import logging
import subprocess
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import requests

from feeding_robot.shared import SharedState
from feeding_robot.speech.server import LocalServer

logger = logging.getLogger(__name__)

PLAY_COMMAND = ("afplay",)
VOICE_SEED = 42  # a fixed seed keeps the delivery consistent across phrases
# Takes chosen by ear, for phrases that sound unnatural with the default seed.
PHRASE_SEEDS = {"プリンをお持ちします。": 99}
SYNTHESIS_TIMEOUT_S = 30.0
OMNIVOICE_HOME = Path.home() / ".local/share/omnivoice.cpp"
OMNIVOICE_MODEL = "omnivoice-base-Q8_0.gguf"
OMNIVOICE_CODEC = "omnivoice-tokenizer-F32.gguf"
OMNIVOICE_LANGUAGE = "Japanese"
OMNIVOICE_PORT = 18_081
CLONED_VOICE = "calm"
SPEECH_CACHE = Path.home() / ".cache/feeding-robot/speech"
CACHE_KEY_LENGTH = 16
REFERENCE_VOICE = Path(__file__).parents[2] / "assets/voice/calm_female.wav"


class TTSError(Exception):
    """Speech synthesis failed."""


class Synthesizer(Protocol):
    identity: str

    def synthesize(self, text: str) -> bytes: ...


def seed_for(text: str) -> int:
    return PHRASE_SEEDS.get(text, VOICE_SEED)


def _post_speech(url: str, body: dict[str, object]) -> bytes:
    try:
        response = requests.post(url, json=body, timeout=SYNTHESIS_TIMEOUT_S)
        response.raise_for_status()
    except requests.RequestException as error:
        raise TTSError(str(error)) from error
    return response.content


class OmniVoiceServer(LocalServer):
    """An OmniVoice server holding one voice cloned from a reference recording."""

    def __init__(self, home: Path = OMNIVOICE_HOME, port: int = OMNIVOICE_PORT) -> None:
        binary = home / "build" / "tts-server"
        if not binary.exists():
            raise TTSError(f"OmniVoice is not built at {home}")
        command = [
            binary,
            "--model",
            home / "models" / OMNIVOICE_MODEL,
            "--codec",
            home / "models" / OMNIVOICE_CODEC,
            "--lang",
            OMNIVOICE_LANGUAGE,
            "--port",
            port,
        ]
        super().__init__([str(part) for part in command], port)

    def clone_voice(self, reference: Path = REFERENCE_VOICE) -> None:
        """Register the reference, so that every phrase uses the same speaker."""
        body = {
            "name": CLONED_VOICE,
            "ref_text": reference.with_suffix(".txt").read_text(encoding="utf-8").strip(),
            "wav_b64": base64.b64encode(reference.read_bytes()).decode("ascii"),
        }
        _post_speech(f"{self.url}/v1/audio/voices", body)


class OmniVoiceSynthesizer:
    """Synthesizes WAV audio in the cloned voice."""

    def __init__(self, url: str, reference: Path = REFERENCE_VOICE) -> None:
        self._url = f"{url}/v1/audio/speech"
        digest = hashlib.sha256(reference.read_bytes()).hexdigest()[:CACHE_KEY_LENGTH]
        self.identity = f"omnivoice-{digest}-{VOICE_SEED}"

    def synthesize(self, text: str) -> bytes:
        body = {
            "input": text,
            "voice": CLONED_VOICE,
            "language": OMNIVOICE_LANGUAGE,
            "response_format": "wav",
            "seed": seed_for(text),
        }
        return _post_speech(self._url, body)


def play_wav(audio: bytes) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "speech.wav"
        path.write_bytes(audio)
        subprocess.run([*PLAY_COMMAND, str(path)], check=False)


class Speaker:
    """Speaks queued text, using cached audio when it has any."""

    def __init__(
        self,
        shared: SharedState,
        synthesizer: Synthesizer,
        cache_dir: Path = SPEECH_CACHE,
    ) -> None:
        self._shared = shared
        self._synthesizer = synthesizer
        self._cache_dir = cache_dir
        self._cache: dict[str, bytes] = {}

    def start(self, phrases: Iterable[str] = ()) -> None:
        """Start speaking, and prepare phrases in the background so they play instantly."""
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._prepare, args=(tuple(phrases),), daemon=True).start()

    def _prepare(self, phrases: tuple[str, ...]) -> None:
        for text in phrases:
            try:
                self._synthesize(text)
            except TTSError as error:
                logger.warning("could not prepare speech: %s", error)
                return
        logger.info("prepared %d phrases", len(phrases))

    def _cache_path(self, text: str) -> Path:
        key = f"{self._synthesizer.identity}:{text}"
        if text in PHRASE_SEEDS:
            key += f":{seed_for(text)}"
        key = key.encode()
        return self._cache_dir / f"{hashlib.sha256(key).hexdigest()}.wav"

    def _synthesize(self, text: str) -> bytes:
        """Return audio for text from memory, then disk, then the synthesizer."""
        if text in self._cache:
            return self._cache[text]
        path = self._cache_path(text)
        if path.exists():
            audio = path.read_bytes()
        else:
            audio = self._synthesizer.synthesize(text)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        self._cache[text] = audio
        return audio

    def _run(self) -> None:
        while True:
            self.speak(self._shared.speech.get())

    def speak(self, text: str) -> None:
        """Speak one phrase. Listening is muted only during playback, not synthesis."""
        try:
            audio = self._synthesize(text)
        except TTSError as error:
            logger.warning("speech synthesis failed: %s", error)
            return
        playing = self._shared.tts_playing
        playing.set()
        try:
            play_wav(audio)
        finally:
            playing.clear()
