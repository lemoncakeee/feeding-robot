"""Audio input: speech detection for the reflex path and recognition for planning.

Speech onset reaches the shared state from the detection thread directly, so stopping the
robot never waits for recognition or planning.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

from feeding_robot.shared import SharedState
from feeding_robot.speech.asr_backend import SAMPLE_RATE, ASRBackend, ASRError

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 512
CHUNK_S = CHUNK_SAMPLES / SAMPLE_RATE
MIC_LATENCY_S = 0.05
SPEECH_THRESHOLD = 0.5
ONSET_S = 0.096
END_SILENCE_S = 0.5
PRE_ROLL_S = 0.32
MAX_UTTERANCE_S = 15.0
_NO_WORDS = re.compile(r"[\s。、．，.,!?！？]*")


class UtteranceSegmenter:
    """Finds utterances in a stream of chunks labelled as speech or not."""

    def __init__(
        self,
        on_onset: Callable[[], None],
        on_utterance: Callable[[np.ndarray], None],
    ) -> None:
        self._on_onset = on_onset
        self._on_utterance = on_utterance
        self._onset_chunks = round(ONSET_S / CHUNK_S)
        self._end_chunks = round(END_SILENCE_S / CHUNK_S)
        self._max_chunks = round(MAX_UTTERANCE_S / CHUNK_S)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=round(PRE_ROLL_S / CHUNK_S))
        self._chunks: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0

    @property
    def in_utterance(self) -> bool:
        return bool(self._chunks)

    def reset(self) -> None:
        """Drop any partial utterance and release its pause."""
        if self.in_utterance:
            self._on_utterance(np.zeros(0, dtype=np.float32))
        self._pre_roll.clear()
        self._chunks = []
        self._speech_run = self._silence_run = 0

    def feed(self, chunk: np.ndarray, is_speech: bool) -> None:
        if self.in_utterance:
            self._continue_utterance(chunk, is_speech)
            return
        self._pre_roll.append(chunk)
        self._speech_run = self._speech_run + 1 if is_speech else 0
        if self._speech_run >= self._onset_chunks:
            self._on_onset()
            self._chunks = list(self._pre_roll)
            self._pre_roll.clear()
            self._speech_run = self._silence_run = 0

    def _continue_utterance(self, chunk: np.ndarray, is_speech: bool) -> None:
        self._chunks.append(chunk)
        self._silence_run = 0 if is_speech else self._silence_run + 1
        if self._silence_run >= self._end_chunks or len(self._chunks) >= self._max_chunks:
            audio = np.concatenate(self._chunks)
            self._chunks = []
            self._silence_run = 0
            self._on_utterance(audio)


class ASRWorker:
    """Transcribes utterances on its own thread and forwards the text to the planner."""

    def __init__(
        self, backend: ASRBackend, shared: SharedState, submit: Callable[[str], None]
    ) -> None:
        self._backend = backend
        self._shared = shared
        self._submit = submit
        self._utterances: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()

    def enqueue(self, audio: np.ndarray) -> None:
        self._utterances.put(audio)

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            self.handle(self._utterances.get())

    def handle(self, audio: np.ndarray) -> None:
        """Transcribe one utterance. Without text for the planner, release the pause here."""
        shared = self._shared
        submitted = False
        try:
            if audio.size == 0:
                return
            text, elapsed_ms = self._backend.transcribe(audio)
            backend = getattr(self._backend, "model", "unknown")
            shared.log.write("asr", text=text, backend=backend, ms=round(elapsed_ms, 1))
            if not _NO_WORDS.fullmatch(text):
                self._submit(text)
                submitted = True
        except ASRError as error:
            logger.warning("recognition failed: %s", error)
            shared.log.write("asr_failed", error=str(error))
        finally:
            if not submitted:
                shared.end_utterance()


def read_wav(path: Path) -> np.ndarray:
    """Load a 16 kHz mono 16-bit WAV file as float samples."""
    with wave.open(str(path), "rb") as file:
        if (file.getframerate(), file.getnchannels(), file.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path} must be {SAMPLE_RATE} Hz mono 16-bit")
        pcm = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32_768


class AudioFrontend:
    """Runs speech detection on chunks from the microphone or from a replayed file."""

    def __init__(
        self,
        shared: SharedState,
        segmenter: UtteranceSegmenter,
        inject: Path | None = None,
    ) -> None:
        self._shared = shared
        self._segmenter = segmenter
        self._inject = inject
        self._chunks: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()
        self._vad = load_silero_vad()
        self._closed = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._detect, daemon=True).start()
        source = self._read_mic if self._inject is None else self._replay
        threading.Thread(target=source, daemon=True).start()

    def close(self) -> None:
        self._closed.set()

    def _deliver(self, chunk: np.ndarray) -> None:
        self._chunks.put(chunk)

    def _read_mic(self) -> None:
        """Read the mic with blocking calls, which lose nothing when the main loop is busy.

        A stream callback needs the interpreter lock in time, and silently lost a fifth of
        the audio while the simulator was running.
        """
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            latency=MIC_LATENCY_S,
        ) as stream:
            while not self._closed.is_set():
                data, overflowed = stream.read(CHUNK_SAMPLES)
                if overflowed:
                    logger.warning("audio input overflowed")
                self._deliver(data[:, 0].copy())

    def _replay(self) -> None:
        """Feed a file in real time, then silence, as a microphone would."""
        audio = read_wav(self._inject)
        for start in range(0, audio.size - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            self._deliver(audio[start : start + CHUNK_SAMPLES])
            time.sleep(CHUNK_S)
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        while True:
            self._deliver(silence)
            time.sleep(CHUNK_S)

    def _detect(self) -> None:
        while True:
            chunk = self._chunks.get()
            if self._shared.tts_playing.is_set():
                self._vad.reset_states()
                self._segmenter.reset()
                continue
            with torch.no_grad():
                probability = self._vad(torch.from_numpy(chunk), SAMPLE_RATE).item()
            self._segmenter.feed(chunk, probability >= SPEECH_THRESHOLD)
