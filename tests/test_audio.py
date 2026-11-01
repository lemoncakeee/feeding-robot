"""Utterance segmentation and recognition failure handling, without audio hardware."""

import io
import wave

import numpy as np
import pytest

from feeding_robot.shared import SharedState
from feeding_robot.speech.asr_backend import SAMPLE_RATE, ASRError, encode_wav
from feeding_robot.speech.audio import (
    CHUNK_S,
    CHUNK_SAMPLES,
    END_SILENCE_S,
    ASRWorker,
    UtteranceSegmenter,
)

END_CHUNKS = round(END_SILENCE_S / CHUNK_S)


class Recorder:
    def __init__(self) -> None:
        self.onsets = 0
        self.utterances: list[np.ndarray] = []
        self.segmenter = UtteranceSegmenter(self.onset, self.utterances.append)

    def onset(self) -> None:
        self.onsets += 1

    def feed(self, *labels: bool) -> None:
        for label in labels:
            self.segmenter.feed(np.zeros(CHUNK_SAMPLES, dtype=np.float32), label)


def test_onset_fires_at_third_consecutive_speech_chunk():
    recorder = Recorder()
    recorder.feed(True, True)
    assert recorder.onsets == 0
    recorder.feed(True)
    assert recorder.onsets == 1


def test_short_noise_does_not_trigger_onset():
    recorder = Recorder()
    recorder.feed(True, True, False, True, True, False)
    assert recorder.onsets == 0


def test_utterance_ends_after_silence_and_includes_pre_roll():
    recorder = Recorder()
    recorder.feed(*[False] * 5, *[True] * 10, *[False] * END_CHUNKS)
    assert recorder.onsets == 1
    assert len(recorder.utterances) == 1
    assert recorder.utterances[0].size >= (5 + 10) * CHUNK_SAMPLES


def test_reset_mid_utterance_releases_with_empty_audio():
    recorder = Recorder()
    recorder.feed(True, True, True)
    recorder.segmenter.reset()
    assert recorder.utterances[0].size == 0


def test_encoded_wav_is_16k_mono_16bit():
    audio = np.linspace(-1, 1, SAMPLE_RATE, dtype=np.float32)
    with wave.open(io.BytesIO(encode_wav(audio))) as file:
        assert file.getparams()[:4] == (1, 2, SAMPLE_RATE, SAMPLE_RATE)


class StubASR:
    def __init__(self, result) -> None:
        self.result = result

    def transcribe(self, audio):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result, 1.0


@pytest.mark.parametrize(
    "result", [ASRError("server down"), "", "。", "  、"], ids=["error", "empty", "dot", "spaces"]
)
def test_no_usable_text_releases_pause(result):
    shared = SharedState()
    submitted = []
    shared.on_speech_onset()
    ASRWorker(StubASR(result), shared, submitted.append).handle(np.ones(10, dtype=np.float32))
    assert submitted == []
    assert not shared.paused.is_set()


def test_text_is_submitted_and_pause_left_to_the_planner():
    shared = SharedState()
    submitted = []
    shared.on_speech_onset()
    ASRWorker(StubASR("ご飯"), shared, submitted.append).handle(np.ones(10, dtype=np.float32))
    assert submitted == ["ご飯"]
    assert shared.paused.is_set()


def test_pause_stays_while_a_second_utterance_is_in_progress():
    shared = SharedState()
    shared.on_speech_onset()
    shared.on_speech_onset()
    shared.end_utterance()
    assert shared.paused.is_set()
    shared.end_utterance()
    assert not shared.paused.is_set()
