"""Speech caching, fallback and muting, without audio hardware."""

import pytest

from feeding_robot.planning.planner import build_plan, common_phrases
from feeding_robot.shared import SharedState
from feeding_robot.speech import tts
from feeding_robot.speech.tts import Speaker, TTSError


class CountingSynthesizer:
    identity = "test-voice"

    def __init__(self, error: bool = False) -> None:
        self.calls = 0
        self.error = error

    def synthesize(self, text):
        self.calls += 1
        if self.error:
            raise TTSError("server down")
        return b"wav"


@pytest.fixture
def played(monkeypatch):
    log = {"wav": 0}

    def play(audio):
        log["wav"] += 1

    monkeypatch.setattr(tts, "play_wav", play)
    return log


def test_repeated_phrase_is_synthesized_once(played, tmp_path):
    synthesizer = CountingSynthesizer()
    speaker = Speaker(SharedState(), synthesizer, tmp_path)
    speaker.speak("こんにちは")
    speaker.speak("こんにちは")
    assert synthesizer.calls == 1
    assert played["wav"] == 2


def test_synthesis_failure_skips_the_phrase(played, tmp_path):
    speaker = Speaker(SharedState(), CountingSynthesizer(error=True), tmp_path)
    speaker.speak("こんにちは")
    assert played == {"wav": 0}


def test_listening_is_muted_only_during_playback(monkeypatch, tmp_path):
    shared = SharedState()
    seen = []

    class Synth:
        identity = "test-voice"

        def synthesize(self, text):
            seen.append(("synthesis", shared.tts_playing.is_set()))
            return b"wav"

    monkeypatch.setattr(
        tts, "play_wav", lambda audio: seen.append(("play", shared.tts_playing.is_set()))
    )
    Speaker(shared, Synth(), tmp_path).speak("こんにちは")
    assert seen == [("synthesis", False), ("play", True)]
    assert not shared.tts_playing.is_set()


def test_common_phrases_include_each_food(scene):
    phrases = common_phrases(scene)
    assert all(any(food.name in phrase for phrase in phrases) for food in scene.foods.values())


def test_voice_is_pinned_on_every_request(monkeypatch):
    bodies = []

    class Response:
        content = b"wav"

        def raise_for_status(self):
            pass

    def post(url, json, timeout):
        bodies.append(json)
        return Response()

    monkeypatch.setattr(tts.requests, "post", post)
    synthesizer = tts.OmniVoiceSynthesizer("http://local")
    synthesizer.synthesize("一つ目")
    synthesizer.synthesize("二つ目")
    assert [(b["voice"], b["seed"]) for b in bodies] == [(tts.CLONED_VOICE, tts.VOICE_SEED)] * 2


def test_reference_voice_has_its_transcript():
    assert tts.REFERENCE_VOICE.exists()
    assert tts.REFERENCE_VOICE.with_suffix(".txt").read_text(encoding="utf-8").strip()


def test_audio_survives_restart_on_disk(played, tmp_path):
    first = CountingSynthesizer()
    Speaker(SharedState(), first, tmp_path).speak("こんにちは")
    second = CountingSynthesizer()
    Speaker(SharedState(), second, tmp_path).speak("こんにちは")
    assert (first.calls, second.calls) == (1, 0)


def test_changing_the_voice_does_not_reuse_audio(played, tmp_path):
    first = CountingSynthesizer()
    Speaker(SharedState(), first, tmp_path).speak("こんにちは")
    other = CountingSynthesizer()
    other.identity = "another-voice"
    Speaker(SharedState(), other, tmp_path).speak("こんにちは")
    assert other.calls == 1


def test_reply_templates_are_all_prepared(scene):
    phrases = common_phrases(scene)
    shared = SharedState()
    plan = build_plan({"mode": "replace", "steps": ["miso", "feed", "rice", "feed"]}, shared, scene)
    assert plan.reply in phrases
    shared.eaten["rice"] = 1
    status = build_plan({"mode": "keep", "steps": ["status"]}, shared, scene)
    assert status.reply in phrases
    for food in scene.foods:
        shared.eaten[food] = 1
    status = build_plan({"mode": "keep", "steps": ["status"]}, shared, scene)
    assert status.reply in phrases


def test_chosen_take_replaces_only_its_phrase(played, tmp_path, monkeypatch):
    plain = CountingSynthesizer()
    Speaker(SharedState(), plain, tmp_path).speak("プリンをお持ちします。")
    Speaker(SharedState(), plain, tmp_path).speak("ご飯をお持ちします。")
    monkeypatch.setitem(tts.PHRASE_SEEDS, "ご飯をお持ちします。", 7)
    retake = CountingSynthesizer()
    Speaker(SharedState(), retake, tmp_path).speak("ご飯をお持ちします。")
    assert retake.calls == 1
    assert tts.seed_for("ご飯をお持ちします。") == 7
    assert tts.seed_for("味噌汁をお持ちします。") == tts.VOICE_SEED
