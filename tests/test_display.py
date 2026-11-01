"""The display renders every pipeline event without needing a terminal."""

import io
import threading

from rich.console import Console

from feeding_robot.display import ConsoleSink, Display, PanelSink
from feeding_robot.shared import EventLog, SharedState


def rendered(scene, events) -> str:
    shared = SharedState(log=EventLog())
    shared.eaten["rice"] = 1
    output = io.StringIO()
    Display(scene, shared, (ConsoleSink(Console(file=output, width=120)),)).attach()
    for event, detail in events:
        shared.log.write(event, **detail)
    return output.getvalue()


def test_pipeline_events_are_shown(scene):
    text = rendered(
        scene,
        [
            ("asr", {"text": "ご飯ちょうだい", "ms": 50.4}),
            (
                "llm",
                {
                    "output": {"mode": "replace", "steps": ["rice", "feed"]},
                    "model": "test-model",
                    "ms": 1200.0,
                    "fallback": False,
                },
            ),
            ("plan_changed", {"reply": "ご飯をお持ちします。", "question": None}),
            ("robot_stopped", {"latency_ms": 3.2}),
            ("step_started", {"skill": "scoop", "args": {"food_id": "rice", "amount": "small"}}),
            ("step_started", {"skill": "cool", "args": {"seconds": 3.0}}),
            ("step_replaced", {"skill": "feed"}),
            ("food_eaten", {"food": "rice", "gap_cm": 1.2}),
        ],
    )
    for expected in (
        "ご飯ちょうだい",
        "rice → feed",
        "1200 ms",
        "3.2 ms",
        "ご飯（小）",
        "冷ます 3秒",
    ):
        assert expected in text


def test_fallback_is_labelled(scene):
    detail = {"output": {"mode": "keep", "steps": []}, "model": "m", "ms": 1.0, "fallback": True}
    assert "fallback" in rendered(scene, [("llm", detail)])


def test_unknown_events_are_ignored(scene):
    assert rendered(scene, [("speech_end", {})]) == ""


def test_panel_draws_new_lines_once(scene):
    shared = SharedState(log=EventLog())
    panel = PanelSink()
    Display(scene, shared, (panel,)).attach()
    blank = panel.take_image()
    assert panel.take_image() is None
    shared.log.write("asr", text="ご飯ちょうだい", ms=50.0)
    drawn = panel.take_image()
    assert drawn.shape == blank.shape
    assert (drawn != blank).any()


def test_panel_keeps_long_text_inside_the_image(scene):
    panel = PanelSink()
    shared = SharedState(log=EventLog())
    Display(scene, shared, (panel,)).attach()
    shared.log.write("asr", text="ご飯" * 80, ms=50.0)
    assert panel.take_image() is not None


def test_panel_survives_events_from_several_threads(scene):
    panel = PanelSink()
    shared = SharedState(log=EventLog())
    Display(scene, shared, (panel,)).attach()
    errors = []

    def spam() -> None:
        try:
            for _ in range(100):
                shared.log.write("asr", text="ご飯ちょうだい", ms=1.0)
        except RuntimeError as error:
            errors.append(error)

    threads = [threading.Thread(target=spam) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
