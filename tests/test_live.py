"""Real Ollama planning. Run with --live."""

import pytest

from feeding_robot.planning.llm_backend import OllamaBackend
from feeding_robot.shared import SharedState, Step

pytestmark = pytest.mark.live
CASES = [
    ("ご飯ちょうだい", "replace", ["rice", "feed"]),
    ("ご飯ちょうらい", "replace", ["rice", "feed"]),
    ("ご飯と味噌汁を交互に2口ずつ", "replace", ["rice", "feed", "miso", "feed"] * 2),
    ("ご飯と味噌汁を交互に1口ずつ", "replace", ["rice", "feed", "miso", "feed"]),
    ("プリンと味噌汁を交互に1口ずつ", "replace", ["pudding", "feed", "miso", "feed"]),
    ("ご飯と煮物を交互に2口ずつ", "replace", ["rice", "feed", "nimono", "feed"] * 2),
    ("やっぱりプリンにして", "replace", ["pudding", "feed"]),
    ("熱そうだから少し冷まして", "keep", ["cool"]),
    ("一口小さめで", "keep", ["small"]),
    ("もうちょっと上", "keep", ["up"]),
    ("待って、まだ噛んでる", "keep", ["wait"]),
    ("もう大丈夫", "keep", ["resume"]),
    ("何か食べたい", "ask", []),
    ("今日いい天気だね", "keep", []),
]


@pytest.fixture(scope="module")
def backend(scene):
    backend = OllamaBackend(scene)
    backend.warm_up()
    return backend


@pytest.mark.parametrize(("text", "mode", "steps"), CASES)
def test_llm_plans(backend, text, mode, steps):
    shared = SharedState()
    shared.current_step = Step("scoop", {"food_id": "rice"})
    raw, _ = backend.plan(text, shared)
    assert raw == {"mode": mode, "steps": steps}
