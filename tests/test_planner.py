"""Plan expansion and the rule fallback."""

import pytest

from feeding_robot.planning.planner import PlanningError, PlanWorker, build_plan, rule_parse
from feeding_robot.shared import SharedState, Step


def scoop(food_id, amount=None):
    return Step("scoop", {"food_id": food_id, "amount": amount})


def test_words_expand_to_steps(scene):
    raw = {"mode": "replace", "steps": ["small", "rice", "feed", "miso", "feed"]}
    plan = build_plan(raw, SharedState(), scene)
    assert plan.steps == (scoop("rice", "small"), Step("feed"), scoop("miso"), Step("feed"))
    assert plan.reply == "ご飯と味噌汁をお持ちします。"


def test_scoop_without_feed_is_completed(scene):
    plan = build_plan({"mode": "replace", "steps": ["rice"]}, SharedState(), scene)
    assert plan.steps == (scoop("rice"), Step("feed"))


def test_task_words_under_keep_still_run(scene):
    plan = build_plan({"mode": "keep", "steps": ["rice", "feed"]}, SharedState(), scene)
    assert plan.mode == "replace"


def test_modifier_under_replace_does_not_clear_the_plan(scene):
    plan = build_plan({"mode": "replace", "steps": ["cool"]}, SharedState(), scene)
    assert plan.mode == "keep"
    assert plan.inserts == (Step("cool", {"seconds": 3.0}),)


def test_trailing_amount_applies_to_next_scoop(scene):
    plan = build_plan({"mode": "keep", "steps": ["small"]}, SharedState(), scene)
    assert plan.control.amount == "small"


def test_ask_produces_a_question_listing_foods(scene):
    plan = build_plan({"mode": "ask", "steps": []}, SharedState(), scene)
    assert "プリン" in plan.question


def test_status_lists_uneaten_foods(scene):
    shared = SharedState()
    shared.eaten["rice"] = 1
    plan = build_plan({"mode": "keep", "steps": ["status"]}, shared, scene)
    assert "味噌汁" in plan.reply
    assert "ご飯" not in plan.reply


@pytest.mark.parametrize(
    ("text", "mode", "steps"),
    [
        ("ご飯ちょうだい", "replace", ["rice", "feed"]),
        ("待って、まだ噛んでる", "keep", ["wait"]),
        ("もう大丈夫", "keep", ["resume"]),
        ("やっぱりプリンにして", "replace", ["pudding", "feed"]),
        ("ご飯と味噌汁を交互に2口ずつ", "replace", ["rice", "feed", "miso", "feed"] * 2),
        ("ご飯と味噌汁を交互に1口ずつ", "replace", ["rice", "feed", "miso", "feed"]),
        ("熱そうだから少し冷まして", "keep", ["cool"]),
        ("一口小さめで", "keep", ["small"]),
        ("もうちょっと上", "keep", ["up"]),
        ("何か食べたい", "ask", []),
        ("今日いい天気だね", "keep", []),
    ],
)
def test_rules(scene, text, mode, steps):
    assert rule_parse(text, scene) == {"mode": mode, "steps": steps}


class FailingBackend:
    name = "failing"

    def plan(self, text, shared):
        raise PlanningError("connection refused")


def test_failure_falls_back_to_rules_and_releases_pause(scene):
    shared = SharedState()
    shared.on_speech_onset()
    plan = PlanWorker(FailingBackend(), shared, scene).handle("ご飯ちょうだい")
    assert plan.steps[0] == scoop("rice")
    assert not shared.paused.is_set()


def test_pause_released_even_on_unexpected_error(scene):
    class BrokenBackend:
        name = "broken"

        def plan(self, text, shared):
            raise RuntimeError("boom")

    shared = SharedState()
    shared.on_speech_onset()
    with pytest.raises(RuntimeError):
        PlanWorker(BrokenBackend(), shared, scene).handle("ご飯ちょうだい")
    assert not shared.paused.is_set()
