"""Limits that hold regardless of what the planner asks for."""

import pytest

from feeding_robot.planning.validator import MAX_STEPS, UNKNOWN_REQUEST_QUESTION, validate
from feeding_robot.shared import Control, Plan, SharedState, Step

FOODS = ("rice", "miso")


def test_unknown_skill_and_food_become_a_question():
    plan = Plan("replace", (Step("dance"), Step("scoop", {"food_id": "steak"}), Step("feed")))
    checked = validate(plan, SharedState(), FOODS)
    assert checked.steps == (Step("feed"),)
    assert checked.question == UNKNOWN_REQUEST_QUESTION


def test_plan_length_is_limited():
    plan = Plan("replace", (Step("feed"),) * (MAX_STEPS + 5))
    assert len(validate(plan, SharedState(), FOODS).steps) == MAX_STEPS


@pytest.mark.parametrize(("seconds", "expected"), [(0, 1), (5, 5), (99, 10)])
def test_cool_seconds_are_clamped(seconds, expected):
    plan = Plan("replace", (Step("cool", {"seconds": seconds}),))
    assert validate(plan, SharedState(), FOODS).steps[0].args["seconds"] == expected


def test_unknown_amount_falls_back_to_default():
    plan = Plan("replace", (Step("scoop", {"food_id": "rice", "amount": "huge"}),))
    assert validate(plan, SharedState(), FOODS).steps[0].args["amount"] is None


@pytest.mark.parametrize(("scale", "expected"), [(100.0, 2.0), (0.01, 0.3), (1.25, 1.25)])
def test_speed_stays_in_range_after_scaling(scale, expected):
    shared = SharedState()
    shared.apply_plan(validate(Plan(control=Control(speed_scale=scale)), shared, FOODS))
    assert shared.speed_factor == pytest.approx(expected)


def test_mouth_offset_is_limited_cumulatively():
    shared = SharedState()
    for z in (2.0, 4.0, 9.0):
        control = Control(mouth_offset_cm=(0.0, 0.0, z))
        shared.apply_plan(validate(Plan(control=control), shared, FOODS))
    assert shared.mouth_offset_cm == (0.0, 0.0, 3.0)
