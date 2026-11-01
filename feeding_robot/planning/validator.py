"""Safety limits applied to every plan, whichever backend produced it."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace

from feeding_robot.shared import Control, Plan, SharedState, Step

logger = logging.getLogger(__name__)

SKILLS = frozenset({"scoop", "feed", "cool", "wait_user", "go_home"})
AMOUNTS = frozenset({"small", "normal", "large"})
MAX_STEPS = 12
COOL_SECONDS = (1.0, 10.0)
SPEED_FACTOR_RANGE = (0.3, 2.0)
MOUTH_OFFSET_LIMIT_CM = 3.0
NEAR_MOUTH_RADIUS_M = 0.15
NEAR_MOUTH_SPEED_CAP = 0.5
UNKNOWN_REQUEST_QUESTION = "すみません、もう一度教えていただけますか？"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _valid_step(step: Step, foods: Iterable[str]) -> Step | None:
    """Return a safe version of step, or None if it cannot be carried out."""
    if step.skill not in SKILLS:
        return None
    if step.skill == "scoop":
        if step.args.get("food_id") not in foods:
            return None
        amount = step.args.get("amount")
        if amount is not None and amount not in AMOUNTS:
            return replace(step, args={**step.args, "amount": None})
    if step.skill == "cool":
        seconds = _clamp(float(step.args.get("seconds", 1.0)), *COOL_SECONDS)
        return replace(step, args={"seconds": seconds})
    return step


def _valid_steps(steps: Iterable[Step], foods: Iterable[str]) -> tuple[list[Step], bool]:
    checked = [_valid_step(step, foods) for step in steps]
    valid = [step for step in checked if step is not None]
    return valid, len(valid) != len(checked)


def _valid_control(control: Control, shared: SharedState) -> Control:
    speed_scale = control.speed_scale
    if speed_scale is not None:
        target = _clamp(shared.speed_factor * speed_scale, *SPEED_FACTOR_RANGE)
        speed_scale = target / shared.speed_factor
    offset = control.mouth_offset_cm
    if offset is not None:
        limit = MOUTH_OFFSET_LIMIT_CM
        offset = tuple(_clamp(axis, -limit, limit) for axis in offset)
    amount = control.amount if control.amount in AMOUNTS else None
    return replace(control, speed_scale=speed_scale, mouth_offset_cm=offset, amount=amount)


def validate(plan: Plan, shared: SharedState, foods: Iterable[str]) -> Plan:
    """Drop what cannot be carried out, clamp values, and ask rather than guess."""
    foods = frozenset(foods)
    steps, dropped_steps = _valid_steps(plan.steps, foods)
    inserts, dropped_inserts = _valid_steps(plan.inserts, foods)
    question = plan.question
    if dropped_steps or dropped_inserts:
        logger.warning("dropped invalid steps from plan: %s", plan)
        question = question or UNKNOWN_REQUEST_QUESTION
    return replace(
        plan,
        steps=tuple(steps[:MAX_STEPS]),
        inserts=tuple(inserts),
        control=_valid_control(plan.control, shared),
        question=question,
    )
