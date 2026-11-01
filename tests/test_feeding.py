"""Headless behavior tests for feeding and pausing."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from feeding_robot.shared import SharedState, Step
from feeding_robot.simulation.sim import FeedingRobot, PyBulletWorld, Scene

SCENE = Scene.load(Path(__file__).parent.parent / "scene.yaml")
BITE = [Step("scoop", {"food_id": "rice"}), Step("feed")]
MOUTH = np.array(SCENE.person["mouth"])


@pytest.fixture
def robot():
    world = PyBulletWorld(SCENE, headless=True, fast=True)
    yield FeedingRobot(world, SCENE, SharedState())
    world.close()


def on_tick(robot: FeedingRobot, callback: Callable[[int], None]) -> None:
    """Call callback with the tick number after every simulation step."""
    step = robot.world.step
    ticks = iter(range(10**9))

    def wrapped() -> None:
        step()
        callback(next(ticks))

    robot.world.step = wrapped


def run_bite(robot: FeedingRobot) -> None:
    for step in BITE:
        robot.run_step(step)


def mouth_distance(robot: FeedingRobot) -> float:
    return float(np.linalg.norm(robot.world.tip_position() - MOUTH))


def test_spoon_reaches_every_bowl(robot):
    for food_id, food in SCENE.foods.items():
        robot.run_step(Step("scoop", {"food_id": food_id}))
        assert robot.spoon_food == food_id
        assert np.linalg.norm(robot.world.tip_position()[:2] - np.array(food.pos[:2])) < 0.02


def test_bite_reaches_mouth_and_counts(robot):
    closest = [1.0]
    on_tick(robot, lambda _: closest.__setitem__(0, min(closest[0], mouth_distance(robot))))
    run_bite(robot)
    assert closest[0] < 0.03
    assert robot.shared.eaten == {"rice": 1}
    assert robot.spoon_food is None


def test_pause_freezes_motion_until_resume(robot):
    shared = robot.shared
    start = 300
    release = start + 240
    positions = {}

    def script(tick: int) -> None:
        if tick == start:
            shared.on_speech_onset()
        if tick in (start + 60, release - 1):
            positions[tick] = robot.world.tip_position()
        if tick == release:
            shared.on_resume()

    on_tick(robot, script)
    run_bite(robot)
    moved = np.linalg.norm(positions[start + 60] - positions[release - 1])
    assert moved < 0.002
    assert shared.eaten == {"rice": 1}
