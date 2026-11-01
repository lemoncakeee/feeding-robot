"""Acceptance scenarios, with utterances planned by rules and run in the simulator."""

from collections.abc import Callable

import numpy as np
import pytest

from feeding_robot.planning.planner import PlanWorker, RulesBackend
from feeding_robot.planning.validator import NEAR_MOUTH_RADIUS_M, NEAR_MOUTH_SPEED_CAP
from feeding_robot.shared import SharedState, Step
from feeding_robot.simulation.sim import FeedingRobot, PyBulletWorld

MAX_TICKS = 240 * 120


class Harness:
    """Runs the robot loop, firing scripted events as conditions become true."""

    def __init__(self, scene) -> None:
        self.world = PyBulletWorld(scene, headless=True, fast=True)
        self.shared = SharedState()
        self.robot = FeedingRobot(self.world, scene, self.shared)
        self.worker = PlanWorker(RulesBackend(scene), self.shared, scene)
        self.mouth = np.array(scene.person["mouth"])
        self.events: list[tuple[Callable[[], bool], Callable[[], None]]] = []
        self.ticks = 0
        step = self.world.step

        def wrapped() -> None:
            step()
            self.ticks += 1
            self._after_tick()

        self.world.step = wrapped

    def say(self, text: str) -> None:
        self.shared.on_speech_onset()
        self.worker.handle(text)

    def on(self, condition: Callable[[], bool], action: Callable[[], None]) -> None:
        self.events.append((condition, action))

    def mouth_distance(self) -> float:
        return float(np.linalg.norm(self.world.tip_position() - self.mouth))

    def _after_tick(self) -> None:
        for event in self.events:
            if event[0]():
                self.events.remove(event)
                event[1]()
                break
        working = self.shared.current_step or self.shared.plan
        if (not working and not self.events and self.ticks > 5) or self.ticks > MAX_TICKS:
            self.shared.stop.set()

    def run(self) -> SharedState:
        self.robot.run()
        assert self.ticks <= MAX_TICKS, "scenario did not finish"
        return self.shared


@pytest.fixture
def h(scene):
    harness = Harness(scene)
    yield harness
    harness.world.close()


def test_1_single_bite(h):
    h.say("ご飯ちょうだい")
    assert h.run().eaten == {"rice": 1}


def test_2_wait_then_resume(h):
    h.say("ご飯ちょうだい")
    held_at = {}

    def wait() -> None:
        h.say("待って、まだ噛んでる")
        held_at["tip"] = h.world.tip_position()
        h.on(lambda: h.ticks > held_at.setdefault("until", h.ticks + 240), check_and_resume)

    def check_and_resume() -> None:
        assert np.linalg.norm(h.world.tip_position() - held_at["tip"]) < 0.02
        assert h.shared.eaten["rice"] == 0
        h.say("いいよ")

    h.on(lambda: h.ticks == 300, wait)
    assert h.run().eaten == {"rice": 1}


def test_3_alternating_bites(h):
    h.say("ご飯と味噌汁を交互に2口ずつ")
    assert h.run().eaten == {"rice": 2, "miso": 2}


def test_4_replace_running_plan(h):
    h.say("ご飯ちょうだい")
    h.on(lambda: h.ticks == 200, lambda: h.say("やっぱりプリンにして"))
    assert h.run().eaten == {"pudding": 1}


def test_5_cool_before_feeding(h):
    h.say("ご飯ちょうだい")
    log = []
    original = h.robot.cool

    def cool(seconds: float) -> None:
        log.append(h.shared.spoon_food)
        original(seconds)

    h.robot._skills["cool"] = cool
    h.on(lambda: h.shared.spoon_food == "rice", lambda: h.say("熱そうだから少し冷まして"))
    assert h.run().eaten == {"rice": 1}
    assert log == ["rice"]


def test_6_small_bite(h):
    sizes = []
    original = h.world.set_food_on_spoon
    h.world.set_food_on_spoon = lambda color, radius: (
        sizes.append(radius),
        original(color, radius),
    )
    h.say("一口小さめで")
    h.say("ご飯ちょうだい")
    h.say("プリンちょうだい")
    h.run()
    assert len(sizes) == 1
    assert sizes[0] < 0.018


def test_6_amount_resets_after_one_bite(h):
    h.say("一口小さめで")
    assert h.shared.take_amount() == "small"
    assert h.shared.take_amount() == "normal"


def test_7_mouth_offset_moves_the_spoon(h):
    h.say("もうちょっと上")
    h.say("もうちょっと上")
    h.say("ご飯ちょうだい")
    raised_mouth = h.mouth + (0, -0.01, 0.02)
    reached = []
    h.on(
        lambda: np.linalg.norm(h.world.tip_position() - raised_mouth) < 0.015,
        lambda: reached.append(h.shared.spoon_food),
    )
    assert h.run().eaten == {"rice": 1}
    assert reached == ["rice"]


def test_9_ambiguous_request_asks(h):
    plan = h.worker.handle("何か食べたい")
    assert plan.question
    assert not h.shared.plan


def test_10_chat_does_nothing(h):
    plan = h.worker.handle("今日いい天気だね")
    assert not plan.steps
    assert not plan.reply
    assert not h.shared.plan


def test_speed_cap_near_mouth_overrides_requests(h):
    h.shared.speed_factor = 2.0
    h.robot._target = h.mouth - (0, NEAR_MOUTH_RADIUS_M - 0.01, 0)
    assert h.robot._speed_factor() == NEAR_MOUTH_SPEED_CAP


def test_wait_step_in_plan_holds_until_resume(h):
    h.worker.handle("ご飯ちょうだい")
    h.shared.plan.append(Step("wait_user"))
    h.on(lambda: h.shared.hold.is_set(), lambda: h.say("いいよ"))
    assert h.run().eaten == {"rice": 1}
