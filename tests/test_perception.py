"""Camera geometry and tracking logic, with a scripted detector."""

import numpy as np
import pytest

from feeding_robot.shared import SharedState, Step
from feeding_robot.simulation.perception import (
    LOST_AFTER_MISSES,
    FaceMeshDetector,
    FallbackDetector,
    Frame,
    LipColorDetector,
    MouthTracker,
    annotate,
    deproject,
    project,
)
from feeding_robot.simulation.sim import FeedingRobot, PyBulletWorld


@pytest.fixture
def world(scene):
    world = PyBulletWorld(scene, headless=True, fast=True)
    yield world
    world.close()


class ScriptedDetector:
    name = "SCRIPTED"

    def __init__(self, *pixels) -> None:
        self.pixels = iter(pixels)

    def detect(self, rgb):
        return next(self.pixels)


def frame_with_depth(world, depth: float = 0.9) -> Frame:
    frame = world.capture(world.wrist_camera())
    return Frame(frame.rgb, np.full_like(frame.depth, depth), frame.view, frame.projection)


def test_project_and_deproject_are_inverses(world):
    frame = world.capture(world.wrist_camera())
    v, u = 120, 160
    point = deproject(np.array((u, v), dtype=float), float(frame.depth[v, u]), frame)
    assert project(point, frame) == pytest.approx((u, v), abs=0.01)


def test_camera_depth_measures_the_scene(world, scene):
    camera = world.wrist_camera()
    frame = world.capture(camera)
    v, u = frame.depth.shape[0] // 2, frame.depth.shape[1] // 2
    point = deproject(np.array((u, v), dtype=float), float(frame.depth[v, u]), frame)
    assert np.linalg.norm(point - np.array(camera.eye)) > 0.05


def test_first_detection_is_trusted_even_far_from_rest(world):
    frame = frame_with_depth(world)
    measured = deproject(np.array((160.0, 120.0)), 0.9, frame)
    tracker = MouthTracker(ScriptedDetector(np.array((160.0, 120.0))), measured + (0.5, 0, 0))
    assert tracker.update(frame, 0.1) == pytest.approx(measured)


def test_jump_is_rejected_after_locking(world):
    frame = frame_with_depth(world)
    near, far = np.array((160.0, 120.0)), np.array((300.0, 20.0))
    tracker = MouthTracker(ScriptedDetector(near, far), np.zeros(3))
    locked = tracker.update(frame, 0.1).copy()
    assert tracker.update(frame, 0.1) == pytest.approx(locked)


def test_tracking_recovers_after_being_lost(world):
    frame = frame_with_depth(world)
    near, far = np.array((160.0, 120.0)), np.array((300.0, 20.0))
    pixels = [near] + [None] * LOST_AFTER_MISSES + [far]
    tracker = MouthTracker(ScriptedDetector(*pixels), np.zeros(3))
    for _ in range(LOST_AFTER_MISSES + 1):
        tracker.update(frame, 0.1)
    assert tracker.lost
    tracker.update(frame, 0.1)
    assert not tracker.lost


def test_annotation_marks_the_mouth_without_changing_the_input():
    rgb = np.full((240, 320, 3), 150, dtype=np.uint8)
    original = rgb.copy()
    view = annotate(rgb, np.array((160.0, 120.0)), "FACE MODEL", 3.2)
    assert (rgb == original).all()
    assert view.shape == rgb.shape
    assert not (view == original).all()


def test_annotation_handles_no_detection():
    rgb = np.full((240, 320, 3), 150, dtype=np.uint8)
    assert annotate(rgb, None, "SEARCHING", 0.0).shape == rgb.shape


def test_lip_detector_finds_a_red_patch():
    rgb = np.full((240, 320, 3), 150, dtype=np.uint8)
    rgb[100:110, 150:170] = (170, 60, 60)
    assert LipColorDetector().detect(rgb) == pytest.approx((159.5, 104.5))


def test_lip_detector_ignores_a_plain_image():
    assert LipColorDetector().detect(np.full((240, 320, 3), 150, dtype=np.uint8)) is None


def test_head_stays_still_until_asked_then_settles(world):
    start = world.true_mouth()
    for _ in range(240):
        world.step()
    assert world.true_mouth() == pytest.approx(start)
    world.shift_head()
    for _ in range(int(2.0 * 240)):
        world.step()
    moved = world.true_mouth()
    assert np.linalg.norm(moved - start) > 0.02
    for _ in range(240):
        world.step()
    assert world.true_mouth() == pytest.approx(moved)


def test_spoon_follows_a_head_that_shifts(scene, world):
    shared = SharedState()
    detector = FallbackDetector(FaceMeshDetector(), LipColorDetector())
    tracker = MouthTracker(detector, np.array(scene.person["mouth"], dtype=float))
    robot = FeedingRobot(world, scene, shared, tracker, head_moves=True)
    gaps = []
    shared.log.subscribe(lambda e, d: gaps.append(d["gap_cm"]) if e == "food_eaten" else None)
    for step in (Step("scoop", {"food_id": "rice"}), Step("feed")):
        robot.run_step(step)
    assert shared.eaten == {"rice": 1}
    assert gaps[0] < 3.5
