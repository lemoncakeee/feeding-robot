"""Simulated scene, arm motion, feeding skills and the plan executor.

All PyBullet calls live in PyBulletWorld and must run on the main thread.
FeedingRobot only uses the World protocol, so the simulator can be replaced.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pybullet as p
import pybullet_data
import yaml

from feeding_robot.planning.validator import NEAR_MOUTH_RADIUS_M, NEAR_MOUTH_SPEED_CAP
from feeding_robot.shared import SharedState, Step
from feeding_robot.simulation import scenery
from feeding_robot.simulation.perception import (
    CameraSpec,
    Frame,
    MouthTracker,
    annotate,
    standby_image,
)
from feeding_robot.simulation.scenery import MONITOR_PIXELS, STATUS_PIXELS

Vec3 = np.ndarray

TIMESTEP_S = 1 / 240
LIGHT_POSITION = (1.5, 1.0, 3.5)
SHADOW_MAP_RESOLUTION = 4096
SHADOW_MAP_WORLD_SIZE = 5
BACKGROUND_COLOR = (0.90, 0.87, 0.80)
FOOD_LIFT_RATIO = 0.55
MAX_LAG_S = 0.1
HEAD_PIVOT_FROM_MOUTH = np.array((0.0, 0.09, -0.12))
HEAD_SHIFT_DELAY_S = 0.4
HEAD_SHIFT_DURATION_S = 1.2
HEAD_LEAN_RANGE_M = (0.03, 0.05)
HEAD_DRIFT_M = np.array((0.0, 0.02, 0.025))
HEAD_YAW_RAD = 0.2
HEAD_UPDATE_TICKS = 4
WINDOW_OPTIONS = "--width=1600 --height=900"
MONITOR_DISTANCE_M = 0.6  # from the viewer, so it sits in front of the scene
MONITOR_VIEW_HEIGHT = 0.32  # as a fraction of the window height
STATUS_VIEW_HEIGHT = 0.34
STATUS_TICKS = 12
MONITOR_MARGIN = 0.02
DEFAULT_VIEW_TANGENTS = (0.77, 0.58)  # used without a window
TRACKING_TICKS = 16
WRIST_CAMERA_FORWARD_M = 0.04
WRIST_CAMERA_UP_M = 0.06
WRIST_CAMERA_TILT = 0.35  # looks down to keep the mouth in frame close up
TRAY_MARGIN_M = 0.09
CAMERA_VIEW = (1.0, 68, -12, (0.30, 0.11, 0.47))  # distance, yaw, pitch, target
ARM_JOINTS = range(7)
FINGER_JOINTS = (9, 10)
GRASP_LINK = 11
FINGER_GRIP_M = 0.01
ARM_REST_POSE = (0.0, -0.6, 0.0, -2.4, 0.0, 1.8, 0.785)
IK_ITERATIONS = 30
MOTOR_FORCE_N = 240.0
SPOON_LENGTH_M = 0.16

SPEED_MPS = 0.30
SLOW_SPEED_MPS = 0.16
ARRIVAL_TOLERANCE_M = 0.01
SETTLE_TIMEOUT_S = 1.0
EAT_TIME_S = 1.5

HOVER_HEIGHT_M = 0.13
DIP_HEIGHT_M = 0.012
PRE_MOUTH_DISTANCE_M = 0.12
MOUTH_GAP_M = 0.01
YAW_TO_FOOD = 0.0
YAW_TO_MOUTH = math.pi / 2
HOME_TIP = np.array([0.42, 0.0, 0.60])
COOL_TIP = np.array([0.45, 0.05, 0.58])
FOOD_RADIUS_M = {"small": 0.012, "normal": 0.018, "large": 0.024}


@dataclass(frozen=True)
class Food:
    id: str
    name: str
    color: tuple[float, float, float]
    pos: tuple[float, float, float]
    aliases: tuple[str, ...] = ()
    style: str = "mound"


@dataclass(frozen=True)
class Scene:
    table: dict
    person: dict
    foods: dict[str, Food]

    @classmethod
    def load(cls, path: Path) -> Scene:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        foods = {
            food["id"]: Food(
                food["id"],
                food["name"],
                tuple(food["color"]),
                tuple(food["pos"]),
                tuple(food.get("aliases", ())),
                food.get("style", "mound"),
            )
            for food in data["foods"]
        }
        return cls(data["table"], data["person"], foods)


class World(Protocol):
    """What the robot needs from a simulator."""

    def command_tip(self, tip: Vec3, yaw: float) -> None: ...
    def tip_position(self) -> Vec3: ...
    def set_food_on_spoon(self, color: tuple[float, float, float], radius: float) -> None: ...
    def clear_spoon(self) -> None: ...
    def step(self) -> None: ...
    def capture(self, camera: CameraSpec) -> Frame: ...
    def wrist_camera(self) -> CameraSpec: ...
    def show_on_monitor(self, rgb: np.ndarray) -> None: ...
    def show_status(self, rgb: np.ndarray) -> None: ...
    def true_mouth(self) -> Vec3: ...
    def shift_head(self) -> None: ...


class PyBulletWorld:
    """Franka Panda holding a level spoon, a table of dishes and a seated person."""

    def __init__(
        self, scene: Scene, *, headless: bool = False, fast: bool = False, head_seed: int = 0
    ) -> None:
        self._fast = fast
        self._client = p.connect(p.DIRECT) if headless else p.connect(p.GUI, options=WINDOW_OPTIONS)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIMESTEP_S)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(
            lightPosition=LIGHT_POSITION,
            shadowMapResolution=SHADOW_MAP_RESOLUTION,
            shadowMapWorldSize=SHADOW_MAP_WORLD_SIZE,
            rgbBackground=BACKGROUND_COLOR,
        )
        p.resetDebugVisualizerCamera(*CAMERA_VIEW)
        self._arm = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for joint, angle in zip(ARM_JOINTS, ARM_REST_POSE, strict=True):
            p.resetJointState(self._arm, joint, angle)
        for joint in FINGER_JOINTS:
            p.resetJointState(self._arm, joint, FINGER_GRIP_M)
        self._limits = [p.getJointInfo(self._arm, j)[8:10] for j in ARM_JOINTS]
        self._assets = tempfile.TemporaryDirectory()
        self._spoon = self._build_scene(scene, Path(self._assets.name))
        self._mouth_rest = np.array(scene.person["mouth"], dtype=float)
        self._wrist_camera_body = scenery.Scenery(Path(self._assets.name)).place(
            "wrist_camera", scenery.box((0.025, 0.09, 0.025)), (0.08, 0.08, 0.09), (0, 0, -5)
        )
        self._head_pivot = self._mouth_rest + HEAD_PIVOT_FROM_MOUTH
        self._food = self._visual(p.GEOM_SPHERE, (1, 1, 1), radius=0.018, position=(0, 0, -5))
        self._food_lift = 0.0
        self._next_tick = time.perf_counter()
        self._ticks = 0
        self._rng = np.random.default_rng(head_seed)
        self._head_offset = np.zeros(3)
        self._head_yaw = 0.0
        self._head_start: tuple[Vec3, float] | None = None
        self._head_goal: tuple[Vec3, float] | None = None
        self._head_shift_tick = 0

    def shift_head(self) -> None:
        """Start one smooth head movement, leaning towards the other side from now."""
        side = -1.0 if self._head_offset[0] > 0 else 1.0
        lean = side * self._rng.uniform(*HEAD_LEAN_RANGE_M)
        drift = self._rng.uniform(-HEAD_DRIFT_M, HEAD_DRIFT_M)
        self._head_start = (self._head_offset.copy(), self._head_yaw)
        self._head_goal = (drift + (lean, 0.0, 0.0), side * self._rng.uniform(0, HEAD_YAW_RAD))
        self._head_shift_tick = self._ticks + int(HEAD_SHIFT_DELAY_S / TIMESTEP_S)

    def _update_head(self) -> None:
        if self._head_goal is None or self._ticks < self._head_shift_tick:
            return
        progress = (self._ticks - self._head_shift_tick) * TIMESTEP_S / HEAD_SHIFT_DURATION_S
        phase = min(progress, 1.0)
        blend = phase**3 * (10 - 15 * phase + 6 * phase**2)  # starts and stops gently
        (offset_from, yaw_from), (offset_to, yaw_to) = self._head_start, self._head_goal
        self._head_offset = offset_from + blend * (offset_to - offset_from)
        self._head_yaw = yaw_from + blend * (yaw_to - yaw_from)
        if phase == 1.0:
            self._head_goal = None
        self._pose_head()

    def _pose_head(self) -> None:
        """Place the head's bodies as one rigid body."""
        turn = p.getQuaternionFromEuler((0, 0, self._head_yaw))
        pivot = self._head_pivot + self._head_offset
        for body, position, orientation in self._head_rest:
            moved, rotated = p.multiplyTransforms(
                pivot.tolist(), turn, (position - self._head_pivot).tolist(), orientation
            )
            p.resetBasePositionAndOrientation(body, moved, rotated)

    def true_mouth(self) -> Vec3:
        """The actual mouth position, for measuring tracking error."""
        turn = p.getMatrixFromQuaternion(p.getQuaternionFromEuler((0, 0, self._head_yaw)))
        arm = np.array(turn).reshape(3, 3) @ (self._mouth_rest - self._head_pivot)
        return self._head_pivot + self._head_offset + arm

    @staticmethod
    def _corner_placement(
        right_side: bool, bottom: bool, view_height: float, pixels: tuple[int, int]
    ) -> tuple[Vec3, Vec3, tuple[float, float]]:
        """Return center, normal and size for a panel in a corner of the view."""
        camera = p.getDebugVisualizerCamera()
        tan_x, tan_y = DEFAULT_VIEW_TANGENTS
        if camera[0] > 0:
            tan_x, tan_y = 1 / camera[3][0], 1 / camera[3][5]
        view = np.array(camera[2]).reshape(4, 4).T
        right, up, back = view[:3, :3]
        eye = -view[:3, :3].T @ view[:3, 3]
        half_x, half_y = MONITOR_DISTANCE_M * tan_x, MONITOR_DISTANCE_M * tan_y
        height = 2 * half_y * view_height
        width = height * pixels[0] / pixels[1]
        margin = 2 * half_y * MONITOR_MARGIN
        across = (half_x - width / 2 - margin) * (1 if right_side else -1)
        along = (half_y - height / 2 - margin) * (-1 if bottom else 1)
        center = eye - back * MONITOR_DISTANCE_M + right * across + up * along
        return center, back, (width, height)

    def show_status(self, rgb: np.ndarray) -> None:
        """Show an image on the status panel."""
        height, width = rgb.shape[:2]
        p.changeTexture(self._status_panel, rgb.flatten().tolist(), width, height)

    def show_on_monitor(self, rgb: np.ndarray) -> None:
        """Show an image on the wall monitor."""
        height, width = rgb.shape[:2]
        p.changeTexture(self._monitor, rgb.flatten().tolist(), width, height)

    def wrist_camera(self) -> CameraSpec:
        """A camera on the gripper, looking along the spoon."""
        position, axis, _ = self._grasp_pose()
        eye = position + WRIST_CAMERA_FORWARD_M * axis + (0, 0, WRIST_CAMERA_UP_M)
        return CameraSpec(tuple(eye), tuple(eye + axis - (0, 0, WRIST_CAMERA_TILT)))

    def capture(self, camera: CameraSpec) -> Frame:
        view = p.computeViewMatrix(camera.eye, camera.target, (0, 0, 1))
        projection = p.computeProjectionMatrixFOV(
            camera.fov_deg, camera.width / camera.height, camera.near, camera.far
        )
        image = p.getCameraImage(
            camera.width, camera.height, view, projection, renderer=p.ER_TINY_RENDERER, shadow=0
        )
        shape = (camera.height, camera.width)
        return Frame(
            rgb=np.reshape(image[2], (*shape, 4))[:, :, :3].astype(np.uint8),
            depth=np.reshape(image[3], shape),
            view=np.array(view).reshape(4, 4).T,
            projection=np.array(projection).reshape(4, 4).T,
        )

    @staticmethod
    def _visual(shape: int, color, position=(0, 0, 0), **geometry) -> int:
        """Create a body that is seen but never collides with the arm."""
        visual = p.createVisualShape(shape, rgbaColor=(*color, 1), **geometry)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=position)

    def _build_scene(self, scene: Scene, directory: Path) -> int:
        """Build the room, table, dishes and person, and return the spoon's body."""
        builder = scenery.Scenery(directory)
        builder.build_room()
        builder.build_table(scene.table["center"], scene.table["half_extents"])
        builder.build_robot_stand()
        x = [food.pos[0] for food in scene.foods.values()]
        y = [food.pos[1] for food in scene.foods.values()]
        margin = TRAY_MARGIN_M
        builder.build_tray((min(x) - margin, max(x) + margin, min(y) - margin, max(y) + margin))
        for food in scene.foods.values():
            builder.build_dish(food.id, food.style, food.color, food.pos)
        builder.build_person(scene.person)
        corner = self._corner_placement(True, True, MONITOR_VIEW_HEIGHT, MONITOR_PIXELS)
        self._monitor = builder.build_monitor("monitor", *corner, MONITOR_PIXELS)
        corner = self._corner_placement(False, False, STATUS_VIEW_HEIGHT, STATUS_PIXELS)
        self._status_panel = builder.build_monitor("status", *corner, STATUS_PIXELS)
        self._head_rest = [
            (body, np.array(position), orientation)
            for body in builder.head_bodies
            for position, orientation in [p.getBasePositionAndOrientation(body)]
        ]
        return builder.place(
            "spoon",
            scenery.spoon(SPOON_LENGTH_M),
            scenery.STEEL,
            (0, 0, -5),
            specular=scenery.SHINY,
        )

    def command_tip(self, tip: Vec3, yaw: float) -> None:
        orientation = p.getQuaternionFromEuler((math.pi, 0, yaw))
        grasp = tip - SPOON_LENGTH_M * np.array([math.cos(yaw), math.sin(yaw), 0.0])
        lower, upper = zip(*self._limits, strict=True)
        angles = p.calculateInverseKinematics(
            self._arm,
            GRASP_LINK,
            grasp.tolist(),
            orientation,
            lowerLimits=[*lower, 0.0, 0.0],
            upperLimits=[*upper, 0.04, 0.04],
            jointRanges=[*(high - low for low, high in self._limits), 0.04, 0.04],
            restPoses=[*ARM_REST_POSE, FINGER_GRIP_M, FINGER_GRIP_M],
            maxNumIterations=IK_ITERATIONS,
        )
        p.setJointMotorControlArray(
            self._arm,
            list(ARM_JOINTS),
            p.POSITION_CONTROL,
            targetPositions=angles[: len(ARM_JOINTS)],
            forces=[MOTOR_FORCE_N] * len(ARM_JOINTS),
        )

    def _grasp_pose(self) -> tuple[Vec3, Vec3, tuple]:
        """Return the grasp position, the spoon axis and the grasp orientation."""
        state = p.getLinkState(self._arm, GRASP_LINK, computeForwardKinematics=True)
        position, orientation = np.array(state[4]), state[5]
        axis = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)[:, 0]
        return position, axis, orientation

    def tip_position(self) -> Vec3:
        position, axis, _ = self._grasp_pose()
        return position + SPOON_LENGTH_M * axis

    def set_food_on_spoon(self, color: tuple[float, float, float], radius: float) -> None:
        p.removeBody(self._food)
        self._food = self._visual(p.GEOM_SPHERE, color, radius=radius, position=(0, 0, -5))
        self._food_lift = radius

    def clear_spoon(self) -> None:
        self._food_lift = 0.0
        p.resetBasePositionAndOrientation(self._food, (0, 0, -5), (0, 0, 0, 1))

    def step(self) -> None:
        p.stepSimulation()
        self._ticks += 1
        if self._ticks % HEAD_UPDATE_TICKS == 0:
            self._update_head()
        position, axis, orientation = self._grasp_pose()
        tip = position + SPOON_LENGTH_M * axis
        p.resetBasePositionAndOrientation(self._spoon, position, orientation)
        p.resetBasePositionAndOrientation(
            self._wrist_camera_body,
            position + WRIST_CAMERA_FORWARD_M * axis + (0, 0, WRIST_CAMERA_UP_M),
            orientation,
        )
        if self._food_lift:
            p.resetBasePositionAndOrientation(
                self._food, tip + (0, 0, self._food_lift * FOOD_LIFT_RATIO), (0, 0, 0, 1)
            )
        if not self._fast:
            self._next_tick = max(self._next_tick + TIMESTEP_S, time.perf_counter() - MAX_LAG_S)
            delay = self._next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def close(self) -> None:
        p.disconnect(self._client)
        self._assets.cleanup()


class StatusSource(Protocol):
    def take_image(self) -> np.ndarray | None: ...


class Replaced(Exception):
    """The current step was abandoned because the plan changed."""


class FeedingRobot:
    """Runs plan steps while reacting to shared flags on every tick."""

    def __init__(
        self,
        world: World,
        scene: Scene,
        shared: SharedState,
        tracker: MouthTracker | None = None,
        *,
        head_moves: bool = False,
        status: StatusSource | None = None,
    ) -> None:
        self.world = world
        self._tracker = tracker
        self._head_moves = head_moves
        self._status = status
        self._watching = False
        if tracker is not None:
            world.show_on_monitor(standby_image(*MONITOR_PIXELS))
        self._ticks = 0
        self.scene = scene
        self.shared = shared
        self._mouth = np.array(scene.person["mouth"], dtype=float)
        self._target = world.tip_position()
        self._yaw = YAW_TO_FOOD
        self._skills = {
            "scoop": self.scoop,
            "feed": self.feed,
            "cool": self.cool,
            "wait_user": self.wait_user,
            "go_home": self.go_home,
        }

    @property
    def spoon_food(self) -> str | None:
        return self.shared.spoon_food

    def _mouth_tip(self) -> Vec3:
        offset = np.array(self.shared.mouth_offset_cm) / 100
        return self._mouth + offset - MOUTH_GAP_M * self._approach_axis()

    def _pre_mouth_tip(self) -> Vec3:
        return self._mouth_tip() - PRE_MOUTH_DISTANCE_M * self._approach_axis()

    @staticmethod
    def _approach_axis() -> Vec3:
        """Direction the spoon travels into the mouth, in the horizontal plane."""
        return np.array((math.cos(YAW_TO_MOUTH), math.sin(YAW_TO_MOUTH), 0.0))

    def _speed_factor(self) -> float:
        factor = self.shared.speed_factor
        if np.linalg.norm(self._target - self._mouth) < NEAR_MOUTH_RADIUS_M:
            return min(factor, NEAR_MOUTH_SPEED_CAP)
        return factor

    def _tick(self) -> None:
        self.world.command_tip(self._target, self._yaw)
        self.world.step()
        self._ticks += 1
        if self._status is not None and self._ticks % STATUS_TICKS == 0:
            image = self._status.take_image()
            if image is not None:
                self.world.show_status(image)
        if self._watching and self._ticks % TRACKING_TICKS == 0:
            frame = self.world.capture(self.world.wrist_camera())
            self._mouth = self._tracker.update(frame, TRACKING_TICKS * TIMESTEP_S)
            correction_cm = float(np.linalg.norm(self._mouth - self._approach_mouth)) * 100
            tracker = self._tracker
            view = annotate(frame.rgb, tracker.pixel, tracker.source, correction_cm)
            self.world.show_on_monitor(view)
            error_cm = float(np.linalg.norm(self._mouth - self.world.true_mouth())) * 100
            self.shared.log.write("mouth_tracked", error_cm=round(error_cm, 2))

    def _tracking_lost(self) -> bool:
        return self._watching and self._tracker.lost

    def _must_hold(self) -> bool:
        shared = self.shared
        held = shared.paused.is_set() or shared.hold.is_set()
        return held and not (shared.replaced.is_set() or shared.stop.is_set())

    def _check_interrupts(self) -> None:
        """Hold still while paused, and abandon the step on a new plan."""
        shared = self.shared
        if self._must_hold():
            self._target = self.world.tip_position()
            if shared.paused.is_set() and shared.onset_time is not None:
                latency_ms = (time.perf_counter() - shared.onset_time) * 1000
                shared.log.write("robot_stopped", latency_ms=round(latency_ms, 1))
            while self._must_hold():
                self._tick()
        if shared.replaced.is_set():
            raise Replaced

    def move_to(self, tip: Vec3 | Callable[[], Vec3], yaw: float, speed: float = SPEED_MPS) -> None:
        """Slide the target point to a fixed or moving tip, checking interrupts every tick."""
        goal = tip if callable(tip) else lambda: tip
        while not self.shared.stop.is_set():
            self._check_interrupts()
            if callable(tip) and self._tracking_lost():
                self._tick()
                continue
            stride = speed * TIMESTEP_S * self._speed_factor()
            remaining = goal() - self._target
            distance = float(np.linalg.norm(remaining))
            if distance <= stride:
                break
            self._target = self._target + remaining / distance * stride
            self._yaw += (yaw - self._yaw) * stride / distance
            self._tick()
        self._yaw = yaw
        for _ in range(int(SETTLE_TIMEOUT_S / TIMESTEP_S)):
            self._target = goal().copy()
            if np.linalg.norm(self.world.tip_position() - self._target) < ARRIVAL_TOLERANCE_M:
                break
            self._check_interrupts()
            self._tick()

    def hold(self, seconds: float, *, at_mouth: bool = False) -> None:
        """Stay in place for simulated seconds, following mouth offset changes if at_mouth."""
        for _ in range(int(seconds / TIMESTEP_S)):
            self._check_interrupts()
            if at_mouth:
                self._target = self._mouth_tip()
            self._tick()

    def scoop(self, food_id: str, amount: str | None = None) -> None:
        food = self.scene.foods[food_id]
        bowl = np.array(food.pos)
        self.move_to(bowl + (0, 0, HOVER_HEIGHT_M), YAW_TO_FOOD)
        self.move_to(bowl + (0, 0, DIP_HEIGHT_M), YAW_TO_FOOD, SLOW_SPEED_MPS)
        self.world.set_food_on_spoon(food.color, FOOD_RADIUS_M[amount or self.shared.take_amount()])
        self.shared.spoon_food = food_id
        self.move_to(bowl + (0, 0, HOVER_HEIGHT_M), YAW_TO_FOOD, SLOW_SPEED_MPS)

    def feed(self) -> None:
        shared = self.shared
        if shared.spoon_food is not None:
            self.move_to(self._pre_mouth_tip, YAW_TO_MOUTH)
            self._watching = self._tracker is not None
            self._approach_mouth = self._mouth.copy()
            if self._head_moves:
                self.world.shift_head()
            self.move_to(self._mouth_tip, YAW_TO_MOUTH, SLOW_SPEED_MPS)
            self.hold(EAT_TIME_S, at_mouth=True)
            gap_m = float(np.linalg.norm(self.world.tip_position() - self.world.true_mouth()))
            shared.eaten[shared.spoon_food] += 1
            shared.log.write("food_eaten", food=shared.spoon_food, gap_cm=round(gap_m * 100, 1))
            shared.spoon_food = None
            self.world.clear_spoon()
        self.move_to(self._pre_mouth_tip, YAW_TO_MOUTH, SLOW_SPEED_MPS)
        if self._watching:
            self.world.show_on_monitor(standby_image(*MONITOR_PIXELS))
        self._watching = False

    def cool(self, seconds: float) -> None:
        self.move_to(COOL_TIP, YAW_TO_FOOD)
        self.hold(seconds)

    def wait_user(self) -> None:
        self.shared.hold.set()
        self._check_interrupts()

    def go_home(self) -> None:
        self.move_to(HOME_TIP, YAW_TO_FOOD)

    def run_step(self, step: Step) -> None:
        """Run one step until it finishes or the plan replaces it."""
        log = self.shared.log
        log.write("step_started", skill=step.skill, args=step.args)
        try:
            self._skills[step.skill](**step.args)
        except Replaced:
            log.write("step_replaced", skill=step.skill)
        else:
            log.write("step_finished", skill=step.skill)

    def run(self) -> None:
        """Execute plan steps until asked to stop. Must run on the main thread."""
        shared = self.shared
        while not shared.stop.is_set():
            step = shared.next_step()
            if step is not None:
                self.run_step(step)
                continue
            try:
                self._check_interrupts()
            except Replaced:
                continue
            self._tick()
