"""Finds the user's mouth from a camera, so the spoon can follow a moving head."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402  (the log level must be set before import)

UPPER_LIP, LOWER_LIP = 13, 14
DETECTION_CONFIDENCE = 0.3
SMOOTHING = 0.6
VELOCITY_GAIN = 0.3
MAX_SPEED_MPS = 0.2
MAX_JUMP_M = 0.08
LOST_AFTER_MISSES = 5
DEPTH_PATCH = 2
LIP_SEARCH_HEIGHT = 0.92
LIP_MIN_RED = 70
LIP_MIN_MARGIN = 45
LIP_BLUE_TO_GREEN = 0.85
LIP_MIN_PIXELS = 12
NO_DETECTION = "SEARCHING"
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
MARKER_RADIUS = 12
MARKER_COLOR = (60, 255, 120)
SEARCH_COLOR = (255, 190, 60)
BANNER_HEIGHT = 44


@dataclass(frozen=True)
class CameraSpec:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    width: int = 320
    height: int = 240
    fov_deg: float = 90.0
    near: float = 0.02
    far: float = 3.0


@dataclass(frozen=True)
class Frame:
    """A camera image with OpenGL-style depth and matrices."""

    rgb: np.ndarray
    depth: np.ndarray
    view: np.ndarray
    projection: np.ndarray


def project(point: np.ndarray, frame: Frame) -> np.ndarray:
    """Return the pixel (u, v) of a world point."""
    clip = frame.projection @ frame.view @ np.append(point, 1.0)
    x, y = clip[:2] / clip[3]
    height, width = frame.depth.shape
    return np.array(((x + 1) / 2 * width, (1 - y) / 2 * height))


def deproject(pixel: np.ndarray, depth: float, frame: Frame) -> np.ndarray:
    """Return the world point seen at a pixel with the given depth buffer value."""
    height, width = frame.depth.shape
    ndc = np.array((2 * pixel[0] / width - 1, 1 - 2 * pixel[1] / height, 2 * depth - 1, 1.0))
    world = np.linalg.inv(frame.projection @ frame.view) @ ndc
    return world[:3] / world[3]


class MouthDetector(Protocol):
    name: str

    def detect(self, rgb: np.ndarray) -> np.ndarray | None: ...


class FaceMeshDetector:
    """Locates the mouth in an image with the MediaPipe face landmark model."""

    name = "FACE MODEL"

    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=DETECTION_CONFIDENCE,
        )

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        height, width = rgb.shape[:2]
        u = (landmarks[UPPER_LIP].x + landmarks[LOWER_LIP].x) / 2 * width
        v = (landmarks[UPPER_LIP].y + landmarks[LOWER_LIP].y) / 2 * height
        return np.array((u, v))


class LipColorDetector:
    """Close-range backup: finds the lips by color when the spoon hides the lower face."""

    name = "LIP COLOR"

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        visible = rgb[: int(rgb.shape[0] * LIP_SEARCH_HEIGHT)].astype(float)
        red, green, blue = visible[..., 0], visible[..., 1], visible[..., 2]
        reddish = (red > LIP_MIN_RED) & (red - green > LIP_MIN_MARGIN)
        lips = reddish & (blue > green * LIP_BLUE_TO_GREEN)
        if lips.sum() < LIP_MIN_PIXELS:
            return None
        rows, columns = np.nonzero(lips)
        return np.array((columns.mean(), rows.mean()))


class FallbackDetector:
    """Tries detectors in order and returns the first result."""

    def __init__(self, *detectors: MouthDetector) -> None:
        self._detectors = detectors
        self.name = NO_DETECTION

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        self.name = NO_DETECTION
        for detector in self._detectors:
            if (pixel := detector.detect(rgb)) is not None:
                self.name = detector.name
                return pixel
        return None


class MouthTracker:
    """Turns detections into a smoothed 3D mouth position."""

    def __init__(self, detector: MouthDetector, resting_mouth: np.ndarray) -> None:
        self._detector = detector
        self.estimate = np.array(resting_mouth, dtype=float)
        self._misses = 0
        self._locked = False
        self.velocity = np.zeros(3)
        self.pixel: np.ndarray | None = None

    @property
    def lost(self) -> bool:
        return self._misses >= LOST_AFTER_MISSES

    def update(self, frame: Frame, dt: float) -> np.ndarray:
        """Process one frame taken dt seconds after the last, and return the estimate."""
        if self._locked and not self.lost:
            self.estimate = self.estimate + self.velocity * dt
        pixel = self.pixel = self._detector.detect(frame.rgb)
        if pixel is None:
            self._misses += 1
            return self.estimate
        measured = self._measure(pixel, frame)
        if measured is None or self._is_jump(measured):
            self._misses += 1
            return self.estimate
        innovation = measured - self.estimate
        if self._locked:
            self.estimate = self.estimate + SMOOTHING * innovation
            velocity = self.velocity + VELOCITY_GAIN * innovation / dt
            speed = float(np.linalg.norm(velocity))
            self.velocity = velocity * min(1.0, MAX_SPEED_MPS / max(speed, 1e-9))
        else:
            self.estimate = measured
        self._misses = 0
        self._locked = True
        return self.estimate

    @property
    def source(self) -> str:
        """Which detector produced the latest result."""
        return self._detector.name

    def _is_jump(self, measured: np.ndarray) -> bool:
        """Reject outliers, but only once tracking is established and still trusted."""
        trusted = self._locked and not self.lost
        return trusted and np.linalg.norm(measured - self.estimate) > MAX_JUMP_M

    @staticmethod
    def _measure(pixel: np.ndarray, frame: Frame) -> np.ndarray | None:
        u, v = np.round(pixel).astype(int)
        height, width = frame.depth.shape
        if not (DEPTH_PATCH <= u < width - DEPTH_PATCH and DEPTH_PATCH <= v < height - DEPTH_PATCH):
            return None
        patch = frame.depth[
            v - DEPTH_PATCH : v + DEPTH_PATCH + 1, u - DEPTH_PATCH : u + DEPTH_PATCH + 1
        ]
        return deproject(pixel, float(np.median(patch)), frame)


def annotate(rgb: np.ndarray, pixel: np.ndarray | None, source: str, correction_cm: float):
    """Draw the detection and the spoon correction on a copy of a camera image."""
    image = rgb.copy()
    found = pixel is not None
    color = MARKER_COLOR if found else SEARCH_COLOR
    if found:
        center = tuple(int(value) for value in pixel)
        cv2.circle(image, center, MARKER_RADIUS, color, 2)
        cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, MARKER_RADIUS * 3, 1)
    cv2.rectangle(image, (0, 0), (image.shape[1], BANNER_HEIGHT), (20, 20, 24), -1)
    cv2.putText(image, f"WRIST CAMERA  {source}", (8, 17), TEXT_FONT, TEXT_SCALE, color, 1)
    correction = f"SPOON CORRECTION {correction_cm:.1f} cm"
    cv2.putText(image, correction, (8, 37), TEXT_FONT, TEXT_SCALE, (240, 240, 240), 1)
    return image


def standby_image(width: int, height: int) -> np.ndarray:
    """The picture shown while the camera is not tracking."""
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    position = (width // 2 - 92, height // 2)
    cv2.putText(image, "WRIST CAMERA STANDBY", position, TEXT_FONT, TEXT_SCALE, (150, 150, 155), 1)
    return image
