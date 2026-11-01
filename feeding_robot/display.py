"""Live view of what the robot hears, plans and does, in the terminal and the demo window."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from feeding_robot.shared import SharedState
from feeding_robot.simulation.scenery import STATUS_PIXELS
from feeding_robot.simulation.sim import Scene

LANE_WIDTH = 6
PANEL_LINES = 8
PANEL_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"
PANEL_FONT_SIZE = 20
PANEL_BACKGROUND = (20, 20, 24)
PANEL_TEXT = (238, 238, 238)
PANEL_NOTE = (150, 150, 158)
PANEL_MARGIN = 14
PANEL_LANE_WIDTH = 86
PANEL_NOTE_GAP = 12
ELLIPSIS = "…"


@dataclass(frozen=True)
class Line:
    lane: str
    color: tuple[int, int, int]
    text: str
    note: str = ""


CYAN = (90, 220, 240)
MAGENTA = (230, 130, 240)
GREEN = (110, 230, 130)
YELLOW = (250, 215, 90)
BLUE = (120, 170, 255)
GRAY = (150, 150, 158)
WHITE = (238, 238, 238)


class Sink(Protocol):
    def show(self, line: Line) -> None: ...


class ConsoleSink:
    """Prints each line to the terminal."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def show(self, line: Line) -> None:
        red, green, blue = line.color
        lane = f"[bold rgb({red},{green},{blue})]{line.lane:<{LANE_WIDTH}}[/]"
        note = f"  [dim]{line.note}[/dim]" if line.note else ""
        self._console.print(f"{lane} {line.text}{note}", highlight=False)


class PanelSink:
    """Draws recent lines into an image for the simulator to show on its main thread."""

    def __init__(self) -> None:
        self._lines: deque[Line] = deque(maxlen=PANEL_LINES)
        self._font = ImageFont.truetype(PANEL_FONT, PANEL_FONT_SIZE)
        self._lock = threading.Lock()
        self._image: np.ndarray | None = self._render()

    def show(self, line: Line) -> None:
        with self._lock:
            self._lines.append(line)
            self._image = self._render()

    def take_image(self) -> np.ndarray | None:
        """Return a new image once, or None if nothing changed."""
        with self._lock:
            image, self._image = self._image, None
        return image

    def _fit(self, draw: ImageDraw.ImageDraw, text: str, width: int) -> str:
        if draw.textlength(text, font=self._font) <= width:
            return text
        while text and draw.textlength(text + ELLIPSIS, font=self._font) > width:
            text = text[:-1]
        return text + ELLIPSIS

    def _render(self) -> np.ndarray:
        width, height = STATUS_PIXELS
        image = Image.new("RGB", STATUS_PIXELS, PANEL_BACKGROUND)
        draw = ImageDraw.Draw(image)
        row_height = (height - 2 * PANEL_MARGIN) / PANEL_LINES
        left = PANEL_MARGIN + PANEL_LANE_WIDTH
        room = width - left - PANEL_MARGIN
        for row, line in enumerate(self._lines):
            y = PANEL_MARGIN + row * row_height
            draw.text((PANEL_MARGIN, y), line.lane, fill=line.color, font=self._font)
            text = self._fit(draw, line.text, room)
            draw.text((left, y), text, fill=PANEL_TEXT, font=self._font)
            used = draw.textlength(text, font=self._font) + PANEL_NOTE_GAP
            if line.note and used < room:
                note = self._fit(draw, line.note, int(room - used))
                draw.text((left + used, y), note, fill=PANEL_NOTE, font=self._font)
        return np.asarray(image)


class Display:
    """Turns pipeline events into lines for each sink."""

    def __init__(self, scene: Scene, shared: SharedState, sinks: Sequence[Sink]) -> None:
        self._scene = scene
        self._shared = shared
        self._sinks = sinks
        self._handlers = {
            "listening": self._listening,
            "asr": self._heard,
            "llm": self._planned,
            "plan_changed": self._plan_changed,
            "robot_stopped": self._stopped,
            "step_started": self._step_started,
            "step_replaced": self._step_cut,
            "food_eaten": self._ate,
        }

    def attach(self) -> None:
        self._shared.log.subscribe(self.on_event)

    def on_event(self, event: str, detail: dict[str, Any]) -> None:
        if handler := self._handlers.get(event):
            handler(detail)

    def _line(self, *parts: Any) -> None:
        for sink in self._sinks:
            sink.show(Line(*parts))

    def _food_name(self, food_id: str) -> str:
        food = self._scene.foods.get(food_id)
        return food.name if food else food_id

    def _step_label(self, skill: str, args: dict[str, Any]) -> str:
        labels = {"feed": "口へ運ぶ", "go_home": "待機位置へ", "wait_user": "待機"}
        if skill == "scoop":
            amount = {"small": "（小）", "large": "（大）"}.get(args.get("amount"), "")
            return f"すくう: {self._food_name(args['food_id'])}{amount}"
        if skill == "cool":
            return f"冷ます {args['seconds']:.0f}秒"
        return labels.get(skill, skill)

    def _listening(self, detail: dict[str, Any]) -> None:
        self._line("READY", WHITE, "話しかけてください")

    def _heard(self, detail: dict[str, Any]) -> None:
        self._line("ASR", CYAN, detail["text"], f"{detail['ms']:.0f} ms")

    def _planned(self, detail: dict[str, Any]) -> None:
        output = detail["output"]
        words = " → ".join(output["steps"]) or "（なし）"
        fallback = ", fallback" if detail["fallback"] else ""
        self._line(
            "PLAN", MAGENTA, f"{output['mode']}: {words}", f"{detail['ms']:.0f} ms{fallback}"
        )

    def _plan_changed(self, detail: dict[str, Any]) -> None:
        for message in (detail.get("reply"), detail.get("question")):
            if message:
                self._line("SAY", GREEN, message)

    def _stopped(self, detail: dict[str, Any]) -> None:
        self._line("STOP", YELLOW, f"発話で停止: {detail['latency_ms']:.1f} ms")

    def _step_started(self, detail: dict[str, Any]) -> None:
        label = self._step_label(detail["skill"], detail["args"])
        self._line("STEP", BLUE, label, f"残り {len(self._shared.plan)} ステップ")

    def _step_cut(self, detail: dict[str, Any]) -> None:
        self._line("STEP", GRAY, f"中断: {detail['skill']}")

    def _ate(self, detail: dict[str, Any]) -> None:
        name = self._food_name(detail["food"])
        self._line("EAT", GREEN, name, f"口元との距離 {detail['gap_cm']:.1f} cm")
