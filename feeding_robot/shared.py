"""State shared between the simulation thread and input threads, plus the event log."""

from __future__ import annotations

import csv
import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_AMOUNT = "normal"


@dataclass(frozen=True)
class Step:
    """One skill invocation in the robot's plan."""

    skill: str
    args: dict[str, Any] = field(default_factory=dict)


class EventLog:
    """Thread-safe event log that writes CSV and notifies subscribers."""

    FIELDS = ("time", "event", "detail")

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[str, dict[str, Any]], None]] = []
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self.FIELDS)

    def subscribe(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        self._subscribers.append(callback)

    def write(self, event: str, **detail: Any) -> None:
        with self._lock:
            if self._file is not None:
                self._writer.writerow((f"{time.time():.6f}", event, detail or ""))
                self._file.flush()
        for callback in self._subscribers:
            callback(event, detail)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


@dataclass(frozen=True)
class Control:
    """Requests applied immediately, without going through the plan queue."""

    hold: bool | None = None
    speed_scale: float | None = None
    mouth_offset_cm: tuple[float, float, float] | None = None
    amount: str | None = None


@dataclass(frozen=True)
class Plan:
    """A planner decision, ready for validation."""

    mode: str = "keep"
    steps: tuple[Step, ...] = ()
    inserts: tuple[Step, ...] = ()
    control: Control = field(default_factory=Control)
    reply: str | None = None
    question: str | None = None


@dataclass
class SharedState:
    """Flags and queues that connect input threads to the simulation thread."""

    log: EventLog = field(default_factory=EventLog)
    plan: deque[Step] = field(default_factory=deque)
    eaten: Counter[str] = field(default_factory=Counter)
    speech: queue.SimpleQueue[str] = field(default_factory=queue.SimpleQueue)
    paused: threading.Event = field(default_factory=threading.Event)
    hold: threading.Event = field(default_factory=threading.Event)
    replaced: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)
    tts_playing: threading.Event = field(default_factory=threading.Event)
    active_utterances: int = 0
    onset_time: float | None = None
    current_step: Step | None = None
    spoon_food: str | None = None
    speed_factor: float = 1.0
    mouth_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amount: str = DEFAULT_AMOUNT
    _plan_lock: threading.Lock = field(default_factory=threading.Lock)
    _utterance_lock: threading.Lock = field(default_factory=threading.Lock)

    def on_speech_onset(self) -> None:
        """Freeze the robot the moment speech starts."""
        with self._utterance_lock:
            self.active_utterances += 1
            self.onset_time = time.perf_counter()
            self.paused.set()
        self.log.write("speech_onset")

    def end_utterance(self) -> None:
        """Release the pause once no utterance is in progress, even after a failure."""
        with self._utterance_lock:
            self.active_utterances = max(0, self.active_utterances - 1)
            if self.active_utterances == 0:
                self.paused.clear()
                self.onset_time = None
        self.log.write("speech_end")

    def on_resume(self) -> None:
        """Release a pause or a hold."""
        with self._utterance_lock:
            self.active_utterances = 0
            self.paused.clear()
        self.hold.clear()
        self.log.write("resume")

    def next_step(self) -> Step | None:
        """Pop the next step and mark it current."""
        with self._plan_lock:
            self.replaced.clear()
            self.current_step = self.plan.popleft() if self.plan else None
            return self.current_step

    def apply_plan(self, plan: Plan) -> None:
        """Apply a validated plan and its controls without touching the simulator."""
        self._apply_control(plan.control)
        with self._plan_lock:
            if plan.mode == "replace":
                self.plan.clear()
                self.plan.extend(plan.steps)
                self.hold.clear()
                self.replaced.set()
            elif plan.mode == "append":
                self.plan.extend(plan.steps)
            self._insert_before_feed(plan.inserts)
        for message in (plan.reply, plan.question):
            if message:
                self.speech.put(message)
        self.log.write(
            "plan_changed",
            mode=plan.mode,
            steps=len(plan.steps),
            reply=plan.reply,
            question=plan.question,
        )

    def _apply_control(self, control: Control) -> None:
        if control.hold is True:
            self.hold.set()
        elif control.hold is False:
            self.on_resume()
        if control.speed_scale is not None:
            self.speed_factor *= control.speed_scale
        if control.mouth_offset_cm is not None:
            self.mouth_offset_cm = control.mouth_offset_cm
        if control.amount is not None:
            self.amount = control.amount

    def _insert_before_feed(self, inserts: tuple[Step, ...]) -> None:
        """Place steps before the next feed, restarting a feed already in progress."""
        if not inserts:
            return
        if self.current_step is not None and self.current_step.skill == "feed":
            self.plan.appendleft(self.current_step)
            self.plan.extendleft(reversed(inserts))
            self.replaced.set()
            return
        steps = list(self.plan)
        index = next((i for i, s in enumerate(steps) if s.skill == "feed"), len(steps))
        steps[index:index] = inserts
        self.plan.clear()
        self.plan.extend(steps)

    def take_amount(self) -> str:
        """Return the requested one-shot amount and reset it."""
        amount, self.amount = self.amount, DEFAULT_AMOUNT
        return amount

    def robot_status(self) -> str:
        if self.hold.is_set():
            return "holding"
        return "working" if self.current_step or self.plan else "idle"
