"""Run the feeding robot, listening to the microphone unless --no-mic is given.

Typed lines also reach the robot. Debug keys: w pauses, r resumes, q quits.
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from feeding_robot.display import ConsoleSink, Display, PanelSink
from feeding_robot.planning.llm_backend import OllamaBackend
from feeding_robot.planning.planner import PlannerBackend, PlanningError, PlanWorker, common_phrases
from feeding_robot.shared import EventLog, SharedState
from feeding_robot.simulation.perception import (
    FaceMeshDetector,
    FallbackDetector,
    LipColorDetector,
    MouthTracker,
)
from feeding_robot.simulation.sim import FeedingRobot, PyBulletWorld, Scene
from feeding_robot.speech.asr_backend import DEFAULT_ASR_MODEL, SpeechCppBackend, SpeechCppServer
from feeding_robot.speech.audio import ASRWorker, AudioFrontend, UtteranceSegmenter
from feeding_robot.speech.tts import OmniVoiceServer, OmniVoiceSynthesizer, Speaker, Synthesizer

ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-mic", action="store_true", help="use keyboard input only")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="speech.cpp model")
    parser.add_argument("--inject-wav", type=Path, help="replay a file instead of the microphone")
    parser.add_argument("--headless", action="store_true", help="run without a window")
    parser.add_argument("--fast", action="store_true", help="run faster than real time")
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.yaml")
    parser.add_argument("--llm-model", help="Ollama model (default: $OLLAMA_MODEL)")
    parser.add_argument("--still-head", action="store_true", help="keep the head still")
    parser.add_argument("--no-tts", action="store_true", help="do not speak replies")
    return parser.parse_args()


def create_backend(args: argparse.Namespace, scene: Scene) -> PlannerBackend:
    backend = OllamaBackend(scene, args.llm_model)
    try:
        backend.warm_up()
    except PlanningError as error:
        logger.warning("%s; falling back to rules until the model responds", error)
    return backend


def read_keyboard(shared: SharedState, worker: PlanWorker) -> None:
    """Treat typed lines as utterances until quit or end of input."""
    actions = {"w": shared.on_speech_onset, "r": shared.on_resume}
    try:
        while (command := input("> ").strip()) != "q":
            if command in actions:
                actions[command]()
            elif command:
                shared.on_speech_onset()
                worker.submit(command)
    except EOFError:
        pass
    finally:
        shared.stop.set()


@contextmanager
def speech_server(args: argparse.Namespace) -> Iterator[SpeechCppServer | None]:
    """Run the recognition server for the duration of the block, if voice input is on."""
    if args.no_mic:
        yield None
        return
    with SpeechCppServer([args.asr_model]) as server:
        yield server


@contextmanager
def voice_input(
    args: argparse.Namespace,
    server: SpeechCppServer | None,
    shared: SharedState,
    worker: PlanWorker,
) -> Iterator[None]:
    """Listen for speech for the duration of the block, unless voice input is off."""
    if args.no_mic:
        yield
        return
    recognizer = ASRWorker(SpeechCppBackend(server.url, args.asr_model), shared, worker.submit)
    recognizer.start()
    segmenter = UtteranceSegmenter(shared.on_speech_onset, recognizer.enqueue)
    frontend = AudioFrontend(shared, segmenter, args.inject_wav)
    frontend.start()
    shared.log.write("listening")
    try:
        yield
    finally:
        frontend.close()


@contextmanager
def voice_synthesizer(args: argparse.Namespace) -> Iterator[Synthesizer | None]:
    """Provide the cloned voice for the duration of the block, unless speech is off."""
    if args.no_tts:
        yield None
        return
    with OmniVoiceServer() as omnivoice:
        omnivoice.clone_voice()
        yield OmniVoiceSynthesizer(omnivoice.url)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    scene = Scene.load(args.scene)
    log_path = ROOT / "logs" / f"events-{datetime.now():%Y%m%d-%H%M%S}.csv"
    shared = SharedState(log=EventLog(log_path))
    panel = PanelSink()
    Display(scene, shared, (ConsoleSink(), panel)).attach()
    world = PyBulletWorld(scene, headless=args.headless, fast=args.fast)
    tracker = None
    if not args.still_head:
        detector = FallbackDetector(FaceMeshDetector(), LipColorDetector())
        tracker = MouthTracker(detector, np.array(scene.person["mouth"]))
    robot = FeedingRobot(
        world, scene, shared, tracker, head_moves=not args.still_head, status=panel
    )
    worker = PlanWorker(create_backend(args, scene), shared, scene)
    worker.start()
    threading.Thread(target=read_keyboard, args=(shared, worker), daemon=True).start()
    try:
        with speech_server(args) as server, voice_synthesizer(args) as synthesizer:
            if synthesizer is not None:
                Speaker(shared, synthesizer).start(common_phrases(scene))
            with voice_input(args, server, shared, worker):
                robot.run()
    finally:
        world.close()
        shared.log.close()


if __name__ == "__main__":
    main()
