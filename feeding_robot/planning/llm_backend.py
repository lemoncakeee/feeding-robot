"""LLM planner backends."""

from __future__ import annotations

import json
import os
import time

import requests

from feeding_robot.planning.planner import PlanningError, plan_schema, system_prompt, user_prompt
from feeding_robot.shared import SharedState
from feeding_robot.simulation.sim import Scene

DEFAULT_MODEL = "qwen3:4b-instruct"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
CONNECT_TIMEOUT_S = 0.5
PLAN_TIMEOUT_S = 5.0  # a slower plan is safe: stopping never waits for the LLM
WARMUP_TIMEOUT_S = 120.0
KEEP_ALIVE = "30m"
OPTIONS = {"temperature": 0, "num_ctx": 4096}


class OllamaBackend:
    """Plans with a local Ollama model using structured output."""

    name = "ollama"

    def __init__(self, scene: Scene, model: str | None = None) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self._schema = plan_schema(scene)
        self._system = system_prompt(scene)

    def _chat(self, content: str, timeout_s: float) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": self._schema,
            "keep_alive": KEEP_ALIVE,
            "options": OPTIONS,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": content},
            ],
        }
        response = requests.post(OLLAMA_CHAT_URL, json=body, timeout=(CONNECT_TIMEOUT_S, timeout_s))
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])

    def warm_up(self) -> None:
        """Load the model so that the first utterance is not slowed by loading."""
        try:
            self._chat("発話: こんにちは", WARMUP_TIMEOUT_S)
        except (requests.RequestException, ValueError, KeyError) as error:
            raise PlanningError(f"warm-up failed: {error}") from error

    def plan(self, text: str, shared: SharedState) -> tuple[dict, float]:
        start = time.perf_counter()
        try:
            raw = self._chat(user_prompt(text, shared), PLAN_TIMEOUT_S)
        except (requests.RequestException, ValueError, KeyError) as error:
            raise PlanningError(str(error)) from error
        return raw, (time.perf_counter() - start) * 1000
