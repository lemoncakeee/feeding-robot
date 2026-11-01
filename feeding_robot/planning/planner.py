"""Turn utterances into plans.

The LLM output is kept to a few tokens: a mode and a list of short words. Local generation
is slow enough that a verbose schema would take seconds, so this module expands the words
into steps and writes the spoken replies from templates.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Iterable
from itertools import chain, combinations, permutations
from typing import Protocol

from feeding_robot.planning.validator import validate
from feeding_robot.shared import Control, Plan, SharedState, Step
from feeding_robot.simulation.sim import Scene

logger = logging.getLogger(__name__)

MODES = ("replace", "append", "keep", "ask")
AMOUNT_WORDS = ("small", "large")
ACTION_WORDS = ("feed", "cool", "home", "wait")
CONTROL_WORDS = ("up", "down", "left", "right", "faster", "slower", "resume", "status")
OFFSET_WORDS_CM = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
}
SPEED_WORDS = {"faster": 1.25, "slower": 0.8}
DEFAULT_COOL_S = 3.0
REPLY_COOL = "少し冷ましますね。"
REPLY_WAIT = "はい、お待ちします。"
REPLY_RESUME = "はい、続けます。"
REPLY_ADJUST = "調整します。"
REPLY_AMOUNT = "はい、そうします。"
REPLY_ALL_EATEN = "全部召し上がりました。"
FIXED_REPLIES = (
    REPLY_COOL,
    REPLY_WAIT,
    REPLY_RESUME,
    REPLY_ADJUST,
    REPLY_AMOUNT,
)

SYSTEM_PROMPT = """手の不自由な方の食事支援ロボットの計画を返す。
料理: {foods}
stepsの語: 料理id=すくう small/large=次の一口の量 feed=口へ運ぶ cool=冷ます home=待機位置へ wait=待つ
resume=再開 up/down/left/right=口元の位置調整 faster/slower=速度 status=食べた料理の確認
mode: replace=今の計画を置換 append=計画の後ろに追加 keep=計画は変えない ask=聞き返す
料理をすくったら必ずfeedを続ける。指定された口数だけ繰り返す。交互なら料理を1口ずつ入れ替える。
音声認識の誤りは文脈から補正する。ロボットへの指示でない雑談は keep で steps は空。
料理が特定できない曖昧な指示は ask。料理名がない量の指定は keep で量の語だけ返す。
例: 煮物とプリンを交互に2口ずつ → {{"mode":"replace","steps":["nimono","feed","pudding","feed","nimono","feed","pudding","feed"]}}
例: 煮物とプリンを交互に1口ずつ → {{"mode":"replace","steps":["nimono","feed","pudding","feed"]}}
例: 煮物を小さめに → {{"mode":"replace","steps":["small","nimono","feed"]}}
例: 次からは小さめで → {{"mode":"keep","steps":["small"]}}
例: 一口大きめがいい → {{"mode":"keep","steps":["large"]}}
例: 冷まして → {{"mode":"keep","steps":["cool"]}}
例: 少し右 → {{"mode":"keep","steps":["right"]}}
例: ちょっと待って → {{"mode":"keep","steps":["wait"]}}
例: いいよ、続けて → {{"mode":"keep","steps":["resume"]}}
例: もう平気です → {{"mode":"keep","steps":["resume"]}}
例: そのあと煮物も → {{"mode":"append","steps":["nimono","feed"]}}
例: 今日は暑いね → {{"mode":"keep","steps":[]}}
例: 何か欲しい → {{"mode":"ask","steps":[]}}"""


def plan_schema(scene: Scene) -> dict:
    """JSON schema for LLM output. It is deliberately flat to keep generation short."""
    words = [*scene.foods, *AMOUNT_WORDS, *ACTION_WORDS, *CONTROL_WORDS]
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": list(MODES)},
            "steps": {"type": "array", "items": {"type": "string", "enum": words}},
        },
        "required": ["mode", "steps"],
    }


def system_prompt(scene: Scene) -> str:
    foods = " ".join(f"{food.id}={food.name}" for food in scene.foods.values())
    return SYSTEM_PROMPT.format(foods=foods)


def describe_state(shared: SharedState) -> str:
    """Summarize the robot's state for the LLM."""
    words = [
        step.args.get("food_id", step.skill)
        for step in ([shared.current_step] if shared.current_step else []) + list(shared.plan)
    ]
    eaten = " ".join(f"{food}={count}" for food, count in shared.eaten.items()) or "なし"
    return (
        f"状態={shared.robot_status()} 計画={','.join(words) or 'なし'} "
        f"スプーン={shared.spoon_food or 'なし'} 食べた回数: {eaten}"
    )


def user_prompt(text: str, shared: SharedState) -> str:
    return f"[{describe_state(shared)}]\n発話: {text}"


def _foods_phrase(steps: list[Step], scene: Scene) -> str:
    names = [scene.foods[s.args["food_id"]].name for s in steps if s.skill == "scoop"]
    return "と".join(dict.fromkeys(names))


def _bringing(foods: str) -> str:
    return f"{foods}をお持ちします。"


def _uneaten_phrase(names: Iterable[str]) -> str:
    listed = "、".join(names)
    return f"まだ{listed}を召し上がっていません。" if listed else REPLY_ALL_EATEN


def common_phrases(scene: Scene) -> tuple[str, ...]:
    """Every reply the templates can produce, apart from plans with three or more foods."""
    names = [food.name for food in scene.foods.values()]
    orders = chain(permutations(names, 1), permutations(names, 2))
    uneaten = chain.from_iterable(combinations(names, n) for n in range(len(names) + 1))
    return (
        *FIXED_REPLIES,
        _clarifying_question(scene),
        *(_bringing("と".join(order)) for order in orders),
        *(_uneaten_phrase(subset) for subset in uneaten),
    )


def _uneaten_reply(shared: SharedState, scene: Scene) -> str:
    return _uneaten_phrase(food.name for food in scene.foods.values() if not shared.eaten[food.id])


def _clarifying_question(scene: Scene) -> str:
    return f"{'、'.join(food.name for food in scene.foods.values())}のどれにしますか？"


def _offset_after(shared: SharedState, words: list[str]) -> tuple[float, float, float] | None:
    deltas = [OFFSET_WORDS_CM[word] for word in words if word in OFFSET_WORDS_CM]
    if not deltas:
        return None
    return tuple(
        base + sum(delta[axis] for delta in deltas)
        for axis, base in enumerate(shared.mouth_offset_cm)
    )


def _speed_scale(words: list[str]) -> float | None:
    scales = [SPEED_WORDS[word] for word in words if word in SPEED_WORDS]
    return scales[-1] if scales else None


def _expand_steps(words: list[str], scene: Scene) -> tuple[list[Step], str | None]:
    """Expand words into steps. Returns any amount word not followed by a food."""
    steps: list[Step] = []
    amount = None
    for word in words:
        if word in AMOUNT_WORDS:
            amount = word
        elif word in scene.foods:
            steps.append(Step("scoop", {"food_id": word, "amount": amount}))
            amount = None
        elif word == "feed":
            steps.append(Step("feed"))
        elif word == "cool":
            steps.append(Step("cool", {"seconds": DEFAULT_COOL_S}))
        elif word == "home":
            steps.append(Step("go_home"))
        elif word == "wait":
            steps.append(Step("wait_user"))
        elif word not in CONTROL_WORDS:
            steps.append(Step(word))
    if steps and steps[-1].skill == "scoop":
        steps.append(Step("feed"))
    return steps, amount


def _reply(mode: str, words: list[str], steps: list[Step], scene: Scene) -> str | None:
    if mode in ("replace", "append") and (foods := _foods_phrase(steps, scene)):
        return _bringing(foods)
    if "cool" in words:
        return REPLY_COOL
    if "wait" in words:
        return REPLY_WAIT
    if "resume" in words:
        return REPLY_RESUME
    if any(word in OFFSET_WORDS_CM or word in SPEED_WORDS for word in words):
        return REPLY_ADJUST
    if any(word in AMOUNT_WORDS for word in words):
        return REPLY_AMOUNT
    return None


def build_plan(raw: dict, shared: SharedState, scene: Scene) -> Plan:
    """Expand compact planner output into a Plan."""
    mode, words = raw["mode"], list(raw["steps"])
    if mode == "ask":
        return Plan(question=_clarifying_question(scene))
    steps, amount = _expand_steps(words, scene)
    control = Control(
        hold=True if words == ["wait"] else False if "resume" in words else None,
        speed_scale=_speed_scale(words),
        mouth_offset_cm=_offset_after(shared, words),
        amount=amount,
    )
    if "status" in words:
        return Plan(control=control, reply=_uneaten_reply(shared, scene))
    has_task = any(step.skill in ("scoop", "feed", "go_home") for step in steps)
    if not has_task:
        inserts = tuple(step for step in steps if step.skill != "wait_user" or words != ["wait"])
        reply = _reply("keep", words, steps, scene)
        return Plan(inserts=inserts, control=control, reply=reply)
    if mode == "keep":
        mode = "replace"
    return Plan(mode, tuple(steps), control=control, reply=_reply(mode, words, steps, scene))


_RESUME = re.compile(r"いいよ|いいです|大丈夫|続けて|再開|どうぞ")
_WAIT = re.compile(r"待って|まって|ストップ|止まって|とまって")
_HOME = re.compile(r"もういい|ごちそうさま|やめて")
_ASK = re.compile(r"何か|なにか|おなか|お腹")
_CHAT = re.compile(r"天気|暑い|寒い")
_ALTERNATE = re.compile(r"交互")
_BITES = re.compile(r"(\d+|[一二三四])口")
_KANJI_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4}
_RULE_WORDS = {
    "cool": re.compile(r"冷ま|さまして"),
    "small": re.compile(r"小さ|少なめ"),
    "large": re.compile(r"大きめ|多め"),
    "up": re.compile(r"上"),
    "down": re.compile(r"下"),
    "faster": re.compile(r"速く|早く"),
    "slower": re.compile(r"ゆっくり"),
}


def rule_parse(text: str, scene: Scene) -> dict:
    """Keyword fallback that returns the same compact form as the LLM."""
    if _WAIT.search(text):
        return {"mode": "keep", "steps": ["wait"]}
    if _RESUME.search(text):
        return {"mode": "keep", "steps": ["resume"]}
    if _HOME.search(text):
        return {"mode": "replace", "steps": ["home"]}
    if _CHAT.search(text):
        return {"mode": "keep", "steps": []}
    mentions = sorted(
        (min(text.index(alias) for alias in found), food_id)
        for food_id, aliases in _aliases(scene).items()
        if (found := [alias for alias in aliases if alias in text])
    )
    modifiers = [word for word, pattern in _RULE_WORDS.items() if pattern.search(text)]
    if not mentions:
        if modifiers:
            return {"mode": "keep", "steps": modifiers}
        return {"mode": "ask" if _ASK.search(text) else "keep", "steps": []}
    foods = [food_id for _, food_id in mentions]
    match = _BITES.search(text)
    bites = 1
    if match:
        bites = _KANJI_DIGITS.get(match[1]) or int(match[1])
    amounts = [word for word in ("small", "large") if word in modifiers]
    order = foods * bites if _ALTERNATE.search(text) or len(foods) == 1 else foods
    steps = [word for food in order for word in (*amounts, food, "feed")]
    return {"mode": "replace", "steps": steps}


def _aliases(scene: Scene) -> dict[str, tuple[str, ...]]:
    return {food.id: (food.name, *food.aliases) for food in scene.foods.values()}


class PlannerBackend(Protocol):
    def plan(self, text: str, shared: SharedState) -> tuple[dict, float]: ...


class RulesBackend:
    """Rule-based planner, used directly or as the LLM fallback."""

    name = "rules"

    def __init__(self, scene: Scene) -> None:
        self._scene = scene

    def plan(self, text: str, shared: SharedState) -> tuple[dict, float]:
        start = time.perf_counter()
        return rule_parse(text, self._scene), (time.perf_counter() - start) * 1000


class PlanWorker:
    """Plans utterances on its own thread so the simulation loop never waits for the LLM."""

    def __init__(self, backend: PlannerBackend, shared: SharedState, scene: Scene) -> None:
        self._backend = backend
        self._rules = RulesBackend(scene)
        self._shared = shared
        self._scene = scene
        self._utterances: queue.SimpleQueue[str] = queue.SimpleQueue()

    def submit(self, text: str) -> None:
        self._utterances.put(text)

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            self.handle(self._utterances.get())

    def _plan_with_fallback(self, text: str) -> dict:
        shared = self._shared
        model = getattr(self._backend, "model", self._backend.name)
        try:
            raw, elapsed_ms = self._backend.plan(text, shared)
            fallback = False
        except PlanningError as error:
            logger.warning("planner failed, using rules: %s", error)
            raw, elapsed_ms = self._rules.plan(text, shared)
            fallback = True
        shared.log.write(
            "llm", text=text, output=raw, model=model, ms=round(elapsed_ms, 1), fallback=fallback
        )
        return raw

    def handle(self, text: str) -> Plan:
        """Plan one utterance. The speech pause is always released afterwards."""
        shared = self._shared
        try:
            raw = self._plan_with_fallback(text)
            plan = validate(build_plan(raw, shared, self._scene), shared, self._scene.foods)
            shared.apply_plan(plan)
            return plan
        finally:
            shared.end_utterance()


class PlanningError(Exception):
    """A planner backend failed. The rule fallback takes over."""
