"""Text prompt -> skill plan, using DeepSeek (OpenAI-compatible API) in JSON mode.

Only text is sent: the user's prompt and the names of visible objects / named locations.
The reply must be {"plan": [{"skill": ..., "obj" | "target": ...}, ...]} using only
SkillExecutor.SKILLS and only visible names. Invalid replies and executor errors
(Unreachable / LookupError) are sent back to the model for a re-plan, at most
planner.max_replans times in total per prompt.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

from skill_executor import SkillExecutor

log = logging.getLogger("planner")


class PlanError(ValueError):
    """The model's output is not a valid plan."""


class PlanningFailed(RuntimeError):
    """Gave up after the re-plan budget was used."""


@dataclass
class PlanResult:
    steps: list
    message: str = ""            # model's note, e.g. why the task is impossible
    attempts: int = 1
    raw: dict = field(default_factory=dict)


def build_system_prompt(skills=None):
    skills = skills or SkillExecutor.SKILLS
    lines = []
    for name, s in skills.items():
        arg = f', "{s["arg"]}": <name>' if s["arg"] else ""
        lines.append(f'- {{"skill": "{name}"{arg}}}: {s["desc"]}')
    return (
        "You control a robot arm with a five-finger hand above a table. Turn the user's request "
        "into a short plan of skills.\n\n"
        "Available skills (use no others):\n" + "\n".join(lines) + "\n\n"
        "Rules:\n"
        '- "obj" must be exactly one of the visible object names. "target" must be exactly one of the '
        "visible object names or named locations. Copy names character for character.\n"
        "- The hand holds one object at a time: pick, then place, before picking again.\n"
        "- Do not add a final home step; the robot goes home automatically.\n"
        "- If the request cannot be done with the visible objects, return an empty plan and explain in "
        '"message".\n\n'
        "Respond with a single JSON object and nothing else, in this format:\n"
        '{"plan": [{"skill": "pick", "obj": "red cup"}, {"skill": "place", "target": "left zone"}], '
        '"message": ""}'
    )


def build_user_message(prompt, objects, locations, holding=None):
    return (f"Visible objects: {json.dumps(sorted(objects))}\n"
            f"Named locations: {json.dumps(sorted(locations))}\n"
            f"Currently holding: {json.dumps(holding) if holding else 'nothing'}\n"
            f"Request: {prompt}")


def parse_json(text):
    if not text or not text.strip():
        raise PlanError("empty response")
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanError(f"response is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise PlanError("response must be a JSON object")
    return data


def validate_plan(data, objects, locations, skills=None, holding=None):
    """Check a parsed reply. Returns (steps, message) or raises PlanError naming the problem."""
    skills = skills or SkillExecutor.SKILLS
    objects, locations = set(objects), set(locations)
    if "plan" not in data or not isinstance(data["plan"], list):
        raise PlanError('missing "plan" list')
    message = data.get("message", "") or ""
    if not isinstance(message, str):
        message = str(message)
    steps, held = [], holding
    for i, step in enumerate(data["plan"]):
        where = f"step {i + 1}"
        if not isinstance(step, dict):
            raise PlanError(f"{where}: must be an object")
        skill = step.get("skill")
        if skill not in skills:
            raise PlanError(f"{where}: unknown skill {skill!r}; allowed: {sorted(skills)}")
        arg = skills[skill]["arg"]
        clean = {"skill": skill}
        if arg:
            name = step.get(arg)
            if not isinstance(name, str):
                raise PlanError(f'{where}: skill "{skill}" needs "{arg}" (a name string)')
            allowed = objects | locations if arg == "target" else objects
            if name not in allowed:
                raise PlanError(f'{where}: "{name}" is not visible; allowed {arg} names: {sorted(allowed)}')
            clean[arg] = name
        if skill == "pick":
            if held:
                raise PlanError(f'{where}: already holding "{held}"; place it before picking "{clean["obj"]}"')
            held = clean["obj"]
        elif skill == "place":
            if not held:
                raise PlanError(f"{where}: place without a preceding pick")
            if clean["target"] == held:
                raise PlanError(f'{where}: cannot place "{held}" on itself')
            held = None
        elif skill == "open_hand":
            held = None
        steps.append(clean)
    return steps, message


class Planner:
    def __init__(self, cfg, client=None):
        p = cfg["planner"]
        self.model = p["model"]
        self.temperature = p.get("temperature", 0.0)
        self.max_replans = int(p["max_replans"])
        self.locations = list((cfg.get("locations") or {}).keys())
        self.system_prompt = build_system_prompt()
        assert "JSON" in self.system_prompt   # DeepSeek JSON mode requires the word in the prompt
        if client is None:
            from openai import OpenAI
            key = os.environ.get(p.get("api_key_env", "DEEPSEEK_API_KEY"))
            if not key:
                raise RuntimeError(f"environment variable {p.get('api_key_env', 'DEEPSEEK_API_KEY')} is not set")
            client = OpenAI(api_key=key, base_url=p["base_url"], timeout=p.get("timeout_s", 30.0))
        self.client = client
        self.messages = []
        self.replans_used = 0
        self._objects = []

    def plan(self, prompt, scene, holding=None):
        """Start a new conversation for this prompt and return a validated PlanResult."""
        self._objects = list(scene)
        self.replans_used = 0
        self.messages = [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": build_user_message(prompt, self._objects, self.locations, holding)}]
        return self._ask(holding)

    def replan(self, error, scene, done_steps=(), holding=None):
        """Feed an executor error back and ask for a new plan (counts against max_replans)."""
        if self.replans_used >= self.max_replans:
            raise PlanningFailed(f"gave up after {self.max_replans} re-plans; last error: {error}")
        self.replans_used += 1
        self._objects = list(scene)
        done = ", ".join(json.dumps(s) for s in done_steps) or "none"
        self.messages.append({"role": "user", "content": (
            f"Executing your plan failed: {error}\nSteps already completed: {done}\n"
            f"Visible objects now: {json.dumps(sorted(self._objects))}\n"
            f"Currently holding: {json.dumps(holding) if holding else 'nothing'}\n"
            "Return a new JSON plan for the remaining work only.")})
        return self._ask(holding)

    def _ask(self, holding):
        while True:
            content = self._call()
            self.messages.append({"role": "assistant", "content": content or ""})
            try:
                data = parse_json(content)
                steps, message = validate_plan(data, self._objects, self.locations, holding=holding)
                return PlanResult(steps, message, attempts=self.replans_used + 1, raw=data)
            except PlanError as e:
                log.warning("invalid plan from model: %s", e)
                if self.replans_used >= self.max_replans:
                    raise PlanningFailed(f"gave up after {self.max_replans} re-plans; last error: {e}") from None
                self.replans_used += 1
                self.messages.append({"role": "user", "content": f"Your reply was invalid: {e}. "
                                      "Reply again with only the corrected JSON object."})

    def _call(self):
        resp = self.client.chat.completions.create(
            model=self.model, messages=self.messages, temperature=self.temperature,
            response_format={"type": "json_object"}, max_tokens=800)
        return resp.choices[0].message.content


class RuleBasedPlanner:
    """Offline fallback (no API key / no network). Understands a few phrasings:
    "put/move/place X on/in/to Y", "pick up X", "point at X", "open hand", "go home".
    """

    def __init__(self, cfg):
        self.locations = list((cfg.get("locations") or {}).keys())
        self.max_replans = 0
        self.replans_used = 0

    def _find(self, text, names):
        text = text.lower()
        best, score = None, 0
        for n in names:
            key = n.lower().replace("_", " ")
            if key in text:
                s = 100 + len(key)
            else:
                words = set(re.findall(r"[a-z0-9]+", key))
                s = len(words & set(re.findall(r"[a-z0-9]+", text)))
            if s > score:
                best, score = n, s
        return best

    def plan(self, prompt, scene, holding=None):
        objects = list(scene)
        text = prompt.lower().strip()
        if re.search(r"\b(home|reset)\b", text):
            return PlanResult([{"skill": "home"}])
        if re.search(r"\bopen\b", text) and "hand" in text:
            return PlanResult([{"skill": "open_hand"}])
        m = re.search(r"(?:put|place|move|drop|stack|set)\s+(.+?)\s+(?:on top of|onto|on|into|in|to|at|next to)\s+(.+)", text)
        if m:
            obj = self._find(m.group(1), objects)
            target = self._find(m.group(2), [o for o in objects if o != obj] + self.locations)
            if obj and target:
                steps = [] if holding == obj else [{"skill": "pick", "obj": obj}]
                return PlanResult(steps + [{"skill": "place", "target": target}])
        m = re.search(r"(?:pick up|pick|grab|lift|take)\s+(.+)", text)
        if m and (obj := self._find(m.group(1), objects)):
            return PlanResult([{"skill": "pick", "obj": obj}])
        m = re.search(r"(?:point at|point to|show me|where is)\s+(.+)", text)
        if m and (obj := self._find(m.group(1), objects)):
            return PlanResult([{"skill": "point", "obj": obj}])
        return PlanResult([], "offline planner did not understand; try 'put the red cup in the left zone'")

    def replan(self, error, scene, done_steps=(), holding=None):
        raise PlanningFailed(f"offline planner cannot re-plan: {error}")
