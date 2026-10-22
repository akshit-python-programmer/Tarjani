import json
from types import SimpleNamespace

import pytest

from planner import (PlanError, Planner, PlanningFailed, RuleBasedPlanner, build_system_prompt, parse_json,
                     validate_plan)
from skill_executor import SkillExecutor

OBJECTS = ["red cup", "blue cup", "yellow ball"]
LOCS = ["left zone", "right zone"]
SCENE = {n: {"x": 0, "y": 0, "width_mm": 50, "height_mm": 50} for n in OBJECTS}


class FakeClient:
    """Stands in for openai.OpenAI; returns queued replies and records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw)))     # deep copy
        content = self.replies.pop(0)
        if not isinstance(content, str):
            content = json.dumps(content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


GOOD = {"plan": [{"skill": "pick", "obj": "red cup"}, {"skill": "place", "target": "left zone"}]}


def test_system_prompt_mentions_json_and_every_skill():
    sp = build_system_prompt()
    assert "JSON" in sp
    for s in SkillExecutor.SKILLS:
        assert f'"{s}"' in sp


def test_valid_plan_passes():
    steps, msg = validate_plan(GOOD, OBJECTS, LOCS)
    assert steps == GOOD["plan"] and msg == ""


@pytest.mark.parametrize("bad,why", [
    ({}, "plan"),
    ({"plan": "pick cup"}, "plan"),
    ({"plan": [{"skill": "throw", "obj": "red cup"}]}, "unknown skill"),
    ({"plan": [{"skill": "pick", "obj": "green cup"}]}, "not visible"),
    ({"plan": [{"skill": "pick"}]}, "needs"),
    ({"plan": [{"skill": "place", "target": "left zone"}]}, "without a preceding pick"),
    ({"plan": [{"skill": "pick", "obj": "red cup"}, {"skill": "pick", "obj": "blue cup"}]}, "already holding"),
    ({"plan": [{"skill": "pick", "obj": "red cup"}, {"skill": "place", "target": "red cup"}]}, "itself"),
    ({"plan": [{"skill": "pick", "obj": "left zone"}]}, "not visible"),      # locations are not objects
])
def test_invalid_plans_rejected(bad, why):
    with pytest.raises(PlanError, match=why):
        validate_plan(bad, OBJECTS, LOCS)


def test_extra_keys_dropped():
    steps, _ = validate_plan({"plan": [{"skill": "pick", "obj": "red cup", "speed": "fast"}]}, OBJECTS, LOCS)
    assert steps == [{"skill": "pick", "obj": "red cup"}]


def test_parse_json_handles_fences_and_garbage():
    assert parse_json('```json\n{"plan": []}\n```') == {"plan": []}
    with pytest.raises(PlanError):
        parse_json("")
    with pytest.raises(PlanError):
        parse_json("sure! here is the plan")


def test_planner_sends_text_only_and_json_mode(cfg):
    client = FakeClient([GOOD])
    res = Planner(cfg, client=client).plan("put the red cup on the left", SCENE)
    assert res.steps == GOOD["plan"]
    req = client.requests[0]
    assert req["response_format"] == {"type": "json_object"}
    assert req["model"] == cfg["planner"]["model"]
    for m in req["messages"]:
        assert isinstance(m["content"], str)            # no image parts
    assert "JSON" in req["messages"][0]["content"]
    user = req["messages"][1]["content"]
    assert "red cup" in user and "put the red cup on the left" in user
    assert '"x"' not in user                            # names only, no coordinates


def test_planner_retries_invalid_output_with_error_feedback(cfg):
    client = FakeClient(["not json", {"plan": [{"skill": "pick", "obj": "purple cup"}]}, GOOD])
    res = Planner(cfg, client=client).plan("put the red cup on the left", SCENE)
    assert res.steps == GOOD["plan"] and res.attempts == 3
    last_msgs = client.requests[2]["messages"]
    assert "purple cup" in last_msgs[-1]["content"] and "invalid" in last_msgs[-1]["content"]


def test_planner_gives_up_after_max_replans(cfg):
    client = FakeClient(["nope"] * 10)
    with pytest.raises(PlanningFailed):
        Planner(cfg, client=client).plan("do it", SCENE)
    assert len(client.requests) == cfg["planner"]["max_replans"] + 1


def test_replan_after_executor_error_shares_budget(cfg):
    client = FakeClient([GOOD, {"plan": [{"skill": "pick", "obj": "blue cup"}, {"skill": "place", "target": "right zone"}]},
                         GOOD])
    p = Planner(cfg, client=client)
    p.plan("move a cup", SCENE)
    res = p.replan("Unreachable: red cup out of reach", SCENE, done_steps=[])
    assert res.steps[0]["obj"] == "blue cup"
    assert "out of reach" in client.requests[1]["messages"][-1]["content"]
    p.replan("again", SCENE)
    with pytest.raises(PlanningFailed):
        p.replan("and again", SCENE)


def test_replan_knows_what_is_held(cfg):
    client = FakeClient([{"plan": [{"skill": "place", "target": "right zone"}]}])
    p = Planner(cfg, client=client)
    p.messages = [{"role": "system", "content": "x"}]
    res = p.replan("left zone unreachable", SCENE, done_steps=[{"skill": "pick", "obj": "red cup"}], holding="red cup")
    assert res.steps == [{"skill": "place", "target": "right zone"}]


def test_rule_based_planner(cfg):
    rb = RuleBasedPlanner(cfg)
    assert rb.plan("put the red cup in the left zone", SCENE).steps == GOOD["plan"]
    assert rb.plan("put the yellow ball on the blue cup", SCENE).steps == [
        {"skill": "pick", "obj": "yellow ball"}, {"skill": "place", "target": "blue cup"}]
    assert rb.plan("go home", SCENE).steps == [{"skill": "home"}]
