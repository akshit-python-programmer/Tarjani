import threading

import numpy as np
import pytest

from skill_executor import ArduinoLink, EmergencyStop, GraspFailed, SkillExecutor, Unreachable
from vision import FakeVision


@pytest.fixture
def setup(cfg, scene_path):
    stop = threading.Event()
    link = ArduinoLink(port=None, abort_event=stop)
    vision = FakeVision(cfg, scene_path)
    ex = SkillExecutor(cfg, link, vision, stop)
    return ex, link, vision, stop


def test_pick_and_place_runs_and_moves_object(setup):
    ex, link, vision, _ = setup
    scene = vision.scene_mm()
    ex.run_plan([{"skill": "pick", "obj": "red cup"}, {"skill": "place", "target": "left zone"}], scene)
    assert vision.scene["red cup"]["x"] == pytest.approx(120)
    assert link.sent[-1] == "J,90,90,90,90\n"            # ends at home
    hand_cmds = [l for l in link.sent if l.startswith("H")]
    assert hand_cmds[1] == "H,80,90,90,85,80\n"          # 70 mm cup -> "wide" preset
    assert ex.holding is None


def test_every_command_respects_limits(setup, cfg):
    ex, link, vision, _ = setup
    ex.run_plan([{"skill": "pick", "obj": "green block_1"}, {"skill": "place", "target": "blue cup"},
                 {"skill": "point", "obj": "yellow ball"}], vision.scene_mm())
    for line in link.sent:
        if line.startswith("J"):
            servo = np.array([int(v) for v in line[2:].split(",")])
            joints = (servo - ex.kin.offset) * ex.kin.sign
            assert not ex.kin.limit_violations(joints), line


def test_unknown_object_raises_lookup_error(setup):
    ex, _, vision, _ = setup
    with pytest.raises(LookupError) as e:
        ex.run_plan([{"skill": "pick", "obj": "purple cup"}], vision.scene_mm())
    assert e.value.step_index == 0


def test_unreachable_raises_and_sends_nothing_for_that_move(setup):
    ex, link, _, _ = setup
    scene = {"far cup": {"x": 1500.0, "y": 1500.0, "width_mm": 70, "height_mm": 90}}
    n = len(link.sent)
    with pytest.raises(Unreachable):
        ex.run_plan([{"skill": "pick", "obj": "far cup"}], scene)
    assert len(link.sent) == n


def test_grasp_retry_then_success(cfg, scene_path):
    link = ArduinoLink(port=None)
    vision = FakeVision(cfg, scene_path, fail_lifts=1)
    ex = SkillExecutor(cfg, link, vision)
    ex.run_plan([{"skill": "pick", "obj": "yellow ball"}], vision.scene_mm())
    assert ex.holding[0] == "yellow ball"
    assert sum(l == "H,10,10,10,10,10\n" for l in link.sent) >= 3     # open, re-open after fail, open again


def test_grasp_gives_up(cfg, scene_path):
    vision = FakeVision(cfg, scene_path, fail_lifts=99)
    ex = SkillExecutor(cfg, ArduinoLink(port=None), vision)
    with pytest.raises(GraspFailed):
        ex.run_plan([{"skill": "pick", "obj": "yellow ball"}], vision.scene_mm())


def test_emergency_stop_then_home(setup):
    ex, link, vision, stop = setup
    stop.set()
    with pytest.raises(EmergencyStop):
        ex.run_plan([{"skill": "pick", "obj": "red cup"}], vision.scene_mm())
    ex.emergency_home()
    assert not stop.is_set()
    assert link.sent[-1] == "J,90,90,90,90\n"


def test_too_wide_object_unreachable(setup):
    ex, _, _, _ = setup
    scene = {"box": {"x": 300.0, "y": 120.0, "width_mm": 200, "height_mm": 50}}
    with pytest.raises(Unreachable, match="wider"):
        ex.run_plan([{"skill": "pick", "obj": "box"}], scene)
