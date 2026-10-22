import threading
import time

from main import App
from planner import RuleBasedPlanner
from skill_executor import ArduinoLink, SkillExecutor
from vision import FakeVision


def make_app(cfg, scene_path, move_time=0.0):
    stop = threading.Event()
    link = ArduinoLink(port=None, mock_move_time_s=move_time, abort_event=stop)
    vision = FakeVision(cfg, scene_path)
    ex = SkillExecutor(cfg, link, vision, stop)
    app = App(cfg, ex, vision, RuleBasedPlanner(cfg), stop, window=False, read_stdin=False)
    return app, link, vision


def run_app(app):
    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    return t


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_prompt_confirm_execute(cfg, scene_path):
    app, link, vision = make_app(cfg, scene_path)
    for line in ["put the red cup in the right zone", "y", "q"]:
        app.lines.put(line)
    t = run_app(app)
    t.join(10)
    assert not t.is_alive()
    assert vision.scene["red cup"]["x"] == 480.0


def test_declining_does_not_move(cfg, scene_path):
    app, link, vision = make_app(cfg, scene_path)
    for line in ["put the red cup in the right zone", "n", "q"]:
        app.lines.put(line)
    run_app(app).join(10)
    assert link.sent == []


def test_emergency_stop_during_execution_goes_home(cfg, scene_path):
    app, link, vision = make_app(cfg, scene_path, move_time=0.05)
    app.lines.put("put the red cup in the right zone")
    app.lines.put("y")
    t = run_app(app)
    assert wait_for(lambda: app.state == "executing" and len(link.sent) > 3)
    app.lines.put("e")
    assert wait_for(lambda: app.state == "idle")
    app.lines.put("q")
    t.join(10)
    assert link.sent[-1] == "J,90,90,90,90\n"            # last command is home
    assert vision.scene["red cup"]["x"] == 220.0          # place never happened
