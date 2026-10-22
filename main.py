"""Text prompt -> scene -> plan -> confirm -> execute.

    python main.py --mock --no-camera              # no hardware at all
    python main.py --port COM5                     # real Arduino + camera
    python main.py --mock                          # camera, but no Arduino

Terminal: type a request, then y/n to confirm. "e" + Enter = emergency stop, "q" = quit.
Video window: press "e" for emergency stop, "q" to quit.
Ctrl+C anywhere: emergency stop, send the hand home, exit.
"""

import argparse
import logging
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config import load_config
from planner import PlanningFailed, Planner, RuleBasedPlanner
from skill_executor import (ArduinoError, ArduinoLink, EmergencyStop, GraspFailed, SkillExecutor,
                            Unreachable)

log = logging.getLogger("main")
WINDOW = "robot hand"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--mock", action="store_true", help="no Arduino: ArduinoLink(port=None)")
    p.add_argument("--port", default=None, help="serial port, e.g. COM5 or /dev/ttyUSB0")
    p.add_argument("--no-camera", nargs="?", const="scenes/demo_scene.json", default=None, metavar="SCENE_JSON",
                   help="use a fake scene from a JSON file (default scenes/demo_scene.json)")
    p.add_argument("--no-window", action="store_true", help="don't open the OpenCV window")
    p.add_argument("--planner", choices=["auto", "deepseek", "offline"], default="auto",
                   help="auto = DeepSeek if the API key is set, otherwise the offline rule-based planner")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def make_planner(cfg, choice):
    if choice == "offline":
        return RuleBasedPlanner(cfg)
    try:
        return Planner(cfg)
    except RuntimeError as e:
        if choice == "deepseek":
            raise SystemExit(f"planner: {e}")
        log.warning("%s -> using the OFFLINE rule-based planner", e)
        return RuleBasedPlanner(cfg)


def format_plan(steps):
    out = []
    for i, s in enumerate(steps, 1):
        arg = s.get("obj") or s.get("target") or ""
        out.append(f"  {i}. {s['skill']:<10} {arg}")
    return "\n".join(out)


class App:
    def __init__(self, cfg, executor, vision, planner, stop_event, window=True, read_stdin=True):
        self.cfg, self.executor, self.vision, self.planner = cfg, executor, vision, planner
        self.stop_event = stop_event
        self.window = window
        self.read_stdin = read_stdin
        self.lines = queue.Queue()
        self.pending = []
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.state = "idle"
        self.steps = []
        self.scene = {}
        self.running = True

    # ---- input

    def _read_stdin(self):
        for line in sys.stdin:
            self.lines.put(line.strip())
        self.lines.put(None)                  # EOF

    def _prompt(self):
        print("\nWhat should the robot do? ('q' quits, 'e' = emergency stop)\n> ", end="", flush=True)

    # ---- actions

    def emergency_stop(self):
        print("\n!!! EMERGENCY STOP - sending the hand home", flush=True)
        self.stop_event.set()
        if self.future is None or self.future.done():
            self._go_home_after_stop()
        # otherwise the worker raises EmergencyStop and _on_done homes

    def _go_home_after_stop(self):
        try:
            self.executor.emergency_home()
            print("hand is home.", flush=True)
        except Exception as e:                # never let homing errors hide the stop
            print(f"homing failed: {e}", flush=True)
        self.state = "idle"
        self._prompt()

    def _start(self, fn, *a, state):
        self.state = state
        self.future = self.pool.submit(fn, *a)

    def _handle_line(self, line):
        if line is None:                      # stdin closed
            if self.state == "idle":
                self.running = False
            return
        low = line.lower()
        if low == "e":
            self.emergency_stop()
            return
        if self.state == "idle":
            if low in ("q", "quit", "exit"):
                self.running = False
            elif line:
                print("planning...", flush=True)
                self.scene = self.vision.scene_mm()
                holding = self.executor.holding[0] if self.executor.holding else None
                self._start(self.planner.plan, line, self.scene, holding, state="planning")
            else:
                self._prompt()
        elif self.state == "confirm":
            if low in ("y", "yes"):
                self.stop_event.clear()
                self._start(self.executor.run_plan, self.steps, self.scene, state="executing")
                print("executing... ('e' + Enter or 'e' in the window to stop)", flush=True)
            elif low in ("n", "no"):
                print("cancelled.")
                self.state = "idle"
                self._prompt()
            else:
                print("please answer y or n: ", end="", flush=True)

    def _on_done(self):
        fut, self.future = self.future, None
        exc = fut.exception()
        if self.state == "planning":
            if exc is not None:
                print(f"planning failed: {exc}")
                self.state = "idle"
                self._prompt()
                return
            result = fut.result()
            if not result.steps:
                print(f"nothing to do. {result.message}".strip())
                self.state = "idle"
                self._prompt()
                return
            self.steps = result.steps
            print("\nPlan:" + (f"  ({result.message})" if result.message else ""))
            print(format_plan(self.steps))
            print("Execute? [y/n] ", end="", flush=True)
            self.state = "confirm"
        elif self.state == "executing":
            if exc is None:
                print("done.")
                self.state = "idle"
                self._prompt()
            elif isinstance(exc, EmergencyStop):
                self._go_home_after_stop()
            elif isinstance(exc, (Unreachable, LookupError)):
                i = getattr(exc, "step_index", 0)
                print(f"step {i + 1} failed: {exc}\nre-planning...", flush=True)
                holding = self.executor.holding[0] if self.executor.holding else None
                self.scene = self.vision.scene_mm()
                self._start(self.planner.replan, str(exc), self.scene, self.steps[:i], holding,
                            state="planning")
            else:
                kind = "grasp failed" if isinstance(exc, GraspFailed) else \
                    "Arduino error" if isinstance(exc, ArduinoError) else "error"
                print(f"{kind}: {exc}\nstopping and going home.")
                self.stop_event.set()
                self._go_home_after_stop()

    # ---- main loop

    def _show(self):
        if hasattr(self.vision, "camera"):
            self.vision.update()
        img = self.vision.render()
        if img is None:
            return -1
        import cv2
        cv2.putText(img, f"state: {self.state}   e=STOP q=quit", (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if self.state == "executing" else (200, 200, 200), 1)
        cv2.imshow(WINDOW, img)
        return cv2.waitKey(30) & 0xFF

    def run(self):
        if self.read_stdin:
            threading.Thread(target=self._read_stdin, daemon=True).start()
        self._prompt()
        try:
            while self.running:
                if self.window:
                    try:
                        key = self._show()
                    except Exception as e:     # e.g. opencv-python-headless has no GUI
                        log.warning("video window disabled: %s", e)
                        self.window = False
                        key = -1
                    if key == ord("e"):
                        self.emergency_stop()
                    elif key == ord("q") and self.state == "idle":
                        break
                try:
                    self.pending.append(self.lines.get(timeout=0.0 if self.window else 0.05))
                    while True:
                        self.pending.append(self.lines.get_nowait())
                except queue.Empty:
                    pass
                if self.state in ("planning", "executing"):
                    # only "e" acts while busy; anything else waits its turn
                    if any(l is not None and l.lower() == "e" for l in self.pending):
                        self.pending = [l for l in self.pending if l is None or l.lower() != "e"]
                        self.emergency_stop()
                elif self.pending:
                    self._handle_line(self.pending.pop(0))
                if self.future is not None and self.future.done():
                    self._on_done()
        except KeyboardInterrupt:
            print("\nCtrl+C - emergency stop")
            self.stop_event.set()
            if self.future is not None:
                try:
                    self.future.result(timeout=15)
                except Exception:
                    pass
            self.executor.emergency_home()
            print("hand is home.")
        finally:
            if self.future is not None and not self.future.done():
                self.stop_event.set()
            self.pool.shutdown(wait=True)
            if self.window:
                import cv2
                cv2.destroyAllWindows()


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")
    cfg = load_config(args.config)

    port = None if args.mock else (args.port or cfg["serial"].get("port"))
    if not args.mock and not port:
        raise SystemExit("no serial port: pass --port COM5 / --port /dev/ttyUSB0, set serial.port, or use --mock")

    stop_event = threading.Event()
    if args.no_camera:
        from vision import FakeVision
        vision = FakeVision(cfg, args.no_camera)
    else:
        from vision import Vision
        vision = Vision(cfg)
    link = ArduinoLink.from_config(cfg, port=port, abort_event=stop_event)
    executor = SkillExecutor(cfg, link, vision, stop_event)
    planner = make_planner(cfg, args.planner)

    print(f"arm: {'MOCK' if link.mock else port} | camera: {'fake ' + args.no_camera if args.no_camera else 'live'}"
          f" | planner: {type(planner).__name__}")
    executor.home()
    try:
        App(cfg, executor, vision, planner, stop_event, window=not args.no_window).run()
    finally:
        link.close()
        vision.close()


if __name__ == "__main__":
    main()
