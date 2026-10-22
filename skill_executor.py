"""Arm kinematics, Arduino serial link and the skills the planner can call.

Layers:
  Kinematics   analytic IK / FK for a yaw + 3-link planar arm (base, shoulder, elbow, wrist)
  ArduinoLink  serial protocol "J,a0,a1,a2,a3" / "H,f0,..,f4", newline-terminated, DONE/ERR replies.
               port=None gives a mock link that just logs.
  SkillExecutor  waypoint moves, pick/place skills, plan runner with retries.

All numbers come from config.yaml (see config.load_config).
"""

import logging
import math
import threading
import time

import numpy as np

log = logging.getLogger("executor")


class Unreachable(Exception):
    """Target pose is outside the workspace or joint limits."""


class GraspFailed(Exception):
    """The object was still on the table after every grasp attempt."""


class EmergencyStop(Exception):
    """The stop event was set; motion was aborted."""


class ArduinoError(Exception):
    """The Arduino replied ERR, or did not reply in time."""


class ArduinoTimeout(ArduinoError):
    pass


# --------------------------------------------------------------------------- kinematics

class Kinematics:
    """Joint angles in degrees: [base, shoulder, elbow, wrist]. Positions in table-frame mm."""

    def __init__(self, arm_cfg):
        self.cfg = arm_cfg
        self.bx, self.by = (float(v) for v in arm_cfg["base_xy_mm"])
        self.yaw0 = float(arm_cfg["base_yaw_offset_deg"])
        self.z0 = float(arm_cfg["shoulder_height_mm"])
        L = arm_cfg["link_lengths_mm"]
        self.l1, self.l2, self.l3 = float(L["upper"]), float(L["fore"]), float(L["hand"])
        self.names = list(arm_cfg["joint_names"])
        self.limits = np.array([arm_cfg["joint_limits_deg"][n] for n in self.names], float)
        self.offset = np.array(arm_cfg["servo_offset_deg"], float)
        self.sign = np.array(arm_cfg["servo_sign"], float)

    def fk(self, joints):
        """Joint angles -> (x, y, z, pitch_deg) of the grasp centre in the table frame."""
        yaw, t1, t2, t3 = (math.radians(a) for a in joints)
        r = self.l1 * math.cos(t1) + self.l2 * math.cos(t1 + t2) + self.l3 * math.cos(t1 + t2 + t3)
        z = self.l1 * math.sin(t1) + self.l2 * math.sin(t1 + t2) + self.l3 * math.sin(t1 + t2 + t3)
        heading = yaw + math.radians(self.yaw0)
        return (self.bx + r * math.cos(heading), self.by + r * math.sin(heading),
                self.z0 + z, _wrap_deg(math.degrees(t1 + t2 + t3)))

    def ik(self, x, y, z, pitch_deg=-90.0):
        """Table-frame grasp-centre position + hand pitch -> joint angles (deg).

        pitch_deg is the hand angle above horizontal, measured in the vertical plane
        through the base axis and the target, facing away from the base (-90 = straight down).
        Tries elbow-up then elbow-down, first facing the target, then with the base turned
        180 deg and the arm reaching back over itself. Raises Unreachable if no solution is
        within the joint limits.
        """
        dx, dy = x - self.bx, y - self.by
        r = math.hypot(dx, dy)
        heading = math.degrees(math.atan2(dy, dx)) if r > 1e-9 else self.yaw0
        errors = []
        for flip in (False, True):
            yaw = _wrap_deg(heading - self.yaw0 + (180.0 if flip else 0.0))
            rr = -r if flip else r
            phi = math.radians(180.0 - pitch_deg if flip else pitch_deg)
            # wrist pivot in the arm plane, relative to the shoulder
            wr = rr - self.l3 * math.cos(phi)
            wz = (z - self.z0) - self.l3 * math.sin(phi)
            d = (wr * wr + wz * wz - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
            if abs(d) > 1.0:
                errors.append("out of reach")
                continue
            for elbow_sign in (-1.0, 1.0):          # -1 = elbow up (forearm bent down)
                t2 = elbow_sign * math.acos(d)
                t1 = math.atan2(wz, wr) - math.atan2(self.l2 * math.sin(t2), self.l1 + self.l2 * math.cos(t2))
                t3 = phi - t1 - t2
                joints = np.array([yaw, _wrap_deg(math.degrees(t1)), math.degrees(t2),
                                   _wrap_deg(math.degrees(t3))])
                bad = self.limit_violations(joints)
                if not bad:
                    return joints
                errors.append(", ".join(bad))
        raise Unreachable(f"({x:.0f},{y:.0f},{z:.0f}) pitch {pitch_deg:.0f}: no solution ({'; '.join(errors)})")

    def limit_violations(self, joints):
        out = []
        for name, a, (lo, hi) in zip(self.names, joints, self.limits):
            if not lo - 1e-6 <= a <= hi + 1e-6:
                out.append(f"{name}={a:.1f} not in [{lo:.0f},{hi:.0f}]")
        return out

    def check_limits(self, joints):
        bad = self.limit_violations(joints)
        if bad:
            raise Unreachable("joint limits: " + ", ".join(bad))

    def to_servo(self, joints):
        """Joint angles -> servo degrees. Enforces joint limits and the 0..180 servo range."""
        self.check_limits(joints)
        servo = self.offset + self.sign * np.asarray(joints, float)
        if np.any(servo < -1e-6) or np.any(servo > 180 + 1e-6):
            raise Unreachable(f"servo command {np.round(servo, 1).tolist()} outside 0..180 "
                              "(check servo_offset_deg / servo_sign / joint_limits_deg)")
        return np.clip(servo, 0, 180)


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- serial link

def format_joint_cmd(servo_deg):
    vals = [int(round(v)) for v in servo_deg]
    if len(vals) != 4:
        raise ValueError("joint command needs 4 values")
    if any(v < 0 or v > 180 for v in vals):
        raise ValueError(f"servo values must be 0..180, got {vals}")
    return "J," + ",".join(str(v) for v in vals) + "\n"


def format_hand_cmd(finger_deg):
    vals = [int(round(v)) for v in finger_deg]
    if len(vals) != 5:
        raise ValueError("hand command needs 5 values")
    if any(v < 0 or v > 180 for v in vals):
        raise ValueError(f"finger values must be 0..180, got {vals}")
    return "H," + ",".join(str(v) for v in vals) + "\n"


class ArduinoLink:
    """Serial link to hand_controller.ino. port=None -> mock mode (no hardware)."""

    def __init__(self, port=None, baud=115200, timeout_s=10.0, boot_wait_s=3.0,
                 retries=1, mock_move_time_s=0.0, abort_event=None, serial_obj=None):
        self.port = port
        self.timeout_s = timeout_s
        self.retries = retries
        self.mock_move_time_s = mock_move_time_s
        self.abort_event = abort_event or threading.Event()
        self.sent = []                      # every line sent (handy for tests and logs)
        self.mock = port is None and serial_obj is None
        self._ser = serial_obj
        if self.mock:
            log.info("ArduinoLink: MOCK mode (no serial port)")
        elif self._ser is None:
            import serial  # pyserial
            self._ser = serial.Serial(port, baud, timeout=0.1)
            self._wait_ready(boot_wait_s)

    @classmethod
    def from_config(cls, cfg, port=None, abort_event=None):
        s = cfg["serial"]
        return cls(port=port, baud=s["baud"], timeout_s=s["move_timeout_s"],
                   boot_wait_s=s.get("boot_wait_s", 3.0), retries=s.get("retries", 1),
                   mock_move_time_s=s.get("mock_move_time_s", 0.0), abort_event=abort_event)

    def _wait_ready(self, wait_s):
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            line = self._ser.readline().decode(errors="replace").strip()
            if line == "READY":
                log.info("Arduino READY on %s", self.port)
                return
        log.warning("no READY from Arduino within %.1fs, continuing anyway", wait_s)

    def send_joints(self, servo_deg):
        self._send(format_joint_cmd(servo_deg))

    def send_hand(self, finger_deg):
        self._send(format_hand_cmd(finger_deg))

    def _send(self, line, ignore_abort=False):
        for attempt in range(self.retries + 1):
            try:
                self._send_once(line, ignore_abort)
                return
            except ArduinoTimeout:
                if attempt == self.retries:
                    raise
                log.warning("timeout on %r, re-sending (%d/%d)", line.strip(), attempt + 1, self.retries)

    def _send_once(self, line, ignore_abort):
        if not ignore_abort and self.abort_event.is_set():
            raise EmergencyStop()
        self.sent.append(line)
        log.log(logging.INFO if self.mock else logging.DEBUG, "-> %s", line.strip())
        if self.mock:
            end = time.monotonic() + self.mock_move_time_s
            while time.monotonic() < end:
                if not ignore_abort and self.abort_event.is_set():
                    raise EmergencyStop()
                time.sleep(0.005)
            return
        self._ser.write(line.encode("ascii"))
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if not ignore_abort and self.abort_event.is_set():
                raise EmergencyStop()
            reply = self._ser.readline().decode(errors="replace").strip()
            if not reply:
                continue
            log.debug("<- %s", reply)
            if reply == "DONE":
                return
            if reply.startswith("ERR"):
                raise ArduinoError(f"Arduino rejected {line.strip()!r}: {reply}")
            # anything else (READY, debug prints) is ignored
        raise ArduinoTimeout(f"no DONE for {line.strip()!r} within {self.timeout_s}s")

    def send_unconditional(self, line):
        """Send ignoring the abort flag (used to go home after an emergency stop)."""
        if not self.mock:
            self._ser.reset_input_buffer()
        self._send(line, ignore_abort=True)

    def close(self):
        if self._ser is not None:
            self._ser.close()


# --------------------------------------------------------------------------- skills

class SkillExecutor:
    # Skills the planner may use: name -> required argument and description (fed to the LLM).
    SKILLS = {
        "pick": {"arg": "obj", "desc": "grasp a visible object and lift it (hand must be empty)"},
        "place": {"arg": "target", "desc": "put the held object on top of a visible object or at a named location"},
        "point": {"arg": "obj", "desc": "hover the hand above a visible object"},
        "open_hand": {"arg": None, "desc": "open the hand (drops anything held where it is)"},
        "home": {"arg": None, "desc": "return the arm to its safe home pose"},
    }

    def __init__(self, cfg, link, vision=None, stop_event=None):
        self.cfg = cfg
        self.g = cfg["grasp"]
        self.hand_cfg = cfg["hand"]
        self.locations = cfg.get("locations", {}) or {}
        self.kin = Kinematics(cfg["arm"])
        self.link = link
        self.vision = vision
        self.stop_event = stop_event or link.abort_event
        self.joints = None              # last commanded joint angles; None = unknown
        self.holding = None             # (name, scene entry) of the held object
        self.home_joints = np.array(cfg["arm"]["home_deg"], float)
        self.kin.check_limits(self.home_joints)

    # ---- low level moves

    def _check_stop(self):
        if self.stop_event.is_set():
            raise EmergencyStop()

    def move_joints(self, joints):
        self._check_stop()
        servo = self.kin.to_servo(joints)       # raises Unreachable on limit violation
        self.link.send_joints(servo)
        self.joints = np.asarray(joints, float)

    def set_hand(self, fingers):
        self._check_stop()
        for i, (v, (lo, hi)) in enumerate(zip(fingers, self.hand_cfg["limits_deg"])):
            if not lo <= v <= hi:
                raise ValueError(f"finger {i} value {v} outside limits [{lo},{hi}]")
        self.link.send_hand(fingers)

    def open_hand(self, step=None):
        self.set_hand(self.hand_cfg["open"])
        self.holding = None

    def home(self, step=None):
        if self.joints is not None:
            x, y, z, pitch = self.pose()
            if z < self.g["safe_z_mm"] - 1:      # rise first so the swing home clears the table
                try:
                    self.move_linear(x, y, self.g["safe_z_mm"], pitch)
                except Unreachable:
                    log.debug("could not rise straight up before homing")
        self.move_joints(self.home_joints)

    def emergency_home(self):
        """After a stop: clear the stop flag, open nothing, drive straight home."""
        self.stop_event.clear()
        servo = self.kin.to_servo(self.home_joints)
        self.link.send_unconditional(format_joint_cmd(servo))
        self.joints = self.home_joints.copy()

    def pose(self):
        return self.kin.fk(self.joints if self.joints is not None else self.home_joints)

    def choose_pitch(self, points):
        """First pitch candidate for which every (x, y, z) point has an IK solution."""
        last = None
        for pitch in self.g["pitch_candidates_deg"]:
            try:
                for p in points:
                    self.kin.ik(*p, pitch_deg=pitch)
                return float(pitch)
            except Unreachable as e:
                last = e
        raise Unreachable(f"no hand pitch reaches all of {[tuple(round(c) for c in p) for p in points]}: {last}")

    def move_linear(self, x, y, z, pitch):
        """Straight-line move of the grasp centre, split into waypoints. All IK is checked before moving."""
        x0, y0, z0, _ = self.pose()
        dist = math.dist((x0, y0, z0), (x, y, z))
        n = max(1, math.ceil(dist / self.g["cartesian_step_mm"]))
        path = [self.kin.ik(x0 + (x - x0) * i / n, y0 + (y - y0) * i / n, z0 + (z - z0) * i / n, pitch)
                for i in range(1, n + 1)]
        prev = self.joints if self.joints is not None else path[0]
        for q in path:
            if np.max(np.abs(q - prev)) > self.g["max_joint_jump_deg"]:
                raise Unreachable(f"straight move to ({x:.0f},{y:.0f},{z:.0f}) needs a joint flip")
            prev = q
        for q in path:
            self.move_joints(q)

    def transit(self, x, y, z, pitch):
        """Rise to safe height, swing over (x, y) in joint space, then descend in a straight line to z."""
        safe = max(self.g["safe_z_mm"], z)
        target_up = self.kin.ik(x, y, safe, pitch)           # check before moving
        self.kin.ik(x, y, z, pitch)
        cx, cy, cz, cp = self.pose()
        if cz < safe - 1:
            try:
                self.move_linear(cx, cy, safe, cp)
            except Unreachable:
                # e.g. starting from home; go up in joint space instead
                self.move_joints(self.kin.ik(cx, cy, safe, self.choose_pitch([(cx, cy, safe)])))
        self.move_joints(target_up)
        if z < safe:
            self.move_linear(x, y, z, pitch)

    # ---- scene helpers

    def _lookup(self, name, scene):
        if name in scene:
            return scene[name]
        raise LookupError(f"'{name}' is not in the scene (visible: {sorted(scene)})")

    def _lookup_target(self, name, scene):
        if name in scene:
            return scene[name], False
        if name in self.locations:
            loc = self.locations[name]
            return {"x": loc["x"], "y": loc["y"], "width_mm": 0, "height_mm": 0}, True
        raise LookupError(f"'{name}' is not a visible object or named location "
                          f"(objects: {sorted(scene)}, locations: {sorted(self.locations)})")

    def grasp_preset(self, width_mm):
        for g in sorted(self.hand_cfg["grasps"], key=lambda g: g["max_width_mm"]):
            if width_mm <= g["max_width_mm"]:
                return g
        raise Unreachable(f"object {width_mm:.0f} mm wide is wider than the widest grasp preset")

    # ---- skills

    def point(self, step, scene):
        o = self._lookup(step["obj"], scene)
        z = o["height_mm"] + self.g["approach_clearance_mm"]
        pitch = self.choose_pitch([(o["x"], o["y"], z), (o["x"], o["y"], max(z, self.g["safe_z_mm"]))])
        self.transit(o["x"], o["y"], z, pitch)

    def pick(self, step, scene):
        name = step["obj"]
        o = self._lookup(name, scene)
        if self.holding:
            raise ValueError(f"cannot pick '{name}': already holding '{self.holding[0]}'")
        preset = self.grasp_preset(o["width_mm"])
        attempts = 1 + int(self.g["grasp_retries"])
        for attempt in range(1, attempts + 1):
            x, y, h = o["x"], o["y"], o["height_mm"]
            grasp_z = max(h - self.g["finger_depth_mm"], self.g["min_grasp_z_mm"])
            above_z = h + self.g["approach_clearance_mm"]
            lift_z = max(self.g["safe_z_mm"], above_z)
            pitch = self.choose_pitch([(x, y, grasp_z), (x, y, above_z), (x, y, lift_z)])
            log.info("pick %s attempt %d/%d at (%.0f, %.0f) z=%.0f pitch=%.0f preset=%s",
                     name, attempt, attempts, x, y, grasp_z, pitch, preset["name"])
            self.set_hand(self.hand_cfg["open"])
            self.transit(x, y, above_z, pitch)
            self.move_linear(x, y, grasp_z, pitch)
            self.set_hand(preset["fingers"])
            self.move_linear(x, y, lift_z, pitch)
            if self._verify_grasp(name, x, y, lift_z, pitch):
                self.holding = (name, dict(o))
                return
            log.warning("grasp check failed for %s", name)
            self.set_hand(self.hand_cfg["open"])
            if self.vision is not None:     # object may have been nudged; re-measure
                fresh = self.vision.scene_mm()
                if name in fresh:
                    o = fresh[name]
        raise GraspFailed(f"could not pick '{name}' after {attempts} attempts")

    def _verify_grasp(self, name, x, y, z, pitch):
        if self.vision is None:
            return True
        # Move towards the base so the arm stops hiding the spot the object came from.
        dx, dy = self.kin.bx - x, self.kin.by - y
        d = math.hypot(dx, dy)
        off = min(self.g["verify_offset_mm"], max(d - 1.0, 0.0))
        if off > 0:
            try:
                self.move_linear(x + dx / d * off, y + dy / d * off, z, pitch)
            except Unreachable:
                log.debug("verify offset move unreachable; checking in place")
        return bool(self.vision.object_lifted(name))

    def place(self, step, scene):
        if not self.holding:
            raise ValueError("place called while not holding anything")
        held_name, held = self.holding
        target = step["target"]
        if target == held_name:
            raise LookupError(f"cannot place '{held_name}' on itself")
        t, _is_location = self._lookup_target(target, scene)
        rest_z = t["height_mm"]                  # bottom of the held object when placed
        release_z = rest_z + max(held["height_mm"] - self.g["finger_depth_mm"], self.g["min_grasp_z_mm"]) \
            + self.g["place_clearance_mm"]
        above_z = rest_z + held["height_mm"] + self.g["approach_clearance_mm"]
        pitch = self.choose_pitch([(t["x"], t["y"], release_z), (t["x"], t["y"], above_z),
                                   (t["x"], t["y"], max(above_z, self.g["safe_z_mm"]))])
        log.info("place %s on %s at (%.0f, %.0f) z=%.0f", held_name, target, t["x"], t["y"], release_z)
        self.transit(t["x"], t["y"], above_z, pitch)
        self.move_linear(t["x"], t["y"], release_z, pitch)
        self.set_hand(self.hand_cfg["open"])
        self.holding = None
        self.move_linear(t["x"], t["y"], above_z, pitch)
        if self.vision is not None and hasattr(self.vision, "notify_moved"):
            self.vision.notify_moved(held_name, t["x"], t["y"])

    # ---- plan runner

    def run_plan(self, plan, scene):
        """Run [{"skill": ..., "obj"/"target": ...}, ...] against scene {name: {x, y, width_mm, height_mm}}.

        Raises Unreachable / LookupError (for re-planning), GraspFailed, EmergencyStop, ArduinoError.
        The failing step index is attached as exc.step_index. Ends at home on success.
        """
        for i, step in enumerate(plan):
            skill = step.get("skill")
            if skill not in self.SKILLS:
                raise LookupError(f"step {i}: unknown skill {skill!r}")
            arg = self.SKILLS[skill]["arg"]
            if arg and arg not in step:
                raise LookupError(f"step {i}: skill '{skill}' needs '{arg}'")
            log.info("step %d/%d: %s %s", i + 1, len(plan), skill, step.get(arg, "") if arg else "")
            try:
                fn = getattr(self, skill)
                fn(step, scene) if skill in ("pick", "place", "point") else fn(step)
            except Exception as e:
                e.step_index = i
                raise
        self.home()
