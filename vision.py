"""YOLOv8 detection -> table-frame object positions.

scene_mm() returns {name: {"x", "y", "width_mm", "height_mm", "class", "colour"}}, the format
SkillExecutor.run_plan expects. Names are "<colour> <class>", with "_1", "_2", ... appended
(left to right) when several objects share a name, e.g. "red cup", "white cup_1", "white cup_2".
"""

import json
import logging
import math
import threading
from pathlib import Path

import cv2
import numpy as np

from camera import Camera, pixel_to_table, table_to_pixel

log = logging.getLogger("vision")


# --------------------------------------------------------------------------- pure functions

def parallax_correct(measured, c, cam_height, h):
    """true = c + (measured - c) * (H - h) / H.

    measured: table-plane point where the ray through the object point lands (mm)
    c: table point directly below the camera; cam_height: H; h: height of the observed point.
    """
    measured, c = np.asarray(measured, float), np.asarray(c, float)
    if h >= cam_height:
        raise ValueError("object height must be below the camera")
    return c + (measured - c) * (cam_height - h) / cam_height


# (name, hue centre in OpenCV units 0..180)
_HUES = [("red", 0), ("orange", 15), ("yellow", 30), ("green", 60), ("cyan", 90),
         ("blue", 115), ("purple", 140), ("pink", 165)]


def colour_name(hsv_pixels):
    """Colour word from an (N,3) array of OpenCV HSV pixels.

    Hue is circular (red sits at both 0 and 180), so its mean is taken as a circular mean
    over the saturated pixels; a plain arithmetic mean would call a red object cyan.
    """
    hsv = np.asarray(hsv_pixels, float).reshape(-1, 3)
    if len(hsv) == 0:
        return "unknown"
    s, v = hsv[:, 1], hsv[:, 2]
    coloured = (s > 70) & (v > 50)
    if coloured.mean() < 0.3:
        mv = v.mean()
        return "black" if mv < 60 else "white" if mv > 170 and s.mean() < 50 else "grey"
    ang = hsv[coloured, 0] * (2 * math.pi / 180.0)
    hue = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180.0
    return min(_HUES, key=lambda t: min(abs(hue - t[1]), 180 - abs(hue - t[1])))[0]


def assign_names(dets):
    """dets: list of dicts with "class", "colour", "x", "y". Adds "name" in place and returns them."""
    groups = {}
    for d in dets:
        groups.setdefault(f"{d['colour']} {d['class']}", []).append(d)
    for base, items in groups.items():
        if len(items) == 1:
            items[0]["name"] = base
        else:
            for k, d in enumerate(sorted(items, key=lambda d: (d["x"], d["y"])), 1):
                d["name"] = f"{base}_{k}"
    return dets


def object_size(cfg_objects, cls):
    s = cfg_objects.get(cls, cfg_objects["default"])
    return float(s["width_mm"]), float(s["height_mm"])


# --------------------------------------------------------------------------- real vision

class Vision:
    def __init__(self, cfg, camera=None, model=None):
        self.cfg = cfg
        v = cfg["vision"]
        self.objects = v["objects"]
        self.conf = v["confidence"]
        self.h_frac = v.get("parallax_height_fraction", 0.5)
        self.cam_h = float(cfg["camera"]["height_mm"])
        self.camera = camera or Camera.from_config(cfg)
        if model is None:
            from ultralytics import YOLO   # heavy import, only when the camera is used
            model = YOLO(v["model"])
        self.model = model
        self.lock = threading.RLock()
        self.frame, self.H, self.dets = None, None, []
        self._snapshot = {}

    def update(self):
        """Grab a frame, detect, map to table mm. Returns the list of detections."""
        with self.lock:
            frame, _markers, H = self.camera.read()
            self.frame, self.H = frame, H
            self.dets = self._detect(frame, H) if H is not None else []
            return self.dets

    def _detect(self, frame, H):
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        if len(res.boxes) == 0:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        c = pixel_to_table(H, [self.camera.principal_point()])[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        dets = []
        for (x1, y1, x2, y2), k, cf in zip(boxes, classes, confs):
            cls = self.model.names[int(k)]
            width, height = object_size(self.objects, cls)
            centre_px = ((x1 + x2) / 2, (y1 + y2) / 2)
            measured = pixel_to_table(H, [centre_px])[0]
            x, y = parallax_correct(measured, c, self.cam_h, height * self.h_frac)
            # colour from the inner half of the box (less table background)
            bw, bh = x2 - x1, y2 - y1
            roi = hsv[int(y1 + bh / 4):int(y2 - bh / 4), int(x1 + bw / 4):int(x2 - bw / 4)]
            dets.append({"class": cls, "colour": colour_name(roi.reshape(-1, 3)), "conf": float(cf),
                         "x": float(x), "y": float(y), "width_mm": width, "height_mm": height,
                         "box": (int(x1), int(y1), int(x2), int(y2))})
        return assign_names(dets)

    def scene_mm(self):
        """Current scene. Also remembers it as the reference for object_lifted()."""
        with self.lock:
            if self.frame is None:
                self.update()
            scene = {d["name"]: {k: d[k] for k in ("x", "y", "width_mm", "height_mm", "class", "colour")}
                     for d in self.dets}
            self._snapshot = scene
            return {k: dict(v) for k, v in scene.items()}

    def object_lifted(self, name):
        """True if no object of the same class is near where `name` was in the last scene_mm() call."""
        ref = self._snapshot.get(name)
        if ref is None:
            log.warning("object_lifted(%r): not in the reference scene", name)
            return False
        n = self.cfg["vision"].get("lift_check_frames", 3)
        radius = max(self.cfg["vision"].get("lift_check_radius_mm", 30.0), ref["width_mm"] / 2)
        still_there = 0
        for _ in range(n):
            dets = self.update()
            if any(d["class"] == ref["class"] and math.hypot(d["x"] - ref["x"], d["y"] - ref["y"]) < radius
                   for d in dets):
                still_there += 1
        return still_there <= n // 2

    def render(self):
        with self.lock:
            if self.frame is None:
                return None
            img = self.frame.copy()
            self.camera.draw_markers(img, self.H)
            for d in self.dets:
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, d["name"], (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(img, f"({d['x']:.0f}, {d['y']:.0f}) mm", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if self.H is not None:
                base = table_to_pixel(self.H, [self.cfg["arm"]["base_xy_mm"]])[0]
                cv2.drawMarker(img, tuple(int(v) for v in base), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            return img

    def close(self):
        self.camera.release()


# --------------------------------------------------------------------------- fake vision (--no-camera)

class FakeVision:
    """Scene loaded from a JSON file; objects move when the executor places them."""

    def __init__(self, cfg, path, fail_lifts=0):
        self.cfg = cfg
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)
        self.scene = {}
        for name, o in raw.items():
            for k in ("x", "y", "width_mm", "height_mm"):
                if k not in o:
                    raise ValueError(f"{path}: object '{name}' is missing '{k}'")
            self.scene[name] = {k: o[k] for k in o}
        self.fail_lifts = fail_lifts       # make the first N grasp checks fail (testing retries)

    def update(self):
        return self.scene

    def scene_mm(self):
        return {k: dict(v) for k, v in self.scene.items()}

    def object_lifted(self, name):
        if self.fail_lifts > 0:
            self.fail_lifts -= 1
            return False
        return name in self.scene

    def notify_moved(self, name, x, y):
        if name in self.scene:
            self.scene[name]["x"], self.scene[name]["y"] = float(x), float(y)

    def render(self, scale=1.2, margin=60):
        """Top-down map of the fake scene in table mm (y up)."""
        markers = self.cfg["camera"]["markers_mm"]
        bx, by = self.cfg["arm"]["base_xy_mm"]
        xs = [p[0] for p in markers.values()] + [bx]
        ys = [p[1] for p in markers.values()] + [by]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = int((x1 - x0) * scale + 2 * margin), int((y1 - y0) * scale + 2 * margin)
        img = np.full((h, w, 3), 40, np.uint8)

        def px(x, y):
            return int((x - x0) * scale + margin), int(h - ((y - y0) * scale + margin))

        corners = [px(*markers[k]) for k in sorted(markers)]
        cv2.polylines(img, [np.array(corners).reshape(-1, 1, 2)], True, (90, 90, 90), 1)
        for k in sorted(markers):
            p = px(*markers[k])
            cv2.rectangle(img, (p[0] - 8, p[1] - 8), (p[0] + 8, p[1] + 8), (255, 255, 255), -1)
            cv2.putText(img, f"ID{k}", (p[0] + 10, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.drawMarker(img, px(bx, by), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, "base", (px(bx, by)[0] + 12, px(bx, by)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        for name, loc in (self.cfg.get("locations") or {}).items():
            p = px(loc["x"], loc["y"])
            cv2.circle(img, p, 25, (120, 120, 120), 1)
            cv2.putText(img, name, (p[0] - 30, p[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        for name, o in self.scene.items():
            p = px(o["x"], o["y"])
            cv2.circle(img, p, max(4, int(o["width_mm"] * scale / 2)), (0, 200, 255), 2)
            cv2.putText(img, name, (p[0] - 30, p[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(img, f"({o['x']:.0f},{o['y']:.0f})", (p[0] - 30, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, "FAKE SCENE (--no-camera)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        return img

    def close(self):
        pass
