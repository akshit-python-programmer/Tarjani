import cv2
import numpy as np
import pytest

from vision import FakeVision, assign_names, colour_name, parallax_correct


def project_to_table(point3d, cam):
    """Where the ray from a pinhole at cam=(cx, cy, H) through point3d hits z=0."""
    (x, y, z), (cx, cy, H) = point3d, cam
    t = H / (H - z)
    return cx + (x - cx) * t, cy + (y - cy) * t


def test_parallax_point_under_camera_unchanged():
    assert np.allclose(parallax_correct((100, 50), (100, 50), 700, 90), (100, 50))


def test_parallax_zero_height_unchanged():
    assert np.allclose(parallax_correct((250, 40), (300, 200), 700, 0), (250, 40))


def test_parallax_formula_values():
    # true = c + (m - c) * (H - h) / H
    out = parallax_correct((400, 300), (300, 200), 700, 70)
    assert np.allclose(out, (300 + 100 * 0.9, 200 + 100 * 0.9))


@pytest.mark.parametrize("obj", [(120, 80, 90), (450, 320, 200), (300, 210, 30)])
def test_parallax_inverts_pinhole_projection(obj):
    cam = (310.0, 190.0, 700.0)
    measured = project_to_table(obj, cam)
    assert np.allclose(parallax_correct(measured, cam[:2], cam[2], obj[2]), obj[:2], atol=1e-9)


def test_parallax_rejects_height_above_camera():
    with pytest.raises(ValueError):
        parallax_correct((0, 0), (0, 0), 700, 700)


def hsv_of(bgr):
    return cv2.cvtColor(np.uint8([[bgr]] * 10), cv2.COLOR_BGR2HSV).reshape(-1, 3)


@pytest.mark.parametrize("bgr,name", [((0, 0, 220), "red"), ((200, 30, 30), "blue"), ((30, 200, 30), "green"),
                                      ((0, 220, 230), "yellow"), ((240, 240, 240), "white"), ((20, 20, 20), "black")])
def test_colour_names(bgr, name):
    assert colour_name(hsv_of(bgr)) == name


def test_red_wraparound_mean():
    # half the pixels hue ~2, half ~178: arithmetic mean (90) would say cyan
    px = np.array([[2, 200, 200]] * 5 + [[178, 200, 200]] * 5)
    assert colour_name(px) == "red"


def test_duplicate_names_numbered_left_to_right():
    dets = [{"class": "cup", "colour": "white", "x": 300, "y": 0},
            {"class": "cup", "colour": "white", "x": 100, "y": 0},
            {"class": "cup", "colour": "red", "x": 200, "y": 0}]
    names = {d["x"]: d["name"] for d in assign_names(dets)}
    assert names == {100: "white cup_1", 300: "white cup_2", 200: "red cup"}


def test_fake_vision_scene_format(cfg, scene_path):
    scene = FakeVision(cfg, scene_path).scene_mm()
    assert scene
    for entry in scene.values():
        for k in ("x", "y", "width_mm", "height_mm"):
            assert isinstance(entry[k], (int, float))


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = (_T(np.array(a, float)) for a in (xyxy, cls, conf))

    def __len__(self):
        return len(self.xyxy.a)


class _T:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeModel:
    names = {0: "cup"}

    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, frame, **kw):
        return [type("R", (), {"boxes": self.boxes})()]


class FakeCamera:
    """1 px = 1 mm, table origin at pixel (200, 600), y up; principal point at table (300, 200)."""

    def __init__(self, frame):
        self.frame = frame
        self.H = np.array([[1, 0, -200], [0, -1, 600], [0, 0, 1]], float)

    def read(self):
        return self.frame.copy(), {}, self.H

    def principal_point(self):
        return 500.0, 400.0

    def draw_markers(self, img, H):
        return img

    def release(self):
        pass


def test_vision_pipeline_parallax_names_and_lift(cfg):
    frame = np.full((720, 1280, 3), 200, np.uint8)
    frame[380:420, 680:720] = (0, 0, 230)       # red patch at pixel centre (700, 400) -> table (500, 200)
    frame[380:420, 280:320] = (230, 230, 230)   # white patch at (300, 400) -> table (100, 200)
    frame[180:220, 280:320] = (230, 230, 230)   # white patch at (300, 200) -> table (100, 400)
    boxes = FakeBoxes([[670, 370, 730, 430], [270, 370, 330, 430], [270, 170, 330, 230]], [0, 0, 0], [.9, .9, .9])
    from vision import Vision
    v = Vision(cfg, camera=FakeCamera(frame), model=FakeModel(boxes))
    scene = v.scene_mm()
    assert set(scene) == {"red cup", "white cup_1", "white cup_2"}
    # cup is 90 mm tall, corrected at 45 mm; camera 700 mm; c = (300, 200)
    k = (700 - 45) / 700
    assert scene["red cup"]["x"] == pytest.approx(300 + 200 * k)
    assert scene["red cup"]["y"] == pytest.approx(200)
    assert scene["red cup"]["width_mm"] == 70 and scene["red cup"]["height_mm"] == 90
    assert v.object_lifted("red cup") is False                  # still detected in place
    v.model = FakeModel(FakeBoxes([[270, 370, 330, 430]], [0], [.9]))
    assert v.object_lifted("red cup") is True                   # gone from its spot
