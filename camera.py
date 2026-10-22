"""Overhead camera: lens undistortion + ArUco table homography (pixel -> table mm).

Library use:
    cam = Camera.from_config(cfg)
    frame, markers, H = cam.read()          # undistorted frame, {id: (u, v)}, 3x3 homography or None

Command line:
    python camera.py calibrate --cols 9 --rows 6 --square 25      # lens calibration (checkerboard)
    python camera.py markers                                       # print-ready ArUco sheet
    python camera.py check                                         # live view; click to print table mm
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("camera")


# --------------------------------------------------------------------------- pure geometry

def compute_homography(pixel_pts, table_pts):
    """Homography mapping pixel points -> table-mm points (needs >= 4 pairs)."""
    src = np.asarray(pixel_pts, np.float64).reshape(-1, 1, 2)
    dst = np.asarray(table_pts, np.float64).reshape(-1, 1, 2)
    if len(src) < 4:
        raise ValueError("need at least 4 point pairs")
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("degenerate marker layout (points collinear?)")
    return H


def pixel_to_table(H, pts):
    """Map pixel point(s) (N,2) through H to table mm (N,2)."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def table_to_pixel(H, pts):
    return pixel_to_table(np.linalg.inv(H), pts)


class HomographyTracker:
    """Recomputes the homography from the 4 table markers every frame.

    If fewer than 4 markers are visible the last good homography is reused (with a
    rate-limited warning), so a hand passing over one marker doesn't break mapping,
    while a bumped camera self-corrects as soon as all 4 are seen again.
    """

    def __init__(self, markers_mm, warn_every_s=2.0):
        self.markers_mm = {int(k): tuple(float(c) for c in v) for k, v in markers_mm.items()}
        self.H = None
        self.warn_every_s = warn_every_s
        self._last_warn = 0.0
        self.stale = True       # True when H was not recomputed this frame

    def update(self, detected):
        """detected: {marker_id: (u, v) pixel centre}. Returns the homography to use (or None)."""
        ids = [i for i in self.markers_mm if i in detected]
        if len(ids) == len(self.markers_mm):
            try:
                self.H = compute_homography([detected[i] for i in ids], [self.markers_mm[i] for i in ids])
                self.stale = False
                return self.H
            except ValueError as e:
                log.warning("homography failed: %s", e)
        self.stale = True
        now = time.monotonic()
        if now - self._last_warn > self.warn_every_s:
            self._last_warn = now
            missing = sorted(set(self.markers_mm) - set(ids))
            if self.H is None:
                log.warning("markers %s not visible and no previous homography: table mapping unavailable", missing)
            else:
                log.warning("markers %s not visible: reusing last good homography", missing)
        return self.H


# --------------------------------------------------------------------------- intrinsics

def load_intrinsics(path):
    path = Path(path)
    if not path.exists():
        return None
    d = np.load(path)
    return {"K": d["K"], "dist": d["dist"], "size": tuple(int(v) for v in d["image_size"])}


class Undistorter:
    """Precomputed undistortion maps. The camera matrix K is kept, so the principal point is unchanged."""

    def __init__(self, intr):
        self.K, self.dist, self.size = intr["K"], intr["dist"], intr["size"]
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, self.size, cv2.CV_16SC2)

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise RuntimeError(f"frame is {w}x{h} but calibration was done at {self.size[0]}x{self.size[1]}; "
                               "set camera.width/height to match or recalibrate")
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- markers

def aruco_detector(dict_name):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters()), dictionary


def detect_markers(frame, detector):
    """Returns {id: (u, v) centre}, raw corners list, ids array."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = tuple(c.reshape(4, 2).mean(axis=0))
    return out, corners, ids


# --------------------------------------------------------------------------- camera

class Camera:
    def __init__(self, index, width, height, markers_mm, dict_name, intrinsics_path):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        intr = load_intrinsics(intrinsics_path)
        if intr is None:
            log.warning("no lens calibration at %s: frames are NOT undistorted "
                        "(run `python camera.py calibrate`)", intrinsics_path)
            self.undistort = None
            self.K = None
        else:
            self.undistort = Undistorter(intr)
            self.K = intr["K"]
        self.detector, _ = aruco_detector(dict_name)
        self.tracker = HomographyTracker(markers_mm)
        self.last_corners, self.last_ids = [], None
        self.frame_size = None

    @classmethod
    def from_config(cls, cfg):
        c = cfg["camera"]
        return cls(c["index"], c["width"], c["height"], c["markers_mm"], c["aruco_dict"], c["intrinsics_path"])

    def principal_point(self):
        """Pixel where the optical axis hits the image (image centre if uncalibrated)."""
        if self.K is not None:
            return float(self.K[0, 2]), float(self.K[1, 2])
        w, h = self.frame_size
        return w / 2.0, h / 2.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if self.undistort is not None:
            frame = self.undistort(frame)
        self.frame_size = (frame.shape[1], frame.shape[0])
        markers, self.last_corners, self.last_ids = detect_markers(frame, self.detector)
        H = self.tracker.update(markers)
        return frame, markers, H

    def draw_markers(self, img, H):
        if self.last_ids is not None and len(self.last_ids):
            cv2.aruco.drawDetectedMarkers(img, self.last_corners, self.last_ids)
        if H is not None:
            colour = (0, 165, 255) if self.tracker.stale else (0, 255, 0)
            pts = table_to_pixel(H, [self.tracker.markers_mm[i] for i in sorted(self.tracker.markers_mm)])
            cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], True, colour, 1)
            if self.tracker.stale:
                cv2.putText(img, "MARKERS HIDDEN - using last homography", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        else:
            cv2.putText(img, "NO HOMOGRAPHY - markers not found", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------- CLI tools

def calibrate(args, cfg):
    """Checkerboard lens calibration. Captures automatically when the board is found and has moved."""
    pattern = (args.cols, args.rows)       # inner corners
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
    obj_pts, img_pts, size = [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    if args.images:
        frames = [cv2.imread(str(p)) for p in sorted(Path().glob(args.images))]
        source = iter(frames)
    else:
        c = cfg["camera"]
        cap = cv2.VideoCapture(c["index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        source = None
        print("Move the checkerboard around the whole table area, tilted at different angles.\n"
              "SPACE = capture, C = calibrate (needs >= 10 views), Q = quit")
    while True:
        if source is not None:
            frame = next(source, None)
            if frame is None:
                break
        else:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if source is not None:
            if found:
                obj_pts.append(objp)
                img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            continue
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"views: {len(obj_pts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            obj_pts.append(objp)
            img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            print(f"captured view {len(obj_pts)}")
        elif key == ord("c") and len(obj_pts) >= 10:
            break
        elif key == ord("q"):
            return
    if source is None:
        cap.release()
        cv2.destroyAllWindows()
    if len(obj_pts) < 10:
        raise SystemExit(f"only {len(obj_pts)} usable views; need at least 10")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    out = Path(cfg["camera"]["intrinsics_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K=K, dist=dist, image_size=np.array(size), rms=rms)
    print(f"saved {out}  (RMS reprojection error {rms:.3f} px; below ~0.5 is good)")


def make_marker_sheet(cfg, out_dir="markers", dpi=300):
    """One A4 page per marker at true size (print at 100% / 'actual size', no scaling)."""
    c = cfg["camera"]
    _, dictionary = aruco_detector(c["aruco_dict"])
    px_per_mm = dpi / 25.4
    side = int(round(c["marker_size_mm"] * px_per_mm))
    page_w, page_h = int(210 * px_per_mm), int(297 * px_per_mm)
    Path(out_dir).mkdir(exist_ok=True)
    for mid in sorted(int(k) for k in c["markers_mm"]):
        page = np.full((page_h, page_w), 255, np.uint8)
        m = cv2.aruco.generateImageMarker(dictionary, mid, side)
        y0, x0 = (page_h - side) // 2, (page_w - side) // 2
        page[y0:y0 + side, x0:x0 + side] = m
        label = f"ID {mid}  {c['aruco_dict']}  {c['marker_size_mm']:.0f} mm - print at 100% scale"
        cv2.putText(page, label, (x0 - 200, y0 + side + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
        path = Path(out_dir) / f"marker_{mid}.png"
        cv2.imwrite(str(path), page)
        print(f"wrote {path}")


def check(cfg):
    """Live view with the homography; click anywhere to print table-frame mm (use it to measure the base)."""
    cam = Camera.from_config(cfg)
    state = {"H": None}

    def on_click(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN and state["H"] is not None:
            x, y = pixel_to_table(state["H"], [(u, v)])[0]
            print(f"pixel ({u}, {v}) -> table ({x:.1f}, {y:.1f}) mm")

    cv2.namedWindow("check")
    cv2.setMouseCallback("check", on_click)
    print("Click points to print table mm (on the table surface only). Q quits.")
    while True:
        frame, markers, H = cam.read()
        state["H"] = H
        cam.draw_markers(frame, H)
        cv2.imshow("check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    from config import load_config
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="checkerboard lens calibration")
    c.add_argument("--cols", type=int, default=9, help="inner corners per row")
    c.add_argument("--rows", type=int, default=6, help="inner corners per column")
    c.add_argument("--square", type=float, default=25.0, help="square size in mm")
    c.add_argument("--images", default=None, help="glob of saved images instead of live capture")
    sub.add_parser("markers", help="write printable ArUco markers to markers/")
    sub.add_parser("check", help="live homography check, click to measure")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.cmd == "calibrate":
        calibrate(args, cfg)
    elif args.cmd == "markers":
        make_marker_sheet(cfg)
    else:
        check(cfg)


if __name__ == "__main__":
    main()
