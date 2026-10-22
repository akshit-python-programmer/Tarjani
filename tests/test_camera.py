import numpy as np
import pytest

from camera import HomographyTracker, compute_homography, pixel_to_table, table_to_pixel

MARKERS = {0: (0.0, 0.0), 1: (600.0, 0.0), 2: (600.0, 400.0), 3: (0.0, 400.0)}

# A plausible table->pixel mapping: scale, rotation, translation and some perspective.
TRUE_T2P = np.array([[1.4, 0.15, 220.0],
                     [0.1, -1.35, 640.0],
                     [0.0001, 0.00005, 1.0]])


def synthetic_pixels(markers=MARKERS):
    return {i: tuple(table_to_pixel(np.linalg.inv(TRUE_T2P), [p])[0]) for i, p in markers.items()}


def test_homography_recovers_table_coordinates():
    px = synthetic_pixels()
    H = compute_homography([px[i] for i in MARKERS], [MARKERS[i] for i in MARKERS])
    # arbitrary table points -> pixels -> back through the estimated homography
    pts = np.array([[100, 50], [300, 200], [555, 390], [12, 300]], float)
    pix = table_to_pixel(np.linalg.inv(TRUE_T2P), pts)
    assert np.allclose(pixel_to_table(H, pix), pts, atol=1e-6)


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        compute_homography([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])


def test_tracker_reuses_last_good_when_marker_hidden(caplog):
    tr = HomographyTracker(MARKERS)
    px = synthetic_pixels()
    H1 = tr.update(px)
    assert H1 is not None and not tr.stale
    partial = {k: v for k, v in px.items() if k != 2}
    H2 = tr.update(partial)
    assert H2 is H1 and tr.stale
    assert "reusing last good homography" in caplog.text


def test_tracker_without_any_homography_returns_none():
    tr = HomographyTracker(MARKERS)
    assert tr.update({0: (1, 1)}) is None


def test_tracker_self_corrects_after_camera_bump():
    tr = HomographyTracker(MARKERS)
    tr.update(synthetic_pixels())
    shifted = {i: (u + 25.0, v - 10.0) for i, (u, v) in synthetic_pixels().items()}   # camera moved
    H = tr.update(shifted)
    probe = table_to_pixel(np.linalg.inv(TRUE_T2P), [[300, 200]])[0] + np.array([25.0, -10.0])
    assert np.allclose(pixel_to_table(H, [probe])[0], [300, 200], atol=1e-6)


def render_table(marker_px, side=60, size=(1280, 720)):
    """White image with ArUco markers drawn centred at the given pixel positions."""
    import cv2
    from camera import aruco_detector
    _, dictionary = aruco_detector("DICT_4X4_50")
    img = np.full((size[1], size[0]), 255, np.uint8)
    for mid, (u, v) in marker_px.items():
        m = cv2.aruco.generateImageMarker(dictionary, mid, side)
        y0, x0 = int(v - side / 2), int(u - side / 2)
        img[y0:y0 + side, x0:x0 + side] = m
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_detects_rendered_markers_and_maps_to_mm():
    from camera import aruco_detector, detect_markers
    # 1 px = 1 mm, table origin at pixel (200, 600), y up
    px = {0: (200, 600), 1: (800, 600), 2: (800, 200), 3: (200, 200)}
    img = render_table(px)
    det, _ = aruco_detector("DICT_4X4_50")
    found, _, _ = detect_markers(img, det)
    assert set(found) == {0, 1, 2, 3}
    for i in px:
        assert np.allclose(found[i], np.array(px[i]) - 0.5, atol=1.0)
    H = HomographyTracker(MARKERS).update(found)
    assert np.allclose(pixel_to_table(H, [(500, 400)])[0], (300, 200), atol=1.5)
