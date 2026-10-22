# Tarjani: text-prompted robotic hand

Type a request ("put the red cup in the left zone"). An overhead camera finds the objects,
an LLM turns the request into a skill plan, you confirm it, and the arm and hand carry it out.

```
prompt ──► planner.py (DeepSeek, text only) ──► plan ──► you confirm (y/n)
              ▲ names of visible objects                       │
              │                                                ▼
camera.py ──► vision.py ──► scene_mm() ──────────────► skill_executor.py ──► serial ──► Arduino
(undistort,   (YOLOv8, table mm,                        (IK, waypoints,          (hand_controller.ino)
 ArUco homography) parallax, names)                      pick/place, retries)
```

| File | Role |
|---|---|
| `config.yaml` | Every hardware number. Values marked `# TODO hardware` are placeholders |
| `config.py` | Loads and sanity-checks `config.yaml` |
| `skill_executor.py` | Analytic IK/FK, `ArduinoLink` serial protocol (mock when `port=None`), skills, `run_plan` |
| `camera.py` | Lens calibration, undistortion, ArUco markers, a homography recomputed every frame, marker printing, click-to-measure tool |
| `vision.py` | YOLOv8 → table mm with parallax correction, colour names, `scene_mm()`, `object_lifted()`; `FakeVision` for `--no-camera` |
| `planner.py` | DeepSeek planner (JSON mode, validation, re-planning) + offline rule-based fallback |
| `main.py` | Terminal loop and live OpenCV window |
| `arduino/hand_controller/hand_controller.ino` | Servo controller for the serial protocol |
| `scenes/demo_scene.json` | Fake scene for `--no-camera` |
| `tests/` | pytest suite |

## 1. Quick start (no hardware)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # skip ultralytics if you only want --no-camera
python main.py --mock --no-camera
```

If `DEEPSEEK_API_KEY` is not set, an offline rule-based planner is used instead. It handles
"put X on/in Y", "pick up X", "point at X", "open hand" and "go home". To use DeepSeek:

```bash
export DEEPSEEK_API_KEY=sk-...       # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
python main.py --mock --no-camera --planner deepseek
```

Example session (mock lines show exactly what would go to the Arduino):

```
> put the red cup in the left zone
Plan:
  1. pick       red cup
  2. place      left zone
Execute? [y/n] y
INFO    executor: pick red cup attempt 1/3 at (220, 140) z=55 pitch=-80 preset=wide
INFO    executor: -> H,10,10,10,10,10
INFO    executor: -> J,112,77,114,9
...
done.
```

### Flags

| Flag | Meaning |
|---|---|
| `--mock` | No Arduino (`ArduinoLink(port=None)`); commands are logged |
| `--port COM5` / `--port /dev/ttyUSB0` | Serial port (or set `serial.port` in config) |
| `--no-camera [scene.json]` | Fake scene from JSON (default `scenes/demo_scene.json`) |
| `--no-window` | No OpenCV window |
| `--planner auto\|deepseek\|offline` | Default `auto`: DeepSeek if the key is set |
| `-v` | Debug logging (shows every serial line on real hardware) |

### Safety controls

- **`e` + Enter** in the terminal, or **`e`** in the video window: emergency stop. Motion is
  aborted at once (including while waiting for an Arduino reply) and the arm is driven home. The Arduino
  replaces its current target as soon as the home command arrives.
- **Ctrl+C**: same, then the program exits.
- Joint limits are enforced twice. Python checks every waypoint against `arm.joint_limits_deg` and the
  0..180 servo range *before anything is sent*, and a straight-line move is checked in full before
  the first step. The Arduino then clamps to 0..180 and to its own per-servo `ARM_MIN/ARM_MAX`.
- Every plan is printed and needs a `y` before the arm moves.

## 2. Hardware setup

### 2.1 Arduino

1. Open `arduino/hand_controller/hand_controller.ino` in the Arduino IDE. The folder must keep the sketch's name.
2. Set the pin numbers, `ARM_MIN/ARM_MAX`, `ARM_HOME` and `HAND_OPEN` at the top. These
   must agree with `config.yaml` (servo = `servo_offset_deg + servo_sign * joint`).
3. Upload. Open the Serial Monitor at 115200 baud with "Newline" line ending. You should see `READY`.
   Try `J,90,90,90,90` → `DONE`, and `J,1,2` → `ERR,bad_count`.
4. Power the servos from a separate 5–6 V supply with enough current (the hand + arm can draw several amps).
   Connect its ground to the Arduino ground. Do not power servos from the Arduino's 5 V pin.

Protocol: `J,a0,a1,a2,a3\n` (base, shoulder, elbow, wrist) and `H,f0,f1,f2,f3,f4\n` (thumb…little),
integer servo degrees. Replies: `READY` on boot, `DONE` when the move finishes, `ERR,<reason>`
(`empty`, `too_long`, `unknown_cmd`, `bad_count`, `not_a_number`).

### 2.2 Calibrating the camera lens (once per camera/resolution)

1. Print a checkerboard (e.g. 10×7 squares → **9×6 inner corners**) and glue it to something flat.
   Measure one square with a ruler, e.g. 25 mm.
2. Mount the camera in its final position and resolution (`camera.width/height` in config).
3. Run:
   ```bash
   python camera.py calibrate --cols 9 --rows 6 --square 25
   ```
   Hold the board at 15–25 different places and tilts, covering the corners of the image.
   Press SPACE to capture each view and C to calibrate.
4. It saves `calib/intrinsics.npz` and prints the RMS error. Below ~0.5 px is good; redo if above 1.
   Every frame is undistorted from then on. If you change the resolution, recalibrate.

### 2.3 Printing and placing the ArUco markers

1. `python camera.py markers` writes `markers/marker_0.png` … `marker_3.png` (A4, 300 dpi, 50 mm markers).
2. Print at **100 % / actual size** (turn off "fit to page"). Exact print size doesn't matter for
   accuracy, because only the marker *centres* are used. Keep the white border around each marker.
3. Tape the markers **flat** on the table, near the corners of the work area, all visible to the camera,
   ideally where the arm will not cover them. The default layout is a 600 × 400 mm rectangle:
   ID 0 at (0, 0), ID 1 at (600, 0), ID 2 at (600, 400), ID 3 at (0, 400).
4. Measure the real centre-to-centre distances with a tape measure and write them into
   `camera.markers_mm`. Marker 0 is the origin, +x points towards marker 1 and +y towards marker 3.
   Use a set square to get the axes perpendicular, or measure the real positions and enter them as they are.
5. `python camera.py check` shows the live view with the marker outline (green = fresh homography,
   orange = a marker is hidden and the last good one is reused). Click on a known point, such as a ruler mark,
   to verify the printed mm are right to within a few mm.

If the camera gets bumped, the homography fixes itself on the next frame where all 4 markers are visible.

### 2.4 Measuring the robot base position

`arm.base_xy_mm` is the point on the table **directly under the base yaw axis** in the table frame.

1. Find that point: e.g. the centre of the base's turntable, or hang a plumb line from the yaw axis.
   Mark it on the table with a small cross of tape. It can be outside the marker rectangle.
2. If the camera can see the mark, run `python camera.py check` and click it. Otherwise measure
   it from marker 0 with a tape measure along +x and +y.
3. `arm.base_yaw_offset_deg`: command base joint 0 (servo at `servo_offset_deg[0]`, e.g.
   `J,90,90,90,90` in the Serial Monitor), lay a ruler along the arm, click two points along it with
   `camera.py check`, and compute `atan2(y2 - y1, x2 - x1)` in degrees.
4. `arm.shoulder_height_mm`: table surface to the shoulder pivot axis.
5. `camera.height_mm`: table surface to the camera lens.

Check: after filling in the numbers, put an object at a marked spot and use the `point` skill
("point at the red cup") with `--port`. The hand should hover centred over it.

## 3. Numbers needed from the hardware team

Everything below is a placeholder in `config.yaml` (marked `# TODO hardware`) or at the top of
the Arduino sketch.

**Arm geometry**
1. `link_lengths_mm.upper`: shoulder pivot → elbow pivot (mm)
2. `link_lengths_mm.fore`: elbow pivot → wrist pivot (mm)
3. `link_lengths_mm.hand`: wrist pivot → grasp centre, i.e. the point between the fingertips when closed (mm)
4. `shoulder_height_mm`: table → shoulder pivot (mm)
5. `base_xy_mm`, `base_yaw_offset_deg`: measured as in 2.4

**Servos (each of the 4 arm joints)**
6. Which way each servo turns, and the servo reading at the kinematic zero:
   - base: servo value when the arm points along `base_yaw_offset_deg`; does increasing it turn CCW (seen from above)?
   - shoulder: servo value when the upper arm is horizontal; does increasing it raise the arm?
   - elbow: servo value when the forearm is in line with the upper arm; does increasing it straighten or bend?
   - wrist: servo value when the hand is in line with the forearm; does increasing it tilt the hand up?

   These become `servo_offset_deg` and `servo_sign` (+1 if increasing the servo matches the
   direction described in the `config.yaml` comments, otherwise −1).
7. Safe mechanical range of each joint (where it hits cables, frame or itself) → `joint_limits_deg`,
   and the same range in servo degrees → `ARM_MIN/ARM_MAX` in the sketch.
8. A safe home pose → `home_deg` and `ARM_HOME`.

**Hand**
9. Finger servo values for fully open → `hand.open` and `HAND_OPEN`
10. Finger servo values for a pinch (~30 mm), medium (~60 mm) and wide (~90 mm) grasp → `hand.grasps`
11. Safe range of each finger servo → `hand.limits_deg`, `HAND_MIN/HAND_MAX`
12. `grasp.finger_depth_mm`: how far below the top of an object the grasp centre should go
13. Does the wrist pitch servo point the hand straight down for a top-down grasp? (The IK assumes the
    hand can pitch to −90°; it falls back to tilted grasps for far objects.)

**Wiring**
14. Arduino pin for every servo → `ARM_PINS`, `HAND_PINS`
15. Board type, and how the servos are powered

**Camera**
16. `camera.height_mm`: table → lens
17. Marker centre positions after taping → `camera.markers_mm`

## 4. Object sizes

`vision.objects` maps YOLO class names to height/width in mm (unknown classes use `default`).
Width chooses the grasp preset; height sets the grasp depth and the parallax correction.
Add an entry for each object you use. The demo uses the default COCO-trained `yolov8n.pt`, so only
COCO classes (cup, bottle, sports ball, apple, …) are detected. Train a custom model for anything else
and set `vision.model`.

## 5. Tests

```bash
python -m pytest -q
```

They cover: IK ↔ FK round trip, homography from synthetic marker points (including the
hidden-marker fallback and a bumped camera), ArUco detection on a rendered image, the parallax
correction math (checked against a pinhole projection), colour naming and duplicate naming,
planner validation and re-planning with a mocked LLM, serial command formatting and the
DONE/ERR/timeout handling, executor skills and retries, and the emergency-stop path in `main.py`.

## 6. Design notes

- **Parallax.** The formula is `true = c + (measured − c)·(H − h)/H`, with `c` = the table point under
  the camera's principal point. A YOLO box centre lies roughly half-way up an object seen from above
  (the box spans the top face and the base), so the correction uses `h = height × vision.parallax_height_fraction`
  (default 0.5). Set it to 1.0 for the literal object-top version.
- **Grasp check.** Seen from straight above, a lifted object still sits at the same x, y. So after lifting,
  the hand first moves `grasp.verify_offset_mm` towards the base, then `object_lifted()` checks that nothing
  of that class remains at the old spot.
- **Re-planning.** Invalid LLM output and executor `Unreachable`/`LookupError` share one budget of
  `planner.max_replans` (2) per prompt. The model is told which steps already finished and what the hand holds.
- **Colour.** Hue is averaged as a circular mean, because red wraps around 0/180 in OpenCV HSV.
