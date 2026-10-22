// Robotic hand + 4-DOF arm controller.
//
// Serial protocol (115200 baud, newline-terminated ASCII, one command at a time):
//   J,a0,a1,a2,a3        arm servos in degrees: base yaw, shoulder, elbow, wrist pitch
//   H,f0,f1,f2,f3,f4     finger servos in degrees: thumb, index, middle, ring, little
// Replies:
//   READY                once after boot
//   DONE                 when every servo has reached its target
//   ERR,<reason>         bad input (nothing moves): empty, too_long, unknown_cmd, bad_count, not_a_number
//
// Values are clamped to 0..180 and then to the per-servo safety limits below.
// Servos move smoothly: at most STEP_DEG every STEP_MS. A new command received while
// moving replaces the targets immediately (used for emergency "go home"); only the
// newest command gets a DONE.

#include <Servo.h>

// ---------------------------------------------------------------- pins (TODO hardware)
const uint8_t ARM_PINS[4]  = {3, 5, 6, 9};          // base, shoulder, elbow, wrist
const uint8_t HAND_PINS[5] = {10, 11, 12, 13, 2};   // thumb, index, middle, ring, little

// ---------------------------------------------------------------- safety limits in SERVO degrees (TODO hardware)
// Keep these in line with config.yaml: servo = servo_offset_deg + servo_sign * joint limit.
const int ARM_MIN[4]  = {5, 10, 20, 0};
const int ARM_MAX[4]  = {175, 170, 180, 180};
const int HAND_MIN[5] = {0, 0, 0, 0, 0};
const int HAND_MAX[5] = {180, 180, 180, 180, 180};

// ---------------------------------------------------------------- start pose (TODO hardware, = config home/open)
const int ARM_HOME[4]  = {90, 90, 90, 90};
const int HAND_OPEN[5] = {10, 10, 10, 10, 10};

// ---------------------------------------------------------------- motion
const unsigned long BAUD = 115200;
const int STEP_DEG = 2;
const unsigned long STEP_MS = 15;

const uint8_t N = 9;          // 4 arm + 5 hand servos, indexed arm first
Servo servos[N];
int current[N];
int target[N];
bool pending = false;         // a command is in progress; send DONE when it finishes
unsigned long lastStep = 0;

const uint8_t BUF_LEN = 64;
char buf[BUF_LEN];
uint8_t bufPos = 0;
bool overflow = false;

int minFor(uint8_t i) { return i < 4 ? ARM_MIN[i] : HAND_MIN[i - 4]; }
int maxFor(uint8_t i) { return i < 4 ? ARM_MAX[i] : HAND_MAX[i - 4]; }

int clampServo(uint8_t i, long v) {
  if (v < 0) v = 0;
  if (v > 180) v = 180;
  if (v < minFor(i)) v = minFor(i);
  if (v > maxFor(i)) v = maxFor(i);
  return (int)v;
}

// Parses "X,v1,...,vn" into vals. Returns NULL on success, or the error reason.
const char* parseValues(char* line, long* vals, uint8_t expected) {
  uint8_t count = 0;
  char* p = line + 1;                       // skip command letter
  if (*p != ',') return "bad_count";
  while (*p == ',') {
    p++;
    if (count >= expected) return "bad_count";
    char* end;
    long v = strtol(p, &end, 10);
    if (end == p) return "not_a_number";
    while (*end == ' ') end++;
    if (*end != ',' && *end != '\0') return "not_a_number";
    vals[count++] = v;
    p = end;
  }
  if (*p != '\0') return "not_a_number";
  if (count != expected) return "bad_count";
  return NULL;
}

void handleLine(char* line) {
  if (line[0] == '\0') { Serial.println(F("ERR,empty")); return; }
  long vals[5];
  uint8_t first, count;
  if (line[0] == 'J') { first = 0; count = 4; }
  else if (line[0] == 'H') { first = 4; count = 5; }
  else { Serial.println(F("ERR,unknown_cmd")); return; }

  const char* err = parseValues(line, vals, count);
  if (err) { Serial.print(F("ERR,")); Serial.println(err); return; }
  for (uint8_t k = 0; k < count; k++) target[first + k] = clampServo(first + k, vals[k]);
  pending = true;
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (overflow) Serial.println(F("ERR,too_long"));
      else { buf[bufPos] = '\0'; handleLine(buf); }
      bufPos = 0;
      overflow = false;
    } else if (bufPos < BUF_LEN - 1) {
      buf[bufPos++] = c;
    } else {
      overflow = true;
    }
  }
}

void stepServos() {
  if (millis() - lastStep < STEP_MS) return;
  lastStep = millis();
  bool moving = false;
  for (uint8_t i = 0; i < N; i++) {
    int d = target[i] - current[i];
    if (d == 0) continue;
    if (d > STEP_DEG) d = STEP_DEG;
    if (d < -STEP_DEG) d = -STEP_DEG;
    current[i] += d;
    servos[i].write(current[i]);
    moving = true;
  }
  if (pending && !moving) {
    pending = false;
    Serial.println(F("DONE"));
  }
}

void setup() {
  Serial.begin(BAUD);
  for (uint8_t i = 0; i < N; i++) {
    int start = clampServo(i, i < 4 ? ARM_HOME[i] : HAND_OPEN[i - 4]);
    current[i] = target[i] = start;
    servos[i].write(start);                 // set position before attach to limit the start-up jump
    servos[i].attach(i < 4 ? ARM_PINS[i] : HAND_PINS[i - 4]);
  }
  Serial.println(F("READY"));
}

void loop() {
  readSerial();
  stepServos();
}
