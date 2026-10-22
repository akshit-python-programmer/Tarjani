"""Load and sanity-check config.yaml."""

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("config.yaml")

_REQUIRED = {
    "serial": ["baud", "move_timeout_s"],
    "arm": ["base_xy_mm", "base_yaw_offset_deg", "shoulder_height_mm", "link_lengths_mm",
            "joint_names", "joint_limits_deg", "servo_offset_deg", "servo_sign", "home_deg"],
    "grasp": ["pitch_candidates_deg", "finger_depth_mm", "safe_z_mm"],
    "hand": ["open", "grasps", "limits_deg"],
    "camera": ["height_mm", "markers_mm", "aruco_dict"],
    "vision": ["objects"],
    "planner": ["base_url", "model", "max_replans"],
}


class ConfigError(ValueError):
    pass


def load_config(path=None):
    path = Path(path or DEFAULT_PATH)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def validate(cfg):
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise ConfigError(f"config: missing section '{section}'")
        for k in keys:
            if k not in cfg[section]:
                raise ConfigError(f"config: missing '{section}.{k}'")
    arm = cfg["arm"]
    names = arm["joint_names"]
    if len(names) != 4:
        raise ConfigError("config: arm.joint_names must list 4 joints")
    for key in ("servo_offset_deg", "servo_sign", "home_deg"):
        if len(arm[key]) != 4:
            raise ConfigError(f"config: arm.{key} must have 4 values")
    for n in names:
        lo, hi = arm["joint_limits_deg"][n]
        if lo >= hi:
            raise ConfigError(f"config: joint limit for {n} has min >= max")
    for s in arm["servo_sign"]:
        if s not in (1, -1):
            raise ConfigError("config: arm.servo_sign values must be 1 or -1")
    hand = cfg["hand"]
    if len(hand["open"]) != 5 or any(len(g["fingers"]) != 5 for g in hand["grasps"]):
        raise ConfigError("config: hand presets must have 5 finger values")
    if len(cfg["camera"]["markers_mm"]) != 4:
        raise ConfigError("config: camera.markers_mm must define exactly 4 markers")
    return cfg
