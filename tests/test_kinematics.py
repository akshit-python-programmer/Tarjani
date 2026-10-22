import math

import numpy as np
import pytest

from skill_executor import Kinematics, Unreachable


@pytest.fixture
def kin(cfg):
    return Kinematics(cfg["arm"])


def hand_direction(kin, q):
    """Unit vector the hand points along, in the table frame."""
    _, _, _, pitch = kin.fk(q)
    heading = math.radians(q[0] + kin.yaw0)
    p = math.radians(pitch)
    return np.array([math.cos(p) * math.cos(heading), math.cos(p) * math.sin(heading), math.sin(p)])


def pitch_for_ik(kin, q):
    """Pitch as ik() defines it: in the plane facing from the base towards the grasp point."""
    x, y, _, pitch = kin.fk(q)
    r_signed = (x - kin.bx) * math.cos(math.radians(q[0] + kin.yaw0)) + \
               (y - kin.by) * math.sin(math.radians(q[0] + kin.yaw0))
    return pitch if r_signed >= 0 else 180.0 - pitch


def test_fk_ik_roundtrip_random_joints(kin):
    rng = np.random.default_rng(0)
    for _ in range(3000):
        q = np.array([rng.uniform(lo, hi) for lo, hi in kin.limits])
        x, y, z, _ = kin.fk(q)
        if math.hypot(x - kin.bx, y - kin.by) < 1.0:
            continue        # directly above the base axis: heading is undefined
        q2 = kin.ik(x, y, z, pitch_for_ik(kin, q))      # valid joints reach it, so IK must too
        assert math.dist((x, y, z), kin.fk(q2)[:3]) < 1e-6
        assert np.allclose(hand_direction(kin, q), hand_direction(kin, q2), atol=1e-6)


@pytest.mark.parametrize("target", [(300, 100, 55), (220, 140, 30), (400, 200, 60), (120, 150, 60)])
def test_ik_hits_table_targets_top_down(kin, target):
    q = kin.ik(*target, pitch_deg=-90)
    x, y, z, pitch = kin.fk(q)
    assert np.allclose((x, y, z), target, atol=1e-6)
    assert pitch == pytest.approx(-90)


def test_unreachable_far_point(kin):
    with pytest.raises(Unreachable):
        kin.ik(2000, 2000, 50)


def test_to_servo_enforces_joint_limits(kin):
    q = np.array(kin.limits[:, 1]) + 5        # every joint past its max
    with pytest.raises(Unreachable):
        kin.to_servo(q)


def test_servo_mapping(kin, cfg):
    home = np.array(cfg["arm"]["home_deg"])
    servo = kin.to_servo(home)
    expected = np.array(cfg["arm"]["servo_offset_deg"]) + np.array(cfg["arm"]["servo_sign"]) * home
    assert np.allclose(servo, expected)
    assert np.all((servo >= 0) & (servo <= 180))


def test_joint_limits_map_inside_servo_range(kin):
    for corner in (kin.limits[:, 0], kin.limits[:, 1]):
        s = kin.offset + kin.sign * corner
        assert np.all((s >= 0) & (s <= 180)), "config joint limits map outside servo 0..180"
