import threading

import pytest

from skill_executor import ArduinoError, ArduinoLink, ArduinoTimeout, EmergencyStop, format_hand_cmd, format_joint_cmd


def test_joint_cmd_format():
    assert format_joint_cmd([90, 45.4, 120.6, 0]) == "J,90,45,121,0\n"


def test_hand_cmd_format():
    assert format_hand_cmd([10, 20, 30, 40, 180]) == "H,10,20,30,40,180\n"


@pytest.mark.parametrize("vals", [[90, 90, 90], [90, 90, 90, 90, 90], [-1, 90, 90, 90], [90, 181, 90, 90]])
def test_joint_cmd_rejects_bad_input(vals):
    with pytest.raises(ValueError):
        format_joint_cmd(vals)


def test_hand_cmd_rejects_bad_input():
    with pytest.raises(ValueError):
        format_hand_cmd([0, 0, 0, 0])
    with pytest.raises(ValueError):
        format_hand_cmd([0, 0, 0, 0, 200])


def test_mock_link_records_lines():
    link = ArduinoLink(port=None)
    link.send_joints([90, 90, 90, 90])
    link.send_hand([1, 2, 3, 4, 5])
    assert link.sent == ["J,90,90,90,90\n", "H,1,2,3,4,5\n"]


class FakeSerial:
    def __init__(self, replies):
        self.replies = list(replies)
        self.written = []

    def write(self, data):
        self.written.append(data)

    def readline(self):
        return self.replies.pop(0) if self.replies else b""

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def test_link_waits_for_done_and_ignores_noise():
    ser = FakeSerial([b"", b"READY\r\n", b"DONE\r\n"])
    link = ArduinoLink(serial_obj=ser)
    link.send_joints([10, 20, 30, 40])
    assert ser.written == [b"J,10,20,30,40\n"]


def test_link_raises_on_err():
    link = ArduinoLink(serial_obj=FakeSerial([b"ERR,bad_count\n"]))
    with pytest.raises(ArduinoError, match="bad_count"):
        link.send_hand([0, 0, 0, 0, 0])


def test_link_timeout_retries_then_fails():
    ser = FakeSerial([])
    link = ArduinoLink(serial_obj=ser, timeout_s=0.05, retries=1)
    with pytest.raises(ArduinoTimeout):
        link.send_joints([90, 90, 90, 90])
    assert len(ser.written) == 2


def test_link_abort():
    ev = threading.Event()
    ev.set()
    link = ArduinoLink(serial_obj=FakeSerial([b"DONE\n"]), abort_event=ev)
    with pytest.raises(EmergencyStop):
        link.send_joints([90, 90, 90, 90])
    link.send_unconditional(format_joint_cmd([90, 90, 90, 90]))     # homing ignores the abort
