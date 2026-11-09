import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from ah_messages.msg import Digits

from .validation_pipeline import (
    HAND_JOINT_ORDER,
    USD_JOINT_ORDER,
    MIMIC_USD_ORDER,
    CONTACT_THRESHOLD,
    remap,
)

# FSR sensor order as published by the real driver on the touch topic.
# 5 fingers x 6 sensors = 30 floats, grouped by finger in this order.
FSR_FINGER_ORDER = ["index", "middle", "ring", "pinky", "thumb"]


class RealAdapter:
    """Translates between real hardware ROS messages and the pipeline's
    internal representation (radians, USD_JOINT_ORDER, no sign flips).

    This is the only place in the codebase that knows about:
      - degree/radian conversion
      - HAND_JOINT_ORDER vs USD_JOINT_ORDER remapping
      - thumb sign convention: hardware reports thumb_rotator position and
        thumb_flexor velocity with opposite sign relative to the USD asset.
        Physical cause: the thumb is mounted in opposition; the hardware ICD
        defines positive as flexion while the USD joint frame defines negative
        as flexion for thumb_q1. Applied on input and inverted on output.
    """

    def feedback_to_pipeline(
        self,
        pos_deg: np.ndarray,
        vel_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert hardware feedback to pipeline input.

        Args:
            pos_deg: joint positions in degrees, HAND_JOINT_ORDER
            vel_deg: joint velocities in degrees/s, HAND_JOINT_ORDER
        Returns:
            pos_rad: positions in radians, USD_JOINT_ORDER
            vel_rad: velocities in radians/s, USD_JOINT_ORDER
        """
        pos_rad = np.radians(remap(pos_deg, HAND_JOINT_ORDER, USD_JOINT_ORDER))
        vel_rad = np.radians(remap(vel_deg, HAND_JOINT_ORDER, USD_JOINT_ORDER))

        pos_rad[-1] *= -1.0  # thumb_rotator: hardware positive = USD negative
        vel_rad[-2] *= -1.0  # thumb_flexor velocity: hardware sign inverted

        return pos_rad, vel_rad

    def target_to_hardware(self, target_deg_usd: np.ndarray) -> np.ndarray:
        """Convert pipeline output to hardware target.

        Args:
            target_deg_usd: targets in degrees, USD_JOINT_ORDER
        Returns:
            target_deg_hw: targets in degrees, HAND_JOINT_ORDER
        """
        target_deg_hw = remap(target_deg_usd, USD_JOINT_ORDER, HAND_JOINT_ORDER)

        target_deg_hw[-1] *= -1.0  # thumb_rotator
        target_deg_hw[-2] *= -1.0  # thumb_flexor

        return target_deg_hw

    def contact_from_touch(self, touch_raw: np.ndarray) -> np.ndarray:
        """Compute binary per-finger contact from raw FSR data.

        Args:
            touch_raw: 30 floats, 5 fingers x 6 sensors, FSR_FINGER_ORDER
        Returns:
            contact: float array shape (5,), USD_JOINT_ORDER finger subset + thumb
        """
        touch = touch_raw.reshape(5, 6)
        per_finger = touch.max(axis=1)
        contact_by_name = dict(zip(FSR_FINGER_ORDER, per_finger))
        return np.array(
            [float(contact_by_name[n] > CONTACT_THRESHOLD)
             for n in ["index", "middle", "pinky", "ring", "thumb"]],
            dtype=np.float32,
        )

    def parse_pos_msg(self, msg: Float32MultiArray) -> np.ndarray:
        return np.array(msg.data, dtype=np.float32)

    def parse_vel_msg(self, msg: Float32MultiArray) -> np.ndarray:
        return np.array(msg.data, dtype=np.float32)

    def parse_touch_msg(self, msg: Float32MultiArray) -> np.ndarray:
        return np.array(msg.data, dtype=np.float32)

    def make_target_msg(self, target_deg_hw: np.ndarray) -> Digits:
        msg = Digits()
        msg.reply_mode = 1
        msg.data = [float(v) for v in target_deg_hw]
        return msg

    def make_zero_msg(self) -> Digits:
        msg = Digits()
        msg.reply_mode = 1
        msg.data = [0.0] * len(HAND_JOINT_ORDER)
        return msg