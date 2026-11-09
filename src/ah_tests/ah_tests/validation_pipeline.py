import math
import numpy as np

# Joint order conventions used throughout this pipeline.
#
# HAND_JOINT_ORDER: order in which the real driver publishes feedback and
# expects targets (Float32MultiArray / Digits messages).
#
# USD_JOINT_ORDER: order Isaac Lab's find_joints returns for this asset.
# pinky and ring are swapped relative to HAND_JOINT_ORDER because the USD
# defines them in that sequence and find_joints follows the articulator order,
# not the request order. This is the order the policy was trained in.
#
# MIMIC_USD_ORDER: q2 mimic joints in USD order, same pinky/ring swap.
#
# Thumb sign convention: thumb_flexor (thumb_q1) has limits [-1.74, 0] in the
# USD because the thumb is mounted in opposition — its joint frame localRot0 is
# rotated ~180 deg relative to the fingers, so negative USD values correspond
# to physical flexion. The real hardware ICD reports thumb flexion as positive
# degrees. The sign flip between hardware and USD is applied only in the real
# adapter, never here.

HAND_JOINT_ORDER = ["index", "middle", "ring",  "pinky", "thumb_flexor", "thumb_rotator"]
USD_JOINT_ORDER  = ["index", "middle", "pinky", "ring",  "thumb_flexor", "thumb_rotator"]
MIMIC_USD_ORDER  = ["index", "middle", "pinky", "ring"]

MIMIC_GEARING = -1.0585132837295532

VALIDATION_AMPLITUDE_RAD = np.array(
    [0.8, 0.8, 0.8, 0.8, -0.6, 0.6], dtype=np.float32
)
VALIDATION_PERIOD_S = 4.0

# Joint limits in radians, USD_JOINT_ORDER.
# Fingers: [0, 1.74]. Thumb flexor: [-1.74, 0] (opposition geometry, see above).
JOINT_LOWER_LIMITS_RAD = np.array([0.0,  0.0,  0.0,  0.0,  -1.74, 0.0],  dtype=np.float32)
JOINT_UPPER_LIMITS_RAD = np.array([1.74, 1.74, 1.74, 1.74,   0.0,  1.74], dtype=np.float32)

ACTION_SCALE      = 0.05
CONTACT_THRESHOLD = 1.0


def remap(values: np.ndarray, source: list, target: list) -> np.ndarray:
    by_name = dict(zip(source, values))
    return np.array([by_name[n] for n in target], dtype=np.float32)


class ValidationPipeline:
    def __init__(self):
        self.finger_joint_target_rad = np.zeros(len(USD_JOINT_ORDER), dtype=np.float32)
        self._seeded = False
        self._t0_ns  = None

    def step(
        self,
        pos_deg_hw: np.ndarray,
        vel_deg_hw: np.ndarray,
        touch_raw: np.ndarray,
        now_ns: int,
    ) -> tuple:
        """
        Args:
            pos_deg_hw: joint positions in degrees, HAND_JOINT_ORDER
            vel_deg_hw: joint velocities in degrees/s, HAND_JOINT_ORDER
            touch_raw:  30 floats from FSR sensors, hardware order
            now_ns:     current ROS clock time in nanoseconds
        Returns:
            target_deg_hw:   targets in degrees, HAND_JOINT_ORDER
            controlled_pos:  current positions in radians, USD_JOINT_ORDER
            mimic_pos:       mimic joint positions in radians, MIMIC_USD_ORDER
            contact:         binary contact per finger, USD_JOINT_ORDER finger subset
            target_deg_usd:  targets in degrees, USD_JOINT_ORDER (for sim adapter)
        """
        if self._t0_ns is None:
            self._t0_ns = now_ns

        # remap hardware order -> USD order, convert to radians
        controlled_pos = np.radians(remap(pos_deg_hw, HAND_JOINT_ORDER, USD_JOINT_ORDER))
        controlled_vel = np.radians(remap(vel_deg_hw, HAND_JOINT_ORDER, USD_JOINT_ORDER))

        # hardware ICD sign flip: thumb_rotator position and thumb_flexor velocity
        # are reported with opposite sign by the real driver relative to USD convention
        controlled_pos[-1] *= -1.0  # thumb_rotator
        controlled_vel[-2] *= -1.0  # thumb_flexor velocity

        mimic_pos = controlled_pos[:4] * MIMIC_GEARING

        if not self._seeded:
            self.finger_joint_target_rad = controlled_pos.copy()
            self._seeded = True

        contact = self._compute_contact(touch_raw)
        action  = self._scripted_action(now_ns)

        self.finger_joint_target_rad = np.clip(
            self.finger_joint_target_rad + action * ACTION_SCALE,
            JOINT_LOWER_LIMITS_RAD,
            JOINT_UPPER_LIMITS_RAD,
        )

        target_deg_usd = np.degrees(self.finger_joint_target_rad)
        target_deg_hw  = remap(target_deg_usd, USD_JOINT_ORDER, HAND_JOINT_ORDER)

        # inverse of input flips: convert USD targets back to hardware convention
        target_deg_hw[-1] *= -1.0  # thumb_rotator
        target_deg_hw[-2] *= -1.0  # thumb_flexor

        return target_deg_hw, controlled_pos, mimic_pos, contact, target_deg_usd

    def _scripted_action(self, now_ns: int) -> np.ndarray:
        elapsed = (now_ns - self._t0_ns) * 1e-9
        phase   = 2.0 * math.pi * elapsed / VALIDATION_PERIOD_S
        desired = VALIDATION_AMPLITUDE_RAD * (0.5 - 0.5 * math.cos(phase))
        desired = np.clip(desired, JOINT_LOWER_LIMITS_RAD, JOINT_UPPER_LIMITS_RAD)
        return ((desired - self.finger_joint_target_rad) / ACTION_SCALE).astype(np.float32)

    def _compute_contact(self, touch_raw: np.ndarray) -> list:
        # 30 floats: 5 fingers x 6 FSR sensors, hardware order [index, middle, ring, pinky, thumb]
        touch = touch_raw.reshape(5, 6)
        per_finger = touch.max(axis=1)
        contact_by_name = dict(zip(
            ["index", "middle", "ring", "pinky", "thumb"], per_finger
        ))
        # return in USD_JOINT_ORDER finger subset to match the policy obs layout
        return [float(contact_by_name[n] > CONTACT_THRESHOLD)
                for n in ["index", "middle", "pinky", "ring", "thumb"]]