import math
import numpy as np


# Isaac publishes radians as joint state, real driver publishes degrees as target and feedback (DIGIT mesg)
# this pipeline showudl always receive degrees for consistency


MIMIC_GEARING = -1.0585132837295532

VALIDATION_AMPLITUDE_RAD = np.array(
    [0.8, 0.8, 0.8, 0.8, -0.6, 0.6], dtype=np.float32
)
VALIDATION_PERIOD_S = 4.0

SIM_JOINT_USD_NAMES = [
    "index_q1", "middle_q1", "pinky_q1", "ring_q1",
    "thumb_q1", "index_q2", "middle_q2", "pinky_q2", "ring_q2", "thumb_q2"]

HAND_JOINT_ORDER = ["index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator"] # order in which the real driver publishes its js
SIM_JOINT_ORDER  = ["index", "middle", "pinky", "ring", "thumb_flexor", "thumb_rotator"] #
MIMIC_SIM_ORDER  = ["index", "middle", "pinky", "ring"]

JOINT_LOWER_LIMITS_RAD = np.array([0.0,  0.0,  0.0,  0.0,  -1.74, 0.0],  dtype=np.float32)
JOINT_UPPER_LIMITS_RAD = np.array([1.74, 1.74, 1.74, 1.74,   0.0,  1.74], dtype=np.float32)

ACTION_SCALE     = 0.05
CONTACT_THRESHOLD = 1.0


def remap(values, source_names, target_names):
    by_name = dict(zip(source_names, values))
    return np.array([by_name[n] for n in target_names], dtype=np.float32)


def hand_deg_to_usd_rad(values_deg: np.ndarray, source_names: list) -> np.ndarray:
    by_name = dict(zip(source_names, values_deg))
    return np.radians(np.array([
        by_name["index"],
        by_name["middle"],
        by_name["ring"],
        by_name["pinky"],
        by_name["thumb_flexor"],
        by_name["thumb_rotator"],
        ], dtype=np.float32))


class ValidationPipeline:
    def __init__(self):
        self.finger_joint_target_rad = np.zeros(len(SIM_JOINT_ORDER), dtype=np.float32)
        self._seeded = False
        self._t0_ns  = None

    def step(
        self,
        pos_deg_hw_order: np.ndarray,
        vel_deg_hw_order: np.ndarray,
        touch_raw: np.ndarray,
        now_ns: int,
    ) -> np.ndarray:
        """
        Args:
            pos_deg_hw_order: joint positions in degrees, hardware order
            vel_deg_hw_order: joint velocities in deg/s, hardware order
            touch_raw:        30 floats from touch sensors, hardware order
            now_ns:           current time in nanoseconds (ROS clock)

        Returns:
            target_deg_hw_order: joint targets in degrees, hardware order
        """
        if self._t0_ns is None:
            self._t0_ns = now_ns

        # remap to sim order
        controlled_pos = np.radians(remap(pos_deg_hw_order, HAND_JOINT_ORDER, SIM_JOINT_ORDER))
        controlled_vel = np.radians(remap(vel_deg_hw_order, HAND_JOINT_ORDER, SIM_JOINT_ORDER))

        # hardware convention flip for rotator
        controlled_pos[-1] *= -1.0
        controlled_vel[-2] *= -1.0

        mimic_pos = controlled_pos[:4] * MIMIC_GEARING

        if not self._seeded:
            self.finger_joint_target_rad = controlled_pos.copy()
            self._seeded = True

        fingers_in_contact = self._compute_contact(touch_raw)
        action = self._scripted_action(now_ns)

        self.finger_joint_target_rad = np.clip(
            self.finger_joint_target_rad + action * ACTION_SCALE,
            JOINT_LOWER_LIMITS_RAD,
            JOINT_UPPER_LIMITS_RAD,
        )

        target_deg_sim  = np.degrees(self.finger_joint_target_rad)
        target_deg_hand = remap(target_deg_sim, SIM_JOINT_ORDER, HAND_JOINT_ORDER)
        target_deg_hand[-1] *= -1.0
        target_deg_hand[-2] *= -1.0 

        return target_deg_hand, controlled_pos, mimic_pos, fingers_in_contact, target_deg_sim

    def _scripted_action(self, now_ns: int) -> np.ndarray:
        elapsed = (now_ns - self._t0_ns) * 1e-9
        phase   = 2.0 * math.pi * elapsed / VALIDATION_PERIOD_S
        desired = VALIDATION_AMPLITUDE_RAD * (0.5 - 0.5 * math.cos(phase))
        desired = np.clip(desired, JOINT_LOWER_LIMITS_RAD, JOINT_UPPER_LIMITS_RAD)
        return ((desired - self.finger_joint_target_rad) / ACTION_SCALE).astype(np.float32)

    def _compute_contact(self, touch_raw: np.ndarray) -> list:
        touch = touch_raw.reshape(5, 6)
        per_finger = touch.max(axis=1)
        contact_by_name = dict(zip(
            ["index", "middle", "ring", "pinky", "thumb"], per_finger
        ))
        return [float(contact_by_name[n] > CONTACT_THRESHOLD)
                for n in ["index", "middle", "pinky", "ring", "thumb"]]

