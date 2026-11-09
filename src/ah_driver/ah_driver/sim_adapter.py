import numpy as np
from sensor_msgs.msg import JointState

from .validation_pipeline import (
    USD_JOINT_ORDER,
    MIMIC_USD_ORDER,
    MIMIC_GEARING,
    remap,
)

# USD joint names as published by Isaac Sim on the JointState topic.
# Controlled q1 joints + mimic q2 joints, in the order Isaac publishes them.
USD_CONTROLLED_NAMES = [
    "index_q1", "middle_q1", "pinky_q1", "ring_q1", "thumb_q1", "thumb_q2",
]
USD_MIMIC_NAMES = [
    "index_q2", "middle_q2", "pinky_q2", "ring_q2",
]

# Finger names matching USD_CONTROLLED_NAMES order, for building the target msg.
USD_FINGER_NAMES = ["index", "middle", "pinky", "ring", "thumb_flexor", "thumb_rotator"]


class SimAdapter:
    """Translates between Isaac Sim JointState messages and the pipeline's
    internal representation (radians, USD_JOINT_ORDER).

    No unit conversion needed: Isaac Sim publishes and expects radians.
    No sign flips: the USD asset already uses the same convention as the pipeline.
    The only work here is mapping USD joint names to USD_JOINT_ORDER indices.
    """

    def feedback_to_pipeline(
        self,
        msg: JointState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract controlled joint pos/vel from a JointState message.

        Args:
            msg: JointState published by Isaac Sim
        Returns:
            pos_rad: positions in radians, USD_JOINT_ORDER
            vel_rad: velocities in radians/s, USD_JOINT_ORDER
        """
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))

        # map USD joint names to the logical finger names used in USD_JOINT_ORDER
        usd_to_finger = dict(zip(USD_CONTROLLED_NAMES, USD_FINGER_NAMES))

        pos_by_finger = {usd_to_finger[n]: name_to_pos[n] for n in USD_CONTROLLED_NAMES}
        vel_by_finger = {usd_to_finger[n]: name_to_vel[n] for n in USD_CONTROLLED_NAMES}

        pos_rad = np.array([pos_by_finger[n] for n in USD_JOINT_ORDER], dtype=np.float32)
        vel_rad = np.array([vel_by_finger[n] for n in USD_JOINT_ORDER], dtype=np.float32)

        return pos_rad, vel_rad

    def compute_mimic(self, pos_rad: np.ndarray) -> np.ndarray:
        """Compute mimic joint positions from q1 positions.

        Args:
            pos_rad: controlled positions in radians, USD_JOINT_ORDER
        Returns:
            mimic_rad: mimic positions in radians, MIMIC_USD_ORDER
        """
        # first 4 entries of USD_JOINT_ORDER are the four fingers with mimic coupling
        return pos_rad[:4] * MIMIC_GEARING

    def make_target_msg(
        self,
        target_deg_usd: np.ndarray,
        mimic_rad: np.ndarray,
        now_ros,
    ) -> JointState:
        """Build a JointState target message for Isaac Sim.

        Args:
            target_deg_usd: targets in degrees, USD_JOINT_ORDER
            mimic_rad:      mimic joint positions in radians, MIMIC_USD_ORDER
            now_ros:        current ROS time (node.get_clock().now().to_msg())
        Returns:
            JointState with all 10 joints (6 controlled + 4 mimic)
        """
        target_rad = np.radians(target_deg_usd)

        msg = JointState()
        msg.header.stamp = now_ros
        msg.name = [
            "index_q1", "middle_q1", "pinky_q1", "ring_q1",
            "thumb_q1", "thumb_q2",
            "index_q2", "middle_q2", "pinky_q2", "ring_q2",
        ]
        msg.position = [
            float(target_rad[0]),  # index_q1
            float(target_rad[1]),  # middle_q1
            float(target_rad[2]),  # pinky_q1
            float(target_rad[3]),  # ring_q1
            float(target_rad[4]),  # thumb_q1
            float(target_rad[5]),  # thumb_q2
            float(mimic_rad[0]),   # index_q2
            float(mimic_rad[1]),   # middle_q2
            float(mimic_rad[2]),   # pinky_q2
            float(mimic_rad[3]),   # ring_q2
        ]
        return msg

    def make_zero_msg(self, now_ros) -> JointState:
        msg = JointState()
        msg.header.stamp = now_ros
        msg.name = [
            "index_q1", "middle_q1", "pinky_q1", "ring_q1",
            "thumb_q1", "thumb_q2",
            "index_q2", "middle_q2", "pinky_q2", "ring_q2",
        ]
        msg.position = [0.0] * 10
        return msg