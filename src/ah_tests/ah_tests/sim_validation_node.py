import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

from .pipeline import ValidationPipeline, HAND_JOINT_ORDER

CONTROL_HZ = 60.0
TOUCH_ZEROS = np.zeros(30, dtype=np.float32)  # sim has no touch sensors


class SimValidationNode(Node):
    def __init__(self):
        super().__init__("ah_validation_sim")

        self.pipeline = ValidationPipeline()

        self.last_position = None
        self.last_velocity = None

        self.create_subscription(JointState, "/ability_hand/right/feedback/position", self._pos_cb, 1)
        self.create_subscription(JointState, "/ability_hand/right/feedback/velocity", self._vel_cb, 1)

        self.target_pub = self.create_publisher(JointState, "/ability_hand/right/target/position", 1)
        self.create_timer(1.0 / CONTROL_HZ, self._loop)
        self.get_logger().info("sim validation node ready")

    def _pos_cb(self, msg): self.last_position = np.array(msg.position, dtype=np.float32)
    def _vel_cb(self, msg): self.last_velocity = np.array(msg.velocity, dtype=np.float32)

    def _loop(self):
        if self.last_position is None or self.last_velocity is None:
            self._pub_zeros()
            return

        target_deg, actual_rad, mimic, contact = self.pipeline.step(
            self.last_position,
            self.last_velocity,
            TOUCH_ZEROS,
            self.get_clock().now().nanoseconds,
        )

        self.get_logger().info(
            f"target={np.round(target_deg, 2)} actual_rad={np.round(actual_rad, 3)}",
            throttle_duration_sec=0.25,
        )

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = HAND_JOINT_ORDER
        msg.position = [float(v) for v in target_deg]
        self.target_pub.publish(msg)

    def _pub_zeros(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = HAND_JOINT_ORDER
        msg.position = [0.0] * len(HAND_JOINT_ORDER)
        self.target_pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(SimValidationNode())
    rclpy.shutdown()