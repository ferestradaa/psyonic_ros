import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from ah_messages.msg import Digits
import numpy as np

from .validation_pipeline import ValidationPipeline, HAND_JOINT_ORDER

CONTROL_HZ = 60.0


class RealValidationNode(Node):
    def __init__(self):
        super().__init__("ah_validation_real")

        self.pipeline = ValidationPipeline()

        self.last_position = None
        self.last_velocity = None
        self.last_touch    = None

        self.create_subscription(Float32MultiArray, "/ability_hand/right/feedback/position", self._pos_cb, 1)
        self.create_subscription(Float32MultiArray, "/ability_hand/right/feedback/velocity", self._vel_cb, 1)
        self.create_subscription(Float32MultiArray, "/ability_hand/right/feedback/touch",    self._touch_cb, 1)

        self.target_pub = self.create_publisher(Digits, "/ability_hand/right/target/position", 1)
        self.create_timer(1.0 / CONTROL_HZ, self._loop)
        self.get_logger().info("real validation node ready")

    def _pos_cb(self, msg):   self.last_position = np.array(msg.data, dtype=np.float32)
    def _vel_cb(self, msg):   self.last_velocity = np.array(msg.data, dtype=np.float32)
    def _touch_cb(self, msg): self.last_touch    = np.array(msg.data, dtype=np.float32)

    def _loop(self):
        if self.last_position is None or self.last_velocity is None or self.last_touch is None:
            self._pub_zeros()
            return

        target_deg, actual_rad, mimic, contact = self.pipeline.step(
            self.last_position,
            self.last_velocity,
            self.last_touch,
            self.get_clock().now().nanoseconds,
        )

        self.get_logger().info(
            f"target={np.round(target_deg, 2)} actual_rad={np.round(actual_rad, 3)} contact={contact}",
            throttle_duration_sec=0.25,
        )

        msg = Digits()
        msg.reply_mode = 1
        msg.data = [float(v) for v in target_deg]
        self.target_pub.publish(msg)

    def _pub_zeros(self):
        msg = Digits()
        msg.reply_mode = 1
        msg.data = [0.0] * len(HAND_JOINT_ORDER)
        self.target_pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(RealValidationNode())
    rclpy.shutdown()