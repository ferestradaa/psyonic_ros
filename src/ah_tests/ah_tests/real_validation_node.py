import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from ah_messages.msg import Digits
import numpy as np

from .validation_pipeline import ValidationPipeline, USD_JOINT_ORDER
from .real_adapter import RealAdapter

CONTROL_HZ = 60.0
TOUCH_ZEROS = np.zeros(30, dtype=np.float32)


class RealValidationNode(Node):
    def __init__(self):
        super().__init__("real_validation_node")

        self.pipeline = ValidationPipeline()
        self.adapter  = RealAdapter()

        self.last_position = None
        self.last_velocity = None
        self.last_touch    = None

        self.create_subscription(Float32MultiArray, "/ability_hand/right/feedback/position", self._pos_cb,   1)
        self.create_subscription(Float32MultiArray, "/ability_hand/right/feedback/velocity", self._vel_cb,   1)
        self.create_subscription(Float32MultiArray, "/ability_hand/right/feedback/touch",    self._touch_cb, 1)

        self.target_pub = self.create_publisher(Digits, "/ability_hand/right/target/position", 1)
        self.create_timer(1.0 / CONTROL_HZ, self._loop)
        self.get_logger().info("real validation node readyaaaaaaaaa")

        self.debug = False

    def _pos_cb(self,   msg): self.last_position = self.adapter.parse_pos_msg(msg)
    def _vel_cb(self,   msg): self.last_velocity = self.adapter.parse_vel_msg(msg)
    def _touch_cb(self, msg): self.last_touch    = self.adapter.parse_touch_msg(msg)

    def _loop(self):
        if self.last_position is None or self.last_velocity is None or self.last_touch is None:
            self.target_pub.publish(self.adapter.make_zero_msg())
            return

        target_deg_hw, controlled_pos, mimic, contact, target_deg_usd = self.pipeline.step(
            self.last_position,
            self.last_velocity,
            self.last_touch,
            self.get_clock().now().nanoseconds,
        )

        if self.debug:
            self.get_logger().info(
                f"target={np.round(target_deg_hw, 2)} contact={contact}",
                throttle_duration_sec=0.25,
            )

        self.target_pub.publish(self.adapter.make_target_msg(target_deg_hw))


def main():
    rclpy.init()
    rclpy.spin(RealValidationNode())
    rclpy.shutdown()