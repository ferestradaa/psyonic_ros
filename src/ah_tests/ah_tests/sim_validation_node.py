import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

from .validation_pipeline import ValidationPipeline, USD_JOINT_ORDER
from .sim_adapter import SimAdapter

CONTROL_HZ = 60.0
TOUCH_ZEROS = np.zeros(30, dtype=np.float32)


class SimValidationNode(Node):
    def __init__(self):
        super().__init__("sim_validation_node")

        self.pipeline = ValidationPipeline()
        self.adapter  = SimAdapter()

        self.last_pos_rad = None
        self.last_vel_rad = None

        self.create_subscription(
            JointState,
            "/ability_hand/right/feedback/position",
            self._feedback_cb,
            10,
        )

        self.target_pub = self.create_publisher(JointState, "/ability_hand/right/target/position", 10)
        self.create_timer(1.0 / CONTROL_HZ, self._loop)
        self.get_logger().info("sim validation node ready")

        self.debug = False

    def _feedback_cb(self, msg: JointState):
        self.last_pos_rad, self.last_vel_rad = self.adapter.feedback_to_pipeline(msg)

    def _loop(self):
        if self.last_pos_rad is None or self.last_vel_rad is None:
            self.target_pub.publish(
                self.adapter.make_zero_msg(self.get_clock().now().to_msg())
            )
            return

        # pipeline expects degrees in HAND_JOINT_ORDER but sim adapter already
        # gave us radians in USD_JOINT_ORDER — convert back to degrees for the
        # pipeline input contract
        pos_deg_hw = np.degrees(self.last_pos_rad)
        vel_deg_hw = np.degrees(self.last_vel_rad)

        target_deg_hw, controlled_pos, mimic, contact, target_deg_usd = self.pipeline.step(
            pos_deg_hw,
            vel_deg_hw,
            TOUCH_ZEROS,
            self.get_clock().now().nanoseconds,
        )

        if self.debug:
            self.get_logger().info(
                f"target={np.round(target_deg_usd, 2)} pos={np.round(controlled_pos, 3)}",
                throttle_duration_sec=0.25,
            )

        self.target_pub.publish(
            self.adapter.make_target_msg(
                target_deg_usd,
                mimic,
                self.get_clock().now().to_msg(),
            )
        )


def main():
    rclpy.init()
    rclpy.spin(SimValidationNode())
    rclpy.shutdown()