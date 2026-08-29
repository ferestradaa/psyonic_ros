import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

from .validation_pipeline import ValidationPipeline, HAND_JOINT_ORDER, SIM_JOINT_ORDER, SIM_JOINT_USD_NAMES, hand_deg_to_usd_rad
CONTROL_HZ = 60.0
TOUCH_ZEROS = np.zeros(30, dtype=np.float32)

#this node suscribes to jointstate (includes position and velocity feedback) published by isaac sim 
#emulates the topic /ability_hand/right/feedback/position which is published by the real hardware
#the control loop executes pipeline which is a simulation of the real inference step for validating sim to real testing

#the ouput emulates the same the real driver expects ../target/position which is read by isaac and applied to the
#articulation inside the sim


class SimValidationNode(Node):
    def __init__(self):
        super().__init__("sim_validation_node")

        self.pipeline = ValidationPipeline()

        self.last_position = None
        self.last_velocity = None

        # single topic carries both position and velocity
        self.create_subscription(JointState, "/ability_hand/right/feedback/position", self._feedback_cb, 10,)

        self.target_pub = self.create_publisher(JointState, "/ability_hand/right/target/position", 10)
        self.create_timer(1.0 / CONTROL_HZ, self._loop)

        self.get_logger().info("sim validation node ready")

    def _feedback_cb(self, msg: JointState):
        # extract only the 6 controlled joints by name, in HAND_JOINT_ORDER
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))

        # USD names to hardware names mapping
        usd_to_hw = {
            "index_q1":  "index",
            "middle_q1": "middle",
            "ring_q1":   "ring",
            "pinky_q1":  "pinky",
            "thumb_q1":  "thumb_flexor",
            "thumb_q2":  "thumb_rotator",
        }

        hw_pos = []
        hw_vel = []
        for hw_name in HAND_JOINT_ORDER:
            usd_name = next(u for u, h in usd_to_hw.items() if h == hw_name)
            hw_pos.append(np.degrees(name_to_pos[usd_name]))
            hw_vel.append(np.degrees(name_to_vel[usd_name]))

        self.last_position = np.array(hw_pos, dtype=np.float32)
        self.last_velocity = np.array(hw_vel, dtype=np.float32)

    def _loop(self):
        if self.last_position is None or self.last_velocity is None:
            return

        target_deg, actual_rad, mimic, contact, target_deg_sim = self.pipeline.step(
            self.last_position,
            self.last_velocity,
            TOUCH_ZEROS,
            self.get_clock().now().nanoseconds,
        )


        self.get_logger().info(
            f"target={np.round(target_deg, 2)} actual_rad={np.round(actual_rad, 3)}",
            throttle_duration_sec=0.25,
        )
        sim_by_name = dict(zip(SIM_JOINT_ORDER, target_deg_sim))
        usd_deg = [sim_by_name[n] for n in ["index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator"]]
        usd_rad = np.radians(np.array(usd_deg, dtype=np.float32))

        # mimic joints follow q1 with gearing, already computed in pipeline
        mimic_rad = mimic  # mimic_pos is already in radians, SIM order index, middle, pinky, ring

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            "index_q1", "middle_q1", "pinky_q1", "ring_q1",
            "thumb_q1", "index_q2", "middle_q2", "pinky_q2", "ring_q2", "thumb_q2"
        ]
        msg.position = [
            float(usd_rad[0]),   # index_q1
            float(usd_rad[1]),   # middle_q1
            float(usd_rad[3]),   # pinky_q1  (usd_rad index 3 = pinky because usd_deg built as index,middle,ring,pinky)
            float(usd_rad[2]),   # ring_q1
            float(usd_rad[4]),   # thumb_q1
            float(mimic_rad[0]), # index_q2
            float(mimic_rad[1]), # middle_q2
            float(mimic_rad[2]), # pinky_q2
            float(mimic_rad[3]), # ring_q2
            float(usd_rad[5]),   # thumb_q2
        ]
        self.target_pub.publish(msg)



    def _pub_zeros(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = SIM_JOINT_USD_NAMES
        msg.position = [0.0] * len(HAND_JOINT_ORDER)
        self.target_pub.publish(msg)


    


def main():
    rclpy.init()
    rclpy.spin(SimValidationNode())
    rclpy.shutdown()