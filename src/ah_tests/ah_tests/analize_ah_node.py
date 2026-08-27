

import time

import rclpy
from rclpy.node import Node
from ah_messages.msg import Digits
from std_msgs.msg import Float32MultiArray

FINGER_NAMES = ["index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator"]
#FINGER_NAMES = ["index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator"]
NUM_DIGITS = len(FINGER_NAMES)

# same target for every finger in deg, for some reason pub uses custom and sub uses ffoat multiarray
# target topic uses ah_messages/msg/Digits, feedback topics use
# Float32MultiArray confirmed with ros2 topic list -t
TARGET_DEG = 99.0
STEP_DEG = 15.0
HOLD_S = 1.0
RATE_HZ = 10.0


class LockupDiagnostic(Node):
    def __init__(self):
        super().__init__("ability_hand_lockup_diagnostic")

        self.last_position = None
        self.last_velocity = None
        self.last_current = None

        self.position_sub = self.create_subscription(
            Float32MultiArray,
            "/ability_hand/right/feedback/position",
            self.on_position,
            50,
        )
        self.velocity_sub = self.create_subscription(
            Float32MultiArray,
            "/ability_hand/right/feedback/velocity",
            self.on_velocity,
            50,
        )
        self.current_sub = self.create_subscription(
            Float32MultiArray,
            "/ability_hand/right/feedback/current",
            self.on_current,
            50,
        )
        self.target_pub = self.create_publisher(
            Digits,
            "/ability_hand/right/target/position",
            50,
        )

        self.get_logger().info(f"diagnostic ready, target={TARGET_DEG} deg for all fingers, step={STEP_DEG} deg")

    def on_position(self, msg):
        self.last_position = list(msg.data)

    def on_velocity(self, msg):
        self.last_velocity = list(msg.data)

    def on_current(self, msg):
        self.last_current = list(msg.data)

    def publish_target(self, value_deg):
        msg = Digits()
        msg.reply_mode = 0
        msg.data = [value_deg] * NUM_DIGITS
        self.target_pub.publish(msg)

    def sample_row(self, t, commanded_deg):
        pos = self.last_position if self.last_position else [float("nan")] * NUM_DIGITS
        vel = self.last_velocity if self.last_velocity else [float("nan")] * NUM_DIGITS
        cur = self.last_current if self.last_current else [float("nan")] * NUM_DIGITS

        row = {"t": t, "commanded_deg": commanded_deg}
        for i, name in enumerate(FINGER_NAMES):
            row[f"{name}_pos"] = pos[i]
            row[f"{name}_vel"] = vel[i]
            row[f"{name}_cur"] = cur[i]

        return row

    def run_step_test(self):
        self.get_logger().info("waiting for first feedback message")
        start_wait = time.time()
        while rclpy.ok() and self.last_position is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_wait > 10.0:
                self.get_logger().error("no feedback received after 10s, aborting")
                return

        period = 1.0 / RATE_HZ
        steps = []
        value = 0.0
        while value < TARGET_DEG:
            steps.append(value)
            value += STEP_DEG
        steps.append(TARGET_DEG)

        t0 = time.time()
        for step_value in steps:
            step_start = time.time()
            while time.time() - step_start < HOLD_S:
                self.publish_target(step_value)
                rclpy.spin_once(self, timeout_sec=period)
                row = self.sample_row(time.time() - t0, step_value)
                summary = " ".join(
                    f"{n}={row[f'{n}_pos']:.1f}/{row[f'{n}_cur']:.2f}A" for n in FINGER_NAMES
                )
                self.get_logger().info(f"t={row['t']:.2f} cmd={step_value:.1f} {summary}")
                time.sleep(period)

def main():
    rclpy.init()
    node = LockupDiagnostic()

    try:
        node.run_step_test()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()