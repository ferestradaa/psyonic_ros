import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from ah_messages.msg import Digits
import numpy as np 
import onnxruntime as ort 
from ament_index_python.packages import get_package_share_directory
import os




class AbilityPolicyNode(Node): 
    def __init__(self): 
        super().__init__("ah_policy_node")

        pkg_path = get_package_share_directory('ah_driver')
        model_path = os.path.join(pkg_path, 'models', 'policy.onnx')

        self.model_path = self.declare_parameter("onnx_path", model_path).value
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name #obs (feecback)
        self.output_name = self.session.get_outputs()[0].name #actions (joint position)        

        self.last_position = None
        self.last_velocity = None
        self.last_touch = None
        self.last_mimic_pos = None

        self.pos_sub = self.create_subscription(Float32MultiArray, '/ability_hand/right/feedback/position', self.position_feedback_cb, 1)
        self.vel_sub = self.create_subscription(Float32MultiArray, '/ability_hand/right/feedback/velocity', self.velocity_feedback_cb, 1)
        self.touch_sub = self.create_subscription(Float32MultiArray, '/ability_hand/right/feedback/touch', self.touch_feedback_cb, 1)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 1)

        self.target_pub = self.create_publisher(Digits, '/ability_hand/right/target/position', 1)

        self.timer= self.create_timer(1.0 / 60.0, self.inference_timer)

        self.HAND_JOINT_ORDER = ["index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator"] #real hardware joint order
        self.SIM_JOINT_ORDER = ["index", "middle", "pinky", "ring", "thumb_flexor", "thumb_rotator"] #sim order used in training
        self.MIMIC_SIM_ORDER = ["index", "middle", "pinky", "ring"] #order of q2 mimic joints matches training's mimic_joint_ids

        # names as published by the driver on /joint_states, q2 is the mimic joint
        self.MIMIC_JOINT_STATE_NAMES = {
            "index": "index_q2",
            "middle": "middle_q2",
            "pinky": "pinky_q2",
            "ring": "ring_q2",
        }

        self.JOINT_LOWER_LIMITS_RAD = np.array([0.0, 0.0, 0.0, 0.0, -1.74, 0.0], dtype=np.float32)
        self.JOINT_UPPER_LIMITS_RAD = np.array([1.74, 1.74, 1.74, 1.74, 0.0, 1.74], dtype=np.float32)
        self.finger_joint_target_rad= np.zeros(len(self.SIM_JOINT_ORDER), dtype = np.float32) #array to storage joint target

        self.cfg_action_scale = 0.05 #scale used in train, matches action_scale in AbilityIsaaclabEnvCfg

        self._seeded = None

        self.contact_thereshold = 1.0

        self._sent_initial_target = False



    def position_feedback_cb(self, msg): 
        self.last_position = msg.data

    def velocity_feedback_cb(self, msg): 
        self.last_velocity = msg.data 

    def touch_feedback_cb(self, msg): 
        self.last_touch = msg.data 

    def joint_state_cb(self, msg):
        # pulls the real q2 mimic joint positions straight from the driver,
        # this is the same source of truth training used (PhysX joint_pos on
        # the mimic joint), instead of approximating q2 from q1 * gearing
        by_name = dict(zip(msg.name, msg.position))
        try:
            self.last_mimic_pos = [
                by_name[self.MIMIC_JOINT_STATE_NAMES[finger]]
                for finger in self.MIMIC_SIM_ORDER
            ]
        except KeyError:
            # driver has not published all q2 names yet, keep last good value
            pass



    def inference_timer(self): 


        if self.last_touch is None or self.last_position is None or self.last_velocity is None or self.last_mimic_pos is None: 
            self._publish_initial_target()
            return 

        pos_deg = self._remap_by_name(self.last_position, self.HAND_JOINT_ORDER, self.SIM_JOINT_ORDER)
        vel_def_per_sec = self._remap_by_name(self.last_velocity, self.HAND_JOINT_ORDER, self.SIM_JOINT_ORDER)

        controlled_pos = np.radians(pos_deg)
        controlled_vel = np.radians(vel_def_per_sec)

        # feedback comes in hardware convention (negative = flexed), sim trained in positive
        # negate rotator position and velocity to match sim convention
        controlled_pos[-1] *= -1.0
        controlled_vel[-1] *= -1.0

        mimic_pos = np.array(self.last_mimic_pos, dtype=np.float32)

        if not self._seeded: 
            self.finger_joint_target_rad = controlled_pos.copy()
            self._seeded = True

        fingers_in_contact = self._compute_fingers_in_contact(self.last_touch)


        self.get_logger().info(
            f"rotator feedback: raw_deg={pos_deg[-1]:.4f} after_negate_rad={controlled_pos[-1]:.4f}",
            throttle_duration_sec=0.5
        )

        obs = np.concatenate([
            controlled_pos,
            controlled_vel,
            mimic_pos,
            np.array(fingers_in_contact, dtype=np.float32),
        ]).astype(np.float32).reshape(1, -1)

        raw_output = self.session.run([self.output_name], {self.input_name: obs})[0]
        action = raw_output.reshape(-1) #aplanar the output because it comes as (6,1)

        finger_delta = action * self.cfg_action_scale #insread of raw angle, uses delta based on current state
        self.finger_joint_target_rad = np.clip(
            self.finger_joint_target_rad + finger_delta, 
            self.JOINT_LOWER_LIMITS_RAD, 
            self.JOINT_UPPER_LIMITS_RAD, 
        )        


        self.get_logger().info(f"action: {action}", throttle_duration_sec=0.5)
        self.get_logger().info(f"target_rad: {self.finger_joint_target_rad}", throttle_duration_sec=0.5)
        self.get_logger().info(f"actual_pos_rad (sim order): {controlled_pos}", throttle_duration_sec=0.5)
        self.get_logger().info(f"fingers in contact: {fingers_in_contact}", throttle_duration_sec=0.5)
        

        action_deg_sim_oder = np.degrees(self.finger_joint_target_rad)
        action_deg_hand_order = self._remap_by_name(action_deg_sim_oder, self.SIM_JOINT_ORDER, self.HAND_JOINT_ORDER)
         # convert back to hardware convention before publishing
        action_deg_hand_order[-1] *= -1.0


        msg = Digits()
        msg.reply_mode = 1 #pos vel and touch
        msg.data = [float(v) for v in action_deg_hand_order]
        self.get_logger().info(f"publishing: {msg.data}", throttle_duration_sec=0.5)
        self.target_pub.publish(msg)


    def _remap_by_name(self, values_in_hand_order, source_names, target_names): 
        by_name = dict(zip(source_names, values_in_hand_order))
        return [by_name[name] for name in target_names]


    def _compute_fingers_in_contact(self, touch_data):
        # touch feedback is 30 floats: 5 fingers x 6 force sensors each, grouped
        # consecutively in hardware order (index, middle, ring, pinky, thumb)
        # scale is 0 to 5, 0 free, 5 max contact force
        touch = np.array(touch_data, dtype=np.float32).reshape(5, 6)
        per_finger_max = touch.max(axis=1)
        contact_by_name = dict(zip(["index", "middle", "ring", "pinky", "thumb"], per_finger_max))

        # training order was index, middle, pinky, ring, thumb, see contact sensor cfg
        sim_contact_order = ["index", "middle", "pinky", "ring", "thumb"]
        return [float(contact_by_name[name] > self.contact_thereshold) for name in sim_contact_order]


    def _publish_initial_target(self):
        # sends a neutral target once at startup, only to make the driver start
        # publishing feedback, the policy has not run yet at this point
        msg = Digits()
        msg.reply_mode = 1
        msg.data = [0.0] * len(self.HAND_JOINT_ORDER)
        self.target_pub.publish(msg)




def main(): 
    rclpy.init()
    node = AbilityPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': 
    main()