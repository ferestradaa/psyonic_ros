import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
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

        self.pos_sub = self.create_subscription(Float32MultiArray, '/ability_hand/right/feedback/position', self.position_feedback_cb, 1)
        self.vel_sub = self.create_subscription(Float32MultiArray, '/ability_hand/right/feedback/velocity', self.velocity_feedback_cb, 1)
        self.touch_sub = self.create_subscription(Float32MultiArray, '/ability_hand/right/feedback/touch', self.touch_feedback_cb, 1)

        self.target_pub = self.create_publisher(Digits, '/ability_hand/right/target/position', 1)

        self.timer= self.create_timer(1.0 / 60.0, self.inference_timer)

        self.HAND_JOINT_ORDER = ["index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator"] #real hardware joint order
        self.SIM_JOINT_ORDER = ["index", "middle", "pinky", "ring", "thumb_flexor", "thumb_rotator"] #sim order used in training

        self.JOINT_LOWER_LIMITS_RAD = np.array([0.0, 0.0, 0.0, 0.0, -1.74, 0.0], dtype=np.float32)
        self.JOINT_UPPER_LIMITS_RAD = np.array([1.74, 1.74, 1.74, 1.74, 0.0, 1.74], dtype=np.float32)
        self.ACTION_SCALE = 0.05 #each step factor (delta) instead of sending raw joint angles idrectly to joints
        self.finger_joint_target_rad= np.zeros(len(self.SIM_JOINT_ORDER), dtype = np.float32) #array to storage joint target

        self.MIMIC_GEARING = -1.0585132837295532    
        self.cfg_action_scale = 0.05 #scale used in train

        self._seeded = None

        self.contact_thereshold = 0.5

        self._sent_initial_target = False



    def position_feedback_cb(self, msg): 
        self.last_position = msg.data

    def velocity_feedback_cb(self, msg): 
        self.last_velocity = msg.data 

    def touch_feedback_cb(self, msg): 
        self.last_touch = msg.data 



    def inference_timer(self): 


        if self.last_touch is None or self.last_position is None or self.last_velocity is None: 
            self._publish_initial_target()
            return 

        pos_deg = self._remap_by_name(self.last_position, self.HAND_JOINT_ORDER, self.SIM_JOINT_ORDER)
        vel_def_per_sec = self._remap_by_name(self.last_velocity, self.HAND_JOINT_ORDER, self.SIM_JOINT_ORDER)

        controlled_pos = np.radians(pos_deg)

        controlled_vel = np.radians(vel_def_per_sec)

        if not self._seeded: 
            self.finger_joint_target_rad = controlled_pos.copy()
            self._seeded = True

        mimic_pos = [
            controlled_pos[self.SIM_JOINT_ORDER.index("index")] * self.MIMIC_GEARING, 
            controlled_pos[self.SIM_JOINT_ORDER.index("middle")] * self.MIMIC_GEARING, 
            controlled_pos[self.SIM_JOINT_ORDER.index("pinky")] * self.MIMIC_GEARING, 
            controlled_pos[self.SIM_JOINT_ORDER.index("ring")] * self.MIMIC_GEARING, 
            
        ]

        fingers_in_contact = self._compute_fingers_in_contact(self.last_touch)

        obs = np.concatenate([
            controlled_pos,
            controlled_vel,
            np.array(mimic_pos, dtype=np.float32),
            np.array(fingers_in_contact, dtype=np.float32),
        ]).astype(np.float32).reshape(1, -1)

        raw_output = self.session.run([self.output_name], {self.input_name: obs})[0]
        action = raw_output.reshape(-1) #aplanar the output because it comes as (6,1)
        self.get_logger().info(f"action: {action}")

        finger_delta = action * self.cfg_action_scale #insread of raw angle, uses delta based on current state
        self.finger_joint_target_rad = np.clip(
            self.finger_joint_target_rad + finger_delta, 
            self.JOINT_LOWER_LIMITS_RAD, 
            self.JOINT_UPPER_LIMITS_RAD, 
        )        


        action_deg_sim_oder = np.degrees(self.finger_joint_target_rad)
        action_deg_hand_order = self._remap_by_name(action_deg_sim_oder, self.SIM_JOINT_ORDER, self.HAND_JOINT_ORDER)


        msg = Digits()
        msg.reply_mode = 1 #pos vel and touch
        msg.data = [float(v) for v in action_deg_hand_order]
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
        msg.reply_mode = 0
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


    


    
