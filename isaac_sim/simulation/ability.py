import omni.graph.core as og


HAND_JOINT_ORDER = ["index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator"]
SIM_JOINT_ORDER  = ["index", "middle", "pinky", "ring", "thumb_flexor", "thumb_rotator"]
CONTROLLED_JOINT_NAMES = [
    "index_q1", "middle_q1", "pinky_q1", "ring_q1", "thumb_q1", "thumb_q2",
]


class AbilityHand:
    def __init__(self, robot_prim_path: str, articulation_root: str, graph_path: str):
        self.robot_prim_path  = robot_prim_path
        self.articulation_root = articulation_root
        self.graph_path        = graph_path
        self._og_controller    = og.Controller()

    def setup_ros2_graph(self, update_callback) -> bool:
        nodes = [
            ("OnPlaybackTick",         "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime",            "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("ROS2Context",            "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishClock",           "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("PublishJointState",      "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeJointState",    "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ]

        connect = [
            ("OnPlaybackTick.outputs:tick",                 "PublishJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick",                 "SubscribeJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick",                 "PublishClock.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime",          "PublishJointState.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime",          "PublishClock.inputs:timeStamp"),
            ("ROS2Context.outputs:context",                 "PublishJointState.inputs:context"),
            ("ROS2Context.outputs:context",                 "SubscribeJointState.inputs:context"),
            ("ROS2Context.outputs:context",                 "PublishClock.inputs:context"),
            ("SubscribeJointState.outputs:execOut",         "ArticulationController.inputs:execIn"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand",   "ArticulationController.inputs:effortCommand"),
        ]

        graph, _, _, _ = self._og_controller.edit(
            {"graph_path": self.graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: nodes,
                og.Controller.Keys.CONNECT:      connect,
            },
        )

        if not graph or not graph.is_valid():
            print("[AbilityHand] ERROR: graph node creation failed")
            return False

        update_callback()

        set_values = [
            ("ROS2Context.inputs:useDomainIDEnvVar",     True),
            ("PublishClock.inputs:topicName",            "clock"),
            ("PublishJointState.inputs:topicName",       "ability_hand/right/feedback/position"),
            ("PublishJointState.inputs:targetPrim",      [self.articulation_root]),
            ("SubscribeJointState.inputs:topicName",     "ability_hand/right/target/position"),
            ("ArticulationController.inputs:robotPath",  self.articulation_root),
            ("ArticulationController.inputs:targetPrim", [self.articulation_root]),
        ]

        _, _, _, _ = self._og_controller.edit(
            graph,
            {og.Controller.Keys.SET_VALUES: set_values},
        )

        print("[AbilityHand] ROS2 graph OK")
        return True