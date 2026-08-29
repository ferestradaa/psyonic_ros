import os
import sys
import time
import yaml
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "width":    1280,
    "height":   720,
    "renderer": "RaytracedLighting",
})

import omni.usd
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdGeom, Gf

import carb.input
import omni.appwindow

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simulation.ability import AbilityHand

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


def setup_world(world_usd_path: str) -> World:
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 30.0,
    )
    add_reference_to_stage(usd_path=world_usd_path, prim_path="/World")
    return world


def load_hand(usd_path: str, prim_path: str) -> None:
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"hand prim not found at {prim_path}")
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir   = os.path.abspath(os.path.join(script_dir, ".."))

    config_path = os.path.join(root_dir, "config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    usd_path       = os.path.join(root_dir, cfg["paths"]["usd_path"])
    world_usd_path = os.path.join(root_dir, cfg["world"]["stage_path"])
    robot_prim     = cfg["robot"]["prim_path"]
    art_root       = cfg["robot"]["articulation_root"]
    graph_path     = cfg["ros2"]["graph_path"]

    ability = AbilityHand(
        robot_prim_path=robot_prim,
        articulation_root=art_root,
        graph_path=graph_path,
    )

    world = setup_world(world_usd_path)
    load_hand(usd_path, robot_prim)
    simulation_app.update()

    if not ability.setup_ros2_graph(simulation_app.update):
        simulation_app.close()
        return

    simulation_app.update()
    world.reset()

    art = Articulation(prim_path=art_root)
    world.scene.add(art)
    world.reset()

    art.initialize()
    dof_names   = art.dof_names
    n_dofs      = len(dof_names)
    current_pos = art.get_joint_positions()

    art.apply_action(ArticulationAction(
        joint_positions=current_pos,
        joint_indices=np.arange(n_dofs),
    ))

    print(f"[sim] DOF names: {dof_names}")

    app_window  = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard    = app_window.get_keyboard()

    TARGET_DT = 1.0 / 60.0

    while simulation_app.is_running():
        t0 = time.time()

        world.step(render=True)

        r_state = input_iface.get_keyboard_button_flags(keyboard, carb.input.KeyboardInput.R)
        if r_state & carb.input.BUTTON_FLAG_PRESSED:
            world.reset()
            art.initialize()
            current_pos = art.get_joint_positions()
            art.apply_action(ArticulationAction(
                joint_positions=current_pos,
                joint_indices=np.arange(n_dofs),
            ))

        try:
            pos_cmd = og.Controller.attribute(
                f"{graph_path}/SubscribeJointState.outputs:positionCommand"
            ).get()
            if pos_cmd is not None and len(pos_cmd) > 0:
                art.apply_action(ArticulationAction(
                    joint_positions=np.array(pos_cmd, dtype=np.float32),
                    joint_indices=np.arange(len(pos_cmd)),
                ))
        except Exception:
            pass

        elapsed = time.time() - t0
        if elapsed < TARGET_DT:
            time.sleep(TARGET_DT - elapsed)

    print("[sim] closing")
    simulation_app.close()


if __name__ == "__main__":
    main()