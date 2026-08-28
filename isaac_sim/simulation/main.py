import argparse, os, sys, yaml

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False, 
    "width":    1280,
    "height":   720,
    "renderer": "RaytracedLighting",
})

import carb
#carb.settings.get_settings().set_bool("/physics/collisionCylinderCustomGeometry", True)
#carb.settings.get_settings().set_bool("/physics/collisionConeCustomGeometry", True)

import omni.usd
import omni.kit.commands
import omni.graph.core as og


from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension


from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf
from isaacsim.core.api.objects import GroundPlane
from pxr import PhysxSchema, UsdPhysics
import omni.usd
from pxr import Sdf
from pxr import Vt, Gf

from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

import carb.input
import omni.appwindow

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pxr import UsdShade, UsdPhysics, PhysxSchema
from isaacsim.core.api.objects import DynamicCuboid
import subprocess
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions

from simulation.ability import AbilityHand
import time


enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.asset.importer.urdf")


simulation_app.update()


def setup_world(world_usd_path) -> World:
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 30.0,
    )
    add_reference_to_stage(usd_path=world_usd_path, prim_path="/World")


    '''
    world.scene.add_default_ground_plane(
        z_position=0.0,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    '''

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/PhysicsMaterials", "Scope")
    mat_prim = stage.DefinePrim("/World/PhysicsMaterials/GroundMat", "Material")
    gnd_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    gnd_mat.CreateStaticFrictionAttr().Set(1.0)
    gnd_mat.CreateDynamicFrictionAttr().Set(1.0)
    gnd_mat.CreateRestitutionAttr().Set(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim).CreateFrictionCombineModeAttr().Set("average")

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "defaultGroundPlane" in path and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material(mat_prim),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics"
            )
            print(f"[sim] GroundMat binded to: {path}")
            break

    return world




def load_robot(usd_path: str, robot_prim_path):
    
    add_reference_to_stage(usd_path=usd_path, prim_path=robot_prim_path)
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(robot_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"No valid prim: {robot_prim_path}")
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))


def main():
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR    = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

    _CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")
    with open(_CONFIG_PATH) as f:
        _cfg=yaml.safe_load(f)


    USD_PATH  = os.path.join(ROOT_DIR, _cfg["paths"]["usd_path"])
    os.makedirs(os.path.dirname(USD_PATH), exist_ok=True)


    WORLD_USD_PATH = os.path.join(ROOT_DIR, _cfg["world"]["stage_path"])

    ROBOT_PRIM_PATH   = _cfg["robot"]["prim_path"]
 


    ability = AbilityHand()




    
    world = setup_world(WORLD_USD_PATH)
    load_robot(USD_PATH, ROBOT_PRIM_PATH)
    simulation_app.update()

    
    simulation_app.update()
    world.reset()



    
    '''
    art = Articulation(ROBOT_PRIM_PATH)
    art.initialize()
    dof_names = art.dof_names

    # set explicit gains before physics has a chance to drive to default target

    current_pos = art.get_joint_positions()
    hold_action = ArticulationActions(
        joint_positions=current_pos,
        joint_indices=np.arange(len(dof_names))
    )
    art.apply_action(hold_action)

    

    limits = art.get_dof_limits()
    lower = limits[0, :, 0]
    upper = limits[0, :, 1]
    print(dof_names)
    print(lower, upper)

    '''
    
    app_window = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()


    TARGET_DT = 1.0 / 30.0 #force the gpu to use real time instad of rushhing up 

    while simulation_app.is_running():
        t0 = time.time()
        world.step(render=True)

        r_state = input_iface.get_keyboard_button_flags(keyboard, carb.input.KeyboardInput.R)
        if r_state & carb.input.BUTTON_FLAG_PRESSED:
            world.reset()

        elapsed = time.time() - t0
        if elapsed < TARGET_DT:
            time.sleep(TARGET_DT - elapsed)


        

    print("[sim] Closing")
    simulation_app.close()


if __name__ == "__main__":
    main()   