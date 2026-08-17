from isaacsim.core.utils.types import ArticulationAction
import numpy as np
from isaacsim.asset.importer.urdf import _urdf
import os
import omni.usd as _usd
import omni.kit.commands
from pxr import UsdGeom, Gf
import omni.replicator.core as rep
import omni.usd
import omni.graph.core as og
from pxr import UsdPhysics
from pxr import PhysxSchema


class AbilityHand():
    def __init__(self):
        pass
        
    def import_urdf_to_usd(self, urdf_path: str, usd_output: str) -> str:

        importer = _urdf.acquire_urdf_interface()
        config   = _urdf.ImportConfig()

        config.set_fix_base(True)
        config.set_merge_fixed_joints(False)
        config.set_self_collision(False)
        config.set_import_inertia_tensor(True)
        config.set_create_physics_scene(False)
        config.set_default_drive_type(2)           # velocity drive
        config.set_default_drive_strength(10.0)
        config.set_default_position_drive_damping(0.0)
        config.set_distance_scale(1.0)
        config.set_density(0.0)
        config.set_convex_decomp(False)
        config.set_make_default_prim(True)

        urdf_dir      = os.path.dirname(os.path.abspath(urdf_path))
        urdf_filename = os.path.basename(urdf_path)

        parsed = importer.parse_urdf(urdf_dir, urdf_filename, config)
        result = importer.import_robot(urdf_dir, urdf_filename, parsed, config, usd_output)

        if not result:
            raise RuntimeError(f"failed import_robot: {urdf_path}")
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(usd_output)
        #UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.GetRootLayer().Save()

        print(f"[sim] URDF imported as {usd_output}  (prim root: {result})")
        return usd_output

        