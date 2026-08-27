#THhsi script reads the isaac sim stage and alizes the usd file that has been loaded
#this might be helpul for you to ssee any potential difference between the OG ability hand asset and the trained asset that 
#isaac lab wrote in runtime before training. 


#RUN THIS INSIDE ISAAC SIM 5.1 script editor once the hand asset has been added to stage!
#you can copy paste it inside sim

import omni.usd
from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom, Gf


def separator(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print('-' * 60)


def inspect_usd() -> None:
    stage = omni.usd.get_context().get_stage()
    if not stage:
        raise RuntimeError("No stage open in Isaac Sim")

    root_path = "/ability_hand_right_small"
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Prim not found: {root_path}")

    print(f"Root prim: {root_path}  type={root_prim.GetTypeName()}")

    all_prims          = list(Usd.PrimRange(root_prim))
    revolute_prims     = [p for p in all_prims if p.GetTypeName() == "PhysicsRevoluteJoint"]
    fixed_prims        = [p for p in all_prims if p.GetTypeName() == "PhysicsFixedJoint"]
    rigid_body_prims   = [p for p in all_prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    collision_prims    = [p for p in all_prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    mimic_prims_rotX   = [p for p in all_prims if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotX")]
    mimic_prims_rotY   = [p for p in all_prims if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotY")]
    mimic_prims_rotZ   = [p for p in all_prims if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotZ")]
    articulation_prims = [p for p in all_prims if p.HasAPI(PhysxSchema.PhysxArticulationAPI)]

    separator("1. ARTICULATION ROOT")
    for p in articulation_prims:
        art = PhysxSchema.PhysxArticulationAPI(p)
        print(f"  path                    : {p.GetPath()}")
        print(f"  enabledSelfCollisions   : {art.GetEnabledSelfCollisionsAttr().Get()}")
        print(f"  solverPositionIterations: {art.GetSolverPositionIterationCountAttr().Get()}")
        print(f"  solverVelocityIterations: {art.GetSolverVelocityIterationCountAttr().Get()}")
        print(f"  sleepThreshold          : {art.GetSleepThresholdAttr().Get()}")
        print(f"  stabilizationThreshold  : {art.GetStabilizationThresholdAttr().Get()}")

    separator("2. REVOLUTE JOINTS  (drives + limits + joint props)")
    for p in revolute_prims:
        body0_rel = p.GetRelationship("physics:body0")
        body1_rel = p.GetRelationship("physics:body1")
        body0 = list(body0_rel.GetTargets()) if body0_rel else []
        body1 = list(body1_rel.GetTargets()) if body1_rel else []
        axis_attr = p.GetAttribute("physics:axis")

        print(f"\n  [{p.GetName()}]  {p.GetPath()}")
        print(f"    body0 : {body0}")
        print(f"    body1 : {body1}")
        print(f"    axis  : {axis_attr.Get() if axis_attr else 'N/A'}")

        for axis_token in ["angular", "linear"]:
            if p.HasAPI(UsdPhysics.LimitAPI, axis_token):
                lim = UsdPhysics.LimitAPI(p, axis_token)
                print(f"    limit [{axis_token}] : low={lim.GetLowAttr().Get()}  high={lim.GetHighAttr().Get()}")

        for axis_token in ["angular", "linear"]:
            if p.HasAPI(UsdPhysics.DriveAPI, axis_token):
                drv = UsdPhysics.DriveAPI(p, axis_token)
                print(f"    drive [{axis_token}]")
                print(f"      type           : {drv.GetTypeAttr().Get()}")
                print(f"      stiffness      : {drv.GetStiffnessAttr().Get()}")
                print(f"      damping        : {drv.GetDampingAttr().Get()}")
                print(f"      maxForce       : {drv.GetMaxForceAttr().Get()}")
                print(f"      targetPosition : {drv.GetTargetPositionAttr().Get()}")
                print(f"      targetVelocity : {drv.GetTargetVelocityAttr().Get()}")

        if p.HasAPI(PhysxSchema.PhysxJointAPI):
            pj = PhysxSchema.PhysxJointAPI(p)
            print(f"    maxJointVelocity : {pj.GetMaxJointVelocityAttr().Get()}")
            print(f"    jointFriction    : {pj.GetJointFrictionAttr().Get()}")
            print(f"    armature         : {pj.GetArmatureAttr().Get()}")

    separator("3. FIXED JOINTS")
    for p in fixed_prims:
        body0_rel = p.GetRelationship("physics:body0")
        body1_rel = p.GetRelationship("physics:body1")
        print(f"  [{p.GetName()}]  {p.GetPath()}")
        print(f"    body0: {list(body0_rel.GetTargets()) if body0_rel else []}")
        print(f"    body1: {list(body1_rel.GetTargets()) if body1_rel else []}")

    separator("4. MIMIC JOINTS")
    all_mimic = set(mimic_prims_rotX + mimic_prims_rotY + mimic_prims_rotZ)
    for p in all_mimic:
        for axis in ["rotX", "rotY", "rotZ"]:
            if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                api = PhysxSchema.PhysxMimicJointAPI(p, axis)
                ref_rel     = api.GetReferenceJointRel()
                ref_targets = list(ref_rel.GetTargets()) if ref_rel else []
                print(f"  [{p.GetName()}] axis={axis}")
                print(f"    referenceJoint : {ref_targets}")
                print(f"    gearing        : {api.GetGearingAttr().Get()}")
                print(f"    offset         : {api.GetOffsetAttr().Get()}")
                for t in ref_targets:
                    if not str(t).startswith(root_path):
                        print(f"    WARNING: {t} is OUTSIDE asset root")

    separator("5. RIGID BODIES + MASSES")
    for p in rigid_body_prims:
        rb = UsdPhysics.RigidBodyAPI(p)
        kinematic_attr = p.GetAttribute("physics:kinematicEnabled")
        print(f"\n  [{p.GetName()}]  {p.GetPath()}")
        print(f"    rigidBodyEnabled : {rb.GetRigidBodyEnabledAttr().Get()}")
        print(f"    kinematic        : {kinematic_attr.Get() if kinematic_attr else False}")

        if p.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(p)
            print(f"    mass            : {m.GetMassAttr().Get()}")
            print(f"    density         : {m.GetDensityAttr().Get()}")
            print(f"    centerOfMass    : {m.GetCenterOfMassAttr().Get()}")
            print(f"    diagonalInertia : {m.GetDiagonalInertiaAttr().Get()}")
        else:
            print(f"    mass            : (no MassAPI)")

        if p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prb = PhysxSchema.PhysxRigidBodyAPI(p)
            print(f"    linearDamping            : {prb.GetLinearDampingAttr().Get()}")
            print(f"    angularDamping           : {prb.GetAngularDampingAttr().Get()}")
            print(f"    maxDepenetrationVelocity : {prb.GetMaxDepenetrationVelocityAttr().Get()}")

    separator("6. COLLISION SHAPES")
    for p in collision_prims:
        coll = UsdPhysics.CollisionAPI(p)
        print(f"\n  [{p.GetName()}]  {p.GetPath()}  enabled={coll.GetCollisionEnabledAttr().Get()}")

        if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
            pc = PhysxSchema.PhysxCollisionAPI(p)
            print(f"    contactOffset : {pc.GetContactOffsetAttr().Get()}")
            print(f"    restOffset    : {pc.GetRestOffsetAttr().Get()}")

        mat_binding = p.GetRelationship("physics:material:binding")
        if mat_binding:
            print(f"    material      : {list(mat_binding.GetTargets())}")

        type_name = p.GetTypeName()
        print(f"    shape type    : {type_name}")
        if type_name == "Mesh":
            points = UsdGeom.Mesh(p).GetPointsAttr().Get()
            print(f"    mesh points   : {len(points) if points else 0}")

    separator("7. PHYSICS MATERIALS")
    for p in [p for p in all_prims if p.HasAPI(UsdPhysics.MaterialAPI)]:
        mat = UsdPhysics.MaterialAPI(p)
        print(f"  [{p.GetName()}]  {p.GetPath()}")
        print(f"    staticFriction  : {mat.GetStaticFrictionAttr().Get()}")
        print(f"    dynamicFriction : {mat.GetDynamicFrictionAttr().Get()}")
        print(f"    restitution     : {mat.GetRestitutionAttr().Get()}")

    separator("8. FULL PRIM HIERARCHY (name | type | APIs)")
    for p in all_prims:
        depth  = len(str(p.GetPath()).split("/")) - 2
        indent = "  " * depth
        apis   = p.GetAppliedSchemas()
        print(f"{indent}{p.GetName()}  [{p.GetTypeName()}]  {', '.join(apis) if apis else ''}")

    separator("9. JOINT ORDER SUMMARY")
    print(f"\n  {'Joint name':<30} {'type':<25} {'body0':<30} {'body1'}")
    print(f"  {'-'*30} {'-'*25} {'-'*30} {'-'*30}")
    for p in revolute_prims + fixed_prims:
        body0_rel = p.GetRelationship("physics:body0")
        body1_rel = p.GetRelationship("physics:body1")
        body0 = str(list(body0_rel.GetTargets())[0]) if body0_rel and body0_rel.GetTargets() else ""
        body1 = str(list(body1_rel.GetTargets())[0]) if body1_rel and body1_rel.GetTargets() else ""
        print(f"  {p.GetName():<30} {p.GetTypeName():<25} {body0.split('/')[-1]:<30} {body1.split('/')[-1]}")

    separator("10. POTENTIAL ISSUES")
    issues = []

    for p in all_mimic:
        for axis in ["rotX", "rotY", "rotZ"]:
            if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                api = PhysxSchema.PhysxMimicJointAPI(p, axis)
                ref_rel = api.GetReferenceJointRel()
                for t in (list(ref_rel.GetTargets()) if ref_rel else []):
                    if not str(t).startswith(root_path):
                        issues.append(f"mimic joint {p.GetName()} references {t} outside asset root")

    for p in revolute_prims:
        for axis_token in ["angular", "linear"]:
            if p.HasAPI(UsdPhysics.DriveAPI, axis_token):
                drv = UsdPhysics.DriveAPI(p, axis_token)
                k = drv.GetStiffnessAttr().Get() or 0.0
                d = drv.GetDampingAttr().Get() or 0.0
                if k == 0.0 and d == 0.0:
                    issues.append(f"joint {p.GetName()} stiffness=0 and damping=0 (passive)")

    for p in revolute_prims:
        if not (p.HasAPI(UsdPhysics.DriveAPI, "angular") or p.HasAPI(UsdPhysics.DriveAPI, "linear")):
            issues.append(f"joint {p.GetName()} has NO DriveAPI")

    for p in rigid_body_prims:
        if p.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(p)
            mass    = m.GetMassAttr().Get()
            density = m.GetDensityAttr().Get()
            if (mass is None or mass == 0.0) and (density is None or density == 0.0):
                issues.append(f"rigid body {p.GetName()} MassAPI present but mass=0 and density=0")
        else:
            issues.append(f"rigid body {p.GetName()} has no MassAPI")

    for p in collision_prims:
        mat_binding = p.GetRelationship("physics:material:binding")
        if not mat_binding or not mat_binding.GetTargets():
            issues.append(f"collision prim {p.GetName()} has no physics material binding")

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"\n  [{i}] {issue}")
    else:
        print("\n  no issues detected")

    separator("DONE")


        # add this at the end of the inspector to catch nested collision prims
    separator("11. NESTED COLLISION PRIMS (inside collisions/ children)")
    for p in all_prims:
        if "collisions" in p.GetPath().pathString and p.GetTypeName() in (
            "Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone", "Plane"
        ):
            has_coll = p.HasAPI(UsdPhysics.CollisionAPI)
            has_mat  = bool(p.GetRelationship("physics:material:binding") and
                            p.GetRelationship("physics:material:binding").GetTargets())
            print(f"  {p.GetPath()}  collision={has_coll}  material={has_mat}  type={p.GetTypeName()}")
            if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                pc = PhysxSchema.PhysxCollisionAPI(p)
                print(f"    contactOffset={pc.GetContactOffsetAttr().Get()}  "
                    f"restOffset={pc.GetRestOffsetAttr().Get()}")


try:
    inspect_usd()
except Exception as e:
    print(f"\nERROR: {e}")