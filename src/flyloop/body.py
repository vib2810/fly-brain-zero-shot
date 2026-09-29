"""Fly + world construction shared by all work items (FlyGym 2.1.0, tutorial 4a settings)."""

from flygym.anatomy import LEGS, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

SPAWN_POS = [0, 0, 0.5]  # tutorial 4a spawn
CUT_LEGS = ("rf", "lh")  # the two legs cut: right front, left hind
CUT_LINKS = ["tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5"]  # cut at the femur-tibia joint


def amputate(fly, legs=CUT_LEGS):
    """Delete each leg's tibia+tarsus subtree from the fly spec and FlyGym's bookkeeping."""
    cut = {seg for seg in fly.bodyseg_to_mjcfbody if seg.pos in legs and seg.link in CUT_LINKS}
    spec = fly.mjcf_root
    spec.delete(fly._neutral_keyframe)  # stale sizes block deletions; rebuilt below
    for leg in legs:
        spec.delete(fly.leg_to_adhesionactuator.pop(leg))
    for acts in fly.jointdof_to_mjcfactuator_by_type.values():
        for dof in [d for d in acts if d.child in cut]:
            spec.delete(acts.pop(dof))
    for neutral in fly.jointdof_to_neutralaction_by_type.values():
        for dof in [d for d in neutral if d.child in cut]:
            del neutral[dof]
    for dof in [d for d in fly.jointdof_to_mjcfjoint if d.child in cut]:
        del fly.jointdof_to_mjcfjoint[dof], fly.jointdof_to_neutralangle[dof]
    for seg in [s for s in cut if s.link == "tibia"]:
        spec.delete(fly.bodyseg_to_mjcfbody[seg])
    for seg in cut:
        del fly.bodyseg_to_mjcfbody[seg], fly.bodyseg_to_mjcfgeom[seg]
    fly._neutral_keyframe = spec.add_key(name="neutral", time=0)
    fly._rebuild_neutral_keyframe()
    # Simulation maps adhesion actuators by iterating get_legs_order().
    intact_legs = [l for l in LEGS if l not in legs]
    fly.get_legs_order = lambda: intact_legs


def make_world(name="nmf", add_camera=True, cut_legs=()):
    fly = make_locomotion_fly(name=name, add_adhesion=True, colorize=True)
    cam = None
    if add_camera:
        cam = fly.add_tracking_camera(
            name="body_cam",
            pos_offset=(-0.5, -7.5, 0.0),
            rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
            fovy=30.0,
        )
    if cut_legs:
        amputate(fly, cut_legs)
    contact_segs = [
        s
        for s in ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
        if s in fly.bodyseg_to_mjcfgeom
    ]
    world = FlatGroundWorld()
    world.add_fly(
        fly, SPAWN_POS, Rotation3D("quat", [1, 0, 0, 0]), bodysegs_with_ground_contact=contact_segs
    )
    return fly, world, cam
