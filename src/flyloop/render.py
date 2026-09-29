"""Rendering helpers shared across projects (offline MuJoCo rendering of recorded qpos)."""

import mujoco as mj
import numpy as np


def add_segment(scene, a, b, width, rgba):
    """Append a capsule from a to b (a sphere when a ~ b) to an MjvScene as a decoration."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                    np.array(rgba, dtype=np.float32))
    mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float),
                     np.asarray(b, float))
    scene.ngeom += 1
