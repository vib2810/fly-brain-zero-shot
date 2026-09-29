"""Warp kernels that record body state every `every` steps."""

import warp as wp


@wp.kernel
def _record(
    xpos: wp.array2d(dtype=wp.vec3f),  # type: ignore
    xquat: wp.array2d(dtype=wp.quatf),  # type: ignore
    body: int,
    counter: wp.array(dtype=wp.int32),  # type: ignore
    every: int,
    pos_out: wp.array2d(dtype=wp.vec3f),  # type: ignore
    quat_out: wp.array2d(dtype=wp.quatf),  # type: ignore
):
    w = wp.tid()
    s = counter[0]
    if s % every == 0:
        pos_out[w, s // every] = xpos[w, body]
        quat_out[w, s // every] = xquat[w, body]


@wp.kernel
def _record_qpos(
    qpos: wp.array2d(dtype=wp.float32),  # type: ignore
    counter: wp.array(dtype=wp.int32),  # type: ignore
    every: int,
    out: wp.array3d(dtype=wp.float32),  # type: ignore
):
    w, j = wp.tid()
    s = counter[0]
    if s % every == 0:
        out[w, s // every, j] = qpos[w, j]


@wp.kernel
def _record_qpos_subset(
    qpos: wp.array2d(dtype=wp.float32),  # type: ignore
    worlds: wp.array(dtype=wp.int32),  # type: ignore
    counter: wp.array(dtype=wp.int32),  # type: ignore
    every: int,
    out: wp.array3d(dtype=wp.float32),  # type: ignore
):
    i, j = wp.tid()
    s = counter[0]
    if s % every == 0:
        out[i, s // every, j] = qpos[worlds[i], j]
