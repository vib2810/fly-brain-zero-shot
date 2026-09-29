"""Clone flies between runners (e.g. intact -> legs cut): body, CPG, body records and brain state."""

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import warp as wp

from flyloop.brain import RING


def _joint_index(m_src, m_dst):
    """(dst, src) qpos and qvel indices of every dst joint, matched by name."""
    width = {mj.mjtJoint.mjJNT_FREE: (7, 6), mj.mjtJoint.mjJNT_BALL: (4, 3)}
    qd, qs, vd, vs = [], [], [], []
    for j in range(m_dst.njnt):
        k = mj.mj_name2id(m_src, mj.mjtObj.mjOBJ_JOINT, mj.mj_id2name(m_dst, mj.mjtObj.mjOBJ_JOINT, j))
        nq, nv = width.get(m_dst.jnt_type[j], (1, 1))
        qd += range(m_dst.jnt_qposadr[j], m_dst.jnt_qposadr[j] + nq)
        qs += range(m_src.jnt_qposadr[k], m_src.jnt_qposadr[k] + nq)
        vd += range(m_dst.jnt_dofadr[j], m_dst.jnt_dofadr[j] + nv)
        vs += range(m_src.jnt_dofadr[k], m_src.jnt_dofadr[k] + nv)
    return map(np.array, (qd, qs, vd, vs))


def _set(dst_arr, value):
    wp.copy(dst_arr, wp.array(np.ascontiguousarray(value), dtype=dst_arr.dtype))


def transfer_body_brain(src, dst, rows, cut_steps):
    """Copy body, CPG, body records and brain state; dst world r takes src world rows[r]."""
    rows = np.asarray(rows)
    sd, dd = src.sim.mjw_data, dst.sim.mjw_data
    qd, qs, vd, vs = _joint_index(src.sim.mj_model, dst.sim.mj_model)
    qpos, qvel = dd.qpos.numpy(), dd.qvel.numpy()
    qpos[:, qd], qvel[:, vd] = sd.qpos.numpy()[rows][:, qs], sd.qvel.numpy()[rows][:, vs]
    _set(dd.qpos, qpos)
    _set(dd.qvel, qvel)
    _set(dd.time, sd.time.numpy()[rows])
    dd.qacc_warmstart.zero_()
    mjw.forward(dst.sim.mjw_model, dd)
    for name in ("phases", "mags", "seeds"):
        _set(getattr(dst.cpg, name), getattr(src.cpg, name).numpy()[rows])
    dst.cpg.params.zero_()  # the bridge rewrites amplitude and asym before the next body step
    _set(dst.cpg.counter, src.cpg.counter.numpy())
    n_rec = cut_steps // src.every + 1  # body records so far
    for name in ("pos", "quat"):
        a = getattr(dst, name).numpy()
        a[:, :n_rec] = getattr(src, name).numpy()[rows, :n_rec]
        _set(getattr(dst, name), a)
    sb, db = src.brain, dst.brain
    for name in ("v", "g", "last", "counts", "rng", "prob"):
        _set(getattr(db, name), getattr(sb, name).numpy()[rows])
    for name in ("counter", "max_n"):
        _set(getattr(db, name), getattr(sb, name).numpy())
    # pending spikes are coded fly * n_neurons + neuron: re-code them for the dst rows
    ring, ring_n, n = sb.ring.numpy(), sb.ring_n.numpy(), sb.n
    new, new_n = np.zeros(db.ring.shape, np.int32), np.zeros(RING, np.int32)
    for s in range(RING):
        codes = ring[s, :min(ring_n[s], ring.shape[1])]
        fly, nrn = codes // n, codes % n
        out = np.concatenate([r * n + nrn[fly == src_row] for r, src_row in enumerate(rows)])
        new[s, :len(out)], new_n[s] = out, len(out)
    _set(db.ring, new)
    _set(db.ring_n, new_n)
