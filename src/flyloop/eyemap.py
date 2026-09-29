"""Eye maps for.

`calibrate_ommatidia()`: the viewing direction of every FlyGym ommatidium (= FlyVis column, same
order) for both eyes, measured by rendering a bright marker at a grid of directions around the
head and recording which ommatidia see it. Directions are in the fly frame: azimuth from the front,
positive to the left; elevation positive up.

  python -m flyloop.eyemap calibrate
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data" / "eyemap"
MARKER_DIST, MARKER_R = 10.0, 0.8  # mm


def _unit(az, el):
    az, el = np.radians(az), np.radians(el)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def calibrate_ommatidia(step=5.0):
    from flygym.compose import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import make_locomotion_fly

    fly = make_locomotion_fly(name="nmf", add_adhesion=True, colorize=True)
    fly.add_vision()
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 20.0], Rotation3D("quat", [1, 0, 0, 0]))  # high: the floor must not hide low markers
    spec = world.mjcf_root
    body = spec.worldbody.add_body(name="marker", mocap=True, pos=[0, 0, -50])
    mat = spec.add_material(name="marker_mat", emission=1.0, rgba=[1, 0, 1, 1])  # self-lit magenta
    body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[MARKER_R, 0, 0], material="marker_mat",
                  contype=0, conaffinity=0)
    sim = Simulation(world)
    m, d = sim.mj_model, sim.mj_data
    mj.mj_forward(m, d)
    head = d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/c_head")] if mj.mj_name2id(
        m, mj.mjtObj.mjOBJ_BODY, "nmf/c_head") >= 0 else d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/c_thorax")] + [0.6, 0, 0.2]

    def hex_view():
        raw = sim.get_raw_vision(fly.name)
        return np.stack([sim.retina.raw_image_to_hex_pxls(img).mean(-1) for img in raw])  # (2 eyes, 721)

    d.mocap_pos[0] = [0, 0, -50]
    mj.mj_forward(m, d)
    base = hex_view()
    azs, els = np.arange(-180, 180, step), np.arange(-75, 90, step)
    dirs, resp = [], []
    for el in els:
        for az in azs:
            d.mocap_pos[0] = head + MARKER_DIST * _unit(az, el)
            mj.mj_forward(m, d)
            resp.append(np.abs(hex_view() - base))  # the marker can be darker than the sky behind it
            dirs.append(_unit(az, el))
    resp, dirs = np.array(resp), np.array(dirs)  # (n_dir, 2, 721), (n_dir, 3)
    w = resp ** 2
    vec = np.einsum("kei,kj->eij", w, dirs)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
    az = np.degrees(np.arctan2(vec[..., 1], vec[..., 0]))
    el = np.degrees(np.arcsin(np.clip(vec[..., 2], -1, 1)))
    seen = resp.max(0) > 0.05
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "ommatidia_dirs.npz", az=az, el=el, seen=seen, head=head)
    for e, name in enumerate(sim._eye_names if hasattr(sim, "_eye_names") else ["eye0", "eye1"]):
        print(f"{name}: seen {seen[e].sum()}/721, azimuth {np.percentile(az[e][seen[e]], [2, 50, 98]).round(0)}, "
              f"elevation {np.percentile(el[e][seen[e]], [2, 50, 98]).round(0)}")
    return az, el, seen


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate_ommatidia()


def _sphere_dirs(P):
    """Unit vectors from the least-squares sphere centre, in the fly frame (forward, left, up).
    FlyWire: +x = fly's right, +y = ventral, +z = posterior."""
    A = np.c_[2 * P, np.ones(len(P))]
    x, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    d = P - x[:3]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.c_[-d[:, 2], -d[:, 0], -d[:, 1]]


def _angles(u):
    return np.degrees(np.arctan2(u[:, 1], u[:, 0])), np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))


# ---------------------------------------------------------------------------------------------
# One physical direction space for FlyVis columns, FlyGym facets and FlyWire neurons.
# FlyVis frame (from its own T4 tuning: T4a front-to-back prefers -X, T4b +X, T4c upward +Y):
# +X = front, +Y = up. Eye field from the FlyGym calibration.
EYES = ("left", "right")


def flyvis_xy():
    from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel
    return np.array(hex_to_pixel(*get_hex_coords(15))).T  # (721, 2)


def flyvis_dirs():
    """{eye: (az, el)} of the 721 FlyVis columns: the lattice's X/Y percentile range stretched
    over the calibrated eye field (azimuth measured from the front, + = the eye's own side)."""
    z = np.load(OUT / "ommatidia_dirs.npz")
    xy = flyvis_xy()
    out = {}
    for e, eye in enumerate(EYES):
        s = z["seen"][e]
        aabs, el = np.abs(z["az"][e][s]), z["el"][e][s]
        a_lo, a_hi = np.percentile(aabs, [2, 98])
        e_lo, e_hi = np.percentile(el, [2, 98])
        x_lo, x_hi = np.percentile(xy[:, 0], [2, 98])
        y_lo, y_hi = np.percentile(xy[:, 1], [2, 98])
        az_abs = a_hi - (xy[:, 0] - x_lo) / (x_hi - x_lo) * (a_hi - a_lo)  # +X = front
        elv = e_lo + (xy[:, 1] - y_lo) / (y_hi - y_lo) * (e_hi - e_lo)  # +Y = up
        out[eye] = (az_abs * (1 if eye == "left" else -1), elv)
    return out


def _unit_deg(az, el):
    return _unit(np.asarray(az), np.asarray(el))


def facet_resampler(k=3):
    """{eye: (idx (721, k), w (721, k))}: FlyVis column i takes the FlyGym facets idx[i] of that
    eye weighted by w[i] (inverse angular distance among calibrated facets)."""
    z = np.load(OUT / "ommatidia_dirs.npz")
    fv = flyvis_dirs()
    out = {}
    for e, eye in enumerate(EYES):
        s = np.flatnonzero(z["seen"][e])
        U = _unit_deg(z["az"][e][s], z["el"][e][s])
        V = _unit_deg(*fv[eye])
        ang = np.degrees(np.arccos(np.clip(V @ U.T, -1, 1)))
        nn = np.argsort(ang, 1)[:, :k]
        d = np.take_along_axis(ang, nn, 1)
        w = 1.0 / (d + 1.0)
        out[eye] = (s[nn], w / w.sum(1, keepdims=True), d[:, 0])
    return out


HANDOFF = ["T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "Tm1", "Tm16",
           "Tm2", "Tm20", "Tm3", "Tm4", "Tm5a", "Tm5b", "Tm5c", "Tm9", "TmY10", "TmY14", "TmY15",
           "TmY3", "TmY4", "TmY5a"]  # FlyVis output types with the same name in FlyWire


def flywire_columns(iters=8):
    """Assign every FlyWire handoff neuron to a FlyVis column of its eye.
    Anchors: Mi1, whose (medulla-chart) positions are stretched over the eye field, with the
    first optic chiasm reversing front and back. Every other optic-lobe neuron of that side gets
    the input-weighted mean direction of its presynaptic partners, iterated. Returns
    {eye: dict(neuron, column, type, spread_deg)}."""
    import pandas as pd
    from flyloop import data
    n = data.neurons()
    t, sd = n.cell_type.astype(str).to_numpy(), n.side.astype(str).to_numpy()
    a = pd.read_csv(data.ANNOT, sep="\t", low_memory=False).drop_duplicates("root_id").set_index("root_id")
    P = a.reindex(n.root_id)[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * [4, 4, 40] / 1000
    pre, post, w = data.edges()
    w = np.abs(w).astype(float)
    fv = flyvis_dirs()
    out = {}
    for eye in EYES:
        az_c, el_c = fv[eye]
        a_lo, a_hi = np.percentile(np.abs(az_c), [2, 98])
        e_lo, e_hi = np.percentile(el_c, [2, 98])
        mi = np.flatnonzero((t == "Mi1") & (sd == eye))
        Q = P[mi][:, [0, 2]] - P[mi][:, [0, 2]].mean(0)
        ax = np.linalg.svd(Q, full_matrices=False)[2][0]
        ax *= np.sign(-ax[1])  # h increases toward anterior (smaller z)
        h, up = Q @ ax, -P[mi][:, 1]
        pc = lambda v: (v - np.percentile(v, 2)) / (np.percentile(v, 98) - np.percentile(v, 2))  # noqa: E731
        az_abs = a_lo + pc(h) * (a_hi - a_lo)  # chiasm: anterior medulla sees the back
        el = e_lo + pc(up) * (e_hi - e_lo)
        # propagate over this side's optic-lobe neurons
        pool = np.flatnonzero((sd == eye) & n.super_class.isin(["optic"]).to_numpy())
        pool = np.union1d(pool, mi)
        loc = {g: k for k, g in enumerate(pool)}
        m = np.isin(pre, pool) & np.isin(post, pool)
        src = np.array([loc[g] for g in pre[m]]); dst = np.array([loc[g] for g in post[m]]); ww = w[m]
        U = np.zeros((len(pool), 3)); known = np.zeros(len(pool), bool)
        anch = np.array([loc[g] for g in mi])
        U[anch] = _unit_deg(az_abs, el); known[anch] = True
        for _ in range(iters):
            acc = np.zeros_like(U); tot = np.zeros(len(pool))
            kw = ww * known[src]
            np.add.at(acc, dst, U[src] * kw[:, None]); np.add.at(tot, dst, kw)
            new = tot > 0
            upd = new & ~np.isin(np.arange(len(pool)), anch)
            U[upd] = acc[upd] / np.linalg.norm(acc[upd], axis=1, keepdims=True)
            known |= new
        # input spread: weighted angular deviation of known inputs from the neuron's direction
        cosd = np.clip((U[src] * U[dst]).sum(1), -1, 1)
        dev = np.degrees(np.arccos(cosd)) * ww * known[src]
        spread = np.zeros(len(pool)); np.add.at(spread, dst, dev)
        wsum = np.zeros(len(pool)); np.add.at(wsum, dst, ww * known[src])
        spread = spread / np.maximum(wsum, 1e-9)
        # each columnar type tiles the whole eye: undo the averaging's pull toward the centre by
        # stretching every handoff type's directions over the eye field again
        V = _unit_deg(np.abs(az_c), el_c)
        hand = np.flatnonzero(np.isin(t[pool], HANDOFF) & known)
        az_h, el_h = _angles(U[hand])
        for ty in np.unique(t[pool][hand]):
            k = t[pool][hand] == ty
            az_h[k] = a_lo + pc(az_h[k]) * (a_hi - a_lo)
            el_h[k] = e_lo + pc(el_h[k]) * (e_hi - e_lo)
        col = np.argmax(_unit_deg(az_h, el_h) @ V.T, 1)
        out[eye] = dict(neuron=pool[hand], column=col, type=t[pool][hand].astype("U16"), spread_deg=spread[hand])
    np.savez(OUT / "flywire_columns.npz", **{f"{e}_{k}": v for e in out for k, v in out[e].items()})
    return out


def column_report(cols):
    import pandas as pd
    rows = []
    for eye, c in cols.items():
        for ty in HANDOFF:
            m = c["type"] == ty
            if m.any():
                rows.append({"eye": eye, "type": ty, "n": int(m.sum()), "columns_covered": len(np.unique(c["column"][m])),
                             "input_spread_deg": float(np.median(c["spread_deg"][m]))})
    return pd.DataFrame(rows)
