"""open-loop gate. Stimuli are drawn in each FlyVis column's viewing
direction (eyemap.flyvis_dirs), both eyes run through FlyVis (flow/0000/000), and each FlyWire
handoff neuron (eyemap.flywire_columns) is driven with its FlyVis cell's evoked activity:
rate = 150 Hz * clip(relu(V - V_background) / max_type, 0, 1), updated every FlyVis frame.

  python -m flyloop.vision_gate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flyloop import data
from flyloop.brain import Brain
from flyloop.eyemap import EYES, HANDOFF, OUT as EYEMAP, flyvis_dirs

OUT = Path("runs/vision_gate")
DT_FV, T_S, R_MAX = 0.01, 2.0, 150.0
BG, DARK = 0.8, 0.0
SEEDS = 8
READ = ["DNa02", "DNa01", "DNg13", "DNp09", "DNg100", "DNg97", "MDN", "DNp01", "LC10a", "LC4", "LPLC2",
        "HSE", "HSN", "HSS"]


def load_flyvis():
    import os
    from flyloop.data import DATA
    os.environ.setdefault("FLYVIS_ROOT_DIR", str(DATA / "flyvis"))  # where `flyvis download-pretrained` put it
    from flyvis import NetworkView
    net = NetworkView("flow/0000/000").init_network()
    nodes = net.connectome.nodes
    types = np.array([x.decode() if isinstance(x, bytes) else x for x in nodes.type[:]])
    return net, types


def stimuli():
    """{name: (frames, 2 eyes, 721) luminance}: dot / bar oscillating +-10 deg at 0.5 Hz around
    azimuth 30/60/90 deg on the left or right; stripes rotating at 60 deg/s either way."""
    fv = flyvis_dirs()
    t = np.arange(int(T_S / DT_FV)) * DT_FV
    out = {}
    for side, sgn in [("left", 1), ("right", -1)]:
        for az0 in (30, 60, 90):
            c = sgn * (az0 + 10 * np.sin(2 * np.pi * 0.5 * t))
            for shape in ("dot", "bar"):
                fr = np.full((len(t), 2, 721), BG)
                for e, eye in enumerate(EYES):
                    az, el = fv[eye]
                    for k in range(len(t)):
                        daz = np.abs(az - c[k])
                        hit = (daz <= 5) & (np.abs(el) <= 5) if shape == "dot" else daz <= 5
                        fr[k, e, hit] = DARK
                out[f"{shape}_{side}_{az0}"] = fr
    for name, omega in [("rotate_left", 60.0), ("rotate_right", -60.0)]:  # left = counter-clockwise
        fr = np.zeros((len(t), 2, 721))
        for e, eye in enumerate(EYES):
            az, _ = fv[eye]
            for k in range(len(t)):
                fr[k, e] = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * (az - omega * t[k]) / 30.0))
        out[name] = fr
    out["background"] = np.full((len(t), 2, 721), BG)
    return out


def run_flyvis(net, stims, keep):
    """{name: (frames, 2, len(keep))} FlyVis activity of the `keep` nodes, one stimulus (both
    eyes) at a time to stay within GPU memory."""
    out = {}
    for name, x in stims.items():
        movie = torch.tensor(x.transpose(1, 0, 2)[:, :, None, :], dtype=torch.float32)  # (2, T, 1, 721)
        with torch.no_grad():
            act = net.simulate(movie.cuda(), DT_FV)[:, :, keep].cpu().numpy()
        out[name] = act.transpose(1, 0, 2)
        torch.cuda.empty_cache()
    return out


def handoff_nodes(fv_types):
    """FlyWire input neurons, their eye (0 left, 1 right) and FlyVis node index."""
    z = np.load(EYEMAP / "flywire_columns.npz")
    col_index = {ty: np.flatnonzero(fv_types == ty) for ty in HANDOFF}  # FlyVis nodes per type, column order
    neurons, eye_i, node = [], [], []
    for e, eye in enumerate(EYES):
        for nrn, c, ty in zip(z[f"{eye}_neuron"], z[f"{eye}_column"], z[f"{eye}_type"]):
            neurons.append(nrn); eye_i.append(e); node.append(col_index[ty][c])
    return np.array(neurons), np.array(eye_i), np.array(node)


def handoff_rates(fv_types, acts, eye_i, node_pos):
    """{stim: (frames, n_inputs) Hz}; acts hold only the kept nodes (node_pos indexes them)."""
    node = node_pos
    bg = acts["background"][:, eye_i, node].mean(0)
    evoked = {k: np.maximum(a[:, eye_i, node] - bg, 0) for k, a in acts.items()}
    types = np.array([str(x) for x in fv_types])
    mx = {ty: max(float(np.percentile(np.concatenate([e[:, types == ty].ravel() for e in evoked.values()]), 99.9)), 1e-6)
          for ty in np.unique(types)}
    rates = {}
    for k, ev in evoked.items():
        r = np.zeros_like(ev)
        for ty, v in mx.items():
            m = types == ty
            r[:, m] = R_MAX * np.clip(ev[:, m] / v, 0, 1)
        rates[k] = r
    return rates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, fv_types = load_flyvis()
    stims = stimuli()
    inputs, eye_i, node = handoff_nodes(fv_types)
    keep, node_pos = np.unique(node, return_inverse=True)
    acts = run_flyvis(net, stims, torch.tensor(keep))
    del net
    torch.cuda.empty_cache()
    rates = handoff_rates(fv_types[node], acts, eye_i, node_pos)
    conds = [k for k in stims if k != "background"]
    n = data.neurons()
    t, sd = n.cell_type.astype(str).to_numpy(), n.side.astype(str).to_numpy()
    b = len(conds) * SEEDS
    br = Brain(b, inputs, seed=150_000, cap=2048 * b)
    frames = int(T_S / DT_FV)
    steps = int(DT_FV / 1e-4)
    counts_prev = np.zeros((b, len(n)))
    for f in range(frames):
        p = np.concatenate([np.repeat(rates[c][f][None], SEEDS, 0) for c in conds])
        br.set_rates(p)
        br.run(steps)
    br.check_overflow()
    hz = br.counts.numpy() / T_S
    drv = np.zeros(len(n), bool)
    drv[inputs] = True
    rows = []
    for k, c in enumerate(conds):
        h = hz[k * SEEDS:(k + 1) * SEEDS]
        row = {"cond": c, "responding": int(((h.mean(0) > 1) & ~drv).sum()),
               "input_mean_hz": float(rates[c].mean())}
        for x in READ:
            for side in ("left", "right"):
                m = (t == x) & (sd == side)
                row[f"{x}_{side[0].upper()}"] = float(h[:, m].mean()) if m.any() else np.nan
        dl = h[:, (t == "DNa02") & (sd == "left")].mean(1) - h[:, (t == "DNa02") & (sd == "right")].mean(1)
        row["DNa02_LmR_seeds"] = dl.round(1).tolist()
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["DNa02_LmR_seeds"]).round(1).to_string(index=False))
    df.to_csv(OUT / "gate.csv", index=False)


if __name__ == "__main__":
    main()
