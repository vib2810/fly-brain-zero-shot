"""closed loop with real eyes. Every 10 ms frame:
  optics  : bright arena + one dark vertical pole -> brightness of 2 x 721 facets (analytic)
  FlyVis  : one step for both eyes of every fly (flow/0000/000)
  handoff : evoked activity of 27 cell types -> Poisson rates into ~31k FlyWire neurons
  brain   : 100 Shiu LIF steps; DNa02 L - R over 50 ms -> CPG turn (tuned gain k, offset b)
  body    : 100 physics steps; a reached pole is replaced 25 mm ahead within +-60 deg

  python -m flyloop.vision_loop check
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from flyloop import data
from flyloop.brain import Brain
from flyloop.clone import transfer_body_brain
from flyloop.eyemap import EYES, OUT as EYEMAP, flyvis_dirs
from flyloop.vision_gate import BG, DARK, R_MAX, handoff_nodes, load_flyvis, run_flyvis, stimuli
from flyloop.body import CUT_LEGS
from flyloop.records import _record_qpos_subset
from flyloop.runner import Runner

OUT = Path("runs/vision_loop")
FRAME, DT_FV, WIN = 100, 0.01, 5
POLE_R, POLE_D, POLE_CONE, REACH, EDGE = 2.5, 25.0, 60.0, 3.0, 2.0
READ = [("DNa02", "left"), ("DNa02", "right"), ("DNg100", "left"), ("DNg100", "right"),
        ("DNa01", "left"), ("DNa01", "right"), ("HSE", "left"), ("HSE", "right")]


@wp.kernel
def _gather(counts: wp.array2d(dtype=wp.int32), idx: wp.array(dtype=wp.int32),  # type: ignore
            out: wp.array2d(dtype=wp.int32)):  # type: ignore
    b, k = wp.tid()
    out[b, k] = counts[b, idx[k]]


def handoff_scale(net, fv_types):
    """Per-type scale (tuned): 99.9th percentile of evoked activity on bar stimuli."""
    path = EYEMAP / "handoff_scale_bars.npz"
    if path.exists():
        z = np.load(path)
        return dict(zip(z["types"], z["scale"]))
    _, eye_i, node = handoff_nodes(fv_types)
    keep, pos = np.unique(node, return_inverse=True)
    st = {k: v for k, v in stimuli().items() if k.startswith("bar") or k == "background"}
    acts = run_flyvis(net, st, torch.tensor(keep))
    bg = acts["background"][:, eye_i, pos].mean(0)
    types = fv_types[node]
    ev = np.concatenate([np.maximum(a[:, eye_i, pos] - bg, 0) for k, a in acts.items() if k != "background"])
    tys = np.unique(types)
    scale = np.array([max(float(np.percentile(ev[:, types == ty], 99.9)), 1e-6) for ty in tys])
    np.savez(path, types=tys, scale=scale)
    return dict(zip(tys, scale))


def warm(net, batch, t=1.0):
    """Uniform-background steady state, one frame at a time (steady_state keeps every step's state)."""
    st = None
    with torch.no_grad():
        for _ in range(round(t / DT_FV)):
            st = net.steady_state(DT_FV, DT_FV, batch, value=BG, state=st)
    return st


def _yaw(q):  # wxyz
    return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


class VisionRunner(Runner):
    """CPG fly + FlyVis eyes + Shiu brain + pole arena. `run(seeds, k, bias, n_frames, steer)`."""

    def __init__(self, n_worlds, max_steps, cut_legs=(), qpos_worlds=(), wiring=None):
        self.qpos_worlds = list(qpos_worlds)
        super().__init__(list(cut_legs), n_worlds, max_steps, record_every=FRAME)
        from flygym.anatomy import BodySegment
        self.thorax = int(self.sim._internal_bodyids_by_fly[self.fly.name][
            self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))])
        self.net, fv_types = load_flyvis()
        self.inputs, eye_i, node = handoff_nodes(fv_types)
        sc = handoff_scale(self.net, fv_types)
        dev = "cuda"
        self._eye = torch.tensor(eye_i, device=dev)
        self._node = torch.tensor(node, device=dev)
        self._scale = torch.tensor([sc[t] for t in fv_types[node]], device=dev, dtype=torch.float32)
        s0 = warm(self.net, 1)
        self._vbg = s0.nodes.activity[0][self._node].float()  # per handoff node, uniform background
        self.brain = Brain(n_worlds, self.inputs, graph_steps=FRAME, cap=2048 * n_worlds, wiring=wiring)
        n = data.neurons()
        t, sd = n.cell_type.astype(str).to_numpy(), n.side.astype(str).to_numpy()
        self._read = wp.array(np.array([np.flatnonzero((t == a) & (sd == b))[0] for a, b in READ], np.int32),
                              dtype=wp.int32)
        self._rc = wp.zeros((n_worlds, len(READ)), dtype=wp.int32)
        fd = flyvis_dirs()
        self.faz = np.stack([fd[e][0] for e in EYES])  # (2, 721) deg, + = left
        self.fel = np.stack([fd[e][1] for e in EYES])

    def _setup_extra(self):
        if self.qpos_worlds:
            nq, n_rec = self.sim.mj_model.nq, self.max_steps // self.every + 1
            self.qsub = wp.zeros((len(self.qpos_worlds), n_rec, nq), dtype=wp.float32)
            self._rec.append((_record_qpos_subset, (len(self.qpos_worlds), nq),
                              [self.sim.mjw_data.qpos, wp.array(self.qpos_worlds, dtype=wp.int32),
                               self.cpg.counter, self.every], [self.qsub]))

    def luminance(self, pos, yaw, poles):
        """(B, 2, 721) facet brightness for thorax positions pos (B, 2), yaw (B,) rad."""
        d = poles - pos
        dist = np.linalg.norm(d, axis=1)
        beta = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        half = np.degrees(np.arcsin(np.clip(POLE_R / np.maximum(dist, POLE_R), 0, 1)))
        waz = np.degrees(yaw)[:, None, None] + self.faz[None]
        delta = np.abs((waz - beta[:, None, None] + 180) % 360 - 180)
        cover = np.clip((half[:, None, None] - delta) / EDGE + 0.5, 0, 1)
        return BG - (BG - DARK) * cover

    def new_pole(self, pos, yaw, rng):
        a = yaw + np.radians(rng.uniform(-POLE_CONE, POLE_CONE))
        return pos + POLE_D * np.array([np.cos(a), np.sin(a)])

    def begin(self, seeds, brain_seed):
        nw = self.n_worlds
        self._start_episode(np.zeros((nw, len(self.param_names))), seeds)
        self.brain.reset(seed=brain_seed)
        self._rngs = [np.random.default_rng(int(s) + 7919) for s in seeds]
        pos, yaw = self.pose()
        self.poles = np.stack([self.new_pole(pos[w], yaw[w], self._rngs[w]) for w in range(nw)])
        self.reached = np.zeros(nw, int)
        self.fv_state = warm(self.net, 2 * nw)
        self._hist = np.zeros((WIN, nw, len(READ)))
        self._prev = np.zeros((nw, len(READ)))
        self.frame = 0

    def pose(self):
        d = self.sim.mjw_data
        p = d.xpos.numpy()[:, self.thorax]
        q = d.xquat.numpy()[:, self.thorax]
        return p[:, :2].astype(float), _yaw(q).astype(float)

    def step_frame(self, k, bias, steer):
        """One 10 ms frame for all worlds. steer: per-world bool (False = steering disconnected)."""
        from flyvis.utils.nn_utils import simulation
        nw = self.n_worlds
        pos, yaw = self.pose()
        lum = self.luminance(pos, yaw, self.poles)
        x = torch.tensor(lum.reshape(2 * nw, 1, 1, 721), dtype=torch.float32, device="cuda")
        with torch.no_grad(), simulation(self.net):
            self.net.stimulus.zero(2 * nw, 1)
            self.net.stimulus.add_input(x)
            self.fv_state = self.net.forward(self.net.stimulus(), DT_FV, self.fv_state, as_states=True)[-1]
            act = self.fv_state.nodes.activity.view(nw, 2, -1)
            self.last_act = act
            a = act[:, self._eye, self._node]
            rate = R_MAX * torch.clamp(torch.relu(a - self._vbg) / self._scale, 0, 1)
            wp.to_torch(self.brain.prob).copy_(rate * 1e-4)
        self.brain.run(FRAME)
        wp.launch(_gather, dim=self._rc.shape, inputs=[self.brain.counts, self._read], outputs=[self._rc])
        c = self._rc.numpy().astype(float)
        self._hist[self.frame % WIN] = c - self._prev
        self._prev = c
        rates = self._hist.sum(0) / (WIN * DT_FV)
        asym = np.clip(-k * (rates[:, 0] - rates[:, 1]) / 100.0 - bias, -0.8, 0.8) * np.asarray(steer)
        params = self.cpg.params.numpy()
        params[:, -1] = asym
        wp.copy(self.cpg.params, wp.array(params, dtype=wp.float64))
        for _ in range(FRAME):
            wp.capture_launch(self._graph)
        pos2, yaw2 = self.pose()
        hit = np.linalg.norm(self.poles - pos2, axis=1) <= REACH
        for w in np.flatnonzero(hit):
            self.reached[w] += 1
            self.poles[w] = self.new_pole(pos2[w], yaw2[w], self._rngs[w])
        self.frame += 1
        bearing = (np.degrees(np.arctan2(*(self.poles - pos2).T[::-1]) - yaw2) + 180) % 360 - 180
        return dict(pos=pos2, yaw=yaw2, pole=self.poles.copy(), rates=rates, asym=asym, bearing=bearing,
                    hit=hit, lum=lum)


def frames(r, n_frames, k, bias, steer, tap=None):
    """Closed loop for n_frames; per-world k, steer. Returns per-frame records (world, frame, ...).
    tap(): called after every frame (video recording)."""
    import time
    keys = ("pos", "yaw", "pole", "rates", "asym", "bearing", "hit")
    rec = {x: [] for x in keys}
    t0 = time.time()
    for f in range(n_frames):
        o = r.step_frame(k, bias, steer)
        for x in keys:
            rec[x].append(o[x])
        if tap is not None:
            tap()
        if f == 49:
            print(f"{(time.time() - t0) / 50 * 1000:.0f} ms/frame for {r.n_worlds} worlds", flush=True)
    r.brain.check_overflow()
    return {x: np.stack(v, 1) for x, v in rec.items()}


def clone(src, dst, rows, cut_steps):
    """Leg-cut clone (after dst.begin): body, CPG and brain (clone.py), plus eyes, arena, DN window."""
    rows = np.asarray(rows)
    transfer_body_brain(src, dst, rows, cut_steps)
    for x, a in src.fv_state.nodes.items():
        dst.fv_state.nodes[x].copy_(a.view(src.n_worlds, 2, -1)[rows].reshape(2 * len(rows), -1))
    dst.poles, dst.reached = src.poles[rows].copy(), src.reached[rows].copy()
    dst._rngs = [copy.deepcopy(src._rngs[w]) for w in rows]
    dst._hist, dst._prev, dst.frame = src._hist[:, rows].copy(), src._prev[rows].copy(), src.frame


def metrics(z, sl=slice(None)):
    """Poles reached and fraction of frames with the pole within +-30 deg of the heading."""
    return z["hit"][:, sl].sum(1), (np.abs(z["bearing"][:, sl]) < 30).mean(1)


def calib():
    """Tuning (disclosed): gain k grid on training seeds; steering-disconnected flies give the
    brain's own left-right offset under side-symmetric poles."""
    ks, n = [0.25, 0.5, 1.0, 2.0, 4.0], 8
    k = np.repeat(ks + [0.0], n)
    steer = k > 0
    seeds = np.tile(np.arange(161_000, 161_000 + n), len(ks) + 1)
    r = VisionRunner(len(k), 1001 * FRAME)
    r.begin(seeds, 161_000)
    z = frames(r, 1000, k, 0.0, steer)
    np.savez(OUT / "calib.npz", k=k, **z)
    hits, front = metrics(z)
    for kk in ks + [0.0]:
        m = k == kk
        lr = z["rates"][m, :, 0] - z["rates"][m, :, 1]
        print(f"k={kk:4}  poles {hits[m].mean():.2f} ({hits[m].tolist()})  front {front[m].mean():.2f}  "
              f"DNa02 L-R {lr.mean():+.1f} Hz  |asym| {np.abs(z['asym'][m]).mean():.2f}")


FV_SHOW = ["Tm3", "T5a"]  # the layers the video shows


class Tap:
    """Video recording for some worlds: which neurons spiked in each frame, and FlyVis activity of
    the FV_SHOW types (per eye, column order)."""

    def __init__(self, r, rows, fv_types, fv_rows):
        self.r, self.rows = r, torch.tensor(rows, device="cuda")
        self.fv_rows = torch.tensor(fv_rows, device="cuda")
        self.nodes = torch.tensor(np.concatenate([np.flatnonzero(fv_types == t) for t in FV_SHOW]), device="cuda")
        self.prev = wp.to_torch(r.brain.counts)[self.rows].clone()
        self.spk, self.fv = [[] for _ in rows], []

    def __call__(self):
        c = wp.to_torch(self.r.brain.counts)[self.rows]
        d = c - self.prev
        self.prev = c.clone()
        for j in range(len(self.rows)):
            self.spk[j].append(torch.nonzero(d[j]).squeeze(1).int().cpu().numpy())
        self.fv.append(self.r.last_act[self.fv_rows][:, :, self.nodes].half().cpu().numpy())

    def arrays(self, tag):
        out = {f"{tag}_fv": np.stack(self.fv, 1)}  # (rows, frames, 2, len(FV_SHOW) * 721)
        for j, s in enumerate(self.spk):
            out[f"{tag}{j}_spk"] = np.concatenate(s)
            out[f"{tag}{j}_off"] = np.r_[0, np.cumsum([len(x) for x in s])]
        return out


def evaluate(k, bias, t_cut=5.0, t_end=15.0, n=16, video=False):
    """Intact brain vs steering disconnected, 0-10 s. Then the brain flies are cut at t_cut
    (rf, lh) and cloned: brain clones vs disconnected clones until t_end.
    video=True: also records every fly's spikes, and the brain flies' FlyVis layers, for the video
    (video_rec.npz; results go to eval_video.npz, a replicate of eval.npz)."""
    seeds = np.arange(162_000, 162_000 + n)
    cut, end = round(t_cut / DT_FV), round(t_end / DT_FV)
    steer = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    src = VisionRunner(2 * n, (end + 1) * FRAME, qpos_worlds=range(2 * n))
    src.begin(np.r_[seeds, seeds], 162_000)
    fv_types = load_flyvis()[1] if video else None
    tap_s = Tap(src, range(n), fv_types, range(n)) if video else None
    pre = frames(src, cut, k, bias, steer, tap_s)
    dst = VisionRunner(2 * n, (end + 1) * FRAME, qpos_worlds=range(2 * n), cut_legs=CUT_LEGS)
    dst.begin(np.r_[seeds, seeds], 162_000)
    clone(src, dst, np.r_[np.arange(n), np.arange(n)], cut * FRAME)
    tap_d = Tap(dst, range(2 * n), fv_types, range(n)) if video else None
    post_cut = frames(dst, end - cut, k, bias, steer, tap_d)
    post_int = frames(src, end - cut, k, bias, steer)
    intact = {x: np.concatenate([pre[x], post_int[x]], 1) for x in pre}
    if video:
        np.savez(OUT / "video_rec.npz", fv_show=np.array(FV_SHOW), **tap_s.arrays("pre"), **tap_d.arrays("cut"))
    np.savez(OUT / ("eval_video.npz" if video else "eval.npz"), k=k, bias=bias, n=n, t_cut=t_cut, seeds=seeds, steer=steer,
             **{f"intact_{x}": v for x, v in intact.items()}, **{f"cut_{x}": v for x, v in post_cut.items()},
             intact_qpos=src.qsub.numpy(), cut_qpos=dst.qsub.numpy())
    h10, f10 = metrics(intact, slice(0, 1000))
    hc, fc = metrics(post_cut)
    hi, fi = metrics(intact, slice(cut, end))
    for name, h, f in [("intact brain 0-10 s", h10[:n], f10[:n]), ("intact disconnected 0-10 s", h10[n:], f10[n:]),
                       ("intact brain 5-15 s", hi[:n], fi[:n]),
                       ("cut brain clone 5-15 s", hc[:n], fc[:n]), ("cut disconnected clone 5-15 s", hc[n:], fc[n:])]:
        print(f"{name:30s} poles {h.mean():.2f} {h.tolist()}  front {f.mean():.2f}")


def shuffle_test(shuffle_seed, ks=(1.0, 2.0, 4.0, 8.0), n=16):
    """the same loop on a degree-preserving shuffled connectome, a gain grid on
    the test flies (best k reported: favours the shuffle), intact body, 10 s."""
    from flyloop.wiring import shuffled_wiring
    seeds = np.arange(162_000, 162_000 + n)
    k = np.repeat(ks, n)
    r = VisionRunner(len(k), 1001 * FRAME, wiring=shuffled_wiring(shuffle_seed))
    r.begin(np.tile(seeds, len(ks)), 162_000)
    z = frames(r, 1000, k, 0.0, np.ones(len(k), bool))
    hz = r.brain.counts.numpy() / 10.0
    drv = np.zeros(hz.shape[1], bool)
    drv[r.inputs] = True
    responding = ((hz > 1) & ~drv).sum(1)
    np.savez(OUT / f"shuffle{shuffle_seed}.npz", k=k, seeds=np.tile(seeds, len(ks)), responding=responding, **z)
    hits, front = metrics(z)
    for kk in ks:
        m = k == kk
        lr = z["rates"][m, :, 0] - z["rates"][m, :, 1]
        print(f"shuffle {shuffle_seed} k={kk:3}  poles {hits[m].mean():.2f} {hits[m].tolist()}  front {front[m].mean():.2f}  "
              f"DNa02 L {z['rates'][m, :, 0].mean():.1f} R {z['rates'][m, :, 1].mean():.1f} Hz  |L-R| {np.abs(lr).mean():.1f}  "
              f"responding {responding[m].mean():.0f}", flush=True)


def scramble_at_cut(k, bias, shuffle_seed=0, t_cut=5.0, t_end=15.0, n=16):
    """intact to t_cut, then each fly is cloned into an injured body with its real
    wiring and one with scrambled wiring (same state, same k). Records every fly for the video:
    eval_scramble.npz + video_rec_scramble.npz (cut rows: real 0..n-1, scrambled n..2n-1)."""
    from flyloop.wiring import shuffled_wiring
    seeds = np.arange(162_000, 162_000 + n)
    cut, end = round(t_cut / DT_FV), round(t_end / DT_FV)
    on = np.ones(n, bool)
    src = VisionRunner(n, (end + 1) * FRAME, qpos_worlds=range(n))
    src.begin(seeds, 162_000)
    fv_types = load_flyvis()[1]
    tap_s = Tap(src, range(n), fv_types, range(n))
    pre = frames(src, cut, k, bias, on, tap_s)
    post, taps, qpos = [], [], []
    for wiring in (None, shuffled_wiring(shuffle_seed)):
        dst = VisionRunner(n, (end + 1) * FRAME, qpos_worlds=range(n), cut_legs=CUT_LEGS, wiring=wiring)
        dst.begin(seeds, 162_000)
        clone(src, dst, np.arange(n), cut * FRAME)
        tap = Tap(dst, range(n), fv_types, range(n) if wiring is None else [0])
        post.append(frames(dst, end - cut, k, bias, on, tap))
        taps.append(tap)
        qpos.append(dst.qsub.numpy())
        del dst
    cut_rec = {x: np.concatenate([post[0][x], post[1][x]]) for x in post[0]}
    rec = {**tap_s.arrays("pre"), "cut_fv": np.stack(taps[0].fv, 1)}
    for j, t in enumerate(taps):
        for i, sp in enumerate(t.spk):
            rec[f"cut{j * n + i}_spk"] = np.concatenate(sp)
            rec[f"cut{j * n + i}_off"] = np.r_[0, np.cumsum([len(x) for x in sp])]
    np.savez(OUT / "video_rec_scramble.npz", fv_show=np.array(FV_SHOW), **rec)
    np.savez(OUT / "eval_scramble.npz", k=k, bias=bias, n=n, t_cut=t_cut, seeds=seeds, shuffle_seed=shuffle_seed,
             **{f"intact_{x}": v for x, v in pre.items()}, **{f"cut_{x}": v for x, v in cut_rec.items()},
             intact_qpos=src.qsub.numpy(), cut_qpos=np.concatenate(qpos))
    h, f = metrics(cut_rec)
    hp, _ = metrics(pre)
    print(f"intact brain 0-5 s               poles {hp.mean():.2f} {hp.tolist()}")
    for name, sl in [("cut, real wiring 5-15 s", slice(0, n)), ("cut, scrambled wiring 5-15 s", slice(n, 2 * n))]:
        print(f"{name:32s} poles {h[sl].mean():.2f} {h[sl].tolist()}  front {f[sl].mean():.2f}  "
              f"DNa02 L {cut_rec['rates'][sl, :, 0].mean():.1f} R {cut_rec['rates'][sl, :, 1].mean():.1f} Hz")


def check():
    """Sanity check: a pole fixed at +60 deg (left) or -60 deg (right), 1 s: does the fly turn toward it?"""
    n = 8
    r = VisionRunner(n, 1000 * FRAME)
    # memory rule: a fresh brain in this context must reproduce sugar -> MN9
    s = data.neuron_sets()["sugar"]
    b2 = Brain(2, s)
    b2.set_rates(np.full((2, len(s)), 150.0))
    b2.run(10_000)
    mn9 = b2.counts.numpy()[:, (data.neurons().cell_type == "CB0701").to_numpy()].mean()
    print(f"runner-context brain check: MN9 {mn9:.1f} Hz (expect ~70)")
    del b2
    r.begin(np.arange(160_000, 160_000 + n), brain_seed=160_000)
    pos, yaw = r.pose()
    side = np.repeat([1, -1], n // 2)
    ang = yaw + np.radians(60) * side
    r.poles = pos + 20.0 * np.stack([np.cos(ang), np.sin(ang)], 1)
    y0 = yaw.copy()
    dl = []
    for f in range(100):
        o = r.step_frame(k=0.6, bias=0.0, steer=np.ones(n, bool))
        r.poles = pos + 20.0 * np.stack([np.cos(ang), np.sin(ang)], 1)  # hold the pole in place
        dl.append(o["rates"][:, 0] - o["rates"][:, 1])
    dyaw = np.degrees(np.unwrap(np.stack([y0, o["yaw"]]), axis=0)[1] - y0)
    print("pole side (+1 left):", side.tolist())
    print("DNa02 L-R mean Hz:", np.mean(dl, 0).round(1).tolist())
    print("yaw change deg (+ = left):", dyaw.round(0).tolist())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "eval":
        evaluate(float(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == "scramble":
        scramble_at_cut(float(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == "shuffle":
        shuffle_test(int(sys.argv[2]))
    elif sys.argv[1] == "record":  # replicate of eval that also records every fly for the video
        evaluate(float(sys.argv[2]), float(sys.argv[3]), video=True)
    else:
        {"check": check, "calib": calib}[sys.argv[1]]()
