"""vertical video of the real-eyes pole walk with the leg cut, from one
run that recorded every fly (`vision_loop scramble`: eval_scramble.npz + video_rec_scramble.npz). Style: ground-frame camera, one sentence per shot, 3 s title and end card). The stage is
drawn dark for looks; the eye panel shows the real input (bright arena, dark pole).

  python -m flyloop.vision_video
"""

from pathlib import Path

import mujoco as mj
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from flyloop import data
from flyloop.eyemap import EYES, OUT as EYEMAP, flyvis_dirs, flyvis_xy
from flyloop.video_kit import Views, leg_icon, schedule, trace
from flyloop.vision_loop import DT_FV, OUT as SRC, POLE_R, VisionRunner
from flyloop.video_kit import (AUTHOR, BG, BLUE, CAP_H, FPS, GREEN, GREY, H, RED, W, WHITE, Writer,
                                    badge, caption, font, styled_model)
from flyloop.render import add_segment

OUT = Path("runs/fly_brain_zero_shot.mp4")
VIEW_H = 560  # smaller 3D view than the other videos: the lower half shows eyes and brain
PANEL_Y = CAP_H + VIEW_H
PANEL_H = H - PANEL_Y
ELEV, DIST, SIGMA_S, POLE_H, LEAN = -58.0, 30.0, 0.6, 8.0, 10.0
INSET, CLOSE_DIST = 300, 7.5
TITLE_S, END_S = 3.0, 3.0
POLE_RGBA, GLOW = [0.02, 0.02, 0.025, 1.0], [1.0, 0.36, 0.75, 1.0]
L_COL, R_COL = (80, 220, 140), (40, 140, 90)
CYAN, AMBER = (60, 200, 255), (255, 190, 70)
PANEL_BG, DIM = (17, 20, 27), (120, 126, 140)
LAYERS = [("what it sees", None), ("medulla (Tm3)", "Tm3"), ("motion cells (T5)", "T5a")]


def speed_label(s):
    return "real time" if abs(s - 1) < 0.05 else (f"{s:.1f}x slow motion" if s < 1 else f"{s:.1f}x speed")


def add_cylinder(scene, xy, h, r, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CYLINDER, np.array([r, h / 2, 0.0]), np.array([xy[0], xy[1], h / 2]),
                    np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


class ArenaView:
    def __init__(self, cut=(), width=W):
        self.m, self.d, _ = styled_model(list(cut))  # the old dark stage; the eye panel shows the real input
        self.r = mj.Renderer(self.m, VIEW_H, width, max_geom=8000)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE

    def render(self, qpos, look, cam, poles, trail):
        """poles: [(xy, core rgba or None, glow rgba)]; trail: (k, 3) ground points."""
        self.d.qpos[:] = qpos
        mj.mj_forward(self.m, self.d)
        self.cam.lookat[:], (self.cam.distance, self.cam.elevation, self.cam.azimuth) = look, cam
        self.r.update_scene(self.d, self.cam)
        sc = self.r.scene
        for a, b in zip(trail[:-1], trail[1:]):
            add_segment(sc, a, b, 0.07, [0.22, 0.63, 1.0, 1.0])
        for xy, rgba, glow in poles:
            if rgba is not None:
                add_cylinder(sc, xy, POLE_H, POLE_R, rgba)
            for z, rad in [(0.04, POLE_R + 0.6), (POLE_H + 0.02, POLE_R + 0.02)]:  # glowing rims
                ring = [[xy[0] + rad * np.cos(th), xy[1] + rad * np.sin(th), z] for th in np.linspace(0, 2 * np.pi, 49)]
                for u, v in zip(ring[:-1], ring[1:]):
                    add_segment(sc, u, v, 0.09, glow[:3] + [0.9 * glow[3]])
        return Image.fromarray(self.r.render())


def inset_frame(img, cut):
    """Square crop of a full-view render, framed, labelled."""
    w, h = img.size
    sq = img.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize((INSET, INSET), Image.LANCZOS)
    dr = ImageDraw.Draw(sq)
    dr.rectangle([0, 0, INSET - 1, INSET - 1], outline=RED if cut else (70, 76, 90), width=3)
    dr.rounded_rectangle([8, 8, 118, 38], radius=12, fill=(10, 12, 16))
    dr.text((18, 12), "LEGS", font=font(19, True), fill=RED if cut else GREY)
    return sq


class Fly:
    """Fly w: intact until the cut, then its clone (brain=True: steering; False: disconnected)."""

    def __init__(self, z, rec, w, brain=True):
        n = int(z["n"])
        c = int(round(float(z["t_cut"]) / DT_FV))
        self.t_cut, self.c = float(z["t_cut"]), c
        row = w if brain else w + n
        self.q_pre, self.q_post = z["intact_qpos"][w], z["cut_qpos"][row]
        get = lambda x: np.concatenate([z[f"intact_{x}"][w][:c], z[f"cut_{x}"][row]])
        self.pos, self.yaw, self.pole, self.rates = get("pos"), get("yaw"), get("pole"), get("rates")
        self.hit, self.bearing = get("hit"), get("bearing")
        self.hit_t = np.flatnonzero(self.hit)
        self.n_rec = len(self.pos)
        self.fv = np.concatenate([rec["pre_fv"][w], rec["cut_fv"][w]]).astype(np.float32) if brain else None
        self._spk = [(rec[f"pre{w}_spk"], rec[f"pre{w}_off"]), (rec[f"cut{row}_spk"], rec[f"cut{row}_off"])]
        # frame record f is the state after frame f, i.e. at t = (f + 1) * DT_FV
        pos3 = np.c_[self.pos, np.full(len(self.pos), 1.2)]
        to = self.pole - self.pos
        to *= np.minimum(0.5, LEAN / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-9))
        pos3[:, :2] += to  # halfway toward the pole, at most LEAN mm
        self.look = gaussian_filter1d(pos3, SIGMA_S / DT_FV, axis=0, mode="nearest")
        self.near = gaussian_filter1d(np.c_[self.pos, np.full(len(self.pos), 0.6)], 0.15 / DT_FV, axis=0,
                                      mode="nearest")
        self.az = float(np.degrees(self.yaw[0]))

    def spikes(self, f):
        """Neurons that spiked in frame f."""
        (s, o), k = (self._spk[0], f) if f < self.c else (self._spk[1], f - self.c)
        return s[o[k]:o[k + 1]]

    def idx(self, t):
        return int(np.clip(round(t / DT_FV) - 1, 0, self.n_rec - 1))

    def reached_by(self, i):
        return int((self.hit_t <= i).sum())

    def since_cut(self, i):
        return self.reached_by(i) - self.reached_by(self.c - 1)

    def poles(self, i):
        """The current pole, plus a reached one dissolving (green rims) for 0.4 s."""
        out = [(self.pole[i], POLE_RGBA, GLOW)]
        recent = self.hit_t[(self.hit_t <= i) & (self.hit_t > i - 0.4 / DT_FV)]
        if len(recent):
            j = recent[-1]
            a = 1 - (i - j) * DT_FV / 0.4
            out.append((self.pole[j - 1] if j > 0 else self.pole[0], None, [0.31, 0.86, 0.55, a]))
        return out

    def qpos(self, t):
        qi = int(round(t / DT_FV))
        if t < self.t_cut:
            return self.q_pre[min(qi, self.c)]
        return self.q_post[int(np.clip(qi, self.c + 1, len(self.q_post) - 1))]  # clone's first own record: c + 1

    def render(self, views, t, inset=True):
        i = self.idx(t)
        v = views.get(t)
        tr = self.pos[max(0, i - 800): i + 1: 3]
        img = v.render(self.qpos(t), self.look[i], (DIST, ELEV, self.az), self.poles(i),
                       np.c_[tr, np.full(len(tr), 0.03)])
        if inset:  # close-up of the legs: same renderer, second camera
            close = v.render(self.qpos(t), self.near[i], (CLOSE_DIST, -30.0, self.az + 60.0), self.poles(i)[:1],
                             np.zeros((0, 3)))  # no dissolving rims: up close they read as a hoop
            img.paste(inset_frame(close, t >= self.t_cut), (img.width - INSET - 16, 16))
        return img, i


# ---------------------------------------------------------------- vision panel
class HexMaps:
    """Both eyes' 721 columns as a dot panorama (left eye left, front in the middle)."""

    def __init__(self, h=118, r=2):
        xy = flyvis_xy()
        x, y = xy[:, 0] - xy[:, 0].min(), -xy[:, 1] - (-xy[:, 1]).min()  # +X = front, +Y = up
        s = (h - 2 * r - 2) / y.max()
        ew = int(x.max() * s) + 2 * r + 2
        self.w, self.h = 2 * ew + 8, h
        pix, fac = [], []
        for e in range(2):  # left eye: front at its right edge; right eye mirrored
            px = x * s + r + 1 if e == 0 else self.w - 1 - r - x * s
            py = y * s + r + 1
            for k in range(721):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            pix.append(int(py[k] + dy) * self.w + int(px[k] + dx))
                            fac.append(e * 721 + k)
        self.pix, self.fac = np.array(pix), np.array(fac)

    def image(self, rgb):
        """rgb: (2 * 721, 3) uint8 -> PIL image."""
        a = np.zeros((self.h * self.w, 3), np.uint8)
        a[:] = PANEL_BG
        a[self.pix] = rgb[self.fac]
        return Image.fromarray(a.reshape(self.h, self.w, 3))


def diverging(v):
    """v in [-1, 1] -> blue (hyperpolarised) .. grey .. orange (depolarised)."""
    v = np.clip(v, -1, 1)[:, None]
    neg, pos, mid = np.array([60, 140, 255]), np.array([255, 170, 40]), np.array([48, 52, 64])
    return np.where(v < 0, mid + (neg - mid) * -v, mid + (pos - mid) * v).astype(np.uint8)


class Vision:
    def __init__(self, fly, fv_show):
        self.maps = HexMaps()
        self.fv_show = [str(x) for x in fv_show]
        fd = flyvis_dirs()
        self.faz = np.stack([fd[e][0] for e in EYES])  # read by VisionRunner.luminance
        self.base = np.median(fly.fv, 0)  # display baseline and scale over the fly's recording
        dev = np.abs(fly.fv - self.base)
        self.scale = {ty: max(float(np.percentile(dev[:, :, self.sl(ty)], 99.5)), 1e-6) for _, ty in LAYERS if ty}

    def sl(self, ty):
        k = self.fv_show.index(ty)
        return slice(k * 721, (k + 1) * 721)

    def panel(self, fly, i, size=(330, PANEL_H)):
        img = Image.new("RGB", size, PANEL_BG)
        dr = ImageDraw.Draw(img)
        dr.text((18, 12), "EYES", font=font(24, True), fill=WHITE)
        dr.text((18, 44), "optic-lobe model (FlyVis)", font=font(17), fill=GREY)
        y = 80
        x0 = (size[0] - self.maps.w) // 2
        for name, ty in LAYERS:
            if ty is None:
                lum = VisionRunner.luminance(self, fly.pos[i][None], np.array([fly.yaw[i]]), fly.pole[i][None])[0]
                g = (40 + 215 * lum.reshape(-1) / 0.8).astype(np.uint8)
                rgb = np.repeat(g[:, None], 3, 1)
            else:
                v = (fly.fv[i][:, self.sl(ty)] - self.base[:, self.sl(ty)]) / self.scale[ty]
                rgb = diverging(v.reshape(-1))
            dr.text((18, y), name, font=font(18, True), fill=WHITE if ty is None else GREY)
            img.paste(self.maps.image(rgb), (x0, y + 26))
            y += self.maps.h + 36
        return img


# ---------------------------------------------------------------- brain panel
class BrainMap:
    """All 138,639 FlyWire neurons at their positions, seen from behind (the fly's left on the left).
    Spikes of the last 30 ms glow: eye-driven optic-lobe inputs cyan, the rest of the brain amber."""

    def __init__(self, w=740, h=330, gain=1.0, base=PANEL_BG):
        n = data.neurons()
        a = pd.read_csv(data.ANNOT, sep="\t", usecols=["root_id", "pos_x", "pos_y"], low_memory=False)
        a = a.drop_duplicates("root_id").set_index("root_id").reindex(n.root_id)
        x, y = a.pos_x.to_numpy(float), a.pos_y.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        side = n.side.astype(str).to_numpy()
        if np.nanmean(x[side == "left"]) > np.nanmean(x[side == "right"]):
            x = -x  # fly's left on the image left
        lo, hi = np.nanpercentile(x, [0.2, 99.8]), np.nanpercentile(y, [0.2, 99.8])
        s = min((w - 10) / (lo[1] - lo[0]), (h - 10) / (hi[1] - hi[0]))
        ox, oy = (w - (lo[1] - lo[0]) * s) / 2, (h - (hi[1] - hi[0]) * s) / 2
        self.px = np.where(ok, np.clip((x - lo[0]) * s + ox, 0, w - 1), 0).astype(int)
        self.py = np.where(ok, np.clip((y - hi[0]) * s + oy, 0, h - 1), 0).astype(int)  # FlyWire +y ventral: down
        self.ok, self.w, self.h, self.gain = ok, w, h, gain
        dens = np.zeros((h, w))
        np.add.at(dens, (self.py[ok], self.px[ok]), 1)
        dens = np.log1p(gaussian_filter(dens, 0.7))
        self.bg = np.array(base)[None, None] + (dens / dens.max())[..., None] * np.array([55, 62, 80])[None, None]
        z = np.load(EYEMAP / "flywire_columns.npz")
        self.visual = np.zeros(len(n), bool)
        self.visual[np.concatenate([z[f"{e}_neuron"] for e in EYES])] = True
        t = n.cell_type.astype(str).to_numpy()
        self.dna02 = {s_: int(np.flatnonzero((t == "DNa02") & (side == s_))[0]) for s_ in ("left", "right")}

    def image(self, fly, i):
        glow_v, glow_c = np.zeros((self.h, self.w)), np.zeros((self.h, self.w))
        for lag, wgt in [(0, 1.0), (1, 0.6), (2, 0.3)]:
            if i - lag < 0:
                continue
            s = fly.spikes(i - lag)
            s = s[self.ok[s]]
            v = self.visual[s]
            np.add.at(glow_v, (self.py[s[v]], self.px[s[v]]), wgt)
            np.add.at(glow_c, (self.py[s[~v]], self.px[s[~v]]), wgt)
        gv = np.clip(gaussian_filter(glow_v, 1.0) * 1.6 * self.gain, 0, 1)[..., None]
        gc = np.clip(gaussian_filter(glow_c, 1.0) * 3.0 * self.gain, 0, 1)[..., None]
        a = self.bg + gv * np.array(CYAN) + gc * np.array(AMBER)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        now = set(fly.spikes(i).tolist()) | (set(fly.spikes(i - 1).tolist()) if i > 0 else set())
        for s_, col in [("left", L_COL), ("right", R_COL)]:
            k = self.dna02[s_]
            on = k in now
            x, y = self.px[k], self.py[k]
            r = 10 if on else 7
            dr.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3, fill=col if on else None)
            f = font(17, True)
            label = f"DNa02 {s_[0].upper()}"
            dr.text((x + 14 if s_ == "right" else x - 14 - dr.textlength(label, font=f), y - 10), label, font=f, fill=col)
        return img


def brain_panel(bm, fly, i, title, status, size):
    """status: (text, colour) of the bottom line."""
    img = Image.new("RGB", size, PANEL_BG)
    dr = ImageDraw.Draw(img)
    dr.text((14, 12), title, font=font(24, True), fill=WHITE)
    sub = f"{len(fly.spikes(i)) * 100:,} spikes/s"
    if size[0] > 600:
        sub = "138,639 neurons at their mapped positions · " + sub
    dr.text((14, 44), sub, font=font(17), fill=GREY)
    img.paste(bm.image(fly, i), (10, 72))
    y = 72 + bm.h + 4
    dr.text((14, y), "●", font=font(17), fill=CYAN)
    dr.text((34, y), "driven by the eyes", font=font(17), fill=GREY)
    dr.text((230, y), "●", font=font(17), fill=AMBER)
    dr.text((250, y), "rest of the brain", font=font(17), fill=GREY)
    j0 = max(0, i - int(3.0 / DT_FV))
    seg = fly.rates[j0: i + 1]
    y += 28
    dr.text((14, y), "turn neurons DNa02 L / R (last 3 s)", font=font(17, True), fill=L_COL)
    y += 24
    th = size[1] - y - 42
    trace(dr, (10, y), (size[0] - 20, th), [seg[:, 0], seg[:, 1]], [L_COL, R_COL], 80, "")
    dr.text((14, y + th + 8), status[0], font=font(21, True), fill=status[1])
    return img


# ---------------------------------------------------------------- shots
def lower(vis, bm, fly, i):
    img = Image.new("RGB", (W, PANEL_H), PANEL_BG)
    img.paste(vis.panel(fly, i), (0, 0))
    img.paste(brain_panel(bm, fly, i, "BRAIN (FlyWire wiring)", ("turn command → legs: ON", GREEN), (W - 330, PANEL_H)), (330, 0))
    dr = ImageDraw.Draw(img)
    dr.line([(330, 16), (330, PANEL_H - 16)], fill=(40, 44, 56), width=2)
    dr.text((316, PANEL_H // 2 - 20), "→", font=font(30, True), fill=DIM)
    return img


def frame_base(img):
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(img, (0, CAP_H))
    return canvas


def single(fly, vis, bm, t0, t1, items, tag, cut_badge=True):
    captions, dur = schedule(items)
    speed = (t1 - t0) / dur
    views = Views(t_cut=fly.t_cut, model=ArenaView)
    for k in range(int(round(dur * FPS))):
        t_vid = k / FPS
        t_sim = t0 + speed * t_vid
        img, i = fly.render(views, t_sim)
        canvas = frame_base(img)
        b = fly.bearing[i]
        badge(canvas, f"pole {abs(b):3.0f}° {'left' if b > 0 else 'right'}", (24, CAP_H + 18))
        badge(canvas, f"poles reached: {fly.reached_by(i)}", (24, CAP_H + 72), GREEN)
        if cut_badge and t_sim >= fly.t_cut:
            badge(canvas, "2 legs cut", (24, CAP_H + 126), RED)
        badge(canvas, speed_label(speed), (24, CAP_H + VIEW_H - 60))
        text, shown = captions(t_vid)
        caption(canvas, text, min(1.0, shown * 3 + 0.2), None, 0, tag)
        canvas.paste(lower(vis, bm, fly, i), (0, PANEL_Y))
        yield np.asarray(canvas)
    views.close()


def freeze(fly, vis, bm, items, tag, zoom_s=2.0):
    """Paused at the cut: zoom to a close-up from above with the cut legs red, then hold."""
    captions, dur = schedule(items)
    views = Views(cut_red=True, t_cut=fly.t_cut, model=ArenaView)
    t = fly.t_cut - 1e-6
    i = fly.idx(t)
    v = views.get(t)
    icon = leg_icon()
    p3 = np.r_[fly.pos[i], 1.2]
    low = lower(vis, bm, fly, i)
    for k in range(int(round(dur * FPS))):
        t_vid = k / FPS
        e = 0.5 - 0.5 * np.cos(np.pi * min(1.0, t_vid / zoom_s))
        look = (1 - e) * fly.look[i] + e * p3
        img = v.render(fly.qpos(t), look, (DIST + (7.0 - DIST) * e, ELEV + (-82.0 - ELEV) * e, fly.az),
                       fly.poles(i), np.zeros((0, 3)))
        canvas = frame_base(img)
        badge(canvas, "paused", (24, CAP_H + VIEW_H - 60))
        if e > 0.99:
            canvas.paste(icon, (W - 264, CAP_H + 18), icon)
        text, shown = captions(t_vid)
        caption(canvas, text, min(1.0, shown * 3 + 0.2), None, 0, tag)
        canvas.paste(low, (0, PANEL_Y))
        yield np.asarray(canvas)
    views.close()


def split(fa, fb, bm, t0, t1, items, tag):
    """bm: a half-width BrainMap."""
    captions, dur = schedule(items)
    speed = (t1 - t0) / dur
    views = Views(width=W // 2, t_cut=fa.t_cut, model=ArenaView)
    for k in range(int(round(dur * FPS))):
        t_vid = k / FPS
        t_sim = t0 + speed * t_vid
        canvas = Image.new("RGB", (W, H), BG)
        low = Image.new("RGB", (W, PANEL_H), PANEL_BG)
        for j, (fly, label, col, on) in enumerate([(fa, "real wiring", GREEN, True),
                                                   (fb, "wiring scrambled", RED, False)]):
            img, i = fly.render(views, t_sim, inset=False)
            canvas.paste(img, (j * W // 2, CAP_H))
            badge(canvas, label, (j * W // 2 + 16, CAP_H + 16), col)
            badge(canvas, f"poles since the cut: {fly.since_cut(i)}", (j * W // 2 + 16, CAP_H + 70),
                  GREEN if on else GREY)
            low.paste(brain_panel(bm, fly, i, "BRAIN: real wiring" if on else "BRAIN: wiring scrambled",
                                  ("wiring as mapped" if on else "same neurons, connections shuffled", col),
                                  (W // 2, PANEL_H)),
                      (j * W // 2, 0))
        ImageDraw.Draw(canvas).line([(W // 2, CAP_H), (W // 2, CAP_H + VIEW_H)], fill=BG, width=6)
        ImageDraw.Draw(low).line([(W // 2, 16), (W // 2, PANEL_H - 16)], fill=(40, 44, 56), width=2)
        badge(canvas, "replay after the cut · " + speed_label(speed), (16, CAP_H + VIEW_H - 60))
        text, shown = captions(t_vid)
        caption(canvas, text, min(1.0, shown * 3 + 0.2), None, 0, tag)
        canvas.paste(low, (0, PANEL_Y))
        yield np.asarray(canvas)
    views.close()


def title_card(fly, seconds):
    """Title over the whole brain flickering with the video fly's own spikes."""
    big = BrainMap(w=1040, h=470, gain=1.8, base=BG)
    n = int(seconds / DT_FV)
    busy = np.convolve([len(fly.spikes(f)) for f in range(fly.c)], np.ones(n), "valid")
    i0 = int(np.argmax(busy[: fly.c - n]))  # the busiest stretch before the cut: a vivid first frame
    for k in range(int(round(seconds * FPS))):
        i = i0 + int(k / FPS / DT_FV)
        img = Image.new("RGB", (W, H), BG)
        img.paste(big.image(fly, i), (20, 500))
        dr = ImageDraw.Draw(img)
        y = 150
        for line in ["Can a fly brain adapt", "zero-shot to unseen", "body dynamics?"]:
            f = font(62, True)
            dr.text(((W - dr.textlength(line, font=f)) // 2, y), line, font=f, fill=WHITE)
            y += 78
        for line, yy in [("Every dot: one of 138,639 mapped neurons. Every flash: a spike.", 1000),
                         ("eye model → whole-brain model (FlyWire wiring) → legs", 1042)]:
            f = font(27)
            dr.text(((W - dr.textlength(line, font=f)) // 2, yy), line, font=f, fill=GREY)
        f = font(30, True)
        dr.text(((W - dr.textlength(AUTHOR, font=f)) // 2, 1160), AUTHOR, font=f, fill=BLUE)
        yield np.asarray(img)


REFERENCES = [
    ("Dorkenwald et al. 2024", "Nature", "FlyWire connectome"),
    ("Shiu et al. 2024", "Nature", "whole-brain spiking model"),
    ("Lappalainen et al. 2024", "Nature", "optic-lobe model (FlyVis)"),
    ("Wang-Chen et al. 2024", "Nature Methods", "NeuroMechFly v2"),
]
CONCLUSION = [
    "FlyVis + a whole-brain FlyWire model + NeuroMechFly simulate the full loop: vision → brain → motor commands.",
    "The simulated fly brain demonstrates zero-shot adaptation: with two legs cut, visual feedback through its "
    "unchanged wiring keeps it on target.",
    "The leg rhythm is a hand-built pattern generator: the brain connectome has no nerve cord, so the brain sends "
    "only turn commands.",
]


def end_card(seconds):
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    x, y = 80, 200
    dr.text((x, y), "CONCLUSION", font=font(26, True), fill=BLUE)
    y += 64
    f = font(34, True)
    for b in CONCLUSION:
        lines = [""]
        for word in b.split():  # wrap by pixel width
            if dr.textlength((lines[-1] + " " + word).strip(), font=f) > W - x - 34 - 50:
                lines.append(word)
            else:
                lines[-1] = (lines[-1] + " " + word).strip()
        dr.ellipse([x, y + 14, x + 12, y + 26], fill=BLUE)
        for line in lines:
            dr.text((x + 34, y), line, font=f, fill=WHITE)
            y += 46
        y += 30
    y = max(y + 40, 840)
    dr.text((x, y), "REFERENCES", font=font(26, True), fill=BLUE)
    y += 56
    for who, journal, what in REFERENCES:
        dr.text((x, y), who, font=font(26, True), fill=GREY)
        dr.text((x + 380, y), f"{journal} · {what}", font=font(26), fill=DIM)
        y += 44
    dr.text((x, H - 110), AUTHOR, font=font(30, True), fill=BLUE)
    frame = np.asarray(img)
    for _ in range(int(round(seconds * FPS))):
        yield frame


def pick(z):
    """Rule fixed before rendering: the brain clone with the median poles reached after the cut
    (ties: the median frontal fraction among them)."""
    n = int(z["n"])
    hits = z["cut_hit"][:n].sum(1)
    front = (np.abs(z["cut_bearing"][:n]) < 30).mean(1)
    return int(np.lexsort((front, hits))[n // 2])


def main():
    z = dict(np.load(SRC / "eval_scramble.npz"))  # cut rows n.. are scrambled clones
    rec = np.load(SRC / "video_rec_scramble.npz")
    w = pick(z)
    t_cut = float(z["t_cut"])
    t_end = t_cut + z["cut_pos"].shape[1] * DT_FV
    fa, fb = Fly(z, rec, w, True), Fly(z, rec, w, False)
    print(f"video fly {w} (seed {int(z['seeds'][w])}): poles before cut {fa.reached_by(fa.c - 1)}, after: "
          f"brain {fa.since_cut(10**9)}, disconnected {fb.since_cut(10**9)}", flush=True)
    vis, bm = Vision(fa, rec["fv_show"]), BrainMap()
    out = Writer(OUT)
    out.part(title_card(fa, TITLE_S))
    out.part(single(fa, vis, bm, 0.0, t_cut, [("It sees a dark pole and walks to it.", 6.0),
                                              ("Its eyes drive the mapped brain; its turn neurons steer the legs.", 6.5)],
                    "01  THE TASK", cut_badge=False))
    out.part(freeze(fa, vis, bm, [("Now we cut two of its legs.", 5.5)], "02  THE CUT"), n_fade=2)
    # same playback speed before and after the cut (0.4x), so the injured fly is not made to look faster
    out.part(single(fa, vis, bm, t_cut, t_cut + 5.0, [("Its brain has never controlled this body.", 6.25),
                                                      ("It keeps steering to the poles.", 6.25)],
                    "03  AFTER THE CUT"), n_fade=2)
    out.part(split(fa, fb, BrainMap(w=520, h=250), t_cut, t_end,
                   [("Same fly, same moment: scramble its wiring and it misses them.", t_end - t_cut)], "04  PROOF"))
    out.part(end_card(END_S))
    out.close()
    print(f"{out.n / FPS:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
