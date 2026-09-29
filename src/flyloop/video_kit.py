"""Frame layout, text, badges, the encoder and the dark render stage shared by the video."""

import textwrap

import imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flyloop.body import CUT_LEGS, make_world

W, H, FPS = 1080, 1350, 30


CAP_H, VIEW_H = 250, 760  # caption band, 3D view; the neural panel fills the rest


BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


BG, WHITE, GREY = (13, 15, 20), (255, 255, 255), (170, 176, 189)


ORANGE, BLUE, RED, GREEN = (255, 158, 26), (57, 160, 255), (255, 77, 94), (80, 220, 140)


AUTHOR = "Vibhakar Mohta"


def font(size, bold=False):
    try:
        return ImageFont.truetype(BOLD if bold else REG, size)
    except OSError:  # DejaVu not installed: PIL's default font
        return ImageFont.load_default(size)


def styled_model(cut):
    """Render-only model: dark floor, dark sky, a directional light with shadows."""
    fly, world, _ = make_world(cut_legs=list(cut), add_camera=False)
    s = world.mjcf_root
    s.texture("checker").rgb1, s.texture("checker").rgb2 = [0.09, 0.10, 0.13], [0.12, 0.135, 0.17]
    s.texture("skybox").rgb1, s.texture("skybox").rgb2 = [0.05, 0.06, 0.09], [0.02, 0.02, 0.03]
    s.material("grid").reflectance = 0.25
    light = s.worldbody.add_light(pos=[0, 0, 60], dir=[0.3, 0.2, -1])
    light.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
    light.castshadow = True
    light.diffuse = [0.7, 0.7, 0.7]
    m, d = world.compile()
    m.vis.headlight.diffuse[:] = [0.35, 0.35, 0.35]
    m.vis.headlight.ambient[:] = [0.25, 0.25, 0.28]
    names = [b.name for b in fly.get_bodysegs_order()]
    ids = {n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{fly.name}/{n}") for n in
           ["l_funiculus", "r_funiculus"] if n in names}
    return m, d, ids


def caption(img, lines, alpha=1.0, sub=None, sub_alpha=0.0, tag=None):
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, CAP_H], fill=BG)
    if tag:
        dr.text((40, 28), tag, font=font(22, True), fill=BLUE)
    y = 66
    for size, wrap, lh in [(46, 34, 56), (40, 40, 48), (34, 48, 42)]:
        wrapped = textwrap.wrap(lines, wrap)
        if len(wrapped) * lh <= (CAP_H - 74) - (46 if sub else 0):
            break
    for line in wrapped:
        c = tuple(int(BG[i] + (WHITE[i] - BG[i]) * alpha) for i in range(3))
        dr.text((40, y), line, font=font(size, True), fill=c)
        y += lh
    if sub:
        c = tuple(int(BG[i] + (ORANGE[i] - BG[i]) * sub_alpha) for i in range(3))
        dr.text((40, y + 8), sub, font=font(36, True), fill=c)


def badge(img, text, xy, color=GREY):
    dr = ImageDraw.Draw(img)
    f = font(24, True)
    w = dr.textlength(text, font=f)
    dr.rounded_rectangle([xy[0], xy[1], xy[0] + w + 28, xy[1] + 44], radius=22, fill=(10, 12, 16))
    dr.text((xy[0] + 14, xy[1] + 8), text, font=f, fill=color)


class Writer:
    """Streams frames to the encoder; crossfades 8 frames between consecutive parts."""

    def __init__(self, path):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", pixelformat="yuv420p",
                                    output_params=["-crf", "19"], macro_block_size=1)
        self.last, self.n = None, 0

    def part(self, frames, n_fade=8):
        first = True
        for f in frames:
            if first and self.last is not None:
                for k in range(1, n_fade):
                    a = k / n_fade
                    self.w.append_data(((1 - a) * self.last + a * f.astype(float)).astype(np.uint8))
                    self.n += 1
            first = False
            self.w.append_data(f)
            self.last, self.n = f.astype(float), self.n + 1

    def close(self):
        self.w.close()


MIN_S, WPS = 5.5, 2.5  # a caption stays up >= MIN_S s; reading speed (words/s)


def need(*texts):
    """Seconds a text must stay on screen: 1.5 s + reading time, at least MIN_S."""
    words = [w for t in texts for w in t.split() if any(c.isalnum() for c in w)]
    return max(MIN_S, 1.5 + len(words) / WPS)


def schedule(items):
    """[(caption, seconds)] shown back to back -> (captions(t) -> (text, seconds shown), total s).
    Refuses captions that would not stay up long enough to read."""
    for text, sec in items:
        assert sec >= need(text) - 1e-9, f"{text!r}: {sec} s < {need(text):.1f} s"
    starts = np.cumsum([0.0] + [sec for _, sec in items])

    def captions(t):
        k = min(int(np.searchsorted(starts, t, side="right")) - 1, len(items) - 1)
        return items[k][0], t - starts[k]
    return captions, float(starts[-1])


class Views:
    """Intact render model for t < T_CUT, injured one after. Only one renderer is alive at a time
    (a second GL context corrupts the first one's frames)."""

    def __init__(self, model, t_cut, width=W, cut_red=False):
        self.width, self.cut_red, self.v, self.kind = width, cut_red, None, None
        self.t_cut, self.model = t_cut, model

    def get(self, t):
        kind = "intact" if t < self.t_cut else "injured"
        if kind != self.kind:
            self.close()
            self.v = self.model(() if kind == "intact" else CUT_LEGS, self.width)
            self.kind = kind
            if kind == "intact" and self.cut_red:  # highlight the legs about to be cut
                m = self.v.m
                for g in range(m.ngeom):
                    b = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
                    if any(f"{leg}_{link}" in b for leg in CUT_LEGS for link in ("tibia", "tarsus")):
                        m.geom_rgba[g] = [1.0, 0.2, 0.25, 1.0]
                    elif b.endswith("_wing"):  # see-through wings: the hind leg is under them
                        m.geom_rgba[g][3] = 0.08
                        m.mat_rgba[m.geom_matid[g]][3] = 0.08 if m.geom_matid[g] >= 0 else 0
        return self.v

    def close(self):
        if self.v is not None:
            self.v.r.close()
            self.v, self.kind = None, None


def leg_icon(size=240):
    """Top-view schematic, head up: the two cut legs (CUT_LEGS) shortened with a red X."""
    img = Image.new("RGBA", (size, size), (10, 12, 16, 225))
    d = ImageDraw.Draw(img)
    cx = size / 2
    d.ellipse([cx - 22, 40, cx + 22, 200], fill=(200, 150, 70, 255))  # body
    d.ellipse([cx - 18, 22, cx + 18, 52], fill=(210, 80, 60, 255))  # head
    for leg, y in [("f", 75), ("m", 110), ("h", 145)]:
        for side, sgn in [("l", -1), ("r", 1)]:
            a = (cx + sgn * 20, y)
            tip = (cx + sgn * 95, y + {"f": -45, "m": 0, "h": 45}[leg])
            if side + leg in CUT_LEGS:
                mid = ((a[0] + tip[0]) / 2, (a[1] + tip[1]) / 2)
                d.line([a, mid], fill=(200, 150, 70, 255), width=6)
                d.line([(mid[0] - 12, mid[1] - 12), (mid[0] + 12, mid[1] + 12)], fill=RED, width=5)
                d.line([(mid[0] - 12, mid[1] + 12), (mid[0] + 12, mid[1] - 12)], fill=RED, width=5)
            else:
                d.line([a, tip], fill=(200, 150, 70, 255), width=6)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=18, outline=(70, 76, 90, 255), width=2)
    d.text((12, 8), "LEGS", font=font(16, True), fill=(170, 176, 189, 255))
    return img


def trace(dr, xy, wh, series, colors, ymax, label):
    x0, y0 = xy
    w, h = wh
    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 56, 70), width=1)
    for s, col in zip(series, colors):
        v = np.clip(np.asarray(s) / ymax, 0, 1)
        pts = [(x0 + w * k / max(len(v) - 1, 1), y0 + h - h * vv) for k, vv in enumerate(v)]
        if len(pts) > 1:
            dr.line(pts, fill=col, width=3)
    dr.text((x0 + 6, y0 + 4), label, font=font(15), fill=GREY)
