"""
sunflower.py - one sunflower made of text, drawn live with numpy + pygame-ce.

No 3D program and no video file. Every frame is just a handful of formulas
turned into characters.

SPACE replays the bloom, S saves a screenshot, ESC or Q quits.
Want a single frame? python sunflower.py --still 14 shot.png
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame

# ----------------------------------------------------------------- settings
W, H = 1000, 760            # window size, in pixels
FONT_SIZE = 8
LOOP = 26.0                 # how long one loop lasts, in seconds
FLASH = 7.0                 # when the light rolls over the seed head

GOLDEN = np.pi * (3 - np.sqrt(5))     # the golden angle, about 137.5 degrees
R0 = 0.165                            # seed head size (1.0 = the window height)
NS = 420                              # how many seeds in the head
N_PETALS = 13                         # petals per layer (a Fibonacci number)

# x, head height, size, where the stem leaves the screen, bloom delay (s), dimming
# (it's a list so you can add more flowers if you ever want to)
FLOWERS = [
    (0.50, 0.36, 0.95, 1.00, 0.2, 1.00),
]

# characters grouped by how heavy they look, faint to bold
POOLS = [" ", ".`'", ",:;", "-~=+", "*^<>", "il|!1", "tfjrx", "nuvczs",
         "oaeCL", "O0QG8", "#&%$", "@MW8B"]
NL = len(POOLS)
POOL_CODES = np.array([[ord(c) for c in (p * 8)[:8]] for p in POOLS])


# ------------------------------------------------------------------ helpers
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def frac(x):
    return x - np.floor(x)


def stem_x(v, ox, vmax):
    """Where the stem is horizontally at height v: it sways, and leans right as it drops."""
    k = np.clip(v / vmax, 0, 1)
    return (ox * (1 - k) ** 1.5
            + 0.02 * np.sin(v * 4.2 + 0.6) * np.clip(v / (0.5 * vmax), 0, 1)
            + 0.10 * k ** 2)


# -------------------------------------------------------------------- scene
class Scene:
    def __init__(self, cols, rows, cw, ch):
        self.cols, self.rows = cols, rows
        self.cw, self.ch = cw, ch
        px = (np.arange(cols) + 0.5) * cw
        py = (np.arange(rows) + 0.5) * ch
        self.X, self.Y = np.meshgrid(px, py)              # pixel centre of every cell
        rng = np.random.default_rng(7)
        shape = (rows, cols)
        self.rnd = rng.integers(0, 1000, shape)           # picks which character from a pool
        self.shim = rng.random(shape) < 0.15              # the cells that flicker
        self.star = rng.random(shape) < 0.012
        self.star_phase = rng.random(shape) * 6.28
        self.star_speed = 0.8 + rng.random(shape) * 2.5
        self.star_ch = np.array([ord(c) for c in ".+*'.."])[rng.integers(0, 6, shape)]
        self.order = sorted(range(len(FLOWERS)), key=lambda i: FLOWERS[i][2])

    # ---------------------------------------------------------------- frame
    def render(self, t):
        rows, cols = self.rows, self.cols
        lum = np.zeros((rows, cols))
        col = np.zeros((rows, cols, 3))
        ov = np.zeros((rows, cols), dtype=np.int32)       # cells where I force a specific character

        # twinkling stars
        tw = 0.5 + 0.5 * np.sin(t * self.star_speed + self.star_phase)
        m = self.star
        lum[m] = (0.12 + 0.30 * tw)[m]
        col[m] = (190, 200, 235)
        ov[m] = self.star_ch[m]

        for i in self.order:
            self.draw_flower(FLOWERS[i], i, t, lum, col, ov)

        # fade in at the start, fade out at the end so the loop is smooth
        fade = float(smooth(t / 1.0) * (1 - smooth((t - (LOOP - 1.5)) / 1.5)))
        if fade < 0.02:
            ov[:] = 0
        return lum * fade, col * fade, ov

    # --------------------------------------------------------------- flower
    def draw_flower(self, f, idx, t, lum, col, ov):
        cx, hy, size, base, delay, dim = f
        tt = t - delay                                    # this flower's own clock
        if tt <= 0:
            return
        unit = H * size                                   # one flower unit, in pixels
        x0 = max(0, int((cx * W - 0.46 * unit) / self.cw))
        x1 = min(self.cols, int((cx * W + 0.46 * unit) / self.cw) + 1)
        y0 = max(0, int((hy * H - 0.40 * unit) / self.ch))
        y1 = min(self.rows, int(base * H / self.ch) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        # only work on the patch of the screen this flower can reach
        U = (self.X[y0:y1, x0:x1] - cx * W) / unit
        V = (self.Y[y0:y1, x0:x1] - hy * H) / unit
        rows, cols = U.shape
        L = np.zeros((rows, cols))
        C = np.zeros((rows, cols, 3))
        occ = np.zeros((rows, cols), bool)
        OV = np.zeros((rows, cols), dtype=np.int32)       # characters this flower forces

        def put(m, l, c, chars=None):
            L[m] = l[m]
            C[m] = c[m]
            OV[m] = 0 if chars is None else chars[m]
            occ[m] = True

        def full(c):
            return np.broadcast_to(np.array(c, dtype=float), (rows, cols, 3))

        def mix(c0, c1, k):
            return np.array(c0) * (1 - k[..., None]) + np.array(c1) * k[..., None]

        # timeline: when each part shows up
        stemp = smooth((tt - 0.1) / 2.4)
        leafp = smooth((tt - 1.0) / 3.0)
        seedp = smooth((tt - 0.3) / 4.5)
        bloom = smooth((tt - 1.2) / 5.5)
        ph = idx * 1.7
        ox = 0.012 * np.sin(t * 0.6 + ph)                 # a little sway
        UF = U - ox
        TH = np.arctan2(V, UF)
        R = np.hypot(UF, V)
        vmax = (base - hy) / size                         # stem length in flower units

        # stem
        sw = 0.016 + 0.012 * np.clip(V / vmax, 0, 1)
        du = (U - stem_x(V, ox, vmax)) / sw
        m = (np.abs(du) < 1) & (V > 0.02) & (V < vmax * stemp)
        l = 0.25 + 0.5 * np.clip(1 - np.abs(du + 0.3), 0, 1) + 0.08 * np.sin(V * 140)
        put(m, l, full((50, 130, 48)))

        # leaves, with a midrib, side veins and a darker edge
        def leaf(bv, dx, dy, length, width, bend, grow):
            n = np.hypot(dx, dy)
            dx, dy = dx / n, dy / n
            bx = float(stem_x(bv, ox, vmax))
            a = (U - bx) * dx + (V - bv) * dy             # distance along the leaf
            b = -(U - bx) * dy + (V - bv) * dx            # distance across the leaf
            Lg = length * max(grow, 1e-3)
            s = np.clip(a / Lg, 0, 1)
            bc = bend * s * s * dx * length               # lets the leaf droop
            hw = width * grow * np.sin(np.pi * s) ** 0.75 + 1e-6
            rel = (b - bc) / hw
            m = (a > 0) & (a < Lg) & (np.abs(rel) < 1) & (grow > 0.03)
            mid = np.exp(-(rel / 0.12) ** 2)
            edge = smooth((np.abs(rel) - 0.78) / 0.2)
            l = (0.34 + 0.28 * np.clip(1 - np.abs(rel + 0.2), 0, 1)
                 + 0.12 * np.sin(a * 110 - np.abs(rel) * 14) + 0.30 * mid)
            l = l * (1 - 0.35 * edge)
            put(m, l, mix((36, 116, 52), (160, 205, 115), mid))

        b1 = float(np.clip(0.60 * vmax, 0.30, 0.95 * vmax))
        b2 = float(np.clip(0.80 * vmax, 0.36, 0.95 * vmax))
        leaf(b1, 0.92, -0.22 + 0.05 * np.sin(t * 0.9 + ph), 0.30, 0.070, 0.5, leafp)
        leaf(b2, -0.92, -0.12 + 0.05 * np.sin(t * 0.9 + ph + 1), 0.22, 0.055, 0.55, leafp)

        # petals: long and pointed, each one outlined, veined and shaded like a drawing
        npet = N_PETALS
        Rin = R0 * 0.9
        Lp = 0.18 * bloom

        def petals(off, length, c_base, c_mid, c_tip, shade):
            s0 = (R - Rin) / (Lp * length + 1e-4)
            th = TH - off - 0.10 * np.clip(s0, 0, 1)      # a slight curl to one side
            k = np.round(th * npet / (2 * np.pi))
            rel = th - k * 2 * np.pi / npet
            flut = 1 + 0.05 * np.sin(t * 1.3 + ph + (k % npet) * 2.4)
            s = (R - Rin) / (Lp * length * flut + 1e-4)   # 0 at the base, 1 at the tip
            sc = np.clip(s, 0, 1)
            half = np.pi / npet
            # petal width along its length: lens shaped, pointed at the tip
            w = np.minimum(0.86 * np.sin(np.pi * sc ** 0.62) ** 0.75 + 0.2 * (1 - sc), 0.88)
            c = np.abs(rel) / (half * w + 1e-6)           # 0 down the middle, 1 at the edge
            m = (s > -0.25) & (s < 1) & (c < 1)

            outline = smooth((c - 0.80) / 0.16)
            vein = np.exp(-(c / 0.07) ** 2)
            col_p = np.where((sc < 0.5)[..., None],
                             mix(c_base, c_mid, sc * 2),
                             mix(c_mid, c_tip, (sc - 0.5) * 2))
            col_p = col_p * (1 - 0.92 * outline[..., None]) + np.array((190, 85, 8)) * 0.92 * outline[..., None]   # deep orange pencil line

            l = 0.72 + 0.16 * sc
            l = l * (1 - 0.40 * outline) * (1 - 0.30 * vein)
            l = l + 0.03 * np.sin(c * 22 + s * 3) * (1 - outline)       # fine streaks
            l = l - 0.10 * (rel / (half * w + 1e-6))                    # light from the left
            l = l * (0.40 + 0.60 * smooth(sc * 2.0))                    # shadow near the head
            l = np.minimum(l * shade, 0.84)                             # no white petals
            # draw each petal with line characters that follow its direction, outlined with #
            phi = np.mod(TH, np.pi)
            lines = np.array([ord("="), ord("\\"), ord("|"), ord("/")])[
                np.rint(phi / (np.pi / 4)).astype(int) % 4]
            chars = np.where(outline > 0.6, ord("#"), lines)
            put(m, l, col_p, chars)

        petals(0.00, 1.12, (200, 95, 8), (240, 150, 14), (252, 190, 25), 0.90)
        petals(np.pi / npet, 0.88, (225, 125, 10), (252, 182, 15), (255, 215, 30), 1.00)
        petals(0.00, 0.62, (240, 150, 12), (255, 200, 20), (255, 225, 55), 1.10)

        # seed head, laid out on the golden-angle spiral
        ns = int(max(40, NS * (size / 0.9) ** 2))         # smaller flower, fewer seeds
        sp = R0 * np.sqrt(np.pi / ns) * 1.05              # roughly how far apart seeds are
        ys, xs = np.nonzero(R < R0 * 1.04)
        if ys.size:
            pu, pv = UF[ys, xs], V[ys, xs]
            nvis = max(1, int(ns * seedp))
            i = np.arange(nvis)
            rho = R0 * np.sqrt((i + 0.5) / ns)
            phi = i * GOLDEN + 0.05 * t + ph
            dx = pu[:, None] - (rho * np.cos(phi))[None, :]
            dy = pv[:, None] - (rho * np.sin(phi))[None, :]
            d2 = dx * dx + dy * dy
            j = d2.argmin(1)
            ar = np.arange(len(j))
            dd = np.sqrt(d2[ar, j])
            ddx, ddy = dx[ar, j], dy[ar, j]

            q = np.clip(dd / (0.55 * sp), 0, 1)
            h = np.sqrt(1 - q * q)                         # each seed is a tiny dome
            lit = (ddx * -0.6 + ddy * -0.75) / (0.55 * sp)  # light comes from the upper left
            l = 0.30 + 0.42 * h + 0.22 * lit
            l = l * (0.75 + 0.25 * smooth(np.hypot(pu, pv) / R0 / 0.8))   # darker toward the middle

            rf = np.hypot(pu, pv) / R0
            # dark brown in the middle, a golden ring of florets at the edge
            stops = [0, .45, .75, .9, 1]
            cr = np.interp(rf, stops, [95, 110, 160, 225, 170])
            cg = np.interp(rf, stops, [58, 68, 105, 160, 110])
            cb = np.interp(rf, stops, [20, 24, 28, 42, 30])
            jit = 0.85 + 0.30 * ((np.sin(j * 12.9898) * 43758.5453) % 1.0)
            l = l * (1 - 0.35 * smooth((rf - 0.94) / 0.08))   # a shadowy rim under the petals

            front = (tt - FLASH) / 2.0                     # the light sweeps outward
            g = np.exp(-((rf - front) / 0.14) ** 2) if 0 <= front <= 1.4 else 0 * rf
            l = np.minimum(l * jit, 0.78) + 0.55 * g       # no white seeds until the light hits
            cr, cg, cb = (cr * (1 - g) + 255 * g, cg * (1 - g) + 236 * g,
                          cb * (1 - g) + 170 * g)

            seeded = dd < sp * 0.9
            L[ys, xs] = np.where(seeded, l, 0.32)          # bare green base where no seed has grown yet
            C[ys, xs] = np.stack([np.where(seeded, cr, 60), np.where(seeded, cg, 105),
                                  np.where(seeded, cb, 36)], axis=1)
            OV[ys, xs] = 0
            occ[ys, xs] = True

        # copy this flower onto the big canvas
        sl = (slice(y0, y1), slice(x0, x1))
        lum[sl][occ] = L[occ] * dim
        col[sl][occ] = C[occ]
        ov[sl][occ] = OV[occ]

        # pollen drifting up after the flash
        K = 60
        k = np.arange(K)
        pt = (t * (0.05 + 0.05 * frac(k * 0.618)) + frac(k * 0.754) + idx * 0.13) % 1
        a = k * GOLDEN
        r0 = R0 * (0.2 + 0.9 * frac(k * 0.37))
        pxu = ox + r0 * np.cos(a) * (1 + 1.4 * pt) + 0.04 * np.sin(pt * 8 + k)
        pyv = r0 * np.sin(a) * (1 + 1.4 * pt) - 0.30 * pt
        vis = frac(k * 0.1337) < smooth((tt - FLASH) / 3.0)
        env = np.sin(np.pi * pt)
        pc = ((cx * W + pxu * unit) / self.cw).astype(int)
        pr = ((hy * H + pyv * unit) / self.ch).astype(int)
        ok = vis & (pc >= 0) & (pc < self.cols) & (pr >= 0) & (pr < self.rows)
        for c_, r_, e_ in zip(pc[ok], pr[ok], env[ok]):
            lum[r_, c_] = (0.30 + 0.5 * e_) * dim
            col[r_, c_] = (255, 215, 90)
            ov[r_, c_] = ord("." if e_ < 0.35 else "+" if e_ < 0.7 else "*")


# ------------------------------------------------------------------ drawing
def draw(screen, font, cache, scene, t):
    lum, col, ov = scene.render(t)
    lvl = np.clip((lum * NL).astype(int), 0, NL - 1)
    rnd = scene.rnd + scene.shim * int(t * 6)             # makes some characters flicker
    codes = np.where(ov > 0, ov, POOL_CODES[lvl, rnd % 8])
    mask = (lvl > 0) | (ov > 0)

    w = np.clip((lum - 0.85) * 3, 0, 1)[..., None]        # the brightest spots go white
    c = col * (0.40 + 0.90 * np.clip(lum, 0, 1.2)[..., None])
    c = c * (1 - w) + 255 * w
    q = np.rint(np.clip(c, 0, 255) / 255 * 7).astype(np.uint8)

    ys, xs = np.nonzero(mask)
    blits = []
    for ch, qc, y, x in zip(codes[ys, xs].tolist(), q[ys, xs].tolist(),
                            ys.tolist(), xs.tolist()):
        key = (ch, qc[0], qc[1], qc[2])
        s = cache.get(key)
        if s is None:
            s = font.render(chr(ch), True, tuple(v * 255 // 7 for v in qc))
            cache[key] = s
        blits.append((s, (x * scene.cw, y * scene.ch)))

    screen.fill((5, 5, 9))
    screen.blits(blits, doreturn=False)
    hud = font.render("SPACE replay   S save   ESC quit", True, (70, 70, 85))
    screen.blit(hud, (10, H - hud.get_height() - 6))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", nargs=2, metavar=("SECONDS", "FILE"),
                    help="render one frame at SECONDS, save it, and exit")
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_caption("ASCII sunflower")
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("consolas,dejavusansmono,menlo,couriernew,monospace",
                               FONT_SIZE)
    cw, ch = font.size("M")[0], font.get_linesize()
    scene = Scene(W // cw, H // ch, cw, ch)
    cache = {}

    if args.still:
        draw(screen, font, cache, scene, float(args.still[0]))
        pygame.image.save(screen, args.still[1])
        return

    clock = pygame.time.Clock()
    t0 = time.time()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if e.key == pygame.K_SPACE:
                    t0 = time.time()
                if e.key == pygame.K_s:
                    name = f"sunflower_{int(time.time())}.png"
                    pygame.image.save(screen, str(Path(name)))
                    print("saved", name)
        draw(screen, font, cache, scene, (time.time() - t0) % LOOP)
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
    pygame.quit()
    sys.exit()
