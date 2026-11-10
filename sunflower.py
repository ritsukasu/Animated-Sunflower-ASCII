"""
sunflower.py - a sunflower made of text, drawn live with numpy + pygame-ce.

No 3D program and no video file. Every frame is just a handful of formulas
turned into characters.

SPACE replays the bloom, S saves a screenshot, ESC or Q quits.
Want a single frame? python sunflower.py --still 12 shot.png
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame

# ----------------------------------------------------------------- settings
W, H = 1000, 760            # window size, in pixels
FONT_SIZE = 10
LOOP = 26.0                 # how long one loop lasts, in seconds
FLASH = 7.0                 # when the light rolls over the seed head
HEAD_Y = 0.37               # where the flower sits, as a fraction of window height

GOLDEN = np.pi * (3 - np.sqrt(5))     # the golden angle, about 137.5 degrees
R0 = 0.165                            # seed head size (1.0 = the full window height)
NS = 340                              # how many seeds in the head
SP = R0 * np.sqrt(np.pi / NS) * 1.05  # roughly how far apart neighbouring seeds are
N_PETALS = 21                         # a Fibonacci number, just like real sunflowers

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


def stem_x(v, ox):
    """Where the stem is horizontally at height v, bending with the sway."""
    return (ox * np.clip(1 - v / 0.62, 0, 1) ** 1.5
            + 0.02 * np.sin(v * 4.2 + 0.6) * np.clip(v / 0.3, 0, 1))


# -------------------------------------------------------------------- scene
class Scene:
    def __init__(self, cols, rows, cw, ch):
        self.cols, self.rows = cols, rows
        px = (np.arange(cols) + 0.5) * cw
        py = (np.arange(rows) + 0.5) * ch
        X, Y = np.meshgrid(px, py)
        self.cw, self.ch = cw, ch
        self.U = (X - W / 2) / H              # turn screen pixels into flower coordinates
        self.V = (Y - H * HEAD_Y) / H
        rng = np.random.default_rng(7)
        shape = (rows, cols)
        self.rnd = rng.integers(0, 1000, shape)           # picks which character from a pool
        self.shim = rng.random(shape) < 0.15              # the cells that flicker
        self.star = rng.random(shape) < 0.012
        self.star_phase = rng.random(shape) * 6.28
        self.star_speed = 0.8 + rng.random(shape) * 2.5
        self.star_ch = np.array([ord(c) for c in ".+*'.."])[rng.integers(0, 6, shape)]

    # ---------------------------------------------------------------- frame
    def render(self, t):
        U, V = self.U, self.V
        rows, cols = U.shape
        lum = np.zeros((rows, cols))
        col = np.zeros((rows, cols, 3))
        ov = np.zeros((rows, cols), dtype=np.int32)       # cells where I force a specific character

        def put(m, l, c):
            lum[m] = l[m]
            col[m] = c[m]
            ov[m] = 0

        def full(c):
            return np.broadcast_to(np.array(c, dtype=float), (rows, cols, 3))

        def mix(c0, c1, tt):
            return np.array(c0) * (1 - tt[..., None]) + np.array(c1) * tt[..., None]

        # timeline: when each part shows up
        fade = float(smooth(t / 1.0) * (1 - smooth((t - (LOOP - 1.5)) / 1.5)))
        stemp = smooth((t - 0.1) / 2.4)
        leafp = smooth((t - 1.0) / 3.0)
        seedp = smooth((t - 0.3) / 4.5)
        bloom = smooth((t - 1.2) / 5.5)
        ox = 0.012 * np.sin(t * 0.6)                      # a little sway
        UF = U - ox
        TH = np.arctan2(V, UF)
        R = np.hypot(UF, V)

        # stars and a faint warm glow
        tw = 0.5 + 0.5 * np.sin(t * self.star_speed + self.star_phase)
        m = self.star
        lum[m] = (0.12 + 0.30 * tw)[m]
        col[m] = (190, 200, 235)
        ov[m] = self.star_ch[m]
        halo = 0.14 * np.exp(-(R / 0.40) ** 2) * bloom
        put(halo > 0.06, halo, full((150, 85, 15)))

        # stem
        sxc = stem_x(V, ox)
        sw = 0.013 + 0.010 * np.clip(V / 0.62, 0, 1)
        du = (U - sxc) / sw
        m = (np.abs(du) < 1) & (V > 0.02) & (V < 0.7 * stemp)
        l = 0.25 + 0.5 * np.clip(1 - np.abs(du + 0.3), 0, 1) + 0.08 * np.sin(V * 140)
        put(m, l, full((50, 130, 48)))

        # leaves
        def leaf(bv, dx, dy, length, width, bend, grow):
            n = np.hypot(dx, dy)
            dx, dy = dx / n, dy / n
            bx = float(stem_x(bv, ox))
            a = (U - bx) * dx + (V - bv) * dy             # distance along the leaf
            b = -(U - bx) * dy + (V - bv) * dx            # distance across the leaf
            Lg = length * max(grow, 1e-3)
            s = np.clip(a / Lg, 0, 1)
            bc = bend * s * s * dx * length               # lets the leaf droop
            hw = width * grow * np.sin(np.pi * s) ** 0.75 + 1e-6
            rel = (b - bc) / hw
            m = (a > 0) & (a < Lg) & (np.abs(rel) < 1) & (grow > 0.03)
            mid = np.exp(-(rel / 0.14) ** 2)
            l = (0.30 + 0.32 * np.clip(1 - np.abs(rel + 0.2), 0, 1)
                 + 0.10 * np.sin(a * 120 - np.abs(rel) * 14) + 0.30 * mid)
            c = mix((36, 116, 52), (160, 205, 115), mid)
            put(m, l, c)

        leaf(0.34, 0.90, -0.42 + 0.05 * np.sin(t * 0.9), 0.24, 0.055, 0.5, leafp)
        leaf(0.47, -0.90, -0.35 + 0.05 * np.sin(t * 0.9 + 1), 0.21, 0.050, 0.55, leafp)

        # petals: two layers, the front one shifted by half a petal
        Rin = R0 * 0.9
        Lp = 0.17 * bloom

        def petals(off, scale, base, tip):
            k = np.round((TH - off) * N_PETALS / (2 * np.pi))
            rel = (TH - off) - k * 2 * np.pi / N_PETALS
            x = rel * N_PETALS / 2
            uu = np.abs(np.sin(x))                        # 0 is the middle of the petal
            flut = 1 + 0.05 * np.sin(t * 1.3 + (k % N_PETALS) * 2.4)
            edge = Rin + Lp * scale * flut * np.clip(1 - uu, 0, 1) ** 1.25
            m = (R < edge) & (R > Rin * 0.7)
            tr = np.clip((R - Rin) / (edge - Rin + 1e-6), 0, 1)
            l = (0.50 + 0.28 * (1 - uu) + 0.18 * tr - 0.10 * np.sin(x * 9) ** 2
                 - 0.08 * np.sin(x))
            l = l * (0.55 + 0.45 * smooth(tr * 3))
            put(m, l, mix(base, tip, tr))

        petals(0.0, 1.00, (200, 105, 12), (250, 170, 30))
        petals(np.pi / N_PETALS, 0.86, (240, 160, 20), (255, 225, 80))

        # seed head, laid out on the golden-angle spiral
        ys, xs = np.nonzero(R < R0 * 1.04)
        if ys.size:
            pu, pv = UF[ys, xs], V[ys, xs]
            nvis = max(1, int(NS * seedp))
            i = np.arange(nvis)
            rho = R0 * np.sqrt((i + 0.5) / NS)
            phi = i * GOLDEN + 0.05 * t
            dx = pu[:, None] - (rho * np.cos(phi))[None, :]
            dy = pv[:, None] - (rho * np.sin(phi))[None, :]
            d2 = dx * dx + dy * dy
            j = d2.argmin(1)
            ar = np.arange(len(j))
            d = np.sqrt(d2[ar, j])
            ddx, ddy = dx[ar, j], dy[ar, j]

            q = np.clip(d / (0.55 * SP), 0, 1)
            h = np.sqrt(1 - q * q)                         # each seed is a tiny dome
            lit = (ddx * -0.6 + ddy * -0.75) / (0.55 * SP)  # light comes from the upper left
            l = 0.22 + 0.50 * h + 0.22 * lit

            rf = np.hypot(pu, pv) / R0
            cr = np.interp(rf, [0, .5, 1], [70, 125, 178])
            cg = np.interp(rf, [0, .5, 1], [40, 72, 112])
            cb = np.interp(rf, [0, .5, 1], [16, 22, 30])
            ring = smooth((rf - 0.7) / 0.25) * 0.5         # the outer florets glow a bit
            cr = cr * (1 - ring) + 240 * ring
            cg = cg * (1 - ring) + 170 * ring
            cb = cb * (1 - ring) + 40 * ring
            jit = 0.85 + 0.30 * ((np.sin(j * 12.9898) * 43758.5453) % 1.0)

            front = (t - FLASH) / 2.0                      # the light sweeps outward
            g = np.exp(-((rf - front) / 0.14) ** 2) if 0 <= front <= 1.4 else 0 * rf
            l = l * jit + 0.55 * g
            cr, cg, cb = (cr * (1 - g) + 255 * g, cg * (1 - g) + 236 * g,
                          cb * (1 - g) + 170 * g)

            seeded = d < SP * 0.9
            l = np.where(seeded, l, 0.32)                  # bare green base where no seed has grown yet
            cr = np.where(seeded, cr, 60)
            cg = np.where(seeded, cg, 105)
            cb = np.where(seeded, cb, 36)
            lum[ys, xs] = l
            col[ys, xs] = np.stack([cr, cg, cb], axis=1)
            ov[ys, xs] = 0

        # pollen drifting up after the flash
        K = 64
        k = np.arange(K)
        ph = (t * (0.05 + 0.05 * frac(k * 0.618)) + frac(k * 0.754)) % 1
        a = k * GOLDEN
        r0 = R0 * (0.2 + 0.9 * frac(k * 0.37))
        pxu = ox + r0 * np.cos(a) * (1 + 1.4 * ph) + 0.04 * np.sin(ph * 8 + k)
        pyv = r0 * np.sin(a) * (1 + 1.4 * ph) - 0.30 * ph
        vis = frac(k * 0.1337) < smooth((t - FLASH) / 3.0)
        env = np.sin(np.pi * ph)
        pc = ((pxu * H + W / 2) / self.cw).astype(int)
        pr = ((pyv * H + H * HEAD_Y) / self.ch).astype(int)
        ok = vis & (pc >= 0) & (pc < cols) & (pr >= 0) & (pr < rows)
        for c_, r_, e_ in zip(pc[ok], pr[ok], env[ok]):
            lum[r_, c_] = 0.30 + 0.5 * e_
            col[r_, c_] = (255, 215, 90)
            ov[r_, c_] = ord("." if e_ < 0.35 else "+" if e_ < 0.7 else "*")

        return lum * fade, col, ov


# ------------------------------------------------------------------ drawing
def draw(screen, font, cache, scene, t):
    lum, col, ov = scene.render(t)
    lvl = np.clip((lum * NL).astype(int), 0, NL - 1)
    rnd = scene.rnd + scene.shim * int(t * 6)             # flicker
    codes = np.where(ov > 0, ov, POOL_CODES[lvl, rnd % 8])
    mask = (lvl > 0) | (ov > 0)

    w = np.clip((lum - 0.85) * 3, 0, 1)[..., None]        # the brightest spots go white
    c = col * (0.30 + 0.85 * np.clip(lum, 0, 1.2)[..., None])
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
