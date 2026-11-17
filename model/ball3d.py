"""Rule 17: the ball in 3D, from where both cameras saw it.

Each export's ball is where its camera's line of sight to the ball meets the grass. For a camera
at C (height h) and a ball at B (height z) the reported ground point is

    G = C + (B - C) h / (h - z)        so, at height z, the ball is at   B(z) = G + (C - G) z / h

-- a point that moves linearly with z along the line of sight. Both cameras' lines must meet at
the ball, which gives z in closed form (least squares); the cameras' centres come from
model/cameras.py (step 1). They are only ~10 m apart on the same side, so a metre of height moves
the two reported points apart by just 1-1.8 m: one frame gives z to about +-0.5-0.8 m. A flight
is fitted as a whole instead, with gravity fixed:

    x(t) = x0 + vx t,  y(t) = y0 + vy t,  z(t) = z0 + vz t - g t^2 / 2

to every camera point in it, through the projection above. Its curvature is what a single camera
sees too, so flights only the tactical camera covers are fitted on model/airborne.py's candidates
-- kept only where the arc fits clearly better than a ball rolling on the grass.

The filter then has two modes. In a flight the height is the fitted arc; elsewhere the ball is on
the grass. Each camera's point is moved to where its line of sight is at that height, and both
go through the two-camera smoother (kalman.ball_kinematics_two) for the ground position, the
velocities and the touches. (Not one joint 3D Kalman filter: a kick is a jump in vertical
velocity that such a filter lags behind; an arc is the exact physics of the flight.)
"""

import numpy as np

import cameras
import data
import kalman
import rules
from data import FPS

# Params
G = 9.81                    # m/s^2
GROUND_GAP = 1.0            # m: frames where the cameras' balls are this close calibrate the
                            # broadcast's systematic offset (a ball on the grass)
Z_SMOOTH = 5                # frames: rolling median of the per-frame height
AIR_SIGMA = 2.5             # a flight candidate: smoothed height this many standard deviations ...
AIR_MIN = 0.5               # ... and this many metres above the grass ...
MIN_FLIGHT = 6              # ... over a span of this many frames (0.24 s, once extended) with
MIN_AIR = 4                 # this many such frames; gaps up to BRIDGE frames bridged, and up to
BRIDGE = 6                  # BRIDGE_UNKNOWN where only one camera saw the ball (height unknown, not
BRIDGE_UNKNOWN = 2 * FPS    # zero: at 1:04 the peak was seen by one camera only)
GROW_MAX = FPS              # frames an arc may be grown either way, to its launch and landing
TURN_UP = 0.25              # m: a flight is extended no further than where the height climbs this
                            # much from one frame to the next (above the smoothed noise)
VALLEY_RISE = 0.4           # m: a flight is split where its height falls and then rises again by
VALLEY_FRAMES = 10          # this much within this many frames either side -- a touch in the air.
                            # The ball headed at 1H 12781 never reached the grass between the long
                            # ball and the header's arc, so extending "while in the air" ran one
                            # into the other and no single arc fitted.
EDGE_Z = 0.2                # m: a flight is extended either way while the smoothed height is above
FIT_ITER = 12
FIT_RMS = 0.6               # m: an arc whose points miss by more is not a flight ...
FIT_RMS_BASE, FIT_RMS_PER_S = 0.4, 0.3  # ... once trimmed: 0.4 m + 0.3 m per second of flight,
                            # as air drag slows a long ball the constant-speed arc does not model
                            # (1:02, a 6.7 m long ball: 0.85 m median residual on a right fit)
TRIM_Z = -0.2               # m: an arc is trimmed to the frames where it is above this
PEAK_MIN = 0.8              # m: nor one that never gets this high
Z_FLOOR = -0.5              # m: nor one that goes this far below the grass
SINGLE_CAMERA = False       # one-camera arcs: off. On the match they found 29 "flights" with a
                            # median peak of 19 m on 0.7 s segments -- too little curvature, so the
                            # fit puts the middle of a tall arc there. (Whole synthetic flights fit.)
SINGLE_GAIN = 0.7           # one camera: the arc's RMS must be under this share of a ground line's
SINGLE_PEAK = 1.0           # m: and peak at least this high
# Rule 34 (docs/VALUES.md): off = the arcs of commit 954604b. iterate.py --flights switches both on.
DRAG = False                # quadratic air drag in the arc model: dv/dt = -DRAG_C |v| v - g
DRAG_C = 0.0133             # 1/m: rho Cd A / (2 m) for a football (1.2 kg/m3, 0.25, 0.038 m2, 0.43 kg)
DRAG_STEP = 0.01            # s: the integration step
JOIN = False                # two flights FLIGHT_JOIN frames or fewer apart are one: the height is
FLIGHT_JOIN = 3             # interpolated across the hole
# Rule 42 (docs/VALUES.md): each camera's own trace fills its short gaps before fusion. iterate.py --traces
# (with kalman.FIX: a filled point's weight is carried as a confidence).
TRACES = False
FILL_MAX = 25               # frames: a camera's gap its own trace may fill
# Rule 47 (docs/VALUES.md): flights one camera alone shows. iterate.py --solo.
SOLO = False
SOLO_BASE = 5               # frames: a ground point's speed is read over 4-6 frames (0.5 m noise -> ~3.5 m/s)
SOLO_COVER = 0.5            # the picks must cover this share of the flight, launch to landing
SOLO_VZ = (5.0, 10.0, 15.0, 20.0)   # m/s: launch speeds the arc search starts from
SOLO_STATS = {}             # diagnostics: stretches ended at each test (rule 47)
# Rule 46 (docs/VALUES.md): arcs grown per end, and the height may not change faster than DZ_MAX a frame.
# iterate.py --heights.
CONTINUITY = False
DZ_MAX = 1.0                # m per frame: 25 m/s vertically, about the hardest lofted kick (35 m/s at 45 degrees)
# Rule 40 (docs/VALUES.md): a ball held up keeps its triangulated height. iterate.py --tri.
TRI_HOLD = False
TRI_MIN_RUN, TRI_MAD, TRI_MISS, TRI_GAP, TRI_ZMAX = 4, 0.4, 1.0, 2.0, 6.0   # frames with both cameras, m, m, m, m
TRI_EXTEND = 25             # frames: carried on through frames only one camera has
COVER = False               # rule 35: a single arc over a merged span must cover its air frames, else the pieces
REACH = 2.3                 # m: no player touches a ball above this (a header)
CONTROL_REACH = 1.2         # m: nor controls one above this


def camera_geometry(tac, bro, alignment):
    """(tactical centre, broadcast centre in tactical coordinates, (n_t,) frames where the
    broadcast is its main camera), all from model/cameras.py."""
    cams_t = cameras.recover(data.homography_matrices("tactical"))
    graded = np.zeros(tac.n, dtype=bool)
    for _, first, last in data.halves():
        graded[first:last + 1] = True
    c_t = np.median(cams_t["centre"][cams_t["ok"] & graded], axis=0)
    H_b = data.homography_matrices("broadcast")
    cams_b = cameras.recover(H_b)
    main_b, c_b, _ = cameras.main_camera(H_b, cams_b, data.homography_errors("broadcast"),
                                         cameras.broadcast_live(alignment, bro.n))
    c_b = c_b.copy()
    flip = tuple(bool(v) for v in alignment["flip"])
    if flip[0]:
        c_b[0] = data.PITCH[0] - c_b[0]
    if flip[1]:
        c_b[1] = data.PITCH[1] - c_b[1]
    b = alignment["b_of_t"]
    main_t = np.zeros(tac.n, dtype=bool)
    mapped = b >= 0
    main_t[mapped] = main_b[b[mapped]]
    return c_t, c_b, main_t


def toward(ground, centre):
    """(n, 2) how a camera's point moves per metre of height: (C - G) / h."""
    return (centre[:2] - ground) / centre[2]


def project(B, centre):
    """Ground points (n, 2) a camera at `centre` reports for balls B (n, 3)."""
    with np.errstate(all="ignore"):
        scale = centre[2] / (centre[2] - B[:, 2])
    return centre[:2] + (B[:, :2] - centre[:2]) * scale[:, None]


def triangulate(g_t, g_b, c_t, c_b, s_t, s_b):
    """Per frame: (B (n, 3), z std (n,)) where both lines of sight meet; NaN where a camera is
    missing. s_t, s_b: each camera's ground-point noise (m)."""
    d_t, d_b = toward(g_t, c_t), toward(g_b, c_b)
    diff = d_t - d_b
    den = (diff ** 2).sum(axis=1)
    with np.errstate(all="ignore"):
        z = (diff * (g_b - g_t)).sum(axis=1) / den
        z_std = np.sqrt(s_t ** 2 + s_b ** 2) / np.sqrt(den)
    w_t, w_b = 1 / s_t ** 2, 1 / s_b ** 2
    xy = (w_t * (g_t + d_t * z[:, None]) + w_b * (g_b + d_b * z[:, None])) / (w_t + w_b)
    return np.concatenate([xy, z[:, None]], axis=1), z_std


def ray_miss(g_t, g_b, c_t, c_b):
    """(n,) m: how far apart the two ground points stay once the best height is taken out -- the
    part of their gap across the lines of sight, which no height explains (two different objects,
    or one camera's point wrong). NaN where a camera is missing."""
    diff = toward(g_t, c_t) - toward(g_b, c_b)
    gap = g_b - g_t
    with np.errstate(all="ignore"):
        z = (diff * gap).sum(axis=1) / (diff ** 2).sum(axis=1)
    return np.linalg.norm(gap - diff * z[:, None], axis=1)


def _model(p, t, cams):
    """Ground points the arc p = (x0, y0, vx, vy, z0, vz) gives at times t, for each camera."""
    B = arc_path(p, t)
    return np.concatenate([project(B, c) for c in cams]), B


def arc_path(p, t):
    """(len(t), 3) the arc p = (x0, y0, vx, vy, z0, vz) at times t (from 0, a frame apart): gravity alone,
    or with quadratic drag (DRAG), integrated in DRAG_STEP steps."""
    if not DRAG:
        return np.stack([p[0] + p[2] * t, p[1] + p[3] * t, p[4] + p[5] * t - G * t * t / 2], axis=1)
    pos = np.array([p[0], p[1], p[4]], dtype=float)
    vel = np.array([p[2], p[3], p[5]], dtype=float)
    at = {int(round(ti / DRAG_STEP)): i for i, ti in enumerate(t)}
    out = np.empty((len(t), 3))
    for i in range(int(round(t[-1] / DRAG_STEP)) + 1):
        if i in at:
            out[at[i]] = pos
        acc = -DRAG_C * np.linalg.norm(vel) * vel
        acc[2] -= G
        vel = vel + acc * DRAG_STEP
        pos = pos + vel * DRAG_STEP
    return out


def arc_z(p, t):
    """(len(t),) the arc's height at times t."""
    return arc_path(p, t)[:, 2]


def fit_arc(t, obs, cams, p0):
    """Gauss-Newton (damped) for the arc through observed ground points. obs: per camera, (len(t),
    2) with NaN where that camera has none. Returns (p, rms) or (None, inf)."""
    seen = [~np.isnan(o[:, 0]) for o in obs]
    target = np.concatenate([o for o in obs])
    use = np.concatenate(seen)
    if use.sum() < 6:
        return None, np.inf
    p, lam = np.asarray(p0, dtype=float), 1e-3

    def residual(q):
        pred, _ = _model(q, t, cams)
        return (pred - target)[use].ravel()

    r = residual(p)
    for _ in range(FIT_ITER):
        J = np.empty((len(r), 6))
        for k in range(6):
            dq = np.zeros(6)
            dq[k] = 1e-4
            J[:, k] = (residual(p + dq) - r) / 1e-4
        if not np.isfinite(J).all() or not np.isfinite(r).all():
            return None, np.inf
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -J.T @ r)
        r_new = residual(p + step)
        if np.isfinite(r_new).all() and r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam / 3
        else:
            lam *= 10
    return p, float(np.sqrt(np.mean(r ** 2)))


def _line_rms(t, g):
    """RMS of the best constant-velocity line on the grass through one camera's points."""
    ok = ~np.isnan(g[:, 0])
    if ok.sum() < 4:
        return np.inf
    A = np.stack([np.ones(ok.sum()), t[ok]], axis=1)
    res = []
    for k in (0, 1):
        coef, *_ = np.linalg.lstsq(A, g[ok, k], rcond=None)
        res.append(g[ok, k] - A @ coef)
    return float(np.sqrt(np.mean(np.concatenate(res) ** 2)))


def _line_speed(t, g):
    """m/s of the best constant-velocity line on the grass through one camera's points (inf: too few)."""
    ok = ~np.isnan(g[:, 0])
    if ok.sum() < 4:
        return np.inf
    A = np.stack([np.ones(ok.sum()), t[ok]], axis=1)
    v = [np.linalg.lstsq(A, g[ok, k], rcond=None)[0][1] for k in (0, 1)]
    return float(np.hypot(*v))


def _start(t, g, d, z):
    """Starting arc: horizontal line through the height-corrected points, vertical from z."""
    pts = g + d * np.nan_to_num(z)[:, None]
    ok = ~np.isnan(pts[:, 0])
    A = np.stack([np.ones(ok.sum()), t[ok]], axis=1)
    (x0, vx), *_ = np.linalg.lstsq(A, pts[ok, 0], rcond=None)
    (y0, vy), *_ = np.linalg.lstsq(A, pts[ok, 1], rcond=None)
    okz = ~np.isnan(z)
    if okz.sum() >= 3:
        (z0, vz), *_ = np.linalg.lstsq(np.stack([np.ones(okz.sum()), t[okz]], axis=1),
                                        z[okz] + G * t[okz] ** 2 / 2, rcond=None)
    else:
        z0, vz = 0.0, G * t[-1] / 2      # launched and landing over the span
    return [x0, y0, vx, vy, z0, vz]


def flights(g_t, g_b, c_t, c_b, z_tri, z_std, candidates):
    """[(start, stop, params, rms, kind)] of fitted flights; kind "two cameras" or "tactical".
    candidates: [(start, stop)] of single-camera candidates (model/airborne.py)."""
    n = len(g_t)
    both = ~np.isnan(z_tri)
    z_s = np.full(n, np.nan)
    for s, e in rules.runs(both):
        v = z_tri[s:e]
        for i in range(e - s):
            lo, hi = max(0, i - Z_SMOOTH // 2), min(e - s, i + Z_SMOOTH // 2 + 1)
            z_s[s + i] = np.median(v[lo:hi])
    with np.errstate(invalid="ignore"):
        air = (z_s > AIR_MIN) & (z_s > AIR_SIGMA * z_std)
    # Bridge short gaps, and longer ones where the height is unknown (one camera only), then
    # extend while still above the grass.
    spans = []                  # (start, stop, [parts merged into it])
    for s, e in rules.runs(air):
        if spans and (s - spans[-1][1] <= BRIDGE or
                      (s - spans[-1][1] <= BRIDGE_UNKNOWN and np.isnan(z_s[spans[-1][1]:s]).all())):
            spans[-1] = (spans[-1][0], e, spans[-1][2] + [(s, e)])
        else:
            spans.append((s, e, [(s, e)]))

    def extended(s, e):
        # Out to where the ball meets the grass -- but no further than where the height turns up
        # again: past a landing comes a bounce, before a launch the previous flight (1H 1:05: the
        # headed ball's span ran on into its bounce and no arc fitted both).
        while s > 0 and z_s[s - 1] > EDGE_Z and not z_s[s - 1] > z_s[s] + TURN_UP:
            s -= 1
        while e < n and z_s[e] > EDGE_Z and not z_s[e] > z_s[e - 1] + TURN_UP:
            e += 1
        return s, e

    def valleys(s, e):
        """Frames in [s, e) where the height dips and climbs again: a touch in the air."""
        out_v = []
        for i in range(s + 1, e - 1):
            if not np.isfinite(z_s[i]):
                continue
            left = z_s[max(s, i - VALLEY_FRAMES):i]
            right = z_s[i + 1:min(e, i + 1 + VALLEY_FRAMES)]
            if not (np.isfinite(left).any() and np.isfinite(right).any()):
                continue
            if np.nanmax(left) - z_s[i] >= VALLEY_RISE and np.nanmax(right) - z_s[i] >= VALLEY_RISE and \
                    z_s[i] <= np.nanmin(z_s[max(s, i - 3):i + 4]):
                if not out_v or i - out_v[-1] > VALLEY_FRAMES:
                    out_v.append(i)
        return out_v

    def regrow(s, p, s2, e2):
        """Rule 46: the arc p (starting at frame s) refitted over [s2, e2), or None if the refit misses by more than the
        tolerance or dips below Z_FLOOR: (s2, e2, t, params, rms)."""
        t2 = (np.arange(s2, e2) - s2) / FPS
        back = (s - s2) / FPS
        q0 = list(p)
        q0[0], q0[1] = p[0] - p[2] * back, p[1] - p[3] * back
        q0[4] = p[4] - p[5] * back - G * back ** 2 / 2
        q0[5] = p[5] + G * back
        q, rms2 = fit_arc(t2, [g_t[s2:e2], g_b[s2:e2]], [c_t, c_b], q0)
        if q is None or rms2 > FIT_RMS_BASE + FIT_RMS_PER_S * (e2 - s2) / FPS:
            return None
        z = arc_z(q, t2)
        if z.min() >= Z_FLOOR:
            return s2, e2, t2, q, rms2
        # Grown past the launch or the landing (1H 18002: to 18023, -0.5 m at the end): cut where the arc meets the grass,
        # as fit_span trims, and refit -- if the cut span still holds the arc it grew from.
        # Only the end being grown is cut: the other is the arc's own.
        top = int(np.argmax(z))
        a, b = 0, len(z)
        if s2 < s:
            a = top
            while a > 0 and z[a - 1] >= TRIM_Z:
                a -= 1
        else:
            b = top + 1
            while b < len(z) and z[b] >= TRIM_Z:
                b += 1
        if s2 + a > s or b - a < MIN_FLIGHT:
            return None
        q1 = list(q)
        q1[0], q1[1] = q[0] + q[2] * a / FPS, q[1] + q[3] * a / FPS
        q1[4], q1[5] = z[a], q[5] - G * a / FPS
        s3, e3 = s2 + a, s2 + b
        t3 = (np.arange(s3, e3) - s3) / FPS
        q, rms3 = fit_arc(t3, [g_t[s3:e3], g_b[s3:e3]], [c_t, c_b], q1)
        if q is not None and rms3 <= FIT_RMS_BASE + FIT_RMS_PER_S * (e3 - s3) / FPS and arc_z(q, t3).min() >= Z_FLOOR:
            return s3, e3, t3, q, rms3
        return None

    def fit_span(s, e, extend=True, limits=(0, None)):
        """(start, stop, params, rms) of the arc for one span, or None. limits: the frames it may
        grow to -- a piece cut at a touch in the air must not grow past the touch (1:04: the long
        ball's arc, grown to where it would have landed, took the headed ball's frames)."""
        lo_lim, hi_lim = limits[0], n if limits[1] is None else limits[1]
        if air[s:e].sum() < MIN_AIR:
            return None
        if extend:
            s, e = extended(s, e)
        if e - s < MIN_FLIGHT:
            return None
        t = (np.arange(s, e) - s) / FPS
        p0 = _start(t, g_t[s:e], toward(g_t[s:e], c_t), z_tri[s:e])
        p, rms = fit_arc(t, [g_t[s:e], g_b[s:e]], [c_t, c_b], p0)
        # Trim to where the arc is above the grass and refit: the span's ends, set by a noisy
        # height, can run past the landing, and an arc forced through them dives below it.
        for _ in range(2):
            if p is None:
                break
            z = arc_z(p, t)
            top = int(np.argmax(z))
            a, b = top, top + 1
            while a > 0 and z[a - 1] >= TRIM_Z:
                a -= 1
            while b < len(z) and z[b] >= TRIM_Z:
                b += 1
            if (a, b) == (0, len(z)) or b - a < MIN_FLIGHT:
                break
            q0 = list(p)
            q0[0], q0[1] = p[0] + p[2] * a / FPS, p[1] + p[3] * a / FPS
            q0[4], q0[5] = z[a], p[5] - G * a / FPS
            s, e = s + a, s + b
            t = (np.arange(s, e) - s) / FPS
            p, rms = fit_arc(t, [g_t[s:e], g_b[s:e]], [c_t, c_b], q0)
        if p is None or rms > FIT_RMS_BASE + FIT_RMS_PER_S * (e - s) / FPS:
            return None
        # Grow the arc to where it meets the grass -- the kick and the landing belong to the
        # flight even where one camera alone saw them -- and refit on the grown span.
        disc = p[5] ** 2 + 2 * G * p[4]
        if disc > 0 and CONTINUITY:
            # Rule 46: each end grown on its own -- a landing end that overreaches must not cost a good launch end.
            up, down = (p[5] - np.sqrt(disc)) / G, (p[5] + np.sqrt(disc)) / G
            s2 = max(lo_lim, s - GROW_MAX, s + int(np.floor(up * FPS)))
            e2 = min(hi_lim, e + GROW_MAX, s + int(np.ceil(down * FPS)) + 1)
            for a, b in ((s2, e), (None, e2)):
                a = s if a is None else a
                if (a, b) == (s, e) or a > s or b < e:
                    continue
                grown = regrow(s, p, a, b)
                if grown is not None:
                    s, e, t, p, rms = grown
        elif disc > 0:
            up, down = (p[5] - np.sqrt(disc)) / G, (p[5] + np.sqrt(disc)) / G
            s2 = max(lo_lim, s - GROW_MAX, s + int(np.floor(up * FPS)))
            e2 = min(hi_lim, e + GROW_MAX, s + int(np.ceil(down * FPS)) + 1)
            if (s2, e2) != (s, e):
                t2 = (np.arange(s2, e2) - s2) / FPS
                q0 = list(p)
                q0[0], q0[1] = p[0] - p[2] * (s - s2) / FPS, p[1] - p[3] * (s - s2) / FPS
                q0[4] = p[4] - p[5] * (s - s2) / FPS - G * ((s - s2) / FPS) ** 2 / 2
                q0[5] = p[5] + G * (s - s2) / FPS
                q, rms2 = fit_arc(t2, [g_t[s2:e2], g_b[s2:e2]], [c_t, c_b], q0)
                # Only if the grown arc stays above the grass: a ball headed or chested starts in
                # the air, and growing it to "launch" reaches back into the flight before it
                # (1:04: the refit dipped 1.9 m below the grass and the whole flight was lost).
                if q is not None and rms2 <= FIT_RMS_BASE + FIT_RMS_PER_S * (e2 - s2) / FPS and \
                        arc_z(q, t2).min() >= Z_FLOOR:
                    s, e, t, p, rms = s2, e2, t2, q, rms2
        z = arc_z(p, t)
        if z.max() < PEAK_MIN or z.min() < Z_FLOOR:
            return None
        return s, e, p, rms

    def solo_flights(taken):
        """Rule 47: flights one camera alone shows. Air frames (a ground point faster than MAX_PACE over ~SOLO_BASE
        frames) outside the flights found so far, joined within BRIDGE; an arc fitted to every pick there through its
        line of sight, trimmed and grown per end; kept if it fits, beats a rolling ball, peaks and stays above the
        grass, and its picks cover SOLO_COVER of the flight. Marks taken; returns the flights."""
        found = []
        have = np.isfinite(g_t[:, 0]) | np.isfinite(g_b[:, 0])
        SOLO_STATS.clear()

        def drop(why):
            SOLO_STATS[why] = SOLO_STATS.get(why, 0) + 1
        for name, g, c in (("tactical", g_t, c_t), ("broadcast", g_b, c_b)):
            ok = np.isfinite(g[:, 0])
            fast = np.zeros(n, dtype=bool)
            for k in (SOLO_BASE - 1, SOLO_BASE, SOLO_BASE + 1):
                with np.errstate(invalid="ignore"):
                    sp = np.linalg.norm(g[k:] - g[:-k], axis=1) * FPS / k
                fast[k:] |= ok[k:] & ok[:-k] & (sp > kalman.MAX_PACE)
            fast &= ~taken
            stretches = []
            for s, e in rules.runs(fast):
                if stretches and s - stretches[-1][1] <= BRIDGE:
                    stretches[-1][1] = e
                else:
                    stretches.append([s, e])
            for s, e in stretches:
                if fast[s:e].sum() < MIN_AIR or taken[s:e].any() or e - s < MIN_FLIGHT:
                    drop("too few air frames")
                    continue
                drop("stretches")
                t = (np.arange(s, e) - s) / FPS
                obs, cams = [g_t[s:e], g_b[s:e]], [c_t, c_b]
                guess = _start(t, g[s:e], toward(g[s:e], c), np.full(e - s, np.nan))
                p, rms = None, np.inf
                for vz in SOLO_VZ:          # launch speeds to start the search from (not a decision)
                    q, r = _fit_nodrag(t, obs, cams, guess[:4] + [0.3, vz])
                    if q is not None and r < rms:
                        p, rms = q, r
                if p is None:
                    drop("no fit")
                    continue
                p, rms = fit_arc(t, obs, cams, p)
                for _ in range(2):          # trimmed to above the grass, as fit_span does
                    if p is None:
                        break
                    z = arc_z(p, t)
                    top = int(np.argmax(z))
                    a, b = top, top + 1
                    while a > 0 and z[a - 1] >= TRIM_Z:
                        a -= 1
                    while b < len(z) and z[b] >= TRIM_Z:
                        b += 1
                    if (a, b) == (0, len(z)) or b - a < MIN_FLIGHT:
                        break
                    q0 = list(p)
                    q0[0], q0[1] = p[0] + p[2] * a / FPS, p[1] + p[3] * a / FPS
                    q0[4], q0[5] = z[a], p[5] - G * a / FPS
                    s, e = s + a, s + b
                    t = (np.arange(s, e) - s) / FPS
                    p, rms = fit_arc(t, [g_t[s:e], g_b[s:e]], cams, q0)
                if p is None or rms > FIT_RMS_BASE + FIT_RMS_PER_S * (e - s) / FPS:
                    drop("fit tolerance")
                    continue
                disc = p[5] ** 2 + 2 * G * p[4]
                if disc > 0:
                    lo_lim = int(np.flatnonzero(taken[:s])[-1]) + 1 if taken[:s].any() else 0
                    hi_lim = s + int(np.flatnonzero(taken[s:])[0]) if taken[s:].any() else n
                    up, down = (p[5] - np.sqrt(disc)) / G, (p[5] + np.sqrt(disc)) / G
                    s2 = max(lo_lim, s - GROW_MAX, s + int(np.floor(up * FPS)))
                    e2 = min(hi_lim, e + GROW_MAX, s + int(np.ceil(down * FPS)) + 1)
                    for a, b in ((s2, e), (None, e2)):
                        a = s if a is None else a
                        if (a, b) == (s, e) or a > s or b < e:
                            continue
                        grown = regrow(s, p, a, b)
                        if grown is not None:
                            s, e, t, p, rms = grown
                z = arc_z(p, t)
                if z.max() < PEAK_MIN or z.min() < Z_FLOOR or taken[s:e].any():
                    drop("peak or floor")
                    continue
                if rms > SINGLE_GAIN * _line_rms(t, g[s:e]) and _line_speed(t, g[s:e]) <= kalman.MAX_PACE:
                    drop("no better than a rolling ball")
                    continue                # a ball rolling on a line explains the picks nearly as well
                disc = p[5] ** 2 + 2 * G * p[4]
                if disc <= 0:
                    drop("peak or floor")
                    continue
                launch = s + int(np.floor((p[5] - np.sqrt(disc)) / G * FPS))
                landing = s + int(np.ceil((p[5] + np.sqrt(disc)) / G * FPS))
                whole = have[max(0, launch):min(n, landing + 1)]
                if landing <= launch or whole.sum() < SOLO_COVER * (landing - launch + 1):
                    drop("coverage")
                    continue                # a piece of a taller arc: too little of the flight to fix its height
                drop("kept")
                found.append((s, e, p, rms, name))
                taken[s:e] = True
        return found

    out = []
    taken = np.zeros(n, dtype=bool)
    for s, e, parts in spans:
        # A touch in the air splits a flight (valleys); and a span merged across frames of
        # unknown height can hold two flights no one arc fits: then its parts, one by one.
        xs, xe = extended(s, e)
        cuts = valleys(xs, xe)
        if cuts:
            bounds = [xs] + cuts + [xe]
            results = [fit_span(a, b, extend=False, limits=(a, b)) for a, b in zip(bounds, bounds[1:])]
        else:
            found = fit_span(s, e)
            if COVER and found and len(parts) > 1:
                # An arc that reaches the grass before the span's air ends leaves the rest uncorrected
                # (1H 15715 with the wider tolerance: 34 of 56 frames): fit the pieces instead.
                frames = np.arange(s, e)
                if int(air[s:e][~((frames >= found[0]) & (frames < found[1]))].sum()) >= MIN_AIR:
                    found = None
            results = [found] if found else ([fit_span(a, b) for a, b in parts] if len(parts) > 1 else [])
        for r in results:
            if r and not taken[r[0]:r[1]].any():
                out.append((r[0], r[1], r[2], r[3], "two cameras"))
                taken[r[0]:r[1]] = True
    for s, e in (candidates if SINGLE_CAMERA else []):
        if taken[s:e].any() or np.mean(~np.isnan(g_b[s:e, 0])) > 0.5:
            continue  # the two-camera fit has this one, or would have
        t = (np.arange(s, e) - s) / FPS
        p0 = _start(t, g_t[s:e], toward(g_t[s:e], c_t), np.full(e - s, np.nan))
        p, rms = fit_arc(t, [g_t[s:e]], [c_t], p0)
        if p is None:
            continue
        z = p[4] + p[5] * t - G * t * t / 2
        if rms > FIT_RMS or rms > SINGLE_GAIN * _line_rms(t, g_t[s:e]) or \
                z.max() < SINGLE_PEAK or z.min() < Z_FLOOR:
            continue
        out.append((s, e, p, rms, "tactical"))
        taken[s:e] = True
    if SOLO:
        out += solo_flights(taken)
    return sorted(out, key=lambda f: f[0])


def _fit_nodrag(t, obs, cams, p0):
    """fit_arc without drag: the fast search for a start (rule 47); the drag refit follows."""
    global DRAG
    keep, DRAG = DRAG, False
    try:
        return fit_arc(t, obs, cams, p0)
    finally:
        DRAG = keep


def camera_traces(t_c, b_c, conf):
    """Rule 42. Each camera's height-corrected points through their own smoother (kalman.smooth_track, the
    ball's acceleration, gaps up to FILL_MAX bridged). Returns, per camera (tactical, broadcast): its points
    with the frames filled where the other camera has a pick and it does not, their confidences (a filled
    point as a pick whose variance adds the trace's), the filled mask, and the whole trace."""
    out = []
    base2 = kalman.MEAS_STD[1] ** 2
    for own, other, c in ((t_c, b_c, conf[0]), (b_c, t_c, conf[1])):
        c = np.asarray(c, dtype=float)
        seen = np.isfinite(own[:, 0])
        good = np.isfinite(c) & (c > 0)
        r2 = base2 * np.clip(kalman.CONF_REF / np.where(good, c, kalman.CONF_REF), 1.0, kalman.VAR_MAX)
        trace, _, sd, bridged = kalman.smooth_track(own.astype(float), r2, kalman.ACCEL_STD, FILL_MAX)
        fill = bridged & ~seen & np.isfinite(other[:, 0]) & np.isfinite(trace[:, 0])
        filled = own.copy()
        filled[fill] = trace[fill]
        cf = c.copy()
        cf[fill] = np.maximum(kalman.CONF_REF / (1.0 + sd[fill] ** 2 / base2), kalman.CONF_REF / kalman.VAR_MAX)
        out.append((filled, cf, fill, trace))
    return out


def solve(g_t, g_b, c_t, c_b, noise, candidates=(), bridge=False, conf=None, image=None):
    """The 3D ball from two cameras' ground points (tactical pitch and timeline, NaN where none):
    kalman.ball_kinematics_two's keys plus z, z_std (m), "flight" (n,) bool, "flights" [(start,
    stop, peak m, rms m, kind)], "z_tri" (per-frame triangulation) and "bias" (broadcast minus
    tactical, calibrated where the cameras agree). noise: each camera's ground-point error, m."""
    n = len(g_t)
    gap = np.linalg.norm(g_b - g_t, axis=1)
    with np.errstate(invalid="ignore"):
        ground = gap <= GROUND_GAP
    bias = np.median((g_b - g_t)[ground], axis=0) if ground.any() else np.zeros(2)
    g_b = g_b - bias
    B_tri, z_std = triangulate(g_t, g_b, c_t, c_b, noise, noise)
    found = flights(g_t, g_b, c_t, c_b, B_tri[:, 2], z_std, candidates)

    z = np.zeros(n)
    zs = np.full(n, 0.3)                       # on the grass: roughly a ball's own size
    flight = np.zeros(n, dtype=bool)
    summary = []
    for s, e, p, rms, kind in found:
        t = (np.arange(s, e) - s) / FPS
        z[s:e] = np.maximum(0.0, arc_z(p, t))
        zs[s:e] = rms
        flight[s:e] = True
        summary.append((s, e, float(z[s:e].max()), rms, kind))
    if JOIN:
        for (s0, e0, *_), (s1, e1, *_) in zip(summary[:-1], summary[1:]):
            if 0 < s1 - e0 <= FLIGHT_JOIN:
                u = np.arange(1, s1 - e0 + 1) / (s1 - e0 + 1)
                z[e0:s1] = z[e0 - 1] + (z[s1] - z[e0 - 1]) * u
                zs[e0:s1] = max(zs[e0 - 1], zs[s1])
                flight[e0:s1] = True
    if TRI_HOLD:
        zt = B_tri[:, 2]
        gap = np.linalg.norm(g_b - g_t, axis=1)
        miss = ray_miss(g_t, g_b, c_t, c_b)
        with np.errstate(invalid="ignore"):
            ok = ~flight & (gap > TRI_GAP) & (miss < TRI_MISS) & (zt > AIR_MIN) & (zt < TRI_ZMAX)
        runs_ = []
        for s0, e0 in rules.runs(ok):
            if runs_ and (s0 - runs_[-1][1] <= BRIDGE or
                          (s0 - runs_[-1][1] <= BRIDGE_UNKNOWN and np.isnan(zt[runs_[-1][1]:s0]).all())):
                runs_[-1][1] = e0
            else:
                runs_.append([s0, e0])
        for s0, e0 in runs_:
            vals = zt[s0:e0]
            have = np.isfinite(vals) & ok[s0:e0]
            if have.sum() < TRI_MIN_RUN:
                continue
            med = np.median(vals[have])
            if np.median(np.abs(vals[have] - med)) > TRI_MAD:
                continue
            a0, a1 = s0, e0
            while a0 > 0 and s0 - a0 < TRI_EXTEND and np.isnan(zt[a0 - 1]) and not flight[a0 - 1]:
                a0 -= 1
            while a1 < n and a1 - e0 < TRI_EXTEND and np.isnan(zt[a1]) and not flight[a1]:
                a1 += 1
            s0, e0 = a0, a1
            vals = zt[s0:e0]
            have = np.isfinite(vals) & ok[s0:e0]
            idx = np.arange(s0, e0)
            filled = np.interp(idx, idx[have], vals[have])
            k = Z_SMOOTH // 2
            smooth_z = np.array([np.median(filled[max(0, i - k):i + k + 1]) for i in range(len(filled))])
            z[s0:e0] = np.maximum(0.0, smooth_z)
            zs[s0:e0] = max(1.5 * np.median(np.abs(vals[have] - med)), 0.3)
            flight[s0:e0] = True
    if CONTINUITY:
        # Rule 46: no ball changes height faster than DZ_MAX a frame; beside a higher frame the height is raised to
        # that slope, so a flight starting or ending in the air ramps the correction instead of switching it.
        # The slope is also held to what does not move the corrected point faster than a ball (V_GAP): a metre of
        # height moves a ground point |C - G| / h, 3-4 m at these distances.
        with np.errstate(invalid="ignore"):
            lever = np.fmax(np.linalg.norm(toward(g_t, c_t), axis=1), np.linalg.norm(toward(g_b, c_b), axis=1))
        slope = np.where(np.isfinite(lever) & (lever > 0), np.minimum(DZ_MAX, kalman.MAX_PACE / FPS / lever), DZ_MAX)
        for i in range(1, n):
            if z[i] < z[i - 1] - slope[i]:
                z[i] = z[i - 1] - slope[i]
        for i in range(n - 2, -1, -1):
            if z[i] < z[i + 1] - slope[i]:
                z[i] = z[i + 1] - slope[i]
    # Each camera's point moved to where its line of sight is at that height.
    t_c = g_t + toward(g_t, c_t) * z[:, None]
    b_c = g_b + toward(g_b, c_b) * z[:, None]
    # Each camera's input to the fusion (after the height correction), for the viewer: rule 42 fills them.
    cams = {"t": t_c.copy(), "b": b_c.copy(), "t_fill": np.zeros(n, bool), "b_fill": np.zeros(n, bool)}
    if TRACES and conf is not None:
        (t_c, ct, ft, tt), (b_c, cb, fb, tb) = camera_traces(t_c, b_c, conf)
        conf = (ct, cb)
        cams.update({"t": t_c.copy(), "b": b_c.copy(), "t_fill": ft, "b_fill": fb, "t_trace": tt, "b_trace": tb})
    out = kalman.ball_kinematics_two(t_c.astype(np.float32), b_c.astype(np.float32), bridge, conf, image)
    out["cam_inputs"] = cams
    out.update({"z": z, "z_std": zs, "flight": flight, "flights": summary, "z_tri": B_tri[:, 2], "bias": bias})
    return out


def kinematics(tac, ball_b, bro, alignment):
    """solve() on the ATD balls (the broadcast's on its main camera only), plus the camera
    geometry ("geometry")."""
    import airborne
    c_t, c_b, main = camera_geometry(tac, bro, alignment)
    g_t = rules.drop_spikes(tac.ball)
    g_b = np.where(main[:, None], rules.drop_spikes(ball_b), np.nan)
    candidates = [(int(s[0]), int(s[1])) for s in airborne.segments(tac.ball) if s[5] == 0]
    out = solve(g_t, g_b, c_t, c_b, kalman.MEAS_STD[1], candidates)
    out["geometry"] = {"tactical": c_t, "broadcast": c_b, "bias": out["bias"], "main": main}
    return out
