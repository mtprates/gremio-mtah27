"""Rule 13: the ball through a Kalman smoother instead of a moving average.

A constant-velocity model per axis (state: position, velocity), run twice:

    forward   over frames 0 .. t       -> v_in  at t: the ball's velocity coming in
    backward  over frames t+1 .. n     -> v_out at t: the velocity going out
    both, combined (two-filter smoother) -> position and velocity at t, with their uncertainty

A touch is a kink between the two: v_out - v_in. Because each filter carries its own uncertainty,
the kink is tested against it (dv_z, in standard deviations), and the filters coast through short
ball gaps -- so a touch hidden by an occlusion at a player's feet still shows as a kink between
the incoming and outgoing straight lines (the "known weakness" of rule 1).

Both axes share the model and the measurement noise, so they share one covariance: the loops
below carry a 2x2 covariance (p00, p01, p11) and a position / velocity per axis.

Rule 13.b (ball_kinematics_two): the same smoother fed by both cameras' balls directly, instead
of rule 8's fused ball. Each camera updates the filter on its own, with its own noise; the
broadcast's offset from the tactical camera is part of the state (a slow random walk); and a ball
too far from where the filter expects it is rejected for that frame (gating) -- which is what
decides, frame by frame, which camera to believe when they disagree.
"""

import numpy as np

from data import FPS

# Params
DT = 1.0 / FPS
ACCEL_STD = 3.0             # m/s^2: white-noise acceleration between touches. A rolling ball
                            # loses ~1 m/s^2 to friction; bounces and spin add the rest. Kicks
                            # are far outside it -- they are what the touch test finds.
VEL0_STD = 15.0             # m/s: velocity uncertainty when a track (re)starts: anything goes
RESET_GAP = 2 * FPS         # frames: a ball unseen this long is lost; the filter restarts
MEAS_STD = {1: 0.5, 2: 0.5, 3: 0.35, 4: 1.0, 5: 1.0}
                            # m, per rules.BALL_SOURCES code. The cameras' balls are 0.77 m apart
                            # at the median on live frames, which is ~0.46 m per camera if both
                            # err alike (Rayleigh median = 1.18 sqrt(2) sigma): one camera 0.5 m,
                            # their mean 0.35 m, a camera that won a disagreement 1.0 m.
POS_STD_MAX = 1.0           # m: positions bridged across a gap are kept while this certain
TOUCH_Z = 3.0               # rule 1 with the smoother: the kink must also be this many standard
                            # deviations (chi-square, 2 dof: p ~ 0.01) -- weak inside a gap
ACCEL_WINDOW = 6            # frames either side for the acceleration estimate (as rules.py)
# Rule 13.b, two cameras
BIAS0_STD = 0.5             # m: typical broadcast offset from the tactical camera ...
BIAS_TAU = 30.0             # s: ... which wanders as the camera pans but decays back toward 0 on
                            # this time scale (the cameras are calibrated to the same pitch). A
                            # free random walk let it absorb the ball's own motion wherever only
                            # the broadcast saw it: 6 m at the 90th percentile. At 1 m / 10 s it
                            # still absorbed a high ball's 6 m camera gap in a second (1H 12785):
                            # that gap is height (rule 17), not offset.
GATE = 18.4                 # chi-square, 2 dof, p = 1e-4: a ball further than this from the
                            # filter's expectation is not this ball, this frame
RESYNC = 3                  # frames of every ball rejected before a restart is considered ...
RESYNC_AGREE = 1.5          # m: ... at once if both cameras' balls agree this closely ...
KICK_MIN_SPEED, KICK_TOL = 5.0, 6.0  # m/s: ... or if one camera's rejected balls move on at least
                            # this fast and within this (or half the jump's speed) of the jump's
                            # velocity -- a kick ...
RESYNC_LONG = FPS           # ... or after this long with every ball rejected; else the rejected
                            # frames count as unseen and are bridged
NOISE_FLOOR, NOISE_CEIL = 0.2, 1.5  # m: bounds on the per-camera noise read off the data.
                            # The ATD balls are smoothed by the provider: frame-to-frame jitter
                            # put both cameras on the 0.2 m floor, so the noise is read from each
                            # camera's residuals against a first smoothing pass instead.

# Rule 31 (docs/VALUES.md): what the picks' confidence says. Off unless FIX and given confidences.
FIX = False
CONF_REF = 0.5              # a pick this confident has the base noise; weaker ones less weight ...
VAR_MAX = 16.0              # ... noise variance x CONF_REF / conf, at most this
CONFLICT_Z = GATE           # the passes differ by more than this (chi-square): the more confident one
RESYNC_CONSISTENT = 8       # rejected picks of one camera on a line within RESYNC_SPREAD m ...
RESYNC_SPREAD = 1.0
RESYNC_BREAK = 5            # ... through gaps of up to this many frames, more confident than what the
RESYNC_MARGIN = 0.1         # filter accepts by this: the filter is on the wrong ball; restart there
# Rule 39 (docs/VALUES.md): no restart, and no bridge, at a pace no ball has (with FIX)
MAX_PACE, PACE_CONF, MAX_PACE_ANY = 35.0, 0.5, 50.0     # m/s, confidence, m/s
RESTART_MIN_CONF = 0.3      # a restart needs picks this confident (two weak ones agreeing are weak evidence)
LONE_FRAMES = 12            # a weak pick with no other pick this close after (or before) it does not end a gap's bridge
WEAK_AGREE = 2.0            # m: between two confident picks a weak pick off their bridge by more than this does not bend it
# Rule 48 (docs/VALUES.md): a gap is bridged only between kept ends, and not where even the straight line is faster
# than MAX_PACE (with FIX). iterate.py --gaps.
GAPS = False
# Rule 50 (docs/VALUES.md): a "two cameras agree" restart on picks under PACE_CONF needs RESYNC agreements. --confirm.
CONFIRM = False
# Rule 51 (docs/VALUES.md): every bridged run, as it finally stands, no faster than MAX_PACE. --bridge-pace.
FINAL_PACE = False

# Rule 44 (docs/VALUES.md): image_track.verdicts' labels, which _run2 decides with when given them
IMG_MEMBER, IMG_OTHER, IMG_SWITCH = 1, -1, 3
DEBUG = None                # a dict: _kinematics' last two-camera call leaves its per-frame choices here (viewer checks)
RESTART_LOG = None          # a list: every restart of the two-camera filter (call, frame, why, camera, confidence)
Q00, Q01, Q11 = (ACCEL_STD ** 2 * DT ** 3 / 3, ACCEL_STD ** 2 * DT ** 2 / 2, ACCEL_STD ** 2 * DT)
BIAS_DECAY = float(np.exp(-DT / BIAS_TAU))
QB = BIAS0_STD ** 2 * (1 - BIAS_DECAY ** 2)  # keeps the offset's spread at BIAS0_STD


def _q(accel_std):
    """Process noise (Q00, Q01, Q11) of a constant-velocity model with white acceleration."""
    a2 = accel_std ** 2
    return a2 * DT ** 3 / 3, a2 * DT ** 2 / 2, a2 * DT


def _run(z, r2, q=None):
    """One Kalman pass in frame order. z (n, 2) measured positions (NaN: none), r2 (n,) measurement
    variance, q the process noise (_q; the ball's ACCEL_STD by default). Returns (prior, post),
    each (n, 7): x, y, vx, vy, p00, p01, p11 -- prior before the frame's measurement, post after
    it; NaN where the filter is not running."""
    q00, q01, q11 = (Q00, Q01, Q11) if q is None else q
    n = len(z)
    prior = np.full((n, 7), np.nan)
    post = np.full((n, 7), np.nan)
    zx, zy, r2 = z[:, 0].tolist(), z[:, 1].tolist(), r2.tolist()
    on, gap = False, 0
    px = py = vx = vy = p00 = p01 = p11 = 0.0
    for t in range(n):
        if on:
            px += DT * vx
            py += DT * vy
            p00 += 2 * DT * p01 + DT * DT * p11 + q00
            p01 += DT * p11 + q01
            p11 += q11
            prior[t] = (px, py, vx, vy, p00, p01, p11)
        r = r2[t]
        if r == r:  # measured
            if not on:
                px, py, vx, vy, p00, p01, p11 = zx[t], zy[t], 0.0, 0.0, r, 0.0, VEL0_STD ** 2
                on = True
            else:
                s = p00 + r
                k0, k1 = p00 / s, p01 / s
                ex, ey = zx[t] - px, zy[t] - py
                px += k0 * ex
                py += k0 * ey
                vx += k1 * ex
                vy += k1 * ey
                p11 -= k1 * p01
                p01 *= 1 - k0
                p00 *= 1 - k0
            gap = 0
        else:
            gap += 1
            if gap > RESET_GAP:
                on = False
        if on:
            post[t] = (px, py, vx, vy, p00, p01, p11)
    return prior, post


def _combine(a, b):
    """Two Gaussian estimates (n, 7) of the same state -> one, per frame; where only one exists,
    that one. Position and velocity are fused each on its own (inverse-variance weights), not as a
    joint state: at a kick the two velocities are both right -- in and out -- and a joint product
    would move the position to split the difference between them."""
    out = np.where(np.isnan(a[:, :1]), b, a)
    both = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    a, b = a[both], b[both]
    fused = np.empty_like(a)
    for var, cols in ((4, (0, 1)), (6, (2, 3))):
        wa, wb = 1 / a[:, var], 1 / b[:, var]
        fused[:, var] = 1 / (wa + wb)
        for c in cols:
            fused[:, c] = (wa * a[:, c] + wb * b[:, c]) * fused[:, var]
    fused[:, 5] = 0.0
    out[both] = fused
    return out


def dv_z_at(a, b, t):
    """The velocity change at frame t between estimates a and b, in standard deviations."""
    return np.linalg.norm(b[t, 2:4] - a[t, 2:4]) / np.sqrt(a[t, 6] + b[t, 6])


def ball_kinematics(ball, source):
    """rules.ball_kinematics on a Kalman smoother: the same keys, plus pos_std (m) and dv_z (the
    velocity change in standard deviations). source (n,) indexes rules.BALL_SOURCES."""
    import rules
    n = len(ball)
    z = rules.drop_spikes(ball)
    seen = ~np.isnan(z[:, 0])
    std = np.array([MEAS_STD.get(int(s), np.nan) for s in range(max(MEAS_STD) + 1)])
    r2 = np.where(seen, std[np.clip(source, 0, len(std) - 1)] ** 2, np.nan)
    r2[seen & np.isnan(r2)] = MEAS_STD[1] ** 2

    forward_prior, forward = _run(z, r2)
    back_prior, back_post = _run(z[::-1], r2[::-1])
    return _kinematics(forward_prior, forward, back_prior, back_post, seen)


def _kinematics(forward_prior, forward, back_prior, back_post, seen, bridge=False, acc=None):
    """Rule 1's kinematics from a forward and a reversed-time Kalman pass ((n, 7) each).

    bridge: every gap the filters span (up to RESET_GAP) gets a position, however uncertain: a
    cubic Hermite curve from the last seen position and velocity to the next seen ones. For the
    detector track, whose gaps are a ball the detector missed (1H 13160: 1 s unseen, 36 m covered
    while the broadcast showed a replay), not a ball nobody knows the whereabouts of. (The kick
    model of the gap -- incoming line, then outgoing -- jumped the 36 m in one frame there: the
    ball arrived slowly.) Without it, gap frames less certain than POS_STD_MAX have no position."""
    import rules
    n = len(seen)

    def flip(est):  # reversed time: velocity and its covariance with position change sign
        est = est[::-1].copy()
        est[:, 2:4] *= -1
        est[:, 5] *= -1
        return est
    backward, backward_post = flip(back_prior), flip(back_post)
    smooth = _combine(forward, backward)
    resolved = np.zeros(len(seen), dtype=bool)
    if acc is not None:
        # Rule 31 (2): two states that differ by metres are not averaged -- on a seen frame the pass
        # that took a pick stands.
        with np.errstate(invalid="ignore", divide="ignore"):
            z2 = np.sum((forward[:, :2] - backward[:, :2]) ** 2, axis=1) / (forward[:, 4] + backward[:, 4])
        bad = seen & np.isfinite(z2) & (z2 > CONFLICT_Z)
        # The pass that took a pick on this frame stands, as its posterior (the measurement is in
        # it); if both did, the more confident. The backward prior is from before any restart on
        # this frame and can be on the wrong ball (1H 23368).
        took_f, took_b = acc[2], acc[3]
        use_f = bad & took_f & (~took_b | (acc[0] >= acc[1]))
        use_b = bad & took_b & (~took_f | (acc[1] > acc[0]))
        if GAPS:
            # Rule 48: where one pass alone took a pick, its posterior stands -- the other coasted on what it rejected.
            use_f |= seen & took_f & ~took_b
            use_b |= seen & took_b & ~took_f
        smooth[use_f] = forward[use_f]
        smooth[use_b] = backward_post[use_b]
        resolved = use_f | use_b
        if DEBUG is not None:
            DEBUG.update(forward=forward[:, :2].copy(), backward=backward[:, :2].copy(),
                         backward_post=backward_post[:, :2].copy(), use_f=use_f.copy(), use_b=use_b.copy(),
                         conflict=bad.copy(), took_f=np.asarray(took_f).copy(), took_b=np.asarray(took_b).copy())

    # Kicks inside a gap are decided below, from the gap as a whole; elsewhere rules.touch_times
    # takes the peaks of dv that pass TOUCH_Z.
    # Inside a gap each filter coasts on a straight line; if the ball was kicked in there, the two
    # lines disagree and their product is confidently wrong (a corner cut). Where unseen, the
    # position is known only to within half the filters' disagreement, whatever they claim.
    apart = np.linalg.norm(forward[:, :2] - backward[:, :2], axis=1)
    apart[resolved] = 0.0
    var = smooth[:, 4] + np.where(seen, 0.0, np.nan_to_num(apart, nan=0.0) / 2) ** 2
    pos_std = np.sqrt(var)
    gaps = []
    for start, stop in rules.runs(~seen):
        lo, hi = start - 1, stop
        if not (lo >= 0 and hi < n and stop - start <= RESET_GAP and np.isfinite(apart[start:stop]).any()):
            continue
        # The kick in a gap is where the incoming and outgoing lines come closest -- searched over
        # the unseen frames only: on a seen frame both filters have just met the same measurement
        # and agree whatever happened. One touch per gap, there (rules.touch_times).
        kink = start + int(np.nanargmin(apart[start:stop]))
        # Is there a kick at all? Either the velocities either side differ beyond their
        # uncertainty, or the line coasted across the gap misses the ball seen at its other end.
        miss = [np.linalg.norm(a[t, :2] - b[t, :2]) / np.sqrt(a[t, 4] + b[t, 4])
                for t, a, b in ((lo, forward, backward), (hi, forward_prior, backward_post))]
        z_gap = np.nanmax([*miss, dv_z_at(forward, backward, kink)])
        dv_gap = np.linalg.norm(backward[kink, 2:4] - forward[kink, 2:4]) if np.isfinite(forward[kink, 0]) else np.nan
        gaps.append((lo, hi, kink, bool(z_gap >= TOUCH_Z and dv_gap >= rules.DV_THRESHOLD)))
    with np.errstate(invalid="ignore"):
        keep = pos_std <= POS_STD_MAX
    pos = np.where(keep[:, None], smooth[:, :2], np.nan)
    bridged = np.zeros(n, dtype=bool)
    anchored = np.zeros(n, dtype=bool)     # placed by rule 39 (8)'s bridge between confident picks
    if bridge:
        for lo, hi, _, _ in gaps:
            p0, p1 = smooth[lo, :2], smooth[hi, :2]
            v0, v1 = smooth[lo, 2:4], smooth[hi, 2:4]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                continue
            if GAPS and FIX and (not keep[lo] or not keep[hi] or
                                 np.linalg.norm(p1 - p0) * FPS / (hi - lo) > MAX_PACE):
                # Rule 48: an end too uncertain to keep, or ends no ball could join: the gap stays unknown.
                pos[lo + 1:hi] = np.nan
                keep[lo + 1:hi] = False
                continue
            if FIX and acc is not None:
                # Rule 39: no bridge toward a lone weak pick -- hold the other end, uncertain by half the way to it.
                pc = acc[4]
                lone_hi = pc[hi] < RESTART_MIN_CONF and not seen[hi + 1:hi + 1 + LONE_FRAMES].any()
                lone_lo = pc[lo] < RESTART_MIN_CONF and not seen[max(0, lo - LONE_FRAMES):lo].any()
                if lone_hi != lone_lo:
                    held_at = p0 if lone_hi else p1
                    for t in (range(lo + 1, hi + 1) if lone_hi else range(lo, hi)):
                        pos[t] = held_at
                        pos_std[t] = max(pos_std[t] if np.isfinite(pos_std[t]) else 0.0, np.linalg.norm(p1 - p0) / 2)
                        bridged[t] = True
                    continue
            v0, v1 = np.nan_to_num(v0), np.nan_to_num(v1)
            span = (hi - lo) / FPS
            for t in range(lo + 1, hi):
                if keep[t] and not FIX:
                    continue            # rule 39 (FIX): inside an unseen gap the bridge is the track throughout
                u = (t - lo) / (hi - lo)
                h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
                h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
                pos[t] = h00 * p0 + h10 * span * v0 + h01 * p1 + h11 * span * v1
                bridged[t] = True
            if FIX:
                # Rule 39: the cubic overshoots its chord where an end's velocity is a guess; a straight line then.
                # (Against the smoothed ends: an end too uncertain to keep has no pos, and skipped this test --
                # 1H 132416-132421 ran 25 m a frame.)
                fast = np.linalg.norm(np.diff(np.vstack([p0, pos[lo + 1:hi], p1]), axis=0), axis=1) * FPS
                if np.isfinite(fast).all() and fast.max() > MAX_PACE:
                    u = (np.arange(lo + 1, hi) - lo) / (hi - lo)
                    for t in range(lo + 1, hi):
                        if bridged[t]:
                            pos[t] = p0 + (p1 - p0) * u[t - lo - 1]
        if FIX and acc is not None:
            # Rule 39 (8): between confident picks the track is their bridge; a weak pick keeps its place only on it.
            pc = acc[4]
            anchors = np.flatnonzero(seen & (pc >= RESTART_MIN_CONF))
            for a, b in zip(anchors[:-1], anchors[1:]):
                if b - a < 2 or b - a > RESET_GAP or not seen[a + 1:b].any():
                    continue
                p0, p1 = smooth[a, :2], smooth[b, :2]
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    continue
                inner = np.arange(a + 1, b)
                u = ((inner - a) / (b - a))[:, None]
                span = (b - a) / FPS
                v0, v1 = np.nan_to_num(smooth[a, 2:4]), np.nan_to_num(smooth[b, 2:4])
                path = ((2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * span * v0 +
                        (-2 * u ** 3 + 3 * u ** 2) * p1 + (u ** 3 - u ** 2) * span * v1)
                if (np.linalg.norm(np.diff(np.vstack([p0, path, p1]), axis=0), axis=1) * FPS).max() > MAX_PACE:
                    path = p0 + (p1 - p0) * u
                    if GAPS and np.linalg.norm(p1 - p0) * FPS / (b - a) > MAX_PACE:
                        continue            # rule 48: no ball joins these two picks; the weak ones keep their place
                with np.errstate(invalid="ignore"):
                    off = np.linalg.norm(pos[inner] - path, axis=1)
                    on_it = seen[inner] & (off <= WEAK_AGREE)
                swap = inner[~on_it]
                anchored[swap] = True
                pos[swap] = path[~on_it]
                spread = np.nan_to_num(off[~on_it], nan=0.0) / 2
                pos_std[swap] = np.fmax(np.nan_to_num(pos_std[swap], nan=0.0), spread)
                bridged[swap] = True
        if FINAL_PACE:
            # Rule 51: the bridges are checked as they finally stand, joins with the frames either side included.
            for start, stop in rules.runs(bridged):
                lo, hi = start - 1, stop
                if lo < 0 or hi >= n or not (np.isfinite(pos[lo]).all() and np.isfinite(pos[hi]).all()):
                    continue
                with np.errstate(invalid="ignore"):
                    fast = np.linalg.norm(np.diff(pos[lo:hi + 1], axis=0), axis=1) * FPS
                if np.isfinite(fast).all() and fast.max() > MAX_PACE:
                    u = ((np.arange(start, stop) - lo) / (hi - lo))[:, None]
                    pos[start:stop] = pos[lo] + (pos[hi] - pos[lo]) * u
        keep = keep | bridged
    if DEBUG is not None and acc is not None:
        DEBUG.update(anchored=anchored.copy(), anchor=(seen & (acc[4] >= RESTART_MIN_CONF)).copy(), bridge=bool(bridge), bridged=bridged.copy(), keep=keep.copy(), pos_std=pos_std.copy(),
                     gap_ends={(lo, hi) for lo, hi, _, _ in gaps}, seen=seen.copy())
    # "present" keeps rule 1's meaning for the rules built on it (carries, dwells, dead balls): a
    # measured ball, or a short gap between two -- the smoother only adds positions, not presence.
    present = seen.copy()
    for start, stop in rules.runs(~seen):
        if start > 0 and stop < n and stop - start <= rules.INTERP_MAX_GAP:
            present[start:stop] = True
    present &= keep & ~bridged

    v_in, v_out = forward[:, 2:4], backward[:, 2:4]
    dv = np.linalg.norm(v_out - v_in, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dv_z = dv / np.sqrt(forward[:, 6] + backward[:, 6])
    velocity = np.where(keep[:, None], smooth[:, 2:4], np.nan)
    later = np.full_like(velocity, np.nan)
    earlier = np.full_like(velocity, np.nan)
    later[:-ACCEL_WINDOW] = velocity[ACCEL_WINDOW:]
    earlier[ACCEL_WINDOW:] = velocity[:-ACCEL_WINDOW]
    return {
        "pos": pos,
        "present": present,
        "seen": seen,
        "v_in": v_in,
        "v_out": v_out,
        "dv": dv,
        "dv_z": dv_z,
        "speed_in": np.linalg.norm(v_in, axis=1),
        "speed_out": np.linalg.norm(v_out, axis=1),
        "speed": np.linalg.norm(velocity, axis=1),
        "accel": np.linalg.norm(later - earlier, axis=1) / (2 * ACCEL_WINDOW / FPS),
        "pos_std": np.where(keep, pos_std, np.nan),
        "gaps": gaps,              # (last seen, first seen again, kick frame, is there a kick)
    }


def _spread(residual):
    """m: robust per-axis standard deviation of (n, 2) residuals, within the noise bounds."""
    r = residual[np.isfinite(residual).all(axis=1)].ravel()
    if not len(r):
        return MEAS_STD[1]
    return float(np.clip(1.4826 * np.median(np.abs(r - np.median(r))), NOISE_FLOOR, NOISE_CEIL))


def _run2(zt, zb, rt2, rb2, ct=None, cb=None, tag=None, img=None):
    """One two-camera pass in frame order. zt, zb (n, 2) tactical / broadcast balls (NaN: none),
    rt2, rb2 their noise variances. State per axis: position, velocity, broadcast offset.
    Returns (prior, post) as _run's (n, 7), used (n,) bit mask -- 1 tactical, 2 broadcast taken;
    4 tactical, 8 broadcast rejected -- the offset (n, 2) after each frame and acc (n,), the
    confidence of what the filter has been accepting (0: nothing lately). ct, cb: the picks'
    confidences (rule 31, with FIX). img: (tactical, broadcast) image_track verdicts in this pass's order
    (rule 44): a member far from the pitch prediction restarts the track there, a non-member is rejected, and
    the image track's switches replace the pitch-motion restarts (kick pattern, consistent run)."""
    n = len(zt)
    fix = FIX and ct is not None
    cts, cbs = (ct.tolist(), cb.tolist()) if fix else (None, None)
    acc = np.zeros(n)

    def scale(c):
        return min(VAR_MAX, max(1.0, CONF_REF / c)) if fix and c == c and c > 0 else 1.0
    prior = np.full((n, 7), np.nan)
    post = np.full((n, 7), np.nan)
    used = np.zeros(n, dtype=np.int8)
    offset = np.full((n, 2), np.nan)
    tx, ty, bx, by = zt[:, 0].tolist(), zt[:, 1].tolist(), zb[:, 0].tolist(), zb[:, 1].tolist()
    st = {"on": False, "gap": 0, "miss": 0,
          "px": 0.0, "py": 0.0, "vx": 0.0, "vy": 0.0, "ox": 0.0, "oy": 0.0,
          # covariance, shared by both axes: position, velocity, offset
          "pp": 0.0, "pv": 0.0, "po": 0.0, "vv": 0.0, "vo": 0.0, "oo": BIAS0_STD ** 2,
          "ac": 0.0, "act": None, "far": {1: 0, 2: 0}, "agreed": []}

    def start(cam, zx, zy, r, *_):
        """(Re)start the track at a measurement, keeping the offset and its uncertainty."""
        if cam == 1:
            st.update(px=zx, py=zy, pp=r, po=0.0)
        else:
            st.update(px=zx - st["ox"], py=zy - st["oy"], pp=r + st["oo"], po=-st["oo"])
        st.update(vx=0.0, vy=0.0, pv=0.0, vo=0.0, vv=VEL0_STD ** 2)

    hist = {1: [], 2: []}       # per camera, the unbroken run of rejected balls: (t, x, y on the
                                # tactical pitch, the measurement)

    def resync(meas):
        """After every ball was rejected: the measurement to restart the track at, or None to keep
        the frames unseen (bridged later). Only (a) both cameras agreeing with each other, (b) one
        camera's rejected balls moving on at the speed of the jump that took them away -- a kick,
        not a jump to something that then stands still -- or (c) a ball unseen for RESYNC_LONG.
        1H 15336: the restart that took the first camera in the list jumped 3.6 m to a motionless
        object while the real ball was hidden at players' feet."""
        if len(meas) == 2:
            p1 = np.array(meas[0][1:3])
            p2 = np.array(meas[1][1:3]) - [st["ox"], st["oy"]]
            if np.linalg.norm(p1 - p2) <= RESYNC_AGREE:
                mid = (p1 + p2) / 2
                choice = (1, mid[0], mid[1], meas[0][3] / 2, (meas[0][4] + meas[1][4]) / 2)
                if not CONFIRM or choice[4] >= PACE_CONF:
                    st["why"] = "two cameras agree"
                    return choice
                # Rule 50: weak picks agreeing once may be one object both cameras take for the ball (14935);
                # RESYNC agreements, each close enough in time and place to the last, restart at the latest.
                ag = st["agreed"]
                if ag and (t - ag[-1][0] > RESYNC_BREAK or np.hypot(mid[0] - ag[-1][1], mid[1] - ag[-1][2]) >
                           RESYNC_AGREE + MAX_PACE * (t - ag[-1][0]) * DT):
                    del ag[:]
                ag.append((t, mid[0], mid[1]))
                if len(ag) >= RESYNC:
                    del ag[:]
                    st["why"] = "two cameras agree, confirmed"
                    return choice
        t0, x0, y0 = st["last"]
        for cam in ((1, 2) if img is None else ()):
            h = hist[cam]
            if len(h) < RESYNC:
                continue
            (ta, xa, ya, _), (tb, xb, yb, m) = h[-RESYNC], h[-1]
            v_own = np.array([xb - xa, yb - ya]) / ((tb - ta) * DT)
            v_jump = np.array([xa - x0, ya - y0]) / (max(1, ta - t0) * DT)
            if np.linalg.norm(v_own) >= KICK_MIN_SPEED and \
                    np.linalg.norm(v_own - v_jump) <= max(KICK_TOL, 0.5 * np.linalg.norm(v_jump)):
                st["why"] = "kick pattern"
                return m
        if st["miss"] >= RESYNC_LONG:
            pred = np.array([st["px"], st["py"]])
            st["why"] = "rejected for a second"
            return min(meas, key=lambda m: np.linalg.norm(
                np.array(m[1:3]) - ([st["ox"], st["oy"]] if m[0] == 2 else 0.0) - pred))
        return None

    def pace_ok(choice, t):
        """Rule 39: a restart at choice must not put the ball farther from its last accepted place than
        MAX_PACE allows (MAX_PACE_ANY for picks of PACE_CONF or more)."""
        if not fix:
            return True
        t0, x0, y0 = st["last"]
        cam, zx, zy, r, cf = choice[:5]
        if cf < RESTART_MIN_CONF and st.get("why") != "rejected for a second":
            return False
        ox, oy = (st["ox"], st["oy"]) if cam == 2 else (0.0, 0.0)
        pace = float(np.hypot(zx - ox - x0, zy - oy - y0)) / (max(1, t - t0) * DT)
        return pace <= MAX_PACE or (cf >= PACE_CONF and pace <= MAX_PACE_ANY)

    def run_restart(t):
        """Rule 31 (1): the measurement to restart at when a camera's rejected picks form a line and
        are more confident than what the filter has been accepting, else None."""
        if img is not None or st["act"] is None or t - st["act"] > RESET_GAP:
            return None
        best = None
        for cam in (1, 2):
            h = hist[cam]
            if len(h) < RESYNC_CONSISTENT:
                continue
            last = h[-RESYNC_CONSISTENT:]
            ts = np.array([x[0] for x in last], dtype=float)
            xy = np.array([[x[1], x[2]] for x in last])
            design = np.column_stack([np.ones(len(ts)), ts - ts[0]])
            line = np.linalg.lstsq(design, xy, rcond=None)[0]
            if np.linalg.norm(xy - design @ line, axis=1).max() > RESYNC_SPREAD:
                continue
            conf = float(np.mean([x[3][4] for x in last]))
            if conf >= st["ac"] + RESYNC_MARGIN and (best is None or conf > best[0]):
                best = (conf, last[-1][3])
        if best is not None:
            st["why"] = "consistent run of rejected picks"
        return None if best is None else best[1]

    st["last"] = (0, 0.0, 0.0)
    for t in range(n):
        if st["on"]:
            st["px"] += DT * st["vx"]
            st["py"] += DT * st["vy"]
            st["pp"] += 2 * DT * st["pv"] + DT * DT * st["vv"] + Q00
            st["pv"] += DT * st["vv"] + Q01
            st["po"] = BIAS_DECAY * (st["po"] + DT * st["vo"])
            st["vo"] *= BIAS_DECAY
            st["vv"] += Q11
            st["ox"] *= BIAS_DECAY
            st["oy"] *= BIAS_DECAY
            st["oo"] = BIAS_DECAY ** 2 * st["oo"] + QB
            prior[t] = (st["px"], st["py"], st["vx"], st["vy"], st["pp"], st["pv"], st["vv"])
        meas = []
        if tx[t] == tx[t]:
            c1 = cts[t] if fix else 1.0
            meas.append((1, tx[t], ty[t], rt2 * scale(c1), c1))
        if bx[t] == bx[t]:
            c2 = cbs[t] if fix else 1.0
            meas.append((2, bx[t], by[t], rb2 * scale(c2), c2))
        if meas and not st["on"] and fix:
            # Rule 39: a filter that is off starts on a confident pick, or on two cameras agreeing.
            meas.sort(key=lambda m: -m[4])
            on_pitch = [(m[1] - (st["ox"] if m[0] == 2 else 0.0), m[2] - (st["oy"] if m[0] == 2 else 0.0)) for m in meas]
            agree = len(meas) == 2 and np.hypot(on_pitch[0][0] - on_pitch[1][0], on_pitch[0][1] - on_pitch[1][1]) <= RESYNC_AGREE
            if meas[0][4] < RESTART_MIN_CONF and not agree:
                for m in meas:
                    used[t] |= 4 if m[0] == 1 else 8
                meas = []
        if meas:
            if not st["on"]:
                start(*meas[0])
                st["on"] = True
                used[t] |= meas[0][0]
                st["ac"], st["act"] = meas[0][4], t
                pending = meas[1:]
            else:
                pending = meas
            # Both balls are tested against the prediction before either updates it: tested in
            # turn, the camera that went first always won a disagreement (1H 12785).
            accepted, far = [], []
            agree2 = False
            if img is not None and len(pending) == 2:
                agree2 = bool(np.hypot(pending[0][1] - pending[1][1] + st["ox"],
                                       pending[0][2] - pending[1][2] + st["oy"]) <= RESYNC_AGREE)
            for cam, zx, zy, r, cf in pending:
                h = 0.0 if cam == 1 else 1.0
                S = st["pp"] + 2 * h * st["po"] + h * st["oo"] + (rt2 if cam == 1 else rb2)
                ex = zx - st["px"] - h * st["ox"]
                ey = zy - st["py"] - h * st["oy"]
                v = int(img[cam - 1][t]) if img is not None else 0
                d2 = ex * ex + ey * ey
                if v == IMG_OTHER and not agree2:
                    used[t] |= 4 if cam == 1 else 8     # rule 44: not the ball this camera's image follows
                elif d2 / S > GATE:
                    used[t] |= 4 if cam == 1 else 8
                    if v in (IMG_MEMBER, IMG_SWITCH):
                        far.append((cam, zx, zy, r, cf, v, d2 / GATE - (S - (rt2 if cam == 1 else rb2))))
                else:
                    accepted.append((cam, zx, zy, r, cf))
                    st["far"][cam] = 0
            if far and accepted:
                for m in far:
                    st["far"][m[0]] += 1
            elif far:
                # Rule 44: the image says this is the ball the camera has been following, so the pitch prediction (an
                # air ball's inflated ground point, coasted) is what is off. A switch, or RESYNC such frames in a row,
                # restart there; else the pick is taken with its noise raised to sit on the gate (a pull, no reset).
                for m in far:
                    st["far"][m[0]] += 1
                hard = [m for m in far if m[5] == IMG_SWITCH or st["far"][m[0]] >= RESYNC]
                st["why"] = "image track"
                refused = [m for m in hard if not pace_ok(m[:5], t)]     # rule 39 holds for these restarts too
                hard = [m for m in hard if pace_ok(m[:5], t)]
                far = [m for m in far if m not in refused]
                if hard:
                    choice = max(hard, key=lambda m: m[4])[:5]
                    st["why"] = "image track switch" if int(img[choice[0] - 1][t]) == IMG_SWITCH else "image track"
                    if RESTART_LOG is not None:
                        RESTART_LOG.append((tag, t, st["why"], choice[0], float(choice[4])))
                    start(*choice)
                    used[t] = (used[t] & ~(4 if choice[0] == 1 else 8)) | choice[0]
                    st["ac"], st["act"] = choice[4], t
                    st["far"] = {1: 0, 2: 0}
                else:
                    for m in far:
                        accepted.append((m[0], m[1], m[2], max(m[3], m[6]), m[4]))
                        used[t] &= ~(4 if m[0] == 1 else 8)
            for cam, zx, zy, r, cf in accepted:
                h = 0.0 if cam == 1 else 1.0
                g0 = st["pp"] + h * st["po"]
                g1 = st["pv"] + h * st["vo"]
                g2 = st["po"] + h * st["oo"]
                S = g0 + h * g2 + r
                ex = zx - st["px"] - h * st["ox"]
                ey = zy - st["py"] - h * st["oy"]
                k0, k1, k2 = g0 / S, g1 / S, g2 / S
                st["px"] += k0 * ex
                st["py"] += k0 * ey
                st["vx"] += k1 * ex
                st["vy"] += k1 * ey
                st["ox"] += k2 * ex
                st["oy"] += k2 * ey
                st["pp"] -= k0 * g0
                st["pv"] -= k0 * g1
                st["po"] -= k0 * g2
                st["vv"] -= k1 * g1
                st["vo"] -= k1 * g2
                st["oo"] -= k2 * g2
                used[t] |= cam
                fresh = st["act"] is not None and t - st["act"] <= RESET_GAP
                st["ac"] = 0.8 * st["ac"] + 0.2 * cf if fresh else cf
                st["act"] = t
            if fix:
                # Rule 31 (1): each camera's own run of rejected picks, through short gaps, not
                # cleared by the other camera being accepted.
                for cam, zx, zy, r, cf in meas:
                    if used[t] & cam:
                        hist[cam].clear()
                        continue
                    if hist[cam] and t - hist[cam][-1][0] > RESYNC_BREAK:
                        hist[cam].clear()
                    hist[cam].append((t, zx - (st["ox"] if cam == 2 else 0.0),
                                      zy - (st["oy"] if cam == 2 else 0.0), (cam, zx, zy, r, cf)))
            if used[t] & 3:
                st["miss"] = 0
                del st["agreed"][:]
                st["last"] = (t, st["px"], st["py"])
                if not fix:
                    hist[1].clear()
                    hist[2].clear()
                choice = run_restart(t) if fix else None
                if choice is not None and not pace_ok(choice, t):
                    choice = None
                if choice is not None:
                    if RESTART_LOG is not None:
                        RESTART_LOG.append((tag, t, st.get("why"), choice[0], float(choice[4])))
                    start(*choice)
                    used[t] |= choice[0]
                    st["ac"], st["act"] = choice[4], t
                    hist[1].clear()
                    hist[2].clear()
            else:
                st["miss"] += 1
                if not fix:
                    for cam, zx, zy, r, cf in meas:
                        if hist[cam] and hist[cam][-1][0] != t - 1:
                            hist[cam].clear()       # only an unbroken run of rejected balls counts
                        hist[cam].append((t, zx - (st["ox"] if cam == 2 else 0.0),
                                          zy - (st["oy"] if cam == 2 else 0.0), (cam, zx, zy, r, cf)))
                choice = resync(meas)
                if choice is not None and not pace_ok(choice, t) and st["miss"] < RESYNC_LONG:
                    choice = None
                if choice is None and fix:
                    choice = run_restart(t)
                    if choice is not None and not pace_ok(choice, t):
                        choice = None
                if choice is not None:
                    if RESTART_LOG is not None:
                        RESTART_LOG.append((tag, t, st.get("why"), choice[0], float(choice[4])))
                    start(*choice)
                    used[t] |= choice[0]
                    st["miss"] = 0
                    st["last"] = (t, st["px"], st["py"])
                    hist[1].clear()
                    hist[2].clear()
            st["gap"] = 0
        else:
            st["gap"] += 1
            if st["gap"] > RESET_GAP:
                st["on"], st["miss"] = False, 0
        if st["on"]:
            post[t] = (st["px"], st["py"], st["vx"], st["vy"], st["pp"], st["pv"], st["vv"])
            offset[t] = (st["ox"], st["oy"])
            acc[t] = st["ac"] if st["act"] is not None and t - st["act"] <= RESET_GAP else 0.0
    return prior, post, used, offset, acc


def ball_kinematics_two(ball_t, ball_b, bridge=False, conf=None, image=None):
    """Rule 13.b: ball_kinematics from both cameras' balls (tactical frames and orientation; the
    broadcast NaN where it is not live), not rule 8's fused ball. Adds "used" (forward pass, the
    bit mask of _run2), "offset" (broadcast minus tactical, m) and "noise" (m per camera).

    Two rounds: the first with MEAS_STD's one-camera noise for both, the second with each
    camera's noise read off its residuals against the first round's smoothed ball.

    conf: (tactical, broadcast) pick confidences (n,), used with FIX (rule 31). image: image_track.match_verdicts
    (rule 44)."""
    import rules
    zt, zb = rules.drop_spikes(ball_t), rules.drop_spikes(ball_b)
    ct, cb = ((np.asarray(conf[0], dtype=float), np.asarray(conf[1], dtype=float))
              if FIX and conf is not None else (None, None))
    seen = ~np.isnan(zt[:, 0]) | ~np.isnan(zb[:, 0])
    noise = {"tactical": MEAS_STD[1], "broadcast": MEAS_STD[1]}
    for _ in range(2):
        rt2, rb2 = noise["tactical"] ** 2, noise["broadcast"] ** 2
        if RESTART_LOG is not None:
            del RESTART_LOG[:]
        img_f = None if image is None else image["forward"]
        img_b = None if image is None else tuple(v[::-1] for v in image["backward"])
        forward_prior, forward, used, off_f, acc_f = _run2(zt, zb, rt2, rb2, ct, cb, "forward", img_f)
        back_prior, back_post, used_b, off_b, acc_b = _run2(
            zt[::-1], zb[::-1], rt2, rb2, None if ct is None else ct[::-1], None if cb is None else cb[::-1], "backward",
            img_b)
        # A frame is seen if either pass took one of its balls; frames whose every ball both
        # passes rejected are gaps, bridged like any other.
        taken = ((used & 3) > 0) | ((used_b[::-1] & 3) > 0)
        out = _kinematics(forward_prior, forward, back_prior, back_post, seen & taken, bridge,
                          None if ct is None else (acc_f, acc_b[::-1], (used & 3) > 0, (used_b[::-1] & 3) > 0,
                                                   np.fmax(np.nan_to_num(ct, nan=0.0), np.nan_to_num(cb, nan=0.0))))
        with np.errstate(all="ignore"):
            offset = np.nanmean(np.stack([off_f, off_b[::-1]]), axis=0)
        # Residuals of the balls both passes took, against the smoothed ball.
        taken = used & used_b[::-1]
        pos = out["pos"]
        noise = {"tactical": _spread(np.where((taken & 1)[:, None] > 0, zt - pos, np.nan)),
                 "broadcast": _spread(np.where((taken & 2)[:, None] > 0, zb - offset - pos, np.nan))}
    out["offset"] = offset
    if conf is not None:
        # Rule 53: the confidence of the picks the smoother used on each frame (best camera; 0 where none).
        took_t = ((used & 1) | (used_b[::-1] & 1)) > 0
        took_b = ((used & 2) | (used_b[::-1] & 2)) > 0
        out["pick_conf"] = np.maximum(np.where(took_t, np.nan_to_num(np.asarray(conf[0], dtype=float), nan=0.0), 0.0),
                                      np.where(took_b, np.nan_to_num(np.asarray(conf[1], dtype=float), nan=0.0), 0.0))
    out["used"] = used
    out["used_b"] = used_b[::-1]
    out["noise"] = {"tactical": rt2 ** 0.5, "broadcast": rb2 ** 0.5}
    return out


# Rule 21: player tracks through the same smoother.
PLAYER_ACCEL = 4.0          # m/s^2: players stop and turn sharply -- more than a rolling ball
VEL_FRAMES = 3              # frames of smoothed positions either side of a gap for its end velocities


def smooth_track(z, r2, accel_std=PLAYER_ACCEL, bridge_max=None):
    """One player's track through the two-pass smoother (no touches, no flights). z (n, 2)
    measured positions (NaN: none), r2 (n,) their variance. Gaps up to bridge_max frames inside the
    track are bridged by a cubic Hermite curve from the last seen smoothed position and velocity to
    the next seen ones -- leaving along his last velocity, arriving along his next -- with the end
    velocities from smoothed positions VEL_FRAMES either side (a filtered velocity overshoots after a
    turn). Returns pos (n, 2), vel (n, 2), std (n,) m and bridged (n,) bool; NaN beyond."""
    import rules
    n = len(z)
    q = _q(accel_std)
    seen = ~np.isnan(z[:, 0])
    r2 = np.where(seen, r2, np.nan)     # _run takes a frame with a variance as measured
    _, forward = _run(z, r2, q)
    back_prior, _ = _run(z[::-1], r2[::-1], q)
    backward = back_prior[::-1].copy()
    backward[:, 2:4] *= -1
    backward[:, 5] *= -1
    smooth = _combine(forward, backward)
    pos = np.where(seen[:, None], smooth[:, :2], np.nan)
    var = np.where(seen, smooth[:, 4], np.nan)
    bridged = np.zeros(n, dtype=bool)
    for start, stop in rules.runs(~seen):
        lo, hi = start - 1, stop
        if lo < 0 or hi >= n or (bridge_max is not None and stop - start > bridge_max):
            continue
        k0 = min(VEL_FRAMES, lo)
        k1 = min(VEL_FRAMES, n - 1 - hi)
        p0, p1 = pos[lo], pos[hi]
        v0 = (pos[lo] - pos[lo - k0]) * FPS / k0 if k0 and np.isfinite(pos[lo - k0]).all() else smooth[lo, 2:4]
        v1 = (pos[hi + k1] - pos[hi]) * FPS / k1 if k1 and np.isfinite(pos[hi + k1]).all() else smooth[hi, 2:4]
        v0, v1 = np.nan_to_num(v0), np.nan_to_num(v1)
        span = (hi - lo) / FPS
        u = (np.arange(start, stop) - lo) / (hi - lo)
        h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
        h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        pos[start:stop] = (h00[:, None] * p0 + h10[:, None] * span * v0 + h01[:, None] * p1 +
                           h11[:, None] * span * v1)
        # Uncertainty: the two filters' fused variance in the gap, plus half their disagreement.
        apart = np.linalg.norm(forward[start:stop, :2] - backward[start:stop, :2], axis=1)
        var[start:stop] = np.nan_to_num(smooth[start:stop, 4], nan=1.0) + (np.nan_to_num(apart) / 2) ** 2
        bridged[start:stop] = True
    known = np.isfinite(pos[:, 0])
    vel = np.full((n, 2), np.nan)
    vel[seen] = smooth[seen, 2:4]
    if bridged.any():
        grad = np.gradient(np.where(known[:, None], pos, 0.0), axis=0) * FPS
        vel[bridged] = grad[bridged]
    return pos, vel, np.sqrt(var), bridged
