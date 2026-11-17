"""Rule 44 (docs/VALUES.md): which picks belong to the ball, by how they move in each camera's image.

An air ball's ground point runs past the ball, away from the camera, at up to twice its speed (1H 85036-85126:
an apparent 33 m/s for a 16-18 m/s ball), so on the pitch a filter coasting on it rejects the real picks. In the
image the ball and its ground point are on one ray, and the ray moves at the ball's own pace.

A pick's image position in frame f is H_f . p, its pitch point p through that frame's pitch -> pixel homography.
A pick of frame f - k compared in frame f's image is H_f . H_(f-k)^-1 . u_(f-k) = H_f . p_(f-k): the pan and
zoom compensation, exact for a camera that turns and zooms about a fixed point. Offsets are divided by the
frame's focal length (cameras.recover), so they are angles and one limit holds at any zoom. Continuity resets at
every broadcast cut (shots.cuts).

Per camera, forward and backward, each pick is tested against that camera's image track: a constant-acceleration
fit (short ballistic) to its last IMG_SPAN frames of members, constant velocity with fewer than IMG_CA_MIN.
verdicts() labels each pick MEMBER, OTHER, SWITCH (a run of OTHERs became the track) or UNKNOWN; the two-camera
filter (kalman._run2) decides with them, and ball_track.associate uses the same angles for its links.
"""

import numpy as np

import data
import kalman
from data import FPS

# Params (docs/VALUES.md, rule 44)
IMG_NOISE_PX = 3.0          # px: a pick's image position (box centre and homography)
IMG_ACCEL = 20.0            # m/s^2: gravity plus drag at 30 m/s; kicks are outside it
IMG_AIR_SLACK = 2.0         # a ball is up to this many times closer to the camera than its ground point
IMG_SPAN = FPS              # frames of members the fit uses, back from the last
IMG_CA_MIN, IMG_CA_FRAMES = 5, 8   # members, and frames they span, for the constant-acceleration fit
V_GAP_IMG = 35.0            # m/s: one member and no velocity yet -- ball_track.V_GAP, the real ball's pace
V_KICK = 70.0               # m/s: a kick's reach from the last member -- ball_track.V_MAX
YOUNG = 3                   # a track with fewer members has no say yet: a failed test replaces it
UNKNOWN, MEMBER, OTHER, SWITCH = 0, kalman.IMG_MEMBER, kalman.IMG_OTHER, kalman.IMG_SWITCH


def angles(H, f, xy):
    """(k, 2) image positions over focal length (radians about the image origin) of pitch points xy (k, 2,
    metres on that camera's pitch) through one frame's pitch (0-1) -> pixel homography H."""
    L, W = data.PITCH
    q = np.column_stack([xy[:, 0] / L, xy[:, 1] / W, np.ones(len(xy))]) @ H.T
    with np.errstate(all="ignore"):
        return q[:, :2] / q[:, 2:3] / f


def eff_distance(xy, centre):
    """(k,) metres: camera centre to the ground points xy, over IMG_AIR_SLACK."""
    return np.sqrt(((xy - centre[:2]) ** 2).sum(axis=1) + centre[2] ** 2) / IMG_AIR_SLACK


def camera(source, centre, cuts=None):
    """One camera's image geometry on its own timeline: H (n, 3, 3), f (n,) px (NaN where none), shot (n,) --
    continuity holds only within one -- and centre (3,), its own pitch metres."""
    import cameras
    H = data.homography_matrices(source).astype(float)
    f = cameras.recover(H)["f"]
    n = len(H)
    shot = np.zeros(n, dtype=np.int64) if cuts is None else np.searchsorted(np.asarray(cuts), np.arange(n), side="right")
    return {"H": H, "f": np.where(np.isfinite(H).all(axis=(1, 2)), f, np.nan), "shot": shot, "centre": np.asarray(centre, float)}


def context(al, bro_n, main, c_t, c_b_own):
    """Both cameras' geometry and the tactical -> broadcast frame map (-1: none) for verdicts()."""
    import align
    import shots
    off = al["offset_b"]
    tac_of_b = np.arange(bro_n) + off.astype(np.int64)
    ok = main & (off != align.MISSING)
    cam_t = camera("tactical", c_t)
    n = len(cam_t["H"])
    b_of_t = np.full(n, -1, dtype=np.int64)
    idx = np.flatnonzero(ok & (tac_of_b >= 0) & (tac_of_b < n))
    first = np.unique(tac_of_b[idx], return_index=True)[1]
    b_of_t[tac_of_b[idx[first]]] = idx[first]
    return {"t": cam_t, "b": camera("broadcast", c_b_own, shots.cuts()), "b_of_t": b_of_t,
            "flip": tuple(bool(v) for v in al["flip"])}


def _test(track, t, xy, H, f, centre):
    """(chi-square, used the constant-acceleration fit) of the pick xy at tactical frame t against a track's
    members [(t, xy)], all in frame t's image."""
    last_t = track[-1][0]
    pts = [m for m in track if abs(m[0] - last_t) <= IMG_SPAN]
    ang = angles(H, f, np.array([m[1] for m in pts] + [xy]))
    if not np.isfinite(ang).all():
        return np.inf, False
    tau = np.array([(m[0] - t) / FPS for m in pts])
    span = abs(pts[-1][0] - pts[0][0])
    deg = 2 if len(pts) >= IMG_CA_MIN and span >= IMG_CA_FRAMES else (1 if len(pts) >= 2 else 0)
    A = np.vander(tau, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(A, ang[:-1], rcond=None)
    lev = float(np.linalg.pinv(A.T @ A)[0, 0])
    d_eff = float(eff_distance(np.array([track[-1][1]]), centre)[0])
    gap = abs(t - last_t) / FPS
    s = (IMG_NOISE_PX / f) ** 2 * (1 + lev) + (IMG_ACCEL / d_eff * gap * gap / 2) ** 2
    if deg == 0:
        s += (V_GAP_IMG / d_eff * gap) ** 2
    r = ang[-1] - coef[0]
    return float(r @ r / s), deg == 2


def verdicts(xy, conf, cam, own, reverse=False):
    """(n,) int8 verdict per tactical frame for one camera's picks. xy (n, 2) on that camera's own pitch
    (NaN: none), conf (n,), cam: camera() of it, own (n,) its own frame for each tactical frame (-1: none)."""
    n = len(xy)
    out = np.zeros(n, dtype=np.int8)
    track, cand, seeded = [], [], False
    order = range(n - 1, -1, -1) if reverse else range(n)
    for t in order:
        if not np.isfinite(xy[t, 0]):
            continue
        fo = own[t]
        if fo < 0 or not np.isfinite(cam["f"][fo]):
            continue
        H, f, shot = cam["H"][fo], cam["f"][fo], cam["shot"][fo]
        member = (t, xy[t], float(conf[t]) if np.isfinite(conf[t]) else 0.0, shot)
        if track and (abs(t - track[-1][0]) > kalman.RESET_GAP or track[-1][3] != shot):
            track, cand = [], []
        if cand and (abs(t - cand[-1][0]) > kalman.RESYNC_BREAK or cand[-1][3] != shot):
            cand = []
        if not track:
            track = [member]
            continue
        chi, _ = _test(track, t, xy[t], H, f, cam["centre"])
        if chi <= kalman.GATE:
            track.append(member)
            cand = []
            out[t] = MEMBER if len(track) > YOUNG else UNKNOWN
            continue
        if len(track) < YOUNG:
            track, cand = [member], []
            continue
        if cand and _test(cand, t, xy[t], H, f, cam["centre"])[0] <= kalman.GATE:
            cand.append(member)
        else:
            # A kick: the ball leaves from where the track last had it, at any pace up to V_MAX. Such a run starts
            # from that member (seeded) and is confirmed by RESYNC picks; any other run by RESYNC_CONSISTENT.
            last = track[-1]
            a = angles(H, f, np.array([last[1], xy[t]]))
            reach = V_KICK * abs(t - last[0]) / FPS / eff_distance(np.array([last[1]]), cam["centre"])[0] + 2 * IMG_NOISE_PX / f
            seeded = bool(np.linalg.norm(a[1] - a[0]) <= reach)
            cand = [last, member] if seeded else [member]
        if seeded and len(cand) - 1 >= kalman.RESYNC or \
                len(cand) >= kalman.RESYNC_CONSISTENT and np.mean([m[2] for m in cand]) >= kalman.RESTART_MIN_CONF:
            track, cand = cand, []
            out[t] = SWITCH
        else:
            out[t] = OTHER
    return out


def match_verdicts(ctx, full_t, full_b, conf):
    """{"forward": (tactical, broadcast), "backward": (...)} verdicts on the tactical timeline for the
    match's picks (tactical pitch and timeline, as ball_track.finish has them)."""
    L, W = data.PITCH
    own_b = np.asarray(full_b, dtype=float).copy()
    if ctx["flip"][0]:
        own_b[:, 0] = L - own_b[:, 0]
    if ctx["flip"][1]:
        own_b[:, 1] = W - own_b[:, 1]
    n = len(full_t)
    own_t = np.arange(n)
    ct = np.asarray(conf[0], float) if conf is not None else np.ones(n)
    cb = np.asarray(conf[1], float) if conf is not None else np.ones(n)
    b_of_t = ctx["b_of_t"][:n]
    out = {}
    for name, rev in (("forward", False), ("backward", True)):
        out[name] = (verdicts(np.asarray(full_t, float), ct, ctx["t"], own_t, rev),
                     verdicts(own_b, cb, ctx["b"], b_of_t, rev))
    return out


def summary(v):
    """Counts of each verdict, for reports."""
    return {k: int((v == c).sum()) for k, c in (("member", MEMBER), ("other", OTHER), ("switch", SWITCH))}
