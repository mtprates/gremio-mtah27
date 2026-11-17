"""Rule 19: a team for every detected player, from the colour of their shirt.

    .venv/Scripts/python model/teams.py tactical 11185 3000
    .venv/Scripts/python model/teams.py broadcast 7171 3000

model/detect_ball.py saves, per player box, the shirt's colour: the median Lab of the torso's
non-grass pixels. Then:

  on the pitch    the feet (box bottom-centre) through the homography; boxes more than OFF_PITCH
                  outside the lines are bench, coaches, ball boys. Broadcast: main camera only.
  tracks          boxes linked frame to frame on the pitch (an optimal assignment, a gate that
                  widens while a player is unseen, up to MAX_GAP frames). A player who leaves the
                  view and comes back is a new track; that is fine for a team.
  kit groups      a Gaussian mixture on shirt colour, one point per track per second (so players
                  long on screen do not dominate), 4-6 groups compared by BIC. A mixture, not
                  k-means: the groups are unequal (two keepers, three officials, twenty players),
                  stretched (one kit in sun and in shade), and every box gets probabilities.
  names           from the ATD, only inside the view (its border taken to the pitch): matched
                  boxes vote home / away per group. A small group living near one goal is that
                  goal's keeper; a group the ATD rarely has a player under is the officials.
  other camera    each box's own-colour vote is shared with the other camera (votes-*.csv): a box
                  the other camera sees on the same spot at the same moment adds that camera's
                  vote to its own. Run both cameras, then the first one again.
  per player      each track takes the label its boxes vote for, weighted by the probabilities;
                  a track whose vote is split (two people under one track) is flagged. A track
                  voted mostly official stays official throughout (sticky), so an official does
                  not flicker into a team and push a real player off it at the team cap.

Writes data/.model/detections/teams-<source>-<first>-<count>.csv (frame, x, y, w, h, track, label,
split; label 0 home, 1 away, 2 home keeper, 3 away keeper, 4 official, -1 unknown) for the viewer.
Prints aggregates only.
"""

import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import data  # noqa: E402
from data import FPS  # noqa: E402

# Params
OFF_PITCH = 1.0             # m outside the lines: not a player on the pitch
SHIRT_MIN_PX = 6            # boxes with fewer shirt pixels give no colour
GATE0, GATE_GROW = 1.0, 0.4  # m: a track's gate, widening per frame unseen ...
MAX_GAP = 10                # ... for up to this many frames
TRACKER = "kalman"          # rule 21: "kalman" gates on the prediction's uncertainty; "average" is
                            # the moving-average velocity and fixed gate above
GATE_CHI2 = 13.8            # chi-square, 2 dof, p = 0.001
VEL0_PLAYER = 4.0           # m/s: a new track's velocity uncertainty
# Rule 22: colour in the matching, merged boxes, hidden players.
COLOUR_STD = 12.0           # Lab units (lightness at L_WEIGHT): colour's share of a match's cost ...
COLOUR_GATE = 35.0          # ... and no match beyond this -- the two outfield kits are ~40 apart
COLOUR_EMA = 0.1            # a track's running shirt colour
PLAYER_HEIGHT_M = 1.8       # m: a box's height, for its width on the pitch
MERGE_MARGIN = 0.7          # m: a track predicted within a box's half-width plus this is under it
MERGE_ASPECT = 0.85         # a box at least this wide for its height is two players
MERGE_MAX = 2 * FPS         # frames a track may stay merged or hidden before it is dropped
GROUPS = (4, 5, 6)          # numbers of kit groups compared by BIC
L_WEIGHT = 0.5              # lightness counts half: shade and distance move it, the hue (a, b)
                            # tells kits apart (an assistant referee 25 units darker than the
                            # referee left the officials' group at 1H 11187)
SPREAD_FLOOR = 16.0         # squared Lab units added to every group's variance: no group so tight
                            # that a small step outside it loses to a broad catch-all group
VOTE_WINDOW = FPS // 2      # frames either side: a box's label is the vote of its track's boxes
                            # this close in time. A whole-track vote carried one player's team
                            # onto another the tracker swapped in (1H 11187, track 11).
MATCH_M = 1.5               # m: a box and an ATD player this close are one person (naming only)
DUPLICATE = 0.3             # a box this much inside a bigger one of the same frame (of its own
                            # area) is the same person: a player cut by a detection tile's edge
                            # comes back whole from one tile and in part from the next
TEAM_MAX = 11               # players of one team in a frame, keeper included
SECOND_MIN = 0.2            # a box moved off a full team takes its next label only with this vote
SIDELINE_M = 2.5            # m: a box this close to a touchline (or beyond it) is moved off a full team
                            # first -- assistant referees stand there, players rarely stay
OFFICIAL_MATCH = 0.3        # a group whose boxes match an ATD player less often than this
KEEPER_SHARE = 0.12         # a group with less than this share of the boxes ...
KEEPER_GOAL_M = 25.0        # ... whose median box is this close to a goal line is that keeper
SPLIT = 0.7                 # a track whose winning label has less of its vote is flagged
# Keepers by position, not colour (a keeper's kit is unlike the team's): one per side.
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16  # m: the penalty area
KEEPER_DEPTH = 30.0         # m from the goal line (rules.GK_RANGE), BOX_HALF_WIDTH wide: where the deepest
                            # person is that goal's keeper -- beyond the penalty area, since a keeper
                            # who comes out alone is still him (1H 18871, 22.6 m; docs/VALUES.md)
OFFICIAL_TRACK = 0.5        # a track whose whole vote is at least this share official is an official
OFFICIAL_LEAVE = 0.15       # ... throughout, except boxes whose own window gives official less than this
                            # (1H 12874, 12920: officials flickered to a team label box by box, and
                            # each time the team cap pushed a real player off it)
PAIR_M = 1.5                # m: a box and the other camera's box this close, same moment, are one person
OTHER_WEIGHT = 1.0          # the other camera's own-colour vote counts as much as this camera's
KEEPER_MIN = 25             # frames: a track that is its goal's deepest visible player at least
KEEPER_SHARE_DEEP = 0.6     # this long, for this share of its frames in the box, is that keeper
LABELS = ("home", "away", "home keeper", "away keeper", "official")


def load(source, first, count):
    """Rows of persons-<source>-<first>-<count>.csv as arrays: frame, box (x, y, w, h), Lab
    (NaN where too few shirt pixels)."""
    path = os.path.join(data.CACHE_DIR, "detections", "persons-{}-{}-{}.csv".format(source, first, count))
    frames, boxes, lab = [], [], []
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            frames.append(int(r["frame"]))
            boxes.append([float(r[k]) for k in ("x", "y", "w", "h")])
            ok = r.get("L") not in (None, "") and int(r.get("shirt_px") or 0) >= SHIRT_MIN_PX
            lab.append([float(r[k]) for k in ("L", "a", "b")] if ok else [np.nan] * 3)
    return np.array(frames), np.array(boxes), np.array(lab)


def to_pitch(H, frames, px):
    """Pixels -> pitch metres through each frame's homography (normalised pitch -> pixels)."""
    out = np.full((len(frames), 2), np.nan)
    for i, (f, (u, v)) in enumerate(zip(frames, px)):
        h = H[f]
        if not np.isnan(h).any():
            p = np.linalg.solve(h, np.array([u, v, 1.0]))
            out[i] = p[0] / p[2] * data.PITCH[0], p[1] / p[2] * data.PITCH[1]
    return out


def view(H, f, size=(1920, 1080), steps=12):
    """The image border taken to the pitch (metres), a polygon; None without a homography."""
    h = H[f]
    if np.isnan(h).any():
        return None
    w, hh = size
    border = [(w * i / steps, 0) for i in range(steps)] + [(w, hh * i / steps) for i in range(steps)] + \
             [(w * (1 - i / steps), hh) for i in range(steps)] + [(0, hh * (1 - i / steps)) for i in range(steps)]
    pts = []
    inv = np.linalg.inv(h)
    for u, v in border:
        p = inv @ np.array([u, v, 1.0])
        if p[2] > 0:
            pts.append((p[0] / p[2] * data.PITCH[0], p[1] / p[2] * data.PITCH[1]))
    return np.array(pts) if len(pts) >= 3 else None


def inside(poly, pts):
    """(k,) whether each point lies inside the polygon (ray casting)."""
    x, y = pts[:, 0], pts[:, 1]
    out = np.zeros(len(pts), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        out ^= cross
        j = i
    return out


def _overlap(a, b):
    """Intersection over the smaller box's area, for (cx, cy, w, h) boxes."""
    ix = min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2)
    iy = min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / max(1e-9, min(a[2] * a[3], b[2] * b[3]))


def track(frames, feet, heights=None):
    """(k,) track id per box: frame-to-frame optimal assignment on the pitch. With TRACKER "kalman"
    (and the boxes' heights, for each foot position's noise): a constant-velocity Kalman filter per
    track predicts it to the frame, and a box joins it only within GATE_CHI2 of that prediction's
    uncertainty plus its own (rule 21); else the moving-average version below."""
    if TRACKER == "kalman" and heights is not None:
        return _track_kalman(frames, feet, heights)[0]
    from scipy.optimize import linear_sum_assignment
    ids = np.full(len(frames), -1)
    active = []                 # [id, last frame, position, velocity]
    next_id = 0
    for f in np.unique(frames):
        rows = np.flatnonzero(frames == f)
        active = [a for a in active if f - a[1] <= MAX_GAP]
        if active and len(rows):
            pred = np.array([a[2] + a[3] * (f - a[1]) for a in active])
            gate = np.array([GATE0 + GATE_GROW * (f - a[1] - 1) for a in active])
            d = np.linalg.norm(pred[:, None] - feet[rows][None], axis=2)
            cost = np.where(d <= gate[:, None], d, 1e6)
            ai, ri = linear_sum_assignment(cost)
            for a, r in zip(ai, ri):
                if cost[a, r] < 1e6:
                    trk = active[a]
                    gap = f - trk[1]
                    vel = (feet[rows[r]] - trk[2]) / gap
                    trk[3] = 0.5 * trk[3] + 0.5 * vel
                    trk[1], trk[2] = f, feet[rows[r]]
                    ids[rows[r]] = trk[0]
        for r in rows:
            if ids[r] < 0:
                ids[r] = next_id
                active.append([next_id, f, feet[r], np.zeros(2)])
                next_id += 1
    return ids


def _track_kalman(frames, feet, heights, widths=None, lab=None):
    """Rule 21/22: returns (ids per box, merged per box, extra rows [(frame, id, position, kind)]).

    Matching uses shirt colour as well as position: each track keeps a running colour, a box of
    another kit (COLOUR_GATE) cannot continue it, and colour adds to the cost (COLOUR_STD). A box
    wider than a player (MERGE_ASPECT) may be two, so its colour is no evidence either way. After
    the matching, a track missed while its prediction lies under a matched box is hidden behind
    that player -- or, if the box is wide, merged into it (1H 17310: a home and an away player
    in one 67 x 49 box); it stays alive at its prediction for up to MERGE_MAX and is
    re-identified by colour when the two part. A track missed otherwise coasts (up to MAX_GAP).
    A track the matching missed is then offered the boxes left over, from its last real observation
    within a sprint's reach (OC-SORT's recovery), and re-found after a gap it restarts from the two
    real observations either side of it.
    Matching first and calling a box merged only for a track left over keeps two players close
    together, each with his own box, from being read as one merge. Extra rows: kind 1 merged or
    hidden, kind 2 coasting -- predictions, not measurements."""
    from scipy.optimize import linear_sum_assignment
    import detector_model
    import kalman
    dt = 1.0 / FPS
    q00, q01, q11 = kalman._q(kalman.PLAYER_ACCEL)
    r_all = np.clip(detector_model.FEET_STD * detector_model.FEET_REF_PX / heights,
                    detector_model.FEET_STD_MIN, detector_model.FEET_STD_MAX) ** 2
    widths = np.full(len(frames), 0.5) * heights if widths is None else widths
    col = np.full((len(frames), 3), np.nan) if lab is None else lab * [L_WEIGHT, 1, 1]
    half_m = 0.5 * widths * PLAYER_HEIGHT_M / heights        # a box's half-width on the pitch, m
    ids = np.full(len(frames), -1)
    merged = np.zeros(len(frames), dtype=bool)
    extra = []
    active = []     # dicts: id, last frame, pos (2,), vel (2,), p00, p01, p11, colour, hidden_since
    next_id = 0

    wide = widths / heights >= MERGE_ASPECT    # maybe two players: its colour is no evidence

    def colour_cost(trk, r):
        if trk["colour"] is None or not np.isfinite(col[r, 0]) or wide[r]:
            return 0.0, True
        d = np.linalg.norm(col[r] - trk["colour"])
        return (d / COLOUR_STD) ** 2, d <= COLOUR_GATE

    def update(trk, r, pred):
        g = f - trk["f"]
        if g > 1:
            # Re-found after a gap (OC-SORT): the prediction ran on at the old velocity, so start
            # again from the two real observations either side of the gap.
            trk["vel"] = (feet[r] - trk["obs"]) / (g * dt)
            trk["pos"] = feet[r].copy()
            trk["p00"], trk["p01"], trk["p11"] = r_all[r], 0.0, 2 * r_all[r] / (g * dt) ** 2
        else:
            p, v, p00, p01, p11 = pred
            s = p00 + r_all[r]
            k0, k1 = p00 / s, p01 / s
            e = feet[r] - p
            trk["pos"], trk["vel"] = p + k0 * e, v + k1 * e
            trk["p11"] = p11 - k1 * p01
            trk["p01"] = p01 * (1 - k0)
            trk["p00"] = p00 * (1 - k0)
        trk["obs"] = feet[r].copy()
        trk["f"] = f
        trk["hidden"] = None
        if np.isfinite(col[r, 0]) and not wide[r]:
            trk["colour"] = col[r].copy() if trk["colour"] is None else \
                (1 - COLOUR_EMA) * trk["colour"] + COLOUR_EMA * col[r]
        ids[r] = trk["id"]

    def predicted(trk, f):
        g = f - trk["f"]
        p00, p01, p11 = trk["p00"], trk["p01"], trk["p11"]
        for _ in range(g):
            p00 += 2 * dt * p01 + dt * dt * p11 + q00
            p01 += dt * p11 + q01
            p11 += q11
        return trk["pos"] + trk["vel"] * g * dt, trk["vel"].copy(), p00, p01, p11

    for f in np.unique(frames):
        rows = np.flatnonzero(frames == f)
        # Alive: seen within MAX_GAP, or hidden / merged (under another box) for under MERGE_MAX.
        active = [a for a in active if f - a["f"] <= MAX_GAP or
                  (a["hidden"] is not None and a["hidden"] == f - 1 and f - a["f"] <= MERGE_MAX)]
        preds = [predicted(a, f) for a in active]
        pos = np.array([p[0] for p in preds]).reshape(-1, 2)
        near = np.linalg.norm(pos[:, None] - feet[rows][None], axis=2) <= \
            (half_m[rows][None] + MERGE_MARGIN) if len(active) else np.zeros((0, len(rows)), bool)
        if len(active) and len(rows):
            var = np.array([p[2] for p in preds])
            d2 = ((pos[:, None] - feet[rows][None]) ** 2).sum(axis=2) / (var[:, None] + r_all[rows][None])
            cost = np.full(d2.shape, 1e6)
            for i, trk in enumerate(active):
                for j, r in enumerate(rows):
                    cc, ok = colour_cost(trk, r)
                    if d2[i, j] <= GATE_CHI2 and ok:
                        cost[i, j] = d2[i, j] + cc
            ai, ri = linear_sum_assignment(cost)
            for i, j in zip(ai, ri):
                if cost[i, j] < 1e6:
                    update(active[i], rows[j], preds[i])
        # Recovery (OC-SORT): a track the matching missed is offered the boxes left over, from its
        # last real observation -- within reach of a sprint since then -- not from its prediction,
        # which ran on at the old velocity (broadcast 14859: the keeper's box missed the gate, started
        # a new track and left the old one "hidden" under it, a ghost with the right label).
        lost = [i for i, trk in enumerate(active) if trk["f"] != f]
        free = [j for j in range(len(rows)) if ids[rows[j]] < 0]
        if lost and free:
            cost = np.full((len(lost), len(free)), 1e6)
            for a, i in enumerate(lost):
                trk = active[i]
                reach = detector_model.RELINK_M + detector_model.RELINK_SPEED * (f - trk["f"]) * dt
                for b, j in enumerate(free):
                    dist = np.linalg.norm(feet[rows[j]] - trk["obs"])
                    cc, ok = colour_cost(trk, rows[j])
                    if dist <= reach and ok:
                        cost[a, b] = (dist / reach) ** 2 + cc
            ai, bi = linear_sum_assignment(cost)
            for a, b in zip(ai, bi):
                if cost[a, b] < 1e6:
                    update(active[lost[a]], rows[free[b]], preds[lost[a]])
        # Missed tracks: under a matched box (merged into it if it is wide, else hidden behind
        # that player), or coasting.
        for a, trk in enumerate(active):
            if trk["f"] == f:
                continue
            under = [j for j in range(len(rows)) if ids[rows[j]] >= 0 and near[a, j]]
            if under:
                trk["hidden"] = f
                extra.append((f, trk["id"], pos[a].copy(), 1))
                for j in under:
                    merged[rows[j]] |= widths[rows[j]] / heights[rows[j]] >= MERGE_ASPECT
            else:
                extra.append((f, trk["id"], pos[a].copy(), 2))
        for r in rows:
            if ids[r] < 0:
                ids[r] = next_id
                active.append({"id": next_id, "f": f, "pos": feet[r].copy(), "obs": feet[r].copy(), "vel": np.zeros(2),
                               "p00": r_all[r], "p01": 0.0, "p11": VEL0_PLAYER ** 2, "hidden": None,
                               "colour": col[r].copy() if np.isfinite(col[r, 0]) else None})
                next_id += 1
    # Extra rows only inside a track's life: drop those after its last box (it left the view).
    last = {}
    for r in range(len(frames)):
        last[ids[r]] = max(last.get(ids[r], -1), frames[r])
    extra = [e for e in extra if e[0] < last.get(e[1], -1)]
    return ids, merged, extra


def track_votes(ids, frames, label_prob):
    """Each box's label: the vote of its track's boxes within VOTE_WINDOW frames. Returns the
    labels, each box's vote as shares, each track's whole-vote label, and whether its whole vote is
    split (two people under one track, somewhere)."""
    final = np.full(len(frames), -1)
    share = np.zeros((len(frames), len(LABELS)))
    track_label, split = {}, {}
    for tid in np.unique(ids):
        rows = np.flatnonzero(ids == tid)
        rows = rows[np.argsort(frames[rows])]
        f = frames[rows]
        votes = label_prob[rows]
        cum = np.vstack([np.zeros(len(LABELS)), np.cumsum(votes, axis=0)])
        lo = np.searchsorted(f, f - VOTE_WINDOW, side="left")
        hi = np.searchsorted(f, f + VOTE_WINDOW, side="right")
        local = cum[hi] - cum[lo]
        final[rows] = np.where(local.sum(axis=1) > 0, local.argmax(axis=1), -1)
        share[rows] = local / np.maximum(local.sum(axis=1, keepdims=True), 1e-9)
        total = votes.sum(axis=0)
        track_label[tid] = int(np.argmax(total)) if total.sum() > 0 else -1
        split[tid] = total.sum() > 0 and total.max() / total.sum() < SPLIT
    return final, share, track_label, split


def official_flips(label, ids, frames):
    """How many times a box's label turns official or stops being official along its track."""
    n = 0
    for tid in np.unique(ids):
        rows = np.flatnonzero(ids == tid)
        off = label[rows[np.argsort(frames[rows])]] == 4
        n += int(np.sum(off[1:] != off[:-1]))
    return n


def keeper_by_goal(label_prob, feet_t, half_t, sides):
    """label_prob with each box's keeper vote (home + away keeper) given to the keeper of the goal
    the box is nearer (feet on the tactical pitch, that half's defending sides). A keeper label is
    "the keeper of this goal": one colour group can hold both keepers' kits (broadcast 7171, group
    0: 45% home), and naming it after one goal made the away keeper "home keeper" at the away goal
    (1H 02:57, 15621-15738), through the cross-camera votes too. Boxes outside the halves keep theirs."""
    out = label_prob.copy()
    keeper = out[:, 2] + out[:, 3]
    left = feet_t[:, 0] < data.PITCH[0] / 2
    for h in (1, 2):
        for team, goal_x in sides[h].items():
            rows = (half_t == h) & (left == (goal_x < data.PITCH[0] / 2)) & np.isfinite(feet_t[:, 0])
            out[rows, 2], out[rows, 3] = 0.0, 0.0
            out[rows, 2 + team] = keeper[rows]
    return out


def cross_camera(source, first, count, t_frames, feet_t, label_prob, assign):
    """The other camera's context. Writes this camera's own-colour vote per box (votes-<source>-
    <first>-<count>.csv: tactical frame, feet on the tactical pitch, a share per label) and adds the
    other camera's own-colour votes, OTHER_WEIGHT each, to the boxes they pair with: same tactical
    frame, within PAIR_M, one-to-one. Only own-colour votes cross over -- never a camera's final
    labels -- so the result does not depend on which camera ran first, once both have run once.
    (1H 12874: the broadcast, closer, saw the referee the tactical camera did not.)"""
    folder = os.path.join(data.CACHE_DIR, "detections")
    with open(os.path.join(folder, "votes-{}-{}-{}.csv".format(source, first, count)), "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["t_frame", "x", "y"] + ["p{}".format(l) for l in range(len(LABELS))])
        for t, p, v in zip(t_frames, feet_t, label_prob):
            if t >= 0 and v.sum() > 0:
                out.writerow([int(t), round(float(p[0]), 2), round(float(p[1]), 2)] + [round(float(x), 3) for x in v])

    other = "broadcast" if source == "tactical" else "tactical"
    if not (t_frames >= 0).any():
        print("  other camera ({}): no frames of this window pair with it".format(other))
        return label_prob
    lo, hi = int(t_frames[t_frames >= 0].min()), int(t_frames.max()) + 1
    theirs = {}
    for path in glob.glob(os.path.join(folder, "votes-{}-*.csv".format(other))):
        for r in csv.DictReader(open(path, newline="")):
            t = int(r["t_frame"])
            if lo <= t < hi:
                theirs.setdefault(t, []).append([float(r["x"]), float(r["y"])] +
                                                [float(r["p{}".format(l)]) for l in range(len(LABELS))])
    if not theirs:
        print("  other camera ({}): no votes for these frames -- run teams.py on it, then this again".format(other))
        return label_prob
    out = label_prob.copy()
    paired = 0
    for t in np.unique(t_frames[t_frames >= 0]):
        if t not in theirs:
            continue
        rows = np.flatnonzero(t_frames == t)
        o = np.array(theirs[t])
        d = np.linalg.norm(feet_t[rows][:, None] - o[None, :, :2], axis=2)
        d = np.where(np.isfinite(d), d, 1e6)
        ri, oi = assign(np.where(d <= PAIR_M, d, 1e6))
        for r, j in zip(ri, oi):
            if d[r, j] <= PAIR_M:
                out[rows[r]] += OTHER_WEIGHT * o[j, 2:]
                paired += 1
    print("  other camera ({}): {} boxes paired ({:.0%} of this camera's)".format(other, paired, paired / len(t_frames)))
    return out


def main():
    global TRACKER
    from scipy.optimize import linear_sum_assignment
    from sklearn.mixture import GaussianMixture
    import align
    import cameras
    import rules
    import run
    source, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if len(sys.argv) > 4:
        TRACKER = sys.argv[4]       # "average" for the moving-average tracker, to compare
    print("tracker: {}".format(TRACKER))
    frames, boxes, lab = load(source, first, count)
    H = data.homography_matrices(source)
    rec = data.load(source)
    tac = data.load("tactical")
    al = run.load_alignment(tac, rec if source == "broadcast" else data.load("broadcast"))

    feet = to_pitch(H, frames, np.column_stack([boxes[:, 0], boxes[:, 1] + boxes[:, 3] / 2]))
    import ball_track
    keep = np.isfinite(feet[:, 0]) & (ball_track.outside(np.nan_to_num(feet, nan=1e9)) <= OFF_PITCH)
    if source == "broadcast":
        # Every frame with a homography outside a cut-away shot (model/cameras.py cuts: image cuts,
        # then each shot by its people's height under its homography), replays included: their
        # players are real and in their real kits.
        keep &= cameras.broadcast_usable()[frames]
    print("\n{} frames {}-{}: {} player boxes, {:.0%} on the pitch{}; {:.0%} of those with a shirt colour".format(
        source, first, first + count - 1, len(frames), keep.mean(), " outside cut-aways" if source == "broadcast" else "",
        np.isfinite(lab[keep, 0]).mean()))
    frames, boxes, lab, feet = frames[keep], boxes[keep], lab[keep], feet[keep]
    # Duplicates: per frame, a box mostly inside a bigger one goes.
    dup = np.zeros(len(frames), dtype=bool)
    for f in np.unique(frames):
        rows = np.flatnonzero(frames == f)
        rows = rows[np.argsort(-boxes[rows, 2] * boxes[rows, 3])]
        for i, a in enumerate(rows):
            for b in rows[:i]:
                if not dup[b] and _overlap(boxes[a], boxes[b]) > DUPLICATE:
                    dup[a] = True
                    break
    print("  duplicate boxes dropped: {} ({:.1%})".format(int(dup.sum()), dup.mean()))
    frames, boxes, lab, feet = frames[~dup], boxes[~dup], lab[~dup], feet[~dup]
    if TRACKER == "kalman":
        ids, merged, extra = _track_kalman(frames, feet, boxes[:, 3], boxes[:, 2], lab)
    else:
        ids, merged, extra = track(frames, feet, boxes[:, 3]), np.zeros(len(frames), bool), []
    lens = np.bincount(ids)
    print("  tracks: {}; length median {:.1f} s, 90th {:.1f} s".format(
        len(lens), float(np.median(lens)) / FPS, float(np.percentile(lens, 90)) / FPS))
    print("  merged boxes (two players): {}; predicted rows: {} merged or hidden, {} coasting".format(
        int(merged.sum()), sum(e[3] == 1 for e in extra), sum(e[3] == 2 for e in extra)))

    # One point per track per second for the fit. A merged box's shirt is two players': no vote.
    colour = np.isfinite(lab[:, 0]) & ~merged
    pts = []
    for tid in np.unique(ids):
        rows = np.flatnonzero((ids == tid) & colour)
        for sec in np.unique(frames[rows] // FPS):
            r = rows[frames[rows] // FPS == sec]
            if len(r) >= 5:
                pts.append(np.median(lab[r], axis=0) * [L_WEIGHT, 1, 1])
    pts = np.array(pts)
    fits = {k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=0,
                               reg_covar=SPREAD_FLOOR).fit(pts) for k in GROUPS}
    bic = {k: g.bic(pts) for k, g in fits.items()}
    k = min(bic, key=bic.get)
    gmm = fits[k]
    print("  kit groups: {} (BIC {}) from {} track-seconds".format(
        k, ", ".join("{}: {:.0f}".format(kk, v) for kk, v in sorted(bic.items())), len(pts)))
    prob = np.zeros((len(frames), k))
    prob[colour] = gmm.predict_proba(lab[colour] * [L_WEIGHT, 1, 1])

    # Name the groups from the ATD, inside the view.
    atd_team = rec.team
    sides = rules.defending_sides(tac)
    match_team = np.full(len(frames), -1)
    for f in np.unique(frames):
        rows = np.flatnonzero(frames == f)
        poly = view(H, f)
        if poly is None:
            continue
        atd = rec.players[f]
        ok = np.isfinite(atd[:, 0])
        ok[ok] = inside(poly, atd[ok])
        if not ok.any():
            continue
        slots = np.flatnonzero(ok)
        d = np.linalg.norm(feet[rows][:, None] - atd[slots][None], axis=2)
        ri, si = linear_sum_assignment(np.where(d <= MATCH_M, d, 1e6))
        for r, s in zip(ri, si):
            if d[r, s] <= MATCH_M:
                match_team[rows[r]] = atd_team[slots[s]]
    # Each box's tactical frame, and its half from the graded halves (data.halves()), not from the
    # kick-off: the second half's range starts ~2 min before its kick-off, with the teams already
    # on their new ends (2H 110808-113741 counted as first half named an away defender in the left
    # box the home keeper, for his whole 8-minute track).
    if source == "tactical":
        t_frames = frames
    else:
        off = al["offset_b"][frames].astype(np.int64)
        t_frames = np.where(al["offset_b"][frames] != align.MISSING, frames + off, -1)
    half_t = rules.half_of_frames(len(frames), t_frames)
    half = np.where(half_t > 0, half_t, 1)
    names = []
    mass = prob.sum(axis=0)
    print("  group  boxes  Lab (median)        ATD match  ATD home  near a goal  ->  name")
    for c in range(k):
        w = prob[:, c]
        matched = match_team >= 0
        match_rate = w[matched].sum() / max(w.sum(), 1e-9)
        home = (w * (match_team == 0)).sum() / max(w[matched].sum(), 1e-9)
        # Distance of the group's boxes to the nearer goal line, and which goal.
        dx = np.minimum(feet[:, 0], data.PITCH[0] - feet[:, 0])
        med_goal = float(np.median(dx[w > 0.5])) if (w > 0.5).any() else np.inf
        share = mass[c] / mass.sum()
        if match_rate < OFFICIAL_MATCH:
            name = 4
        elif share < KEEPER_SHARE and med_goal <= KEEPER_GOAL_M:
            sel = w > 0.5
            left = np.median(feet[sel, 0]) < data.PITCH[0] / 2
            goal_x = 0.0 if left else data.PITCH[0]
            h0 = int(np.bincount(half[sel]).argmax()) if sel.any() else 1
            defender = [t for t, gx in sides[h0].items() if gx == goal_x]
            name = 2 + (defender[0] if defender else (0 if home >= 0.5 else 1))
        else:
            name = 0 if home >= 0.5 else 1
        names.append(name)
        print("  {:>5}  {:>5.0%}  {:>4.0f} {:>4.0f} {:>4.0f}      {:>6.0%}     {:>5.0%}   {:>6.1f} m   ->  {}".format(
            c, share, *gmm.means_[c], match_rate, home, med_goal, LABELS[name]))
    names = np.array(names)

    # Per track: the label its boxes vote for.
    label_prob = np.zeros((len(frames), len(LABELS)))
    for c in range(k):
        label_prob[:, names[c]] += prob[:, c]

    # Each box's tactical frame and its feet on the tactical pitch: the other camera's boxes are
    # paired on these.
    if source == "tactical":
        feet_t = feet
    else:
        flip = tuple(bool(v) for v in al["flip"])
        feet_t = np.column_stack([data.PITCH[0] - feet[:, 0] if flip[0] else feet[:, 0],
                                  data.PITCH[1] - feet[:, 1] if flip[1] else feet[:, 1]])
    label_prob = keeper_by_goal(label_prob, feet_t, half_t, sides)
    own_prob = label_prob
    # Only live broadcast frames pair with the other camera: a replay's players stood there at
    # another moment.
    t_pair = t_frames if source == "tactical" else np.where(cameras.broadcast_live(al, rec.n)[frames], t_frames, -1)
    label_prob = cross_camera(source, first, count, t_pair, feet_t, label_prob, linear_sum_assignment)
    label_prob = keeper_by_goal(label_prob, feet_t, half_t, sides)
    final, share, track_label, split = track_votes(ids, frames, label_prob)
    flagged = np.array([split[t] for t in ids])
    # The keepers are found on this camera's own colour: the other camera's vote calls a keeper
    # his team's outfield kit (the broadcast has no keeper group), and a track whose shirt says
    # outfield is taken for a defender -- with it, a spot check lost 4 of its 5 keeper tracks.
    own_final = track_votes(ids, frames, own_prob)[0]

    # Keepers: per goal, the deepest visible person within KEEPER_DEPTH of it (the penalty area's
    # width); a track that is that most of its time there is the defending team's keeper, the whole track; one per
    # side at a time. Sides from the kick-off (rules.defending_sides), per half.
    keepers, kit_rejected = [], []
    colour_label = own_final        # by this camera's shirt alone, before any keeper is named
    for h0 in (1, 2):
        for team, goal_x in sides[h0].items():
            depth = np.abs(feet[:, 0] - goal_x)
            # Officials' boxes included: a keeper's kit can fall in an officials' colour group (2H
            # 110808: both keepers voted official or unknown, never keeper). The referee is rarely
            # the deepest in the box with the keeper in view; the assistants stand wider than it.
            in_box = (half_t == h0) & (depth <= KEEPER_DEPTH) & (np.abs(feet[:, 1] - data.PITCH[1] / 2) <= BOX_HALF_WIDTH)
            deepest = np.zeros(len(frames), dtype=bool)
            for f in np.unique(frames[in_box]):
                rows = np.flatnonzero(in_box & (frames == f))
                deepest[rows[np.argmin(depth[rows])]] = True
            cands = []
            for tid in np.unique(ids[deepest]):
                rows = ids == tid
                n_deep, n_in = int((deepest & rows).sum()), int((in_box & rows).sum())
                if n_deep < KEEPER_MIN or n_deep < KEEPER_SHARE_DEEP * n_in:
                    continue
                # A keeper's kit is unlike his own team's: a track whose shirt says that team's
                # outfield kit is a defender who was the deepest one in view -- on the zoomed
                # broadcast, with the keeper out of view, most of them (red home defenders "kept"
                # the home goal there). Broadcast only: the tactical view has the keeper in it,
                # and a keeper's kit can fall in his team's outfield group (1H 04:04, track 423:
                # deepest in the home box on all its 817 frames there, on the goal line, dropped as
                # a "defender"). An official's kit is no reason to drop him (above).
                own = np.bincount(colour_label[rows & (colour_label >= 0)], minlength=len(LABELS))
                if source == "broadcast" and own.sum() and own.argmax() == team:
                    kit_rejected.append((h0, team, tid, n_deep))
                    continue
                cands.append((n_deep, tid))
            taken = set()
            for n_deep, tid in sorted(cands, reverse=True):
                fr = set(frames[ids == tid].tolist())
                if fr & taken:
                    continue        # one keeper per side at a time
                taken |= fr
                keepers.append((h0, team, tid, n_deep))
                final[ids == tid] = 2 + team
    for h0, team, tid, n_deep in keepers:
        rows = ids == tid
        print("  keeper: {} (half {}), track {}: {:.1f} s, deepest in the box {:.1f} s; shirt Lab {:.0f} {:+.0f} {:+.0f}".format(
            LABELS[2 + team], h0, tid, rows.sum() / FPS, n_deep / FPS,
            *(np.nanmedian(lab[rows], axis=0) - [0, 128, 128])))
    if kit_rejected:
        print("  not keepers (shirt says their team's outfield kit): {} tracks, {:.0f} s deepest in a box in all".format(
            len(kit_rejected), sum(k[3] for k in kit_rejected) / FPS))
    if not keepers:
        print("  keepers: none in the window (no track deepest in a penalty area for {:.0f} s)".format(KEEPER_MIN / FPS))

    # Sticky officials: a track mostly voted official is an official throughout, but for boxes
    # whose own window clearly says otherwise (someone else under the track for a while). After
    # the keepers, who are found by position and are left alone: a keeper's kit is unlike his
    # team's and can vote official (sticky first lost every keeper in a spot check).
    windowed = final.copy()
    keeper_tracks = {tid for _, _, tid, _ in keepers}
    sticky = 0
    for tid in np.unique(ids):
        if tid in keeper_tracks:
            continue
        rows = np.flatnonzero(ids == tid)
        total = label_prob[rows].sum(axis=0)
        if total.sum() > 0 and total[4] / total.sum() >= OFFICIAL_TRACK:
            keep_off = rows[share[rows, 4] >= OFFICIAL_LEAVE]
            final[keep_off] = 4
            sticky += 1
    print("  officials: {} tracks sticky; official label changes along tracks {} -> {}; boxes changed {}".format(
        sticky, official_flips(windowed, ids, frames), official_flips(final, ids, frames),
        int((final != windowed).sum())))
    frame_label = np.where(colour, label_prob.argmax(axis=1), -1)

    # At most TEAM_MAX of a team per frame: boxes on a touchline move first (an assistant referee
    # lives there; 1H 12874, 12920: a linesman in a darker kit voted home and pushed a real home
    # player off), then the boxes least sure of the team. A moved box takes its next label (an
    # official or the other team) if its vote gives it SECOND_MIN, else unknown.
    on_line = np.minimum(feet[:, 1], data.PITCH[1] - feet[:, 1]) <= SIDELINE_M
    moved, moved_line = 0, 0
    for f in np.unique(frames):
        rows = np.flatnonzero(frames == f)
        for members in ((0, 2), (1, 3)):
            mine = rows[np.isin(final[rows], members)]
            excess = len(mine) - TEAM_MAX
            if excess <= 0:
                continue
            sure = share[mine, final[mine]]
            for r in mine[np.lexsort((sure, ~on_line[mine]))][:excess]:
                alt = share[r].copy()
                alt[list(members)] = 0
                final[r] = int(alt.argmax()) if alt.max() >= SECOND_MIN else -1
                moved += 1
                moved_line += int(on_line[r])
    print("  boxes moved off a team over {} in a frame: {} ({} from a touchline)".format(TEAM_MAX, moved, moved_line))

    team_of = {0: 0, 1: 1, 2: 0, 3: 1}
    def agree(lbl):
        sel = (match_team >= 0) & np.isin(lbl, [0, 1, 2, 3])
        return np.mean([team_of[l] == m for l, m in zip(lbl[sel], match_team[sel])]) if sel.any() else np.nan, int(sel.sum())
    a_trk, n_trk = agree(final)
    a_frm, n_frm = agree(frame_label)
    n_tracks = len(track_label)
    print("  team vs ATD on matched boxes: per track {:.1%} ({} boxes), per frame without the vote {:.1%}".format(a_trk, n_trk, a_frm))
    print("  labels (boxes): " + ", ".join("{} {:.0%}".format(LABELS[l], np.mean(final == l)) for l in range(len(LABELS))) +
          ", unknown {:.0%}".format(np.mean(final < 0)))
    print("  tracks with a split vote: {} of {} ({:.0%} of boxes)".format(sum(split.values()), n_tracks, flagged.mean()))
    per_frame = [np.sum((frames == f) & np.isin(final, [0, 2])) for f in np.unique(frames)]
    per_frame_a = [np.sum((frames == f) & np.isin(final, [1, 3])) for f in np.unique(frames)]
    print("  visible per frame: home median {:.0f} (max {}), away median {:.0f} (max {}); frames with more than 11 of one team {:.1%}".format(
        np.median(per_frame), max(per_frame), np.median(per_frame_a), max(per_frame_a),
        np.mean((np.array(per_frame) > 11) | (np.array(per_frame_a) > 11))))

    # Predicted rows: the track's label and box size from its nearest real box, the box placed
    # where the prediction's feet project into the image. kind: 0 a box, 1 a merged box (two
    # players), 2 a track merged or hidden behind another box, 3 a track coasting.
    rows_of = {tid: np.flatnonzero(ids == tid) for tid in np.unique(ids)}
    pred_rows = []
    for f, tid, p, kind in extra:
        h = H[f]
        if np.isnan(h).any():
            continue
        q = h @ np.array([p[0] / data.PITCH[0], p[1] / data.PITCH[1], 1.0])
        if q[2] <= 0:
            continue
        rows = rows_of[tid]
        r = rows[np.argmin(np.abs(frames[rows] - f))]
        w, hh = boxes[r, 2], boxes[r, 3]
        pred_rows.append((f, [q[0] / q[2], q[1] / q[2] - hh / 2, w, hh], tid, final[r], flagged[r], 1 + kind))
    real_rows = [(f, b, t, l, s, int(m)) for f, b, t, l, s, m in zip(frames, boxes, ids, final, flagged, merged)]

    path = os.path.join(data.CACHE_DIR, "detections", "teams-{}-{}-{}.csv".format(source, first, count))
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["frame", "x", "y", "w", "h", "track", "label", "split", "kind"])
        for f, b, t, l, s, kind in sorted(real_rows + pred_rows, key=lambda row: (row[0], row[2])):
            out.writerow([int(f)] + [round(float(v), 1) for v in b] + [int(t), int(l), int(s), int(kind)])
    print("  -> {}".format(os.path.relpath(path, os.path.dirname(HERE))))


if __name__ == "__main__":
    main()
