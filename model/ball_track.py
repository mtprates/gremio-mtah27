"""Rule 18: a ball track from our own detections (model/detect_ball.py), gaps bridged.

    py model/ball_track.py 11185 3000          # tactical first frame and frame count of a window

The detector's ball is more accurate than the ATD's where it finds one, but it is a list of
candidates per frame, not a track: some frames have several (a wrong object among them), many
have none. So, per camera:

  prefilter candidates that are not the ball are dropped before anything else (prefilter): one
            near an official's feet -- a shoe, to a detector not trained on football (69362) --
            unless it is very confident. More filters of this kind go there.
  pick      one candidate per frame, or none, by the cheapest path through the whole window
            (associate): confident candidates are cheap; standing still, a penalty or centre
            spot and being outside the lines are expensive; moves faster than 40 m/s are
            impossible; frames with no pick cost a little and restarting costs more. When the ball
            disappears the path takes no pick -- the smoother bridges it -- rather than jumping
            to a spot or a ball by the line.
  3D        the broadcast's picks (live frames outside cut-away shots, main_broadcast) are put on the tactical
            pitch and timeline, and both cameras go through rule 17 (ball3d.solve): per-frame
            triangulation, ballistic arcs for flights, each pick moved to where its line of
            sight is at that height, then the two-camera Kalman smoother, which bridges the gaps.
            On these picks -- pure pixel measurements, unlike the ATD balls -- the geometry holds:
            where the two cameras agree the height is -0.09 m at the median, 90% within 0.31 m.

Writes data/.model/detections/balltrack-<first>-<count>.csv (tactical frame, x, y metres, std m --
negative where no camera picked the ball and the smoother bridged it -- and which cameras did:
1 tactical, 2 broadcast; 4 held by a keeper, held_by_keeper) for the viewer, and prints aggregates only.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
from data import FPS  # noqa: E402

# Params
CONF_TRACK = 0.1            # candidates the tracker may take
KEEPER_HOLD_MIN = FPS       # frames: shorter gaps are missed frames, which the smoother bridges well, not catches
KEEPER_HOLD_RADIUS = 1.5    # m: rule 36 -- the ball this close to a keeper in his box is in his hands (rules.CARRY_RADIUS;
                            # iterate.py --keeper: 3.0)
# Rule 38 (docs/VALUES.md): a ball nobody sees is with the player who reaches it. Off; iterate.py --cover.
HOLD_PLAYER = False
COVER_R, BALL_TAU, HOLD_PLAYER_MAX = 2.5, 2.0, 30 * FPS   # m, s, frames
RELEASE_M, RELEASE_FRAMES, OCC_M = 3.0, 5, 0.6             # m, frames, m
HOLD_MIN, HOLD_CONF = 10, 0.4                              # frames, confidence of a pick that ends the stretch
HOLD_RELEASE = False        # rule 48 (docs/VALUES.md): a keeper hold releases at ball pace; iterate.py --gaps
KEEPER_HOLD_MAX = 8 * FPS   # frames: a ball gap starting at a keeper's hands in his own penalty area follows
                            # him at most this long -- Law 12, eight seconds in the hands (docs/VALUES.md)
OFFICIAL_FEET = 0.3         # box heights: a candidate this close to an official's feet (box bottom-
OFFICIAL_KEEP = 0.7         # centre) is a shoe, dropped unless this confident (shoe candidates: median
                            # 0.17, 99th pct 0.61-0.65; docs/VALUES.md)
DETECTION_NOISE = 0.25      # m: one camera's detection on the grass (the height spread above)
TRACK_OUT = 0.5             # m: no candidate further outside a line is taken at all. A ball out
                            # there is out of play, and the tactical image's bottom band (pitch-side
                            # boards and people, 2-6 m beyond the near touchline) held 60% of all
                            # candidates.
# The tracker picks the cheapest path through the whole window (Viterbi), not frame by frame:
# the greedy first version widened its gate while the ball was unseen until something else fell
# in -- at 1H 11670 an object by the far touchline, 60 m from the ball, for seconds.
# Rule 52 (docs/VALUES.md): the detector's measured full-frame precision per confidence band (lower edges), public
# images (COCO yolo11m, 14 clips); with PRICE_PRECISION a candidate's confidence term is -log of it. iterate.py --precision.
PRECISION = ((0.05, 0.08), (0.10, 0.17), (0.15, 0.18), (0.20, 0.30), (0.25, 0.38), (0.35, 0.76), (0.50, 0.87))
PRICE_PRECISION = False
REWARD = 2.0                # nats: taking a candidate earns this, less -log(conf) ...
STATIC_PEN = 3.0            # ... minus this if it stands still: a candidate within STATIC_M of
STATIC_M = 0.5              # it on at least STATIC_HITS of the frames sampled every STATIC_STEP
STATIC_HITS = 3             # over STATIC_FRAMES either side, at least one before and one after
STATIC_FRAMES = 2 * FPS     # beyond STATIC_NEAR (penalty spots, marks, a spare ball by the line;
STATIC_STEP = 5             # on the pitch, as the view pans). A share of the frames missed the
STATIC_NEAR = 10            # object by the far touchline at 11670, detected only now and then ...
SPOT_PEN = 2.0              # ... minus this within SPOT_M of a penalty or the centre spot ...
SPOT_M = 0.6
OUT_PEN = 4.0               # ... minus this per metre outside a line ...
V_AIR, V_LANDED = 38.0, 25.0  # m/s: rule 18.b -- faster ground point is a ball in the air; landed
                            # when seen again at the slower speed (the window's seen frames: 99th
                            # percentile 24 m/s once fast ground points are kept)
HINT_MAX = 4 * FPS          # frames: an air hint lasts at most this long
HINT_Z = 3.5                # m: the height assumed while hinted -- out of reach (REACH 2.3 m)
AIR_MAX = 25.0              # ... -- or, further out, if some height up to this on its line of sight
AIR_PEN, AIR_PEN_PER_M = 0.5, 0.1  # puts it over the pitch, this plus this per metre of that height.
                            # 1H 0:03: a ball kicked toward the far line left the pitch on the
                            # tactical image at 0.9 m up and the "never outside" rule dropped it.
LONELY_M = 15.0             # ... and LONELY_PEN if no player of that camera's tracking is within
LONELY_PEN = 1.0            # LONELY_M: a spare ball by the line with play at the other end (11670)
MISS = 0.5                  # nats per frame with no pick
START = 12.0                # nats to start a new run of picks (after a gap longer than SKIP, or
                            # without a plausible move from the last pick). At 3 a run of weak
                            # detections 60 m away paid for itself once the ball was lost (11670): a
                            # teleport. 12 needs ~10 confident frames -- a real restart.
SKIP = 3 * FPS              # frames a run of picks may skip without restarting
V_MAX, V_SLACK = 70.0, 1.0  # m/s and m: no faster move between picks. These are ground points: a
                            # rising ball's runs away from the camera faster than the ball (h / (h -
                            # z)), most for the low broadcast camera; at 40 m/s the 6.8 m long ball
                            # at 1:02 lost its picks and its flight
V_GAP, GAP_FRAMES = 35.0, 5  # m/s: across more than GAP_FRAMES unseen, the real ball's pace. At
                            # 70 m/s a run could legally move 60 m once the ball had been unseen for
                            # a second (11670), and pick up the spare ball by the far line.
SPEED_COST, V_REF = 0.3, 30.0  # nats: SPEED_COST * (v / V_REF)^2 for a move at v m/s
# Rule 28 (docs/VALUES.md): a pick that leaves the line the path was on, with no player near to
# have kicked it, costs TURN_PEN * (excess miss / TURN_SCALE)^2 nats, at most TURN_CAP. Off at 0.
TURN_PEN = 0.0
TURN_BASE, TURN_BASE_MAX = 4, 12    # frames: the path's velocity over at least / at most this long
TURN_FREE = 2 * DETECTION_NOISE     # m: a miss this small is detection noise
TURN_SCALE, TURN_CAP = 1.0, 6.0
TURN_NEAR = 4.0                     # m: a player this close (rules.TOUCH_RADIUS) could have kicked it
# Rules 29-30 (docs/VALUES.md): switches. All off = the pipeline as committed in 954604b.
#   persons  "lonely" (no player within LONELY_M) from our detected people, not the provider's tracking
#   refs     a second pass in which each camera's candidates are priced against the other camera's picks
#   feet     rule 32: a weak candidate at a player's feet, far from the last confident ball, pays
#   out      rule 33: candidates beyond TRACK_OUT that no height explains may continue a run (the ball going out)
#   image    rule 44: links between picks judged by their angular move in the camera's image (image_track.py)
#   restart  rule 45: no ball appears from nowhere -- reach from the last confident ball, feet, exits, isolated picks
OPTS = {"persons": False, "refs": False, "feet": False, "out": False, "image": False, "restart": False}
# Rule 44 (docs/VALUES.md): the filter decides with each camera's image track. Off; iterate.py --image (with OPTS["image"]).
IMAGE = False
OUT_SOFT, OUT_CAP, OUT_MAX = 0.5, 3.0, 10.0   # rule 33: nats per metre outside, at most, and the farthest kept (m)
REF_CONF = 0.2              # the other camera's picks at least this confident are anchors ...
ANCHOR_FRAMES = 2 * FPS     # ... for this long either side
AGREE_M = 3.0               # m: ... a candidate is supported within this + V_GAP * dt of an anchor
AGREE_STATIC = 1.5          # m: ... one this close to an anchor within REF_NEAR frames is no static object
REF_NEAR = 3
ANCHOR_PEN, ANCHOR_SCALE, ANCHOR_STRONG = 6.0, 5.0, 0.5   # nats, m, confidence
START_REF = 3.0             # nats: a supported candidate starts a run for this, not START
FEET_M, HC_CONF, FEET_PEN = 1.5, 0.5, 6.0   # rule 32: m from a detected person, confidence, nats
# Rule 45 (docs/VALUES.md), OPTS "restart" (second pass): a candidate must be reachable from the last confident ball
MEMORY = 8 * FPS            # frames: the last confident ball is remembered this long (Law 12: a keeper's 8 s)
V_PLAYER = 8.0              # m/s: a ball out of sight beside a player is covered and moves with him (RELINK_SPEED)
EXIT_M, THROW_M, ZONE_SLACK = 3.0, 5.0, 2.0   # m: an exit (AGREE_M); a throw-in's reach of it (Law 15); slack on a zone
GOAL_AREA, CORNER_R = (5.5, 9.16), 1.0      # m: the goal area's depth and half-width (Law 16), the corner arc (Law 17)
ISO_GAP = 2                 # frames: a pick with no other this close, either camera, is isolated


def outside(xy):
    """(k,) metres outside the pitch lines (0 inside)."""
    L, W = data.PITCH
    dx = np.maximum(0, np.maximum(-xy[:, 0], xy[:, 0] - L))
    dy = np.maximum(0, np.maximum(-xy[:, 1], xy[:, 1] - W))
    return np.hypot(dx, dy)


def spots():
    L, W = data.PITCH
    return np.array([[11.0, W / 2], [L - 11.0, W / 2], [L / 2, W / 2]])


def air_height(xy, centre):
    """(k,) the least height (m) at which a ball on each ground point's line of sight to the camera
    at `centre` (x, y, height; that source's pitch) is over the pitch, within TRACK_OUT; inf if
    none up to AIR_MAX. A ball in the air projects away from the camera, past the far lines; the
    clutter by the near touchline is between the camera and the pitch, and no height explains it."""
    z = np.arange(0.0, AIR_MAX + 1e-9, 0.25)
    toward_cam = (centre[None, :2] - xy) / centre[2]                     # metres per metre of height
    B = xy[:, None, :] + toward_cam[:, None, :] * z[None, :, None]       # (k, Z, 2)
    ok = outside(B.reshape(-1, 2)).reshape(len(xy), len(z)) <= TRACK_OUT
    return np.where(ok.any(axis=1), z[np.argmax(ok, axis=1)], np.inf)


PREFILTER_DROPPED = {}      # (source, filter) -> candidates dropped, over every window built


def prefilter(source, first, count, rows):
    """The candidates (frame, x, y px, conf) without those that are not the ball: near an official's
    feet unless OFFICIAL_KEEP or more (a shoe; tactical 69362 was picked at 0.30, 7 px from one).
    Officials from model/teams.py's labels of the same window; without them, nothing is dropped."""
    path = os.path.join(data.CACHE_DIR, "detections", "teams-{}-{}-{}.csv".format(source, first, count))
    if not os.path.isfile(path):
        print("  prefilter: no {} -- run model/teams.py first; officials' feet not filtered".format(os.path.basename(path)))
        return rows
    feet = {}
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            if int(r["label"]) == 4 and int(r.get("kind") or 0) <= 1:
                h = float(r["h"])
                feet.setdefault(int(r["frame"]), []).append((float(r["x"]), float(r["y"]) + h / 2, h))
    out, dropped = [], 0
    for f, x, y, c in rows:
        if c < OFFICIAL_KEEP and any(np.hypot(x - fx, y - fy) <= OFFICIAL_FEET * h for fx, fy, h in feet.get(f, ())):
            dropped += 1
            continue
        out.append((f, x, y, c))
    key = (source, "official's feet")
    PREFILTER_DROPPED[key] = PREFILTER_DROPPED.get(key, 0) + dropped
    return out


def persons_pitch(source, first, count, H):
    """{frame: (k, 2)} pitch positions (that source's pitch) of the people our detector found
    (model/teams.py's file: boxes, kind 0 and 1; feet = box bottom centre), None without the file."""
    import compare_ball
    path = os.path.join(data.CACHE_DIR, "detections", "teams-{}-{}-{}.csv".format(source, first, count))
    if not os.path.isfile(path):
        return None
    frames, px = [], []
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            if int(r.get("kind") or 0) <= 1:
                frames.append(int(r["frame"]))
                px.append((float(r["x"]), float(r["y"]) + float(r["h"]) / 2))
    out = {}
    if frames:
        for f, p in zip(frames, compare_ball.to_pitch(H, np.array(frames), np.array(px))):
            if np.isfinite(p[0]):
                out.setdefault(f, []).append(p)
    return {f: np.array(v) for f, v in out.items()}


def candidates(source, first, count, centre=None):
    """Per frame of the window, (k, 5) candidates: x, y metres (that source's pitch), conf, the cost
    of taking it (nats, lower is better), and whether it may start a run of picks. On the pitch
    (within TRACK_OUT), or -- given the camera's centre -- outside it where some height up to
    AIR_MAX puts the ball over the pitch (air_height): those continue a run, never start one."""
    import compare_ball
    path = os.path.join(data.CACHE_DIR, "detections", "{}_{}_{}.csv".format(source, first, count))
    rows = prefilter(source, first, count, [r for r in compare_ball.load(path) if r[3] >= CONF_TRACK])
    by_frame = {}
    for f, x, y, c in rows:
        by_frame.setdefault(f, []).append((x, y, c))

    H = data.homography_matrices(source)
    players = data.load(source).players
    persons = persons_pitch(source, first, count, H)
    pitch = {}
    for f, found in by_frame.items():
        xy = compare_ball.to_pitch(H, np.full(len(found), f), np.array([(x, y) for x, y, _ in found]))
        fin = np.isfinite(xy[:, 0])
        out_m = outside(np.nan_to_num(xy, nan=1e9))
        air = np.zeros(len(xy))
        far = fin & (out_m > TRACK_OUT)
        if centre is not None and far.any():
            air[far] = air_height(xy[far], centre)
        else:
            air[far] = np.inf
        # Rule 33: beyond TRACK_OUT and no height explains it -- kept, flagged "off the pitch" (pick_match
        # drops these unless OPTS["out"]), so a ball rolling out of play can be followed.
        off = far & ~np.isfinite(air) & (out_m <= OUT_MAX)
        air[off] = 0.0
        keep = fin & np.isfinite(air)
        pitch[f] = np.column_stack([xy, [c for _, _, c in found], air, off])[keep]

    offsets = range(-STATIC_FRAMES, STATIC_FRAMES + 1, STATIC_STEP)
    sp = spots()
    out = []
    for f in range(first, first + count):
        c = pitch.get(f, np.zeros((0, 5)))
        if not len(c):
            out.append(np.zeros((0, 12)))
            continue
        in_air = c[:, 3] > 0
        hits, before, after = np.zeros(len(c)), np.zeros(len(c), bool), np.zeros(len(c), bool)
        for o in offsets:
            other = pitch.get(f + o)
            if abs(o) <= STATIC_NEAR or other is None or not len(other):
                continue
            hit = np.linalg.norm(c[:, None, :2] - other[None, :, :2], axis=2).min(axis=1) <= STATIC_M
            hits += hit
            before |= hit & (o < 0)
            after |= hit & (o > 0)
        static = (hits >= STATIC_HITS) & before & after
        near_spot = np.linalg.norm(c[:, None, :2] - sp[None], axis=2).min(axis=1) <= SPOT_M
        with np.errstate(invalid="ignore"):
            d_player = np.linalg.norm(c[:, None, :2] - players[f][None, :, :], axis=2)
        lonely = ~(np.nan_to_num(d_player, nan=np.inf).min(axis=1) <= LONELY_M)
        place = np.where(in_air, AIR_PEN + AIR_PEN_PER_M * c[:, 3],
                         np.where(c[:, 4] > 0, np.minimum(OUT_SOFT * outside(c[:, :2]), OUT_CAP), OUT_PEN * outside(c[:, :2])))
        cost = (-np.log(c[:, 2]) - REWARD + STATIC_PEN * static + SPOT_PEN * near_spot + place +
                LONELY_PEN * lonely)
        near = np.nan_to_num(d_player, nan=np.inf).min(axis=1) if d_player.shape[1] else np.full(len(c), np.inf)
        near_p = np.full(len(c), np.inf)
        if persons is None:
            near_p = near                                   # no detections of people: the provider's
        elif len(persons.get(f, ())):
            near_p = np.linalg.norm(c[:, None, :2] - persons[f][None, :, :], axis=2).min(axis=1)
        # x, y, conf, cost, may start a run, nearest provider player, static, near a spot, placement
        # cost, nearest detected person, cost of starting a run here: the pieces recost() reprices.
        out.append(np.column_stack([c[:, :3], cost, ~in_air & (c[:, 4] == 0), near, static, near_spot, place, near_p,
                                    np.full(len(c), START), c[:, 4]]))
    return out


def precision(conf):
    """Rule 52: the measured chance a detection of this confidence is the ball, on a full frame (PRECISION's band)."""
    conf = np.asarray(conf, dtype=float)
    edges = np.array([lo for lo, _ in PRECISION])
    values = np.array([p for _, p in PRECISION])
    return values[np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(values) - 1)]


def recost(cands, persons=False, ref=None, feet_ref=None, last=None):
    """candidates() priced again. persons: "lonely" from our detected people (rule 30). ref = (pos
    (n, 2), dt (n,)) per frame: the nearest anchor from the other camera and its distance in frames
    (NaN: none), rule 29 -- a weak candidate beyond the anchor's reach pays, a supported one starts a
    run cheaply, one on the anchor is no static object. feet_ref, the same for the nearest confident
    ball (rule 32): a weak candidate within FEET_M of a person and beyond its reach pays FEET_PEN. last
    (rule 45): the last confident ball (p0, v0, dt) per frame; the candidates it rules out are removed."""
    out = []
    for f, c in enumerate(cands):
        if len(c) and last is not None:
            c = c[reachable(c, (last[0][f], last[1][f], last[2][f]), persons)]
        if not len(c):
            out.append(c)
            continue
        lonely = ~(c[:, 9 if persons else 5] <= LONELY_M)
        static = c[:, 6].copy()
        start = np.full(len(c), START)
        pen = np.zeros(len(c))
        if ref is not None and np.isfinite(ref[1][f]):
            dt = ref[1][f]
            d = np.linalg.norm(c[:, :2] - ref[0][f], axis=1)
            reach = AGREE_M + V_GAP * dt / FPS
            weak = np.clip(1.0 - c[:, 2] / ANCHOR_STRONG, 0.0, 1.0)
            pen = ANCHOR_PEN * weak * np.clip((d - reach) / ANCHOR_SCALE, 0.0, 1.0)
            if dt <= REF_NEAR:
                static = np.where(d <= AGREE_STATIC, 0.0, static)
            start = np.where(d <= reach, START_REF, START)
        if feet_ref is not None and np.isfinite(feet_ref[1][f]):
            d = np.linalg.norm(c[:, :2] - feet_ref[0][f], axis=1)
            reach = AGREE_M + V_GAP * feet_ref[1][f] / FPS
            weak = np.clip(1.0 - c[:, 2] / ANCHOR_STRONG, 0.0, 1.0)
            at_feet = c[:, 9 if persons else 5] <= FEET_M
            pen = pen + FEET_PEN * weak * np.clip((d - reach) / ANCHOR_SCALE, 0.0, 1.0) * at_feet
        new = c.copy()
        conf_cost = -np.log(precision(c[:, 2])) if PRICE_PRECISION else -np.log(c[:, 2])    # rule 52
        new[:, 3] = (conf_cost - REWARD + STATIC_PEN * static + SPOT_PEN * c[:, 7] + c[:, 8] +
                     LONELY_PEN * lonely + pen)
        new[:, 10] = start
        out.append(new)
    return out


def restart_zone(c, p0):
    """Rule 45. (k,) bool: candidates c (k, 2) where the ball comes back into play after leaving the pitch near p0
    (outside the lines, or within EXIT_M of one), by the Laws: a throw-in where it left the touchline (THROW_M), or
    after the goal line a goal kick from the goal area, a corner from the corner arc on that side, or a kick-off from
    the centre spot after a goal; each zone ZONE_SLACK wider. All False if p0 is no exit. Symmetric, so either
    camera's own pitch works."""
    L, W = data.PITCH
    out_m = outside(p0[None])[0]
    dx, dy = min(p0[0], L - p0[0]), min(p0[1], W - p0[1])
    if out_m == 0 and min(dx, dy) > EXIT_M:
        return np.zeros(len(c), dtype=bool)
    if out_m > 0:
        touch = 0 <= p0[0] <= L     # outside beyond a touchline (not past a goal line)
    else:
        touch = dy <= dx
    if touch:
        exit_pt = np.array([np.clip(p0[0], 0, L), 0.0 if p0[1] < W / 2 else W])
        return np.linalg.norm(c - exit_pt, axis=1) <= THROW_M
    end = 0.0 if p0[0] < L / 2 else L
    goal_area = (np.abs(c[:, 0] - end) <= GOAL_AREA[0] + ZONE_SLACK) & (np.abs(c[:, 1] - W / 2) <= GOAL_AREA[1] + ZONE_SLACK)
    corner = np.linalg.norm(c - [end, 0.0 if p0[1] < W / 2 else W], axis=1) <= CORNER_R + ZONE_SLACK
    centre = np.linalg.norm(c - [L / 2, W / 2], axis=1) <= ZONE_SLACK
    return goal_area | corner | centre


def last_ball(pos, conf, near):
    """Rule 45. The last confident ball on the tactical timeline: (p0 (n, 2), v0 (n, 2) m/s, dt (n,) frames, NaN beyond
    MEMORY or before any). pos, conf, near: per camera (tactical, broadcast) first-pass picks (tactical pitch), their
    confidences and metres to the nearest detected person; velocities from the same camera's picks. A
    confident pick is a ball only if it passes the same test a candidate does against the ball before it (reach,
    feet, exit), so a confident pick at a player's feet with no pass toward him does not become the reference."""
    n = len(pos[0])
    frames = []
    for cam in (0, 1):
        ok = np.flatnonzero(np.isfinite(pos[cam][:, 0]) & (np.nan_to_num(conf[cam], nan=0.0) >= HC_CONF))
        frames += [(int(f), cam) for f in ok]
    frames.sort()
    p0, v0, dt = np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full(n, np.nan)
    acc = None                          # (frame, position, velocity) of the last accepted ball
    marks = []
    for f, cam in frames:
        p = pos[cam][f].astype(float)
        v = np.zeros(2)
        for k in range(1, GAP_FRAMES + 1):
            q = pos[cam][f - k] if f - k >= 0 else np.array([np.nan, np.nan])
            if np.isfinite(q[0]):
                vel = (p - q) * FPS / k
                if np.hypot(*vel) <= V_MAX:
                    v = vel
                break
        if acc is not None and f - acc[0] <= MEMORY:
            gap = (f - acc[0]) / FPS
            d0 = np.linalg.norm(p - acc[1])
            feet = np.nan_to_num(near[cam][f], nan=np.inf) <= FEET_M
            ok = d0 <= AGREE_M + V_GAP * gap if not feet else                 min(d0, np.linalg.norm(p - acc[1] - acc[2] * gap)) <= AGREE_M + V_PLAYER * gap
            if not ok and not restart_zone(p[None], acc[1])[0]:
                continue
        acc = (f, p, v)
        marks.append(acc)
    for (f, p, v), nxt in zip(marks, marks[1:] + [(n, None, None)]):
        stop = min(nxt[0], f + MEMORY + 1, n)
        p0[f:stop], v0[f:stop] = p, v
        dt[f:stop] = np.arange(stop - f)
    return p0, v0, dt


def reachable(c, last_f, persons):
    """Rule 45. (k,) bool: the candidates c (one frame) that may be the ball given the last confident ball
    (p0, v0, dt frames) on that frame -- see last_ball."""
    p0, v0, dt = last_f
    if not np.isfinite(dt):
        return np.ones(len(c), dtype=bool)
    gap = dt / FPS
    d0 = np.linalg.norm(c[:, :2] - p0, axis=1)
    dcoast = np.linalg.norm(c[:, :2] - (p0 + v0 * gap), axis=1)
    feet = c[:, 9 if persons else 5] <= FEET_M
    ok = np.where(feet, np.minimum(d0, dcoast) <= AGREE_M + V_PLAYER * gap, d0 <= AGREE_M + V_GAP * gap)
    ok |= restart_zone(c[:, :2], p0)
    return ok | (~feet & (c[:, 2] >= HC_CONF))


def isolated(pos, other):
    """Rule 45. (n,) bool: picks of one camera (pos, tactical timeline) with no pick of its own within ISO_GAP frames
    (within AGREE_M + V_MAX * dt) and none of the other camera within ISO_GAP frames and AGREE_M + V_GAP * dt."""
    n = len(pos)
    have = np.isfinite(pos[:, 0])
    alone = have.copy()
    for k in range(-ISO_GAP, ISO_GAP + 1):
        lo, hi = max(0, -k), min(n, n - k)
        idx = np.arange(lo, hi)
        dt = abs(k) / FPS
        for src, v in ((pos, V_MAX), (other, V_GAP)):
            if k == 0 and src is pos:
                continue
            q = src[idx + k]
            with np.errstate(invalid="ignore"):
                near = np.isfinite(q[:, 0]) & (np.linalg.norm(q - pos[idx], axis=1) <= AGREE_M + v * dt)
            alone[idx[near]] = False
    return alone & have


def references(pos, conf, min_conf=REF_CONF):
    """(n, 2) positions and (n,) distance in frames of the nearest pick of confidence min_conf or more
    within ANCHOR_FRAMES, for every frame (NaN where there is none)."""
    n = len(pos)
    ref, dt = np.full((n, 2), np.nan), np.full(n, np.nan)
    ok = np.flatnonzero(np.isfinite(pos[:, 0]) & (np.nan_to_num(conf, nan=0.0) >= min_conf))
    if not len(ok):
        return ref, dt
    f = np.arange(n)
    j = np.searchsorted(ok, f)
    lo, hi = ok[np.clip(j - 1, 0, len(ok) - 1)], ok[np.clip(j, 0, len(ok) - 1)]
    near = np.where(np.abs(f - lo) <= np.abs(hi - f), lo, hi)
    keep = np.abs(f - near) <= ANCHOR_FRAMES
    ref[keep], dt[keep] = pos[near[keep]], np.abs(f - near)[keep]
    return ref, dt


def _path_velocity(cands, link, t):
    """(k, 2) m/s: each candidate at t, along its best path, over the last TURN_BASE frames (the first
    pick at least that far back), NaN with no such pick within TURN_BASE_MAX or where the path started."""
    c = cands[t]
    out = np.full((len(c), 2), np.nan)
    for j in range(len(c)):
        tt, jj = t, j
        while t - tt < TURN_BASE:
            tt, jj = int(link[tt][jj][0]), int(link[tt][jj][1])
            if tt < 0:
                break
        if tt >= 0 and t - tt <= TURN_BASE_MAX:
            out[j] = (c[j, :2] - cands[tt][jj, :2]) * FPS / (t - tt)
    return out


def _turn_cost(p, c, v, dt):
    """(prev, now) nats: how far each new pick c lies from where the path of each previous pick p
    (velocity v) would have it dt s on, past TURN_FREE, unless a player was near either pick."""
    pred = p[:, None, :2] + v[:, None, :] * dt
    with np.errstate(invalid="ignore"):
        miss = np.linalg.norm(c[None, :, :2] - pred, axis=2)
        pen = np.minimum(TURN_PEN * (np.maximum(0.0, miss - TURN_FREE) / TURN_SCALE) ** 2, TURN_CAP)
    gate = np.minimum(p[:, None, 5], c[None, :, 5]) > TURN_NEAR
    return np.where(gate & np.isfinite(pen), pen, 0.0)


def associate(cands, img=None, weak_v=None, timing=None):
    """(n, 2) the picked candidate per frame (NaN: none) and (n,) its confidence: the cheapest
    path through the window. A frame either takes one candidate or none (MISS); consecutive picks
    are linked by a move no faster than V_MAX over up to SKIP frames (SPEED_COST, plus MISS per
    skipped frame), or a new run starts (START). img (rule 44): the window's image geometry (H, f,
    shot per frame and the camera's centre, image_track.camera sliced): within a shot the move is an
    angle in frame t's image, its limit and price V / D_eff; the ground points still never faster than V_MAX.
    weak_v (rule 45): the pace limit of a link into a weak candidate (under HC_CONF) at every gap. timing (rule 45):
    the candidates' nearest-person column; a run may then start on a weak candidate, or one at a person's feet, only
    where the path's last pick (within MEMORY) could have got to by then -- V_GAP, V_PLAYER at feet -- or in the restart
    zone of an exit there."""
    import image_track
    n = len(cands)
    vel = [None] * n             # per frame, (k, 2) the best path's velocity into each candidate (rule 28)
    best = [None] * n            # per frame, (k,) cheapest path ending on each candidate
    link = [None] * n            # per frame, (k, 2): where it came from (frame, candidate; -1: a start)
    none_cost = np.zeros(n)      # cheapest path ending with no pick on this frame
    none_from = np.full((n, 2), -1)
    none_last = [None] * n      # rule 45: (frame, place) of the last pick on the cheapest path with no pick at t
    for t in range(n):
        prev_none = none_cost[t - 1] if t else 0.0
        # No pick at t: from no pick at t-1, or from a run ending at t-1.
        none_cost[t], none_from[t] = prev_none + MISS, (-1, -1)
        none_last[t] = none_last[t - 1] if t else None
        if t and best[t - 1] is not None and len(best[t - 1]):
            j = int(np.argmin(best[t - 1]))
            if best[t - 1][j] + MISS < none_cost[t]:
                none_cost[t], none_from[t] = best[t - 1][j] + MISS, (t - 1, j)
                none_last[t] = (t - 1, cands[t - 1][j, :2])
        c = cands[t]
        if not len(c):
            best[t], link[t] = np.zeros(0), np.zeros((0, 2), dtype=int)
            continue
        starts = c[:, 10] if c.shape[1] > 10 else START
        cost = np.where(c[:, 4] > 0, starts + prev_none, np.inf)   # a ball in the air starts no run
        if timing is not None and t and none_last[t - 1] is not None and t - none_last[t - 1][0] <= MEMORY:
            # Rule 45: a far restart waits as long as the ball would need to get there from where it was last seen.
            lf, lp = none_last[t - 1]
            gap = (t - lf) / FPS
            d = np.linalg.norm(c[:, :2] - lp, axis=1)
            feet = c[:, timing] <= FEET_M
            ok = np.where(feet, d <= AGREE_M + V_PLAYER * gap, d <= AGREE_M + V_GAP * gap) | restart_zone(c[:, :2], lp)
            cost[((c[:, 2] < HC_CONF) | feet) & ~ok] = np.inf
        frm = np.full((len(c), 2), -1)
        ang_now = None
        if img is not None and np.isfinite(img["f"][t]):
            ang_now = image_track.angles(img["H"][t], img["f"][t], c[:, :2])
            slack = 2 * image_track.IMG_NOISE_PX / img["f"][t]
        vfast = np.where(c[:, 2] < HC_CONF, weak_v, V_MAX)[None] if weak_v is not None else V_MAX
        for k in range(1, min(SKIP, t) + 1):
            p = cands[t - k]
            if not len(p):
                continue
            d = np.linalg.norm(p[:, None, :2] - c[None, :, :2], axis=2)        # (prev, now)
            dt = k / FPS
            vlim = vfast if k <= GAP_FRAMES else V_GAP
            if ang_now is not None and img["shot"][t - k] == img["shot"][t]:
                # Rule 44: the earlier picks in frame t's image (H_t . H_(t-k)^-1 . u = H_t . p), as angles.
                a = np.linalg.norm(image_track.angles(img["H"][t], img["f"][t], p[:, :2])[:, None] - ang_now[None], axis=2)
                d_eff = image_track.eff_distance(p[:, :2], img["centre"])[:, None]
                with np.errstate(invalid="ignore"):
                    ok = (a <= vlim * dt / d_eff + slack) & (d <= V_MAX * dt + V_SLACK)
                v = np.nan_to_num(a * d_eff / dt, nan=np.inf)
            else:
                ok = d <= vlim * dt + V_SLACK
                v = d / dt
            total = best[t - k][:, None] + MISS * (k - 1) + SPEED_COST * (v / V_REF) ** 2
            if TURN_PEN and k <= GAP_FRAMES and vel[t - k] is not None:
                total = total + _turn_cost(p, c, vel[t - k], dt)
            total = np.where(ok, total, np.inf)
            i = np.argmin(total, axis=0)
            m = total[i, np.arange(len(c))]
            better = m < cost
            cost[better] = m[better]
            frm[better] = np.column_stack([np.full(better.sum(), t - k), i[better]])
        best[t], link[t] = cost + c[:, 3], frm
        if TURN_PEN:
            vel[t] = _path_velocity(cands, link, t)
    # Backtrack from the cheapest ending.
    picked = np.full((n, 2), np.nan)
    conf = np.full(n, np.nan)
    t, j = n - 1, -1
    if best[t] is not None and len(best[t]) and best[t].min() < none_cost[t]:
        j = int(np.argmin(best[t]))
    while t >= 0:
        if j < 0:
            # No pick at t: reached from no pick at t-1 (-1), or from a run ending at t-1.
            if none_from[t][0] < 0:
                t -= 1
            else:
                t, j = int(none_from[t][0]), int(none_from[t][1])
            continue
        picked[t], conf[t] = cands[t][j, :2], cands[t][j, 2]
        pt, pj = int(link[t][j][0]), int(link[t][j][1])
        if pt < 0:   # a run started here, after frames with no pick
            t, j = t - 1, -1
        else:
            t, j = pt, pj
    return picked, conf


def air_hint(kin):
    """Rule 18.b: a ball whose ground point moves faster than a ball on the grass can (V_AIR) is in
    the air -- the ground point of a rising ball runs away from the camera. Marked in the air from
    there, through the gap that follows, until it is seen again at V_LANDED or slower (at most
    HINT_MAX): no reach for players (z HINT_Z, the HMM's distance evidence and touches), unless a
    fitted flight (rule 17) already says how high. One camera is enough -- at 1H 0:03 the broadcast
    had no view, the tactical ground point ran at 52 m/s toward the far line and was lost against
    the crowd, and the bridge across the gap passed home players as if the ball rolled by them."""
    seen, speed = kin["seen"], np.nan_to_num(kin["speed"], nan=0.0)
    n = len(seen)
    hint = np.zeros(n, dtype=bool)
    t = 0
    while t < n:
        if seen[t] and speed[t] > V_AIR:
            start = t
            while t < n and t - start < HINT_MAX and not (seen[t] and speed[t] <= V_LANDED and t > start + 2):
                hint[t] = True
                t += 1
        t += 1
    fresh = hint & ~kin["flight"]
    kin["z"] = np.where(fresh, HINT_Z, kin["z"])
    kin["z_std"] = np.where(fresh, 1.0, kin["z_std"])
    kin["flight"] = kin["flight"] | hint
    kin["air_hint"] = hint
    return kin


def build(first, count):
    """The fused, smoothed detection track on the tactical window: dict of (count,) / (count, 2)
    arrays -- pos, pos_std, bridged, tactical pick, broadcast pick (tactical pitch) -- plus the
    kinematics over the whole match (mostly empty) for the rules."""
    import align
    import cameras
    import kalman
    import run
    tac, bro = data.load("tactical"), data.load("broadcast")
    al = run.load_alignment(tac, bro)
    import ball3d
    c_t, c_b, _ = ball3d.camera_geometry(tac, bro, al)
    # The broadcast's centre on its own pitch (camera_geometry gives it on the tactical one).
    flip = tuple(bool(v) for v in al["flip"])
    c_b_own = np.array([data.PITCH[0] - c_b[0] if flip[0] else c_b[0],
                        data.PITCH[1] - c_b[1] if flip[1] else c_b[1], c_b[2]])
    pick_t, _ = associate(candidates("tactical", first, count, centre=c_t))

    # The broadcast window: the broadcast frames that map into the tactical window.
    off = al["offset_b"]
    b_frames = np.flatnonzero((off != align.MISSING))
    t_of_b = b_frames + off[b_frames].astype(np.int64)
    inside = (t_of_b >= first) & (t_of_b < first + count)
    pick_b = np.full((count, 2), np.nan)
    if inside.any():
        b0, b1 = int(b_frames[inside].min()), int(b_frames[inside].max()) + 1
        path = os.path.join(data.CACHE_DIR, "detections")
        runs_b = [f for f in os.listdir(path) if f.startswith("broadcast_")]
        for name in runs_b:
            _, s, c = os.path.splitext(name)[0].split("_")
            s, c = int(s), int(c)
            if s > b0 or s + c < b1:
                continue
            main = main_broadcast(al, bro)
            picks, _ = associate(candidates("broadcast", s, c, centre=c_b_own))
            to_tactical(picks, s, main, al, pick_b, first)
            break

    full_t = np.full((tac.n, 2), np.nan, dtype=np.float32)
    full_b = np.full((tac.n, 2), np.nan, dtype=np.float32)
    full_t[first:first + count] = pick_t
    full_b[first:first + count] = pick_b
    # bridge: the detector's gaps are a ball it missed, so they follow the ball's lines through
    # (kalman._kinematics) -- 1H 13160, 1 s unseen while the broadcast showed a replay.
    kin = ball3d.solve(full_t, full_b, c_t, c_b, DETECTION_NOISE, bridge=True)
    air_hint(kin)
    w = slice(first, first + count)
    return {"pos": kin["pos"][w], "pos_std": kin["pos_std"][w], "seen": kin["seen"][w], "z": kin["z"][w],
            "z_tri": kin["z_tri"][w], "flights": [f for f in kin["flights"] if first <= f[0] < first + count],
            "pick_t": pick_t, "pick_b": pick_b, "kin": kin, "atd": tac.ball[w].astype(float)}


def main_broadcast(al, bro):
    """(n,) bool: broadcast frames whose picks are used -- outside cut-away shots (model/cameras.py
    cuts: image cuts, then each shot by its people's height under its homography) and live (the
    alignment matched the tactical players then): a replay's ball is from another moment. The
    broadcast's main camera is one camera that pans and zooms, so no test of its direction."""
    import cameras
    return cameras.broadcast_usable() & cameras.broadcast_live(al, bro.n)


def to_tactical(picks, s, main, al, out, first, conf=None, conf_out=None):
    """Broadcast picks of frames s, s + 1, ... into out (rows = tactical frames from `first`), on the
    tactical pitch: where `main` (main_broadcast) allows it and the alignment maps the frame. conf
    (their confidences) go to conf_out the same way."""
    import align
    off = al["offset_b"]
    flip = tuple(bool(v) for v in al["flip"])
    for k in range(len(picks)):
        b = s + k
        if np.isnan(picks[k, 0]) or not main[b] or off[b] == align.MISSING:
            continue
        t = b + int(off[b]) - first
        if 0 <= t < len(out):
            x, y = picks[k]
            out[t] = (data.PITCH[0] - x if flip[0] else x, data.PITCH[1] - y if flip[1] else y)
            if conf_out is not None:
                conf_out[t] = conf[k]


def base_candidates(tac_chunks, bro_chunks, c_t, c_b_own):
    """candidates() for every chunk: ([tactical chunk], [broadcast chunk]). The slow part of picking."""
    return ([candidates("tactical", first, count, centre=c_t) for first, count in tac_chunks],
            [candidates("broadcast", s, count, centre=c_b_own) for s, count in bro_chunks])


def pick_match(cand_t, cand_b, tac_chunks, bro_chunks, al, bro, n, opts=None, first=None, centres=None):
    """(full_t, full_b, conf_t, conf_b): each camera's picks and their confidences over the match,
    on the tactical timeline and pitch. One association per chunk; with OPTS["refs"] a second, in
    which each camera's candidates are priced against the other camera's first-pass picks (rule 29).
    opts overrides OPTS. first: the first pass's result, if already known (it does not depend on refs
    and feet). centres: (tactical, broadcast's own pitch) camera centres, for OPTS["image"] (rule 44)."""
    import align
    opts = dict(OPTS, **(opts or {}))
    main = main_broadcast(al, bro)
    geo = None
    if opts["image"]:
        import image_track
        import shots
        geo = (image_track.camera("tactical", centres[0]), image_track.camera("broadcast", centres[1], shots.cuts()))

    def sliced(g, s, count):
        return None if g is None else {"H": g["H"][s:s + count], "f": g["f"][s:s + count],
                                       "shot": g["shot"][s:s + count], "centre": g["centre"]}
    off = al["offset_b"]
    flip = tuple(bool(v) for v in al["flip"])

    def priced(c, ref, feet=None, last=None):
        if not opts["out"]:
            c = [x[x[:, 11] == 0] if len(x) and x.shape[1] > 11 else x for x in c]
        return (recost(c, opts["persons"], ref, feet, last)
                if opts["persons"] or ref is not None or feet is not None or last is not None else c)

    def broadcast_ref(ref, s, count):
        """The tactical-timeline anchors (ref) on broadcast frames s.. and the broadcast's own pitch."""
        idx = np.arange(s, s + count)
        o = off[idx]
        t = idx + o.astype(np.int64)
        ok = main[idx] & (o != align.MISSING) & (t >= 0) & (t < n)
        pos, dt = np.full((count, 2), np.nan), np.full(count, np.nan)
        pos[ok], dt[ok] = ref[0][t[ok]], ref[1][t[ok]]
        if flip[0]:
            pos[:, 0] = data.PITCH[0] - pos[:, 0]
        if flip[1]:
            pos[:, 1] = data.PITCH[1] - pos[:, 1]
        return pos, dt

    def broadcast_last(last, s, count):
        """Rule 45's last confident ball (tactical timeline and pitch) on broadcast frames s.. and its own pitch."""
        idx = np.arange(s, s + count)
        o = off[idx]
        t = idx + o.astype(np.int64)
        ok = main[idx] & (o != align.MISSING) & (t >= 0) & (t < n)
        p0, v0, dt = np.full((count, 2), np.nan), np.full((count, 2), np.nan), np.full(count, np.nan)
        p0[ok], v0[ok], dt[ok] = last[0][t[ok]], last[1][t[ok]], last[2][t[ok]]
        for ax in (0, 1):
            if flip[ax]:
                p0[:, ax] = data.PITCH[ax] - p0[:, ax]
                v0[:, ax] = -v0[:, ax]
        return p0, v0, dt

    def near_person(full_t, full_b):
        """Rule 45: metres from each first-pass pick to the nearest detected person (the candidate's own column)."""
        col = 9 if opts["persons"] else 5
        near_t, near_b = np.full(n, np.nan), np.full(n, np.nan)
        for (first, count), c in zip(tac_chunks, cand_t):
            for i, x in enumerate(c):
                p = full_t[first + i]
                if len(x) and np.isfinite(p[0]):
                    j = np.argmin(np.linalg.norm(x[:, :2] - p, axis=1))
                    near_t[first + i] = x[j, col]
        for (s, count), c in zip(bro_chunks, cand_b):
            for i, x in enumerate(c):
                b = s + i
                if not len(x) or not main[b] or off[b] == align.MISSING:
                    continue
                t = b + int(off[b])
                if not 0 <= t < n or not np.isfinite(full_b[t, 0]):
                    continue
                p = np.array([data.PITCH[0] - full_b[t, 0] if flip[0] else full_b[t, 0],
                              data.PITCH[1] - full_b[t, 1] if flip[1] else full_b[t, 1]])
                j = np.argmin(np.linalg.norm(x[:, :2] - p, axis=1))
                near_b[t] = x[j, col]
        return near_t, near_b

    def run(ref_for_t, ref_for_b, feet_for_t=None, feet_for_b=None, last=None):
        timing = None if last is None else (9 if opts["persons"] else 5)
        full_t = np.full((n, 2), np.nan, dtype=np.float32)
        full_b = np.full((n, 2), np.nan, dtype=np.float32)
        conf_t = np.full(n, np.nan, dtype=np.float32)
        conf_b = np.full(n, np.nan, dtype=np.float32)
        for (first, count), c in zip(tac_chunks, cand_t):
            ref = None if ref_for_t is None else (ref_for_t[0][first:first + count], ref_for_t[1][first:first + count])
            feet = None if feet_for_t is None else (feet_for_t[0][first:first + count], feet_for_t[1][first:first + count])
            lt = None if last is None else tuple(a[first:first + count] for a in last)
            picks, conf = associate(priced(c, ref, feet, lt), sliced(geo and geo[0], first, count),
                                    V_GAP if last is not None else None, timing)
            full_t[first:first + count], conf_t[first:first + count] = picks, conf
        for (s, count), c in zip(bro_chunks, cand_b):
            ref = None if ref_for_b is None else broadcast_ref(ref_for_b, s, count)
            feet = None if feet_for_b is None else broadcast_ref(feet_for_b, s, count)
            lb = None if last is None else broadcast_last(last, s, count)
            picks, conf = associate(priced(c, ref, feet, lb), sliced(geo and geo[1], s, count),
                                    V_GAP if last is not None else None, timing)
            to_tactical(picks, s, main, al, full_b, 0, conf, conf_b)
        return full_t, full_b, conf_t, conf_b

    out = first if first is not None else run(None, None)
    if opts["refs"] or opts["feet"] or opts["restart"]:
        ref_t = references(out[1], out[3]) if opts["refs"] else None      # tactical's, from the broadcast picks
        ref_b = references(out[0], out[2]) if opts["refs"] else None
        feet = None
        if opts["feet"]:
            # The confident ball: whichever camera's pick is the more confident on a frame.
            ct, cb = np.nan_to_num(out[2], nan=-1.0), np.nan_to_num(out[3], nan=-1.0)
            use_t = ct >= cb
            conf = np.maximum(ct, cb)
            feet = references(np.where(use_t[:, None], out[0], out[1]), np.where(conf >= 0, conf, np.nan), HC_CONF)
        last = None
        if opts["restart"]:
            near = near_person(out[0], out[1])
            last = last_ball((out[0], out[1]), (out[2], out[3]), near)
        out = run(ref_t, ref_b, feet, feet, last)
        if opts["restart"]:
            full_t, full_b, conf_t, conf_b = (np.array(a) for a in out)
            drop_t, drop_b = isolated(full_t.astype(float), full_b.astype(float)), isolated(full_b.astype(float), full_t.astype(float))
            full_t[drop_t], conf_t[drop_t], full_b[drop_b], conf_b[drop_b] = np.nan, np.nan, np.nan, np.nan
            print("  rule 45: isolated picks dropped: tactical {}, broadcast {}".format(int(drop_t.sum()), int(drop_b.sum())))
            out = (full_t, full_b, conf_t, conf_b)
    return out


def build_match(tac_chunks, bro_chunks):
    """build() over the whole match: each chunk's picks (model/detect_all.py's chunks, associated
    chunk by chunk), then one smoother over the match, so no track ends at a chunk's edge. Returns
    the kinematics and the tactical / broadcast picks on the tactical timeline."""
    import ball3d
    import run
    tac, bro = data.load("tactical"), data.load("broadcast")
    al = run.load_alignment(tac, bro)
    c_t, c_b, _ = ball3d.camera_geometry(tac, bro, al)
    flip = tuple(bool(v) for v in al["flip"])
    c_b_own = np.array([data.PITCH[0] - c_b[0] if flip[0] else c_b[0],
                        data.PITCH[1] - c_b[1] if flip[1] else c_b[1], c_b[2]])
    cand_t, cand_b = base_candidates(tac_chunks, bro_chunks, c_t, c_b_own)
    full_t, full_b, conf_t, conf_b = pick_match(cand_t, cand_b, tac_chunks, bro_chunks, al, bro, tac.n)
    return finish(full_t, full_b, c_t, c_b, tac_chunks, conf=(conf_t, conf_b)), full_t, full_b


# Rule 41 (docs/VALUES.md): drop the broadcast where it contradicts the tactical camera for long. Off; iterate.py --side.
DROP_SIDE = False
SIDE_MISS, SIDE_MIN, SIDE_JOIN, SIDE_AGREE, SIDE_END_GAP = 3.0, 8, 6, 1.0, 12   # m, frames, frames, m, frames


def drop_side_camera(full_t, full_b, c_t, c_b, conf):
    """Rule 41. (full_b, conf) without the broadcast's picks over stretches where its lines of sight miss
    the tactical camera's by over SIDE_MISS for SIDE_MIN frames or more: a shot whose homography does not
    fit (1H 27496-27517, a side-field camera). Extended each way over frames of broadcast picks until the
    two cameras agree again (SIDE_AGREE) or the picks stop for SIDE_END_GAP."""
    import ball3d
    import rules
    n = len(full_t)
    t, b = full_t.astype(float), full_b.astype(float)
    both = np.isfinite(t[:, 0]) & np.isfinite(b[:, 0])
    gap = np.linalg.norm(b - t, axis=1)
    ground = both & (gap <= ball3d.GROUND_GAP)
    bias = np.median((b - t)[ground], axis=0) if ground.any() else np.zeros(2)
    miss = ball3d.ray_miss(t, b - bias, c_t, c_b)
    with np.errstate(invalid="ignore"):
        bad = both & (miss > SIDE_MISS)
        agree = both & (miss <= SIDE_AGREE)
    stretches = []
    for s, e in rules.runs(bad):
        if stretches and s - stretches[-1][1] <= SIDE_JOIN:
            stretches[-1][1] = e
        else:
            stretches.append([s, e])
    bpick = np.isfinite(b[:, 0])
    out_b, out_conf = full_b.copy(), conf[1].copy()
    dropped = 0
    for s, e in stretches:
        if e - s < SIDE_MIN:
            continue
        a, z = s, e
        while a > 0 and bpick[max(0, a - SIDE_END_GAP):a].any() and not agree[a - 1]:
            a -= 1
        while z < n and bpick[z:z + SIDE_END_GAP].any() and not agree[z]:
            z += 1
        out_b[a:z] = np.nan
        out_conf[a:z] = np.nan
        dropped += int(bpick[a:z].sum())
    print("  side-field cuts: {} stretches, {} broadcast picks dropped".format(sum(1 for s, e in stretches if e - s >= SIDE_MIN), dropped))
    return out_b, (conf[0], out_conf)


def finish(full_t, full_b, c_t, c_b, tac_chunks, conf=None, players=None, image=None):
    """The match's picks (tactical timeline and pitch) -> the kinematics: 3D and smoother, air
    hints, keeper holds. conf: the picks' confidences (tactical, broadcast), for kalman.FIX. image:
    image_track.context, for IMAGE (rule 44)."""
    import ball3d
    if DROP_SIDE and conf is not None:
        full_b, conf = drop_side_camera(full_t, full_b, c_t, c_b, conf)
    verdicts = None
    if IMAGE and image is not None:
        import image_track
        verdicts = image_track.match_verdicts(image, full_t, full_b, conf)
    kin = ball3d.solve(full_t, full_b, c_t, c_b, DETECTION_NOISE, bridge=True, conf=conf, image=verdicts)
    kin["image"] = verdicts
    kin["cams"] = (c_t, c_b)
    air_hint(kin)
    held_by_keeper(kin, tac_chunks)
    if HOLD_PLAYER and players is not None and conf is not None:
        held_by_player(kin, players, conf, c_t)
    return kin


def held_by_player(kin, det, conf, c_t):
    """Rule 38. A stretch of HOLD_MIN frames or more with no pick of confidence HOLD_CONF or more, after a
    confident one on the ground, belongs to the player who first comes within COVER_R of where the ball
    would be (its last velocity, slowing with time constant BALL_TAU): the ball follows that path until he
    arrives and is then with him, OCC_M behind him as the tactical camera sees him -- an unseen ball is not in
    front of the player it would be visible past. It stays with him until the next confident pick (at most
    HOLD_PLAYER_MAX); if that pick is over RELEASE_M from him, the last RELEASE_FRAMES run to it. Weak picks
    in the stretch are overridden. kin["held"] marks the frames (as for a keeper)."""
    import rules
    n = len(kin["pos"])
    P = det.players
    pos, seen = kin["pos"], kin["seen"]
    ct, cb = (np.nan_to_num(np.asarray(c, dtype=float), nan=0.0) for c in conf)
    confident = seen & (np.maximum(ct, cb) >= HOLD_CONF)
    held = kin.get("held")
    if held is None:
        held = kin["held"] = np.zeros(n, dtype=bool)
    stretches = 0
    for s, e in rules.runs(~confident):
        f = s - 1
        if f < 1 or e - s < HOLD_MIN or not confident[f] or not np.isfinite(pos[f, 0]) or kin["z"][f] > 0.5 or \
                kin["flight"][f] or held[s:e].any():
            continue
        stop = min(e, s + HOLD_PLAYER_MAX)
        v0 = np.nan_to_num(kin["v_in"][f])
        t = (np.arange(s, stop) - f) / FPS
        path = pos[f] + v0 * BALL_TAU * (1.0 - np.exp(-t / BALL_TAU))[:, None]
        with np.errstate(invalid="ignore"):
            near = np.linalg.norm(P[s:stop] - path[:, None, :], axis=2) <= COVER_R
        if not near.any():
            continue
        first = np.where(near.any(axis=0), near.argmax(axis=0), stop - s)   # per column: first frame within reach
        j = int(np.argmin(first))
        fa = s + int(first[j])
        pos[s:fa] = path[:fa - s]
        last = None
        for u in range(fa, stop):
            q = P[u, j]
            if np.isfinite(q[0]):
                last = q
            if last is None:
                continue
            away = last - c_t[:2]
            away = away / max(float(np.linalg.norm(away)), 1e-9)
            pos[u] = last + OCC_M * away
        if stop == e and e < n and confident[e]:
            lead = max(fa, stop - RELEASE_FRAMES)
            if np.linalg.norm(pos[e] - pos[stop - 1]) > RELEASE_M and lead < stop:
                w = (np.arange(lead, stop) - lead + 1) / (stop - lead + 1)
                pos[lead:stop] = pos[lead - 1] + (pos[e] - pos[lead - 1]) * w[:, None]
        kin["pos_std"][s:stop] = COVER_R / 2
        kin["z"][s:stop] = 0.0
        held[s:stop] = True
        stretches += 1
    print("  held by a player: {} stretches, {:.0f} s".format(stretches, held.sum() / FPS))
    return kin


def held_by_keeper(kin, tac_chunks):
    import kalman
    """A ball gap that starts within rules.CARRY_RADIUS of a keeper inside his own penalty area is the
    ball in his hands: bridged with his position (his team's keeper box nearest the last held spot,
    frame by frame), at most KEEPER_HOLD_MAX, instead of along its last flight -- a caught ball does
    not fly on (1H 29947: 4.8 s without a candidate until his release). The frames stay unseen
    ("present" False), so the rules built on seen frames are unchanged; kin["held"] marks them."""
    import rules
    import teams
    tac = data.load("tactical")
    n = len(kin["pos"])
    H = data.homography_matrices("tactical")
    sides = rules.defending_sides(tac)
    half = rules.half_of_frames(n)
    keepers = {}            # frame -> [(x, y, team)]
    for first, count in tac_chunks:
        path = os.path.join(data.CACHE_DIR, "detections", "teams-tactical-{}-{}.csv".format(first, count))
        if not os.path.isfile(path):
            continue
        rows = [r for r in csv.DictReader(open(path, newline="")) if int(r["label"]) in (2, 3) and int(r.get("kind") or 0) <= 1]
        if not rows:
            continue
        f = np.array([int(r["frame"]) for r in rows])
        feet = teams.to_pitch(H, f, np.array([[float(r["x"]), float(r["y"]) + float(r["h"]) / 2] for r in rows]))
        for fr, p, r in zip(f, feet, rows):
            if np.isfinite(p[0]):
                keepers.setdefault(int(fr), []).append((p[0], p[1], int(r["label"]) - 2))
    seen, pos = kin["seen"], kin["pos"]
    held = np.zeros(n, dtype=bool)
    for s, e in rules.runs(~seen):
        f = s - 1
        if f < 0 or not seen[f] or half[f] == 0 or e - s < KEEPER_HOLD_MIN or not keepers.get(f):
            continue
        k = np.array(keepers[f])
        d = np.hypot(k[:, 0] - pos[f, 0], k[:, 1] - pos[f, 1])
        j = int(np.argmin(d))
        team, (x, y) = int(k[j, 2]), k[j, :2]
        goal_x = sides[half[f]][team]
        in_area = abs(x - goal_x) <= teams.BOX_DEPTH and abs(y - data.PITCH[1] / 2) <= teams.BOX_HALF_WIDTH
        if d[j] > KEEPER_HOLD_RADIUS or not in_area:
            continue
        last = np.array([x, y])
        stop = min(e, s + KEEPER_HOLD_MAX)
        for t in range(s, stop):
            mine = [q for q in keepers.get(t, ()) if q[2] == team]
            if mine:
                q = min(mine, key=lambda q: np.hypot(q[0] - last[0], q[1] - last[1]))
                # Rule 48: only his box -- another detection with the keeper's label is not where the ball went.
                if not HOLD_RELEASE or np.hypot(q[0] - last[0], q[1] - last[1]) <= kalman.RESYNC_AGREE:
                    last = np.array(q[:2])
            pos[t] = last
            kin["pos_std"][t] = rules.CARRY_RADIUS / 2
            kin["z"][t] = 0.0
            held[t] = True
        if HOLD_RELEASE and stop < n and np.isfinite(pos[stop, 0]):
            # Rule 48: released toward where the ball is next, at no more than MAX_PACE -- not in one frame.
            target = pos[stop].copy()
            r = min(stop - s, int(np.ceil(np.linalg.norm(target - last) * FPS / kalman.MAX_PACE)))
            for i, t in enumerate(range(stop - r, stop)):
                pos[t] = last + (target - last) * (i + 1) / (r + 1)
    kin["held"] = held
    print("  held by a keeper: {} gaps, {:.0f} s".format(len(rules.runs(held)), held.sum() / FPS))
    return kin


def window(kin, full_t, full_b, first, count):
    """build()'s dict for one window of build_match()'s output (what write() takes)."""
    w = slice(first, first + count)
    out = {"pos": kin["pos"][w], "pos_std": kin["pos_std"][w], "z": kin["z"][w],
           "pick_t": full_t[w], "pick_b": full_b[w], "z_tri": kin["z_tri"][w],
           "held": kin.get("held", np.zeros(len(kin["pos"]), bool))[w]}
    if "cam_inputs" in kin:
        ci = kin["cam_inputs"]
        out.update({"in_t": ci["t"][w], "in_b": ci["b"][w], "fill_t": ci["t_fill"][w], "fill_b": ci["b_fill"][w]})
    if "cams" in kin:
        import ball3d
        # How far the two cameras' lines of sight miss each other (m): no height explains it.
        out["miss"] = ball3d.ray_miss(full_t.astype(float), full_b.astype(float) - kin["bias"], *kin["cams"])[w]
    return out


def _gaps(mask):
    import rules
    lens = np.array([e - s for s, e in rules.runs(~mask)]) / FPS
    return "{} gaps, median {:.2f} s, 90th {:.2f} s, longest {:.1f} s".format(
        len(lens), float(np.median(lens)) if len(lens) else 0, float(np.percentile(lens, 90)) if len(lens) else 0,
        float(lens.max()) if len(lens) else 0)


def main():
    first, count = int(sys.argv[1]), int(sys.argv[2])
    tr = build(first, count)
    atd = ~np.isnan(tr["atd"][:, 0])
    t_on, b_on = ~np.isnan(tr["pick_t"][:, 0]), ~np.isnan(tr["pick_b"][:, 0])
    known = ~np.isnan(tr["pos"][:, 0])
    print("\ndetection ball track, tactical frames {}-{}".format(first, first + count - 1))
    print("  picked by the tracker: tactical {:.1%}, broadcast {:.1%}, either {:.1%} of frames (ATD ball {:.1%})".format(
        t_on.mean(), b_on.mean(), (t_on | b_on).mean(), atd.mean()))
    print("  after smoothing, position known on {:.1%} of frames ({:.1%} bridged); where the ATD has none: {:.1%}".format(
        known.mean(), (known & ~(t_on | b_on)).mean(), (known & ~atd).mean()))
    print("  gaps: picks {}".format(_gaps(t_on | b_on)))
    print("        smoothed {}".format(_gaps(known)))
    print("        ATD {}".format(_gaps(atd)))
    for name, a in (("tactical picks", tr["pick_t"]), ("broadcast picks", tr["pick_b"]), ("smoothed track", tr["pos"])):
        both = ~np.isnan(a[:, 0]) & atd
        d = np.linalg.norm(a[both] - tr["atd"][both], axis=1)
        print("  {:<16} vs ATD where both: median {:.2f} m, within 1 m {:.0%}, 2 m {:.0%}, beyond 5 m {:.0%}".format(
            name, float(np.median(d)), np.mean(d <= 1), np.mean(d <= 2), np.mean(d > 5)))
    both = t_on & b_on
    d = np.linalg.norm(tr["pick_t"][both] - tr["pick_b"][both], axis=1)
    print("  tactical vs broadcast picks where both: median {:.2f} m, within 1 m {:.0%}, beyond 5 m {:.0%}".format(
        float(np.median(d)), np.mean(d <= 1), np.mean(d > 5)))
    fl = tr["flights"]
    if fl:
        peaks = np.array([f[2] for f in fl])
        print("  flights (rule 17 on the picks): {}, {:.0f} s in all ({:.1%} of frames); peak median {:.1f} m, 90th {:.1f} m; "
              "arc fit RMS median {:.2f} m".format(len(fl), sum(f[1] - f[0] for f in fl) / 25, np.mean(tr["z"] > 0),
                                                   float(np.median(peaks)), float(np.percentile(peaks, 90)),
                                                   float(np.median([f[3] for f in fl]))))
    else:
        print("  flights (rule 17 on the picks): none")
    lo, hi = 12785 - first, 12835 - first
    if 0 <= lo and hi <= count:
        print("  1:04 airball: picked on {} of 50 frames, smoothed known on {}".format(
            int((t_on | b_on)[lo:hi].sum()), int(known[lo:hi].sum())))
        f = [x for x in fl if x[0] - first <= lo + 25 and x[1] - first >= lo]
        print("    flight: {}; height every 5th frame from 1:04: {}".format(
            "{:+.2f}..{:+.2f} s, peak {:.1f} m".format((f[0][0] - 12785) / 25, (f[0][1] - 12785) / 25, f[0][2]) if f else "none",
            " ".join("{:.1f}".format(v) for v in tr["z"][lo:hi:5])))

    print("  track -> {}".format(write(tr, first, count)))


def write(tr, first, count):
    """The track for the viewer: data/.model/detections/balltrack-<first>-<count>.csv (tactical
    frame, x, y m, std m -- negative where no camera picked the ball -- cameras, height m), then
    each camera's own pick (tx, ty tactical; bx, by broadcast, both on the tactical pitch), the
    triangulated height of the two picks (ztri, m) and how far their lines of sight miss (miss, m).
    Written wherever the track is built for scoring, so the viewer shows what was scored."""
    known = ~np.isnan(tr["pos"][:, 0])
    t_on, b_on = ~np.isnan(tr["pick_t"][:, 0]), ~np.isnan(tr["pick_b"][:, 0])
    held = tr.get("held", np.zeros(count, bool))
    path = os.path.join(data.CACHE_DIR, "detections", "balltrack-{}-{}.csv".format(first, count))
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["frame", "x", "y", "std", "src", "z", "tx", "ty", "bx", "by", "ztri", "miss",
                      "itx", "ity", "ibx", "iby", "fill"])
        cell = lambda v: round(float(v), 2) if np.isfinite(v) else ""
        for i in range(count):
            has_in = "in_t" in tr and (np.isfinite(tr["in_t"][i, 0]) or np.isfinite(tr["in_b"][i, 0]))
            if not (known[i] or t_on[i] or b_on[i] or has_in):
                continue
            std = float(tr["pos_std"][i]) if np.isfinite(tr["pos_std"][i]) else 1.0
            src = int(t_on[i]) | (2 * int(b_on[i])) or (4 if held[i] else 0)
            x, y = (float(tr["pos"][i, 0]), float(tr["pos"][i, 1])) if known[i] else (-1.0, -1.0)
            out.writerow([first + i, round(x, 2), round(y, 2),
                          round(std if src & 3 else -max(std, 0.01), 2), src, round(float(tr["z"][i]), 2),
                          cell(tr["pick_t"][i, 0]), cell(tr["pick_t"][i, 1]), cell(tr["pick_b"][i, 0]),
                          cell(tr["pick_b"][i, 1]), cell(tr["z_tri"][i]) if "z_tri" in tr else "",
                          cell(tr["miss"][i]) if "miss" in tr else ""] +
                         ([cell(tr["in_t"][i, 0]), cell(tr["in_t"][i, 1]), cell(tr["in_b"][i, 0]), cell(tr["in_b"][i, 1]),
                           int(tr["fill_t"][i]) | 2 * int(tr["fill_b"][i])] if "in_t" in tr else ["", "", "", "", ""]))
    return os.path.relpath(path, os.path.dirname(HERE))


if __name__ == "__main__":
    main()
