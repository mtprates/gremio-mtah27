"""Rules-based possession: who touched the ball, who controls it, and when it is dead.

Each rule is numbered as in docs/TECHNICAL.md, which has the reasoning behind every
parameter. Parameters come from football and physics, not from the public 5 minutes.

Pipeline (Context holds everything that does not depend on which rules are switched on):

    ball fused across the two cameras (8) -> kinematics -> touches (1) / dwells (1.a) / carries (2)
    trackers standing still for seconds dropped from the player set (9)
    each touch / control checked against the broadcast's nearest player to the fused ball (10)
    team per slot per frame, with the goalkeeper fix (3), the cross-camera check (3.a) and its
    resolution by track consistency (3.b)
    dead-ball intervals: out of bounds + referee delay (5), stillness, local stillness around a
    lost ball (11), broadcast cut-away (4), each grown over the still players around it (12)
    ball direction changes, player or not: a viewer marker only
    a small state machine turning events into H / A / D per frame

Rule 13 (kalman.py) can replace the moving-average ball, and rule 14 (hmm.py) the state machine.
"""

import copy

import numpy as np

import align
from data import AWAY, FPS, HOME, KICKOFFS, PITCH, halves

# Params
INTERP_MAX_GAP = 6          # frames: ball gaps this short are interpolated
SPIKE = 1.6                 # m in one frame (40 m/s) away and back: a tracking glitch, dropped
SMOOTH = 2                  # frames either side in the ball position smoother
DV_WINDOW = 4               # frames either side for v_in / v_out (0.16 s)
DV_THRESHOLD = 3.5          # m/s: rule 1, ~3 sigma of the smoothed velocity noise (median 0.9)
TOUCH_NMS = 6               # frames: minimum spacing of touches
TOUCH_RADIUS = 4.0          # m: rule 1, nearest player must be this close ...
TOUCH_CLOSE = 2.0           # ... and within this, a plain "touch"; between the two, a "far
                            # touch" -- ball tracking is sometimes off by metres (1H 11551),
                            # kept as its own event kind so its effect can be measured
TOUCH_SLACK = 3             # ... at some frame within this many of the peak (smoothing blurs it)
TOUCH_CAP = False           # rule 49 (docs/VALUES.md): no touch where the ball makes a step faster than
TOUCH_MAX_STEP = 40.0       # this (m/s) within TOUCH_SLACK of the peak -- a jump between objects, not a kick
TOUCH_ECHO = False          # rule 58 (docs/VALUES.md): a kicked gap's kink gives way to a stronger peak beside it
TOUCH_LOCAL = False         # rule 54 (docs/VALUES.md): a touch peak must be a local maximum of the score (no
                            # lower than the frame either side) -- a kick's dv hump outlasts +-TOUCH_NMS on the
                            # Kalman ball, and its slopes are no touches
CONTROL_RADIUS = 1.3        # m: rule 1.a dwell
CONTROL_FRAMES = 10         # 0.4 s of the same team at the ball
CARRY_SEEN = 3              # rule 2: tracked frames at the player's feet before the gap
CARRY_RADIUS = 1.5          # m
CARRY_SPEED = (0.5, 1.6)    # player keeps this share of their speed at the moment of loss
CARRY_SLOW = 1.5            # m/s: below this, "similar pace" means staying below 2.5 m/s
CARRY_MAX = 5 * FPS
CARRY_MIN_GAP = INTERP_MAX_GAP + 1
GK_DEPTH = 16.5             # m: rule 3, penalty area depth
GK_HALF_WIDTH = 20.16       # m: half the penalty area width
GK_WINDOW = 3 * FPS         # the goalkeeper pattern must hold over 3 s ...
GK_SHARE = 0.6              # ... for this share of it
GK_RANGE = 30.0             # m: rule 3.c, a goalkeeper's track stays this close to its goal line ...
GK_HOME_SHARE = 0.9         # ... for this share of its frames. Without it an outfielder deepest in
                            # the box for 3 s of his 4 s there was a keeper for his 13-minute track.
NO_KEEPER, KEEPER_UNSEEN = -2, -1  # rule 3.c, what one camera tells the other of a keeper
GK_DEAD_MARGIN = 3 * FPS    # rule 3 exception: no correction within this long of a dead ball.
                            # Around stoppages the goalkeeper is often not the last man (goal
                            # kicks, corners, the keeper out after a catch) -- 1H 39:00. Only as
                            # good as the dead-ball detection it leans on.
UNCERTAIN_WINDOW = FPS      # rule 3.a: 1 s of cross-camera pairings ...
UNCERTAIN_SHARE = 0.6       # ... mostly disagreeing
UNCERTAIN_MIN_PAIRS = 5
CONSISTENCY_WINDOW = 5 * FPS  # rule 3.b: look this far either side of a doubt ...
CONSISTENCY_MARGIN = FPS      # ... and believe the camera whose track of the player covers
                              # at least this much more of it without a break
BROADCAST_DEAD_MIN = 0      # rule 4 (simple): frames with no broadcast homography are dead,
                            # exactly as they are -- no minimum length, no corroboration.
                            # The base the dead-ball rules will build on.
ACCEL_WINDOW = 6            # frames either side for the acceleration estimate (0.24 s)
ACCEL_NEAR_ZERO = 3.0       # m/s^2: about the measurement noise at that window ...
ACCEL_MIN = FPS             # ... for at least 1 s -> POSSIBLE dead ball (evidence only)
AWAY_MIN = 4 * FPS          # rule 4 (corroborated variant): broadcast away this long ...
AWAY_CORROBORATE = 0.7      # ... with the ball untracked or the players still for most of it.
                            # The broadcast is in close-up ~30% of graded frames, live play
                            # included, so a cut-away alone is weak evidence.
REF_DELAY = FPS             # rule 5: referee delay after the ball leaves the pitch
OOB_MARGIN = 0.5            # m beyond a line before the ball counts as out ...
OOB_FRAMES = 3              # ... for at least this many frames, so line-hugging jitter is not
STILL_SPEED = 0.9           # m/s mean player speed: the slowest ~quarter of graded frames
                            # (median 1.45); 1.3 caught ordinary slow build-up play
STILL_BALL = 1.0            # m/s ball speed (or ball untracked)
STILL_MIN = 3 * FPS
LOCAL_RADIUS = 4.0          # m: rule 11, the players this close to the last seen ball ...
LOCAL_SPEED = 1.0           # m/s: ... all slower than this, at least one from each team ...
LOCAL_MIN = 2 * FPS         # ... for this long, with the ball untracked or still: a foul (2H 118553)
LOCAL_MAX_AGE = 10 * FPS    # a ball last seen longer ago than this anchors nothing
LOCAL_BACKDATE = 3 * FPS    # the stretch starts where the ball went quiet, at most this early
QUIET_SPEED = 1.0           # m/s: rule 12, a player slower than this is standing still ...
QUIET_SHARE = 0.6           # ... and this share of the tracked players (at least QUIET_MIN_PLAYERS)
QUIET_MIN_PLAYERS = 10      # standing still, averaged over QUIET_WINDOW, is "players mostly still".
QUIET_WINDOW = FPS          # All trackers count: rule 9 would drop exactly the players who stand.
EXTEND_MAX = 10 * FPS       # rule 12 grows a dead stretch at most this far either way ...
EXTEND_SLACK = FPS // 2     # ... to a touch at most this far outside the still players
TURN_ANGLE = 20.0          # degrees between v_in and v_out: a ball direction change (viewer
TURN_SPEED = 1.5            # marker only) when both sides move at least this fast, m/s ...
TURN_DV = 2.5               # ... or a velocity change this big (rule 1 needs DV_THRESHOLD)
TURN_NMS = 4                # frames: minimum spacing of direction changes
MIN_DEAD = int(1.5 * FPS)   # a restart cannot come sooner than this
RESTART_IN_SPEED = 2.0      # m/s: a restart is a kick from (near) rest ...
RESTART_OUT_SPEED = 3.0     # m/s: ... that sends the ball off at least this fast ...
THROW_IN_LOOKBACK = 3 * FPS # ... or the ball coming back from outside the pitch
RESTART_NEAR = 3 * FPS      # stillness / cut-away stretches end at a restart this close to
                            # where their evidence ends, or else where it ends
RESUME_FRAMES = 12          # out of bounds with no restart found: play resumes once the ball is
RESUME_SPEED = 3.0          # back inside and moving this fast for this many frames
RESTART_MAX = 60 * FPS
FALLBACK_RADIUS = 3.0       # m: nearest-player vote for an alive stretch with no events
BALL_AGREE = 2.0            # m: rule 8, the two cameras' balls this close are averaged (about
                            # the 75th percentile of their distance on live frames) ...
BALL_RECENT = 5             # ... further apart, the one nearer the fused ball of the last this
                            # many frames wins (tactical when there is none)
BALL_BIAS_DECAY = 1.0       # s: a lone camera keeps its recent offset from the fused ball,
                            # fading on this time constant, so switching source is not a jump
                            # (hard switching made up to 1 m steps that read as touches)
CROSS_MARGIN = 0.5          # m: rule 10, the broadcast's nearest player must be this much
                            # closer to the ball than the tactical one to overrule it
STATIC_WINDOW = 3 * FPS     # rule 9: a tracker that stays within ...
STATIC_EXTENT = 1.0         # ... a box this many metres across (diagonal) for this long is
                            # dropped for that time: a stuck tracker or a non-player

LABELS = np.array(["H", "A"])
RESOLVED = "resolved"       # key of the team / doubt sets after rule 3.b
KINDS = ("touch", "control", "carry", "far touch")

GUARD = FPS                 # rule 6: frames after a near-ball dropout during which a switch
                            # needs confirming

# broadcast_dead: False, "no_homography" (simple rule 4) or "corroborated" (the earlier variant)
# Rule 8 (ball fusion) is not a switch here: it changes everything built on the ball, so it is
# a Context argument (Context(..., fuse=False) for the ablation).
V1 = dict(confirm=True, control=True, carry=True, goalie=True, crosscheck=True, resolve=True,
          broadcast_dead="no_homography", ref_delay=REF_DELAY, dropouts=True, static=True,
          crosscam=True, touch_radius=TOUCH_RADIUS, local_still=True, dead_extend="both")
R1 = dict(confirm=False, control=False, carry=False, goalie=False, crosscheck=False,
          resolve=False, broadcast_dead=False, ref_delay=0, dropouts=False, static=False,
          crosscam=False, touch_radius=TOUCH_RADIUS, local_still=False, dead_extend=False)


# -- small array helpers -------------------------------------------------------------------------

def runs(mask):
    """[(start, stop)] of the True runs in a boolean array, stop exclusive."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def window_sum(x, width):
    """Centred moving sum along axis 0, for (n,) or (n, k) arrays."""
    x = np.asarray(x, dtype=np.float64)
    half = width // 2
    padded = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    index = np.arange(len(x))
    hi = np.minimum(len(x), index + half + 1)
    lo = np.maximum(0, index - half)
    return padded[hi] - padded[lo]


def value_runs(values, ignore=-1):
    """[(start, stop, value)] of constant runs, skipping runs equal to `ignore`."""
    values = np.asarray(values)
    if not len(values):
        return []
    change = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(values)]))
    return [(int(s), int(e), int(values[s])) for s, e in zip(starts, stops) if values[s] != ignore]


# -- ball and players ----------------------------------------------------------------------------

BALL_SOURCES = ("none", "tactical only", "broadcast only", "averaged", "tactical (disagree)",
                "broadcast (disagree)", "dropped (lone camera distrusted)")


def fuse_ball(ball_t, ball_b):
    """Rule 8: one ball from both cameras, per tactical frame. ball_b is the broadcast ball on
    the tactical timeline and orientation (NaN where the broadcast is not live).

    Both seen and within BALL_AGREE: their mean. One seen: that one. Both seen but apart: one of
    them is wrong -- either can be (tactical tracking is sometimes the better one), so take the
    one continuing the fused ball of the last BALL_RECENT frames, else tactical. Whenever one
    camera stands alone, it is shifted by its offset from the last mean, fading over
    BALL_BIAS_DECAY, so the fused ball has no step where the source changes.
    A camera that lost a disagreement is not trusted alone until the two agree again: those
    frames have no ball. At 1H 17042 the broadcast ball, 3.3 m off just before the tactical ball
    was lost at an away player's feet, carried on alone towards a home player -- a fake home
    touch, and no ball gap for rule 2's carry.
    Returns (ball (n, 2), source (n,) indexing BALL_SOURCES).
    """
    n = len(ball_t)
    seen_t, seen_b = ~np.isnan(ball_t[:, 0]), ~np.isnan(ball_b[:, 0])
    dist = np.linalg.norm(ball_t - ball_b, axis=1)
    out = np.where(seen_t[:, None], ball_t, ball_b).astype(np.float32)
    source = np.where(seen_t, 1, np.where(seen_b, 2, 0)).astype(np.int8)
    with np.errstate(invalid="ignore"):
        agree = seen_t & seen_b & (dist <= BALL_AGREE)
    source[agree] = 3
    fade = np.exp(-1.0 / (BALL_BIAS_DECAY * FPS))
    bias = {1: np.zeros(2), 2: np.zeros(2)}  # fused minus camera, per camera
    trusted = {1: True, 2: True}             # a camera that lost a disagreement is not
    last, last_t = None, -BALL_RECENT - 1
    for t in range(n):
        if agree[t]:
            out[t] = (ball_t[t] + ball_b[t]) / 2
            bias[1], bias[2] = out[t] - ball_t[t], out[t] - ball_b[t]
            trusted[1] = trusted[2] = True
        else:
            cam = int(source[t])
            if seen_t[t] and seen_b[t]:
                near_b = last is not None and t - last_t <= BALL_RECENT and \
                    np.linalg.norm(ball_b[t] + bias[2] - last) < np.linalg.norm(ball_t[t] + bias[1] - last)
                cam = 2 if near_b else 1
                source[t] = 5 if near_b else 4
                trusted[cam], trusted[3 - cam] = True, False
            elif cam and not trusted[cam]:
                out[t], source[t], cam = np.nan, 6, 0
            if cam:
                out[t] = (ball_t[t] if cam == 1 else ball_b[t]) + bias[cam]
            bias[1] *= fade
            bias[2] *= fade
        if 0 < source[t] < 6:
            last, last_t = out[t], t
    return out, source


def static_trackers(players):
    """Rule 9: (n, slots) True where a tracker stays within a STATIC_EXTENT box for a whole
    STATIC_WINDOW -- every frame of every such window."""
    from numpy.lib.stride_tricks import sliding_window_view
    n, n_slots = players.shape[:2]
    out = np.zeros((n, n_slots), dtype=bool)
    if n < STATIC_WINDOW:
        return out
    for slot in range(n_slots):
        extent = np.zeros(n - STATIC_WINDOW + 1)
        for k in (0, 1):
            w = sliding_window_view(players[:, slot, k], STATIC_WINDOW)
            extent = extent + (w.max(axis=1) - w.min(axis=1)) ** 2  # NaN if any frame untracked
        with np.errstate(invalid="ignore"):
            still = np.sqrt(extent) < STATIC_EXTENT
        out[:, slot] = _cover(still, STATIC_WINDOW, n)
    return out


def broadcast_nearest(ball, bro_players, b_team, alignment):
    """Rule 10: per tactical frame, the broadcast's nearest player to `ball` (the fused ball,
    tactical orientation) and its team; inf / -1 where the broadcast is not live."""
    n = len(ball)
    dist = np.full(n, np.inf, dtype=np.float32)
    team = np.full(n, -1, dtype=np.int8)
    live = np.flatnonzero((alignment["state_t"] == align.STATE_LIVE) & ~np.isnan(ball[:, 0]))
    for start in range(0, len(live), 5000):
        t = live[start:start + 5000]
        b = alignment["b_of_t"][t]
        d = np.linalg.norm(bro_players[b] - ball[t][:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        k = d.argmin(axis=1)
        dist[t] = d[np.arange(len(t)), k]
        team[t] = b_team[b, k]
    return dist, team


def _cover(starts, width, n):
    """Frames covered by a window of `width` beginning at any True index of `starts`."""
    marks = np.zeros(n + 1, dtype=np.int64)
    idx = np.flatnonzero(starts)
    np.add.at(marks, idx, 1)
    np.add.at(marks, idx + width, -1)
    return np.cumsum(marks)[:n] > 0

def drop_spikes(ball):
    """The ball (float64 copy) without one-frame glitches: far from both neighbours, which are close
    to each other -- out and back. Left in, each one is a huge fake velocity change; the first
    version found most of its "touches" on these. (Without the "and back", a ball moving steadily
    faster than SPIKE per frame -- a rising ball's ground point, 52 m/s at 1H 0:03 -- lost every
    point.)"""
    n = len(ball)
    pos = ball.astype(np.float64).copy()
    step_prev = np.full(n, np.nan)
    step_prev[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    step_next = np.full(n, np.nan)
    step_next[:-1] = step_prev[1:]
    across = np.full(n, np.nan)
    across[1:-1] = np.linalg.norm(pos[2:] - pos[:-2], axis=1)
    with np.errstate(invalid="ignore"):
        pos[(step_prev > SPIKE) & (step_next > SPIKE) & (across < SPIKE)] = np.nan
    return pos


def ball_kinematics(ball):
    """Interpolated, smoothed ball position and the velocity change at every frame (rule 1)."""
    n = len(ball)
    pos = drop_spikes(ball)
    seen = ~np.isnan(pos[:, 0])
    for start, stop in runs(~seen):
        if start > 0 and stop < n and stop - start <= INTERP_MAX_GAP:
            for k in (0, 1):
                pos[start:stop, k] = np.interp(np.arange(start, stop), [start - 1, stop],
                                               [pos[start - 1, k], pos[stop, k]])
    present = ~np.isnan(pos[:, 0])

    # Centred mean over 2*SMOOTH+1 frames, only where every frame of the window is present.
    width = 2 * SMOOTH + 1
    filled = np.where(present[:, None], pos, 0.0)
    total = window_sum(filled, width)
    count = window_sum(present, width)
    smooth = np.where((count == width)[:, None], total / np.maximum(count, 1)[:, None], np.nan)
    smooth[~present] = np.nan

    def shifted(k):
        out = np.full_like(smooth, np.nan)
        if k > 0:
            out[:-k] = smooth[k:]
        else:
            out[-k:] = smooth[:n + k]
        return out

    v_in = (smooth - shifted(-DV_WINDOW)) / DV_WINDOW * FPS
    v_out = (shifted(DV_WINDOW) - smooth) / DV_WINDOW * FPS
    speed = np.linalg.norm(shifted(1) - shifted(-1), axis=1) / 2 * FPS
    # Acceleration over a longer window than the touch detector's, so noise stays near 3 m/s^2:
    # velocity change across +-ACCEL_WINDOW frames, divided by the time between the two halves.
    a_in = (smooth - shifted(-ACCEL_WINDOW)) / ACCEL_WINDOW * FPS
    a_out = (shifted(ACCEL_WINDOW) - smooth) / ACCEL_WINDOW * FPS
    accel = np.linalg.norm(a_out - a_in, axis=1) / (ACCEL_WINDOW / FPS)
    return {
        "pos": smooth,
        "present": present & ~np.isnan(smooth[:, 0]),
        "seen": seen,
        "v_in": v_in,
        "v_out": v_out,
        "dv": np.linalg.norm(v_out - v_in, axis=1),
        "speed_in": np.linalg.norm(v_in, axis=1),
        "speed_out": np.linalg.norm(v_out, axis=1),
        "speed": speed,
        "accel": accel,
    }


def possible_dead(kin):
    """[(start, stop)] where the ball's acceleration stays near zero for ACCEL_MIN frames.

    A POSSIBLE dead ball, evidence only: a ball placed for a set piece has zero acceleration,
    but so does one rolling freely. Not used for labels yet.
    """
    with np.errstate(invalid="ignore"):
        calm = kin["present"] & (kin["accel"] < ACCEL_NEAR_ZERO)
    return [r for r in runs(calm) if r[1] - r[0] >= ACCEL_MIN]


def nearest_players(players, ball_pos):
    """Per frame: nearest tracked slot to the ball and its distance (inf where none)."""
    dist = np.linalg.norm(players - ball_pos[:, None, :].astype(np.float32), axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist).astype(np.float32)
    slot = dist.argmin(axis=1)
    return slot, dist[np.arange(len(slot)), slot]


def player_speed(rec):
    """m/s per slot per frame: the exported speed, or from positions where it is missing."""
    derived = np.full(rec.speed.shape, np.nan, dtype=np.float32)
    derived[1:-1] = np.linalg.norm(rec.players[2:] - rec.players[:-2], axis=2) / 2 * FPS
    return np.where(np.isfinite(rec.speed), rec.speed, derived)


def half_of_frames(n, frames_to_tactical=None):
    """Half (1, 2) of every frame, 0 outside the graded halves. For the broadcast, through the
    alignment's mapping onto tactical frames."""
    out = np.zeros(n, dtype=np.int8)
    if frames_to_tactical is None:
        for half, first, last in halves():
            out[first:last + 1] = half
        return out
    t = frames_to_tactical
    for half, first, last in halves():
        out[(t >= first) & (t <= last)] = half
    return out


# -- rule 3: goalkeepers -------------------------------------------------------------------------

def defending_sides(tac):
    """{half: {team: goal x}} read off the kick-off, when every player is in their own half."""
    sides = {}
    length = PITCH[0]
    for half, kickoff in KICKOFFS.items():
        x = tac.players[kickoff:kickoff + 13, :, 0]
        medians = {}
        for team in (HOME, AWAY):
            values = x[:, tac.team == team]
            medians[team] = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
        home_left = not (medians[HOME] > medians[AWAY])
        sides[half] = {HOME: 0.0 if home_left else length, AWAY: length if home_left else 0.0}
    return sides


def goalkeeper_fix(rec, half_of, sides, exempt=None, keepers=None):
    """Team per slot per frame, with each team's goalkeeper on that team. Returns (team, fixes).

    3.c, first: one continuous track is one person. A track that is the deepest object in a
    penalty area for GK_WINDOW frames or more, and for GK_SHARE of its frames inside that area,
    and stays within GK_RANGE of that goal line for GK_HOME_SHARE of its frames, is the defending
    team's goalkeeper for the whole track -- from its first frame to its last,
    not only where the pattern holds (1H 19820: a keeper re-acquired under an away slot was home
    for 10 s, away either side).
    Then, only while no such keeper track is on the pitch for that goal, the deepest object inside
    each penalty area is given to the team defending it wherever that holds for most of 3 s. One
    goalkeeper per team: at 1H 15655 the windowed rule handed the role from the keeper's track to
    an attacker on the goal line and back.

    exempt (n,) marks frames where neither pattern is counted: around dead balls the deepest
    player is often not the goalkeeper.
    keepers {team: (n,) int}: what another camera knows of that team's keeper per frame --
    NO_KEEPER, KEEPER_UNSEEN (it has a keeper track, paired with no object here) or the object
    here that is the keeper. Where a keeper is known, that object takes the team and nobody else
    becomes the keeper: the broadcast, zoomed in, often has the keeper out of view, and its
    deepest visible player is an attacker (1H 15655, broadcast object 26).
    Returns (team, fixes, keepers found here: {team: (n,) int}, the keeper's slot or NO_KEEPER)."""
    n, n_slots = rec.players.shape[:2]
    exempt = np.zeros(n, dtype=bool) if exempt is None else exempt
    team = np.broadcast_to(rec.team, (n, n_slots)).copy()
    x, y = rec.players[..., 0], rec.players[..., 1]
    tracked = ~np.isnan(x)
    fixes = []
    found = {team: np.full(n, NO_KEEPER, dtype=np.int64) for goals in sides.values() for team in goals}
    for half, goals in sides.items():
        frames = np.flatnonzero(half_of == half)
        if not len(frames):
            continue
        xs, ys, tr = x[frames], y[frames], tracked[frames]
        for defender, goal_x in goals.items():
            depth = np.where(tr, np.abs(xs - goal_x), np.inf)
            deepest = depth.argmin(axis=1)
            rows = np.arange(len(frames))
            in_box = (depth[rows, deepest] <= GK_DEPTH) & \
                     (np.abs(ys[rows, deepest] - PITCH[1] / 2) <= GK_HALF_WIDTH)
            allowed = ~exempt[frames]
            given = keepers[defender][frames] if keepers else np.full(len(frames), NO_KEEPER)
            known = given != NO_KEEPER
            # The other camera's keeper, where it is paired with an object here, is that keeper.
            for slot in np.unique(given[given >= 0]):
                on = (given == slot) & tr[:, slot] & (rec.team[slot] != defender)
                team[frames[on], slot] = defender
                for start, stop in runs(on):
                    fixes.append((int(slot), int(frames[start]), int(frames[stop - 1]) + 1, int(defender)))
            # Where the keeper is known, only that object can count as the keeper here.
            allowed_here = allowed & (~known | (deepest == given))

            # 3.c: goalkeeper tracks, whole.
            with np.errstate(invalid="ignore"):
                inside = (depth <= GK_DEPTH) & (np.abs(ys - PITCH[1] / 2) <= GK_HALF_WIDTH) & allowed[:, None]
            keeper_on = np.zeros(len(frames), dtype=bool)
            deep_counts = np.bincount(deepest[in_box & allowed_here], minlength=n_slots)
            for slot in np.flatnonzero(deep_counts >= GK_WINDOW):
                for s, e in runs(tr[:, slot]):
                    n_deep = int(np.sum((deepest[s:e] == slot) & in_box[s:e] & allowed_here[s:e]))
                    if n_deep < GK_WINDOW or n_deep < GK_SHARE * inside[s:e, slot].sum():
                        continue
                    if np.mean(depth[s:e, slot] <= GK_RANGE) < GK_HOME_SHARE:
                        continue
                    keeper_on[s:e] = True
                    found[defender][frames[s:e]] = slot
                    if rec.team[slot] != defender:
                        team[frames[s:e], slot] = defender
                        fixes.append((int(slot), int(frames[s]), int(frames[e - 1]) + 1, int(defender)))

            keeper_on |= known
            wrong = in_box & (rec.team[deepest] != defender) & allowed & ~keeper_on
            for slot in np.unique(deepest[wrong]):
                candidate = wrong & (deepest == slot)
                share = window_sum(candidate, GK_WINDOW) / window_sum(np.ones(len(frames)), GK_WINDOW)
                flip = (share >= GK_SHARE) & tr[:, slot] & allowed & ~keeper_on
                if not flip.any():
                    continue
                team[frames[flip], slot] = defender
                # Report in frame units; frames of one half are contiguous for the tactical
                # recording, and for the broadcast a run here can span a mapping gap -- fine
                # for display.
                for start, stop in runs(flip):
                    fixes.append((int(slot), int(frames[start]), int(frames[stop - 1]) + 1, int(defender)))
    return team, fixes, found


# -- rule 15: cross-camera pairing by identity ---------------------------------------------------

PAIR_STICK = 1.0            # m: rule 15, what leaving a still-visible partner costs -- above the
                            # cameras' ~0.7 m typical disagreement, so only a clearly better
                            # partner wins
_PAIR_CACHE = {}


def sticky_pairs(players_t, players_b, alignment):
    """Rule 15: per tactical frame, each tactical slot's broadcast object (-1: none), one to one
    and keeping yesterday's partner.

    align.py pairs each slot with its nearest broadcast object, every frame on its own and not one
    to one. At 1H 11400 an away player (slot 31) went behind a home player (slot 18): slot 18
    took the away player's broadcast object, rule 3.a raised a doubt, rule 3.b believed the
    broadcast and made slot 18 away, and rule 6 inserted slot 18's own broadcast object as an
    extra home player. Here each live frame is an optimal assignment within align.PAIR_RADIUS,
    where pairing a slot with anything but its previous partner, while that partner is still in
    view, costs PAIR_STICK -- and so does taking an object whose previous slot is still in view.
    players_b are the broadcast players in tactical orientation, on broadcast frames."""
    key = (id(alignment), id(players_t))  # players_b follows from the alignment
    if key in _PAIR_CACHE:
        return _PAIR_CACHE[key]
    from scipy.optimize import linear_sum_assignment
    n, n_t = players_t.shape[:2]
    n_b = players_b.shape[1]
    pairs = np.full((n, n_t), -1, dtype=np.int16)
    b_of_t = alignment["b_of_t"]
    owner = np.full(n_b, -1, dtype=np.int64)       # broadcast object -> its slot last frame
    partner = np.full(n_t, -1, dtype=np.int64)     # slot -> its object last frame
    last = -2
    for t in np.flatnonzero(alignment["state_t"] == align.STATE_LIVE):
        if t != last + 1:
            owner[:], partner[:] = -1, -1
        last = t
        tp, bp = players_t[t], players_b[b_of_t[t]]
        ti = np.flatnonzero(~np.isnan(tp[:, 0]))
        bi = np.flatnonzero(~np.isnan(bp[:, 0]))
        if not len(ti) or not len(bi):
            owner[:], partner[:] = -1, -1
            continue
        d = np.linalg.norm(tp[ti][:, None, :] - bp[bi][None, :, :], axis=2)
        cost = d.copy()
        mine = partner[ti]
        keeps = np.isin(mine, bi)                              # previous partner still in view
        cost += np.where(keeps[:, None] & (bi[None, :] != mine[:, None]), PAIR_STICK, 0.0)
        theirs = owner[bi]
        held = np.isin(theirs, ti)                             # previous slot still in view
        cost += np.where(held[None, :] & (theirs[None, :] != ti[:, None]), PAIR_STICK, 0.0)
        feasible = d <= align.PAIR_RADIUS
        # Staying unpaired is an option at the radius' cost, one dummy column per slot -- else
        # the assignment maximises the number of pairs and pushes a slot that lost its object to
        # someone else within reach, often on the other team.
        unpaired = np.full((len(ti), len(ti)), 1e6)
        np.fill_diagonal(unpaired, align.PAIR_RADIUS)
        rows, cols = linear_sum_assignment(np.hstack([np.where(feasible, cost, 1e6), unpaired]))
        real = cols < len(bi)
        rows, cols = rows[real], cols[real]
        ok = feasible[rows, cols]
        pairs[t, ti[rows[ok]]] = bi[cols[ok]]
        owner[:], partner[:] = -1, -1
        partner[ti[rows[ok]]] = bi[cols[ok]]
        owner[bi[cols[ok]]] = ti[rows[ok]]
    _PAIR_CACHE[key] = pairs
    return pairs


def pairing_switches(pairs):
    """Times a slot's broadcast partner changes from one object to another on consecutive frames."""
    a, b = pairs[:-1].astype(np.int64), pairs[1:].astype(np.int64)
    return int(np.sum((a >= 0) & (b >= 0) & (a != b)))


# -- rule 3.a: cross-camera team check -----------------------------------------------------------

def crosscheck(team_t, team_b, alignment):
    """Per tactical slot per frame: does the paired broadcast object mostly carry the other team?

    Returns (uncertain, agreement share, swapped) -- swapped if the two exports list the teams
    in opposite order, which is detected rather than assumed.
    """
    pairs = alignment["pairs_t"].astype(np.int64)
    b_of_t = alignment["b_of_t"]
    has = pairs >= 0
    rows, cols = np.nonzero(has)
    b_team = team_b[b_of_t[rows], pairs[rows, cols]]
    agree = b_team == team_t[rows, cols]
    share = float(agree.mean()) if len(agree) else float("nan")
    swapped = bool(len(agree)) and share < 0.5
    if swapped:
        agree = ~agree
        share = 1 - share
    disagree = np.zeros(team_t.shape, dtype=np.float32)
    disagree[rows, cols] = ~agree
    pairs_in_window = window_sum(has.astype(np.float32), UNCERTAIN_WINDOW)
    against = window_sum(disagree, UNCERTAIN_WINDOW)
    uncertain = (pairs_in_window >= UNCERTAIN_MIN_PAIRS) & \
                (against >= UNCERTAIN_SHARE * np.maximum(pairs_in_window, 1))
    return uncertain, share, swapped


def _segment_at(segments, t):
    """(start, stop) of the run in `segments` (sorted starts, stops) holding frame t, or None."""
    starts, stops = segments
    i = np.searchsorted(starts, t, side="right") - 1
    return (int(starts[i]), int(stops[i])) if i >= 0 and t < stops[i] else None


def _segments(tracked):
    r = runs(tracked)
    return (np.array([a for a, _ in r], dtype=np.int64), np.array([b for _, b in r], dtype=np.int64))


def resolve_doubt(players_t, team_t, players_b, team_b, alignment, uncertain):
    """Rule 3.b: settle each rule-3.a doubt by which camera tracked the player consistently.

    Tracked players do not vanish mid-view, so a track that breaks is the one to distrust: a
    player the tactical camera dropped often comes back in a slot of the wrong team (1H 11409).
    For each doubtful run, compare how much of the CONSISTENCY_WINDOW before it the tactical
    slot's track and its paired broadcast object's track cover without a break (a track born
    mid-view is as suspect as one that vanishes); on a tie, the same for the window after it.
      broadcast clearly longer -> the broadcast team, for the whole tactical track (one person)
                                  if the broadcast disagrees over most of it, else for the doubt
      tactical clearly longer  -> the tactical team stands; the doubt is cleared
      otherwise                -> still in doubt
    Returns (team, uncertain, fixes [(slot, start, stop, team)], outcome counts).
    """
    n, n_slots = team_t.shape
    pairs = alignment["pairs_t"].astype(np.int64)
    b_of_t = alignment["b_of_t"].astype(np.int64)
    offset_b = alignment["offset_b"].astype(np.int64)
    team, left = team_t.copy(), uncertain.copy()
    fixes, outcome = [], {"broadcast": 0, "tactical": 0, "undecided": 0}
    tracked_b = ~np.isnan(players_b[..., 0])
    b_segments = {}
    for slot in range(n_slots):
        t_segments = _segments(~np.isnan(players_t[:, slot, 0]))
        for s, e in runs(uncertain[:, slot]):
            if not left[s:e, slot].any():
                continue  # settled with an earlier run of the same track
            paired = np.flatnonzero(pairs[s:e, slot] >= 0) + s
            if not len(paired):
                outcome["undecided"] += 1
                continue
            objs, counts = np.unique(pairs[paired, slot], return_counts=True)
            obj = int(objs[counts.argmax()])
            at = int(paired[pairs[paired, slot] == obj][0])
            if obj not in b_segments:
                b_segments[obj] = _segments(tracked_b[:, obj])
            bf = int(b_of_t[at])
            seg_t = _segment_at(t_segments, at)
            seg_b = _segment_at(b_segments[obj], bf)
            if seg_t is None or seg_b is None:
                outcome["undecided"] += 1
                continue
            # The broadcast run, onto the tactical timeline.
            seg_b = (seg_b[0] + int(offset_b[seg_b[0]]), seg_b[1] + int(offset_b[seg_b[1] - 1]))
            # History first: a track that begins mid-view has no identity to carry over, and
            # the team during the doubt is inherited from before it. What follows the doubt
            # (does the track vanish?) only breaks a tie.
            def cover(seg, lo, hi):
                return max(0, min(hi, seg[1]) - max(lo, seg[0]))
            before = (max(0, s - CONSISTENCY_WINDOW), s)
            after = (e, min(n, e + CONSISTENCY_WINDOW))
            lead = cover(seg_b, *before) - cover(seg_t, *before)
            if abs(lead) < CONSISTENCY_MARGIN:
                lead = cover(seg_b, *after) - cover(seg_t, *after)
            other = int(team_b[bf, obj])
            if lead >= CONSISTENCY_MARGIN and other != team_t[at, slot]:
                # The whole track only if the broadcast has it on the other team for most of
                # its paired frames; a re-acquired track can also start on the wrong person and
                # then settle on the right one, so otherwise just the doubtful stretch.
                rows = np.flatnonzero(pairs[seg_t[0]:seg_t[1], slot] >= 0) + seg_t[0]
                against = (team_b[b_of_t[rows], pairs[rows, slot]] == other).mean()
                a, b = seg_t if against >= UNCERTAIN_SHARE else (max(s, seg_t[0]), min(e, seg_t[1]))
                team[a:b, slot] = other
                left[s:e, slot] = False
                left[a:b, slot] = False
                fixes.append((slot, a, b, other))
                outcome["broadcast"] += 1
            elif lead <= -CONSISTENCY_MARGIN:
                left[s:e, slot] = False
                outcome["tactical"] += 1
            else:
                outcome["undecided"] += 1
    return team, left, fixes, outcome


# -- rules 1, 1.a, 2: events ---------------------------------------------------------------------

def _peaks(score, threshold, spacing, local=False):
    """Frames where score exceeds threshold, strongest first, no two within spacing. local: only
    local maxima, no lower than the frame either side (rule 54) -- else a hump wider than spacing
    gives extra picks on its slopes."""
    score = np.where(np.isfinite(score), score, 0.0)
    above = score > threshold
    if local and len(score) > 1:
        above[1:] &= score[1:] >= score[:-1]
        above[:-1] &= score[:-1] >= score[1:]
    candidates = np.flatnonzero(above)
    taken = np.zeros(len(score), dtype=bool)
    picked = []
    for t in candidates[np.argsort(-score[candidates], kind="stable")]:
        if not taken[max(0, t - spacing):t + spacing + 1].any():
            taken[t] = True
            picked.append(int(t))
    return np.array(sorted(picked), dtype=np.int64)


def touch_times(kin):
    """Local maxima of the ball's velocity change above DV_THRESHOLD, TOUCH_NMS apart. On the
    Kalman ball (rule 13) the change must also stand out from the filters' own uncertainty."""
    score = kin["dv"]
    if "dv_z" in kin:
        import kalman
        with np.errstate(invalid="ignore"):
            score = np.where(kin["dv_z"] >= kalman.TOUCH_Z, score, 0.0)
    peaks = _peaks(score, DV_THRESHOLD, TOUCH_NMS, local=TOUCH_LOCAL)
    if "gaps" in kin:
        # Rule 13: a ball gap (its last and first seen frames included) has one touch or none,
        # decided by the smoother over the whole gap; peaks inside it, and the touch's echo just
        # outside it, give way.
        keep = np.ones(len(peaks), dtype=bool)
        kinks = []
        clear = np.ones(len(peaks), dtype=bool)     # rule 58: peaks outside every gap, the only ones that can stand
        for lo, hi, _, _ in kin["gaps"]:
            clear &= (peaks < lo) | (peaks > hi)
        for lo, hi, kink, kicked in kin["gaps"]:
            keep &= (peaks < lo) | (peaks > hi)
            if kicked:
                near = clear & (np.abs(peaks - kink) <= TOUCH_NMS)
                if TOUCH_ECHO and near.any() and np.nanmax(kin["dv"][peaks[near]]) > np.nan_to_num(kin["dv"][kink]):
                    continue    # rule 58: a stronger measured peak beside the gap; the kink is its echo
                kinks.append(kink)
                keep &= np.abs(peaks - kink) > TOUCH_NMS
        peaks = np.unique(np.concatenate([peaks[keep], np.array(kinks, dtype=np.int64)]))
    if TOUCH_CAP and len(peaks):
        # Rule 49: a step no ball could make, near the peak, means the track jumped; the "kick" is that jump.
        step = np.zeros(len(kin["pos"]))
        with np.errstate(invalid="ignore"):
            step[1:] = np.nan_to_num(np.linalg.norm(np.diff(kin["pos"], axis=0), axis=1)) * FPS
        fast = np.flatnonzero(step > TOUCH_MAX_STEP)
        if len(fast):
            i = np.searchsorted(fast, peaks - TOUCH_SLACK)
            near = (i < len(fast)) & (fast[np.minimum(i, len(fast) - 1)] <= peaks + TOUCH_SLACK)
            peaks = peaks[~near]
    return peaks


def direction_changes(kin):
    """Every frame the ball turns by TURN_ANGLE or changes velocity by TURN_DV, player or not.

    A viewer marker, more sensitive than rule 1 and not used for labels: a direction change
    nobody is credited with is a header, a deflection or a tracking artefact worth looking at.
    [(frame, angle in degrees or -1 when the ball was too slow to have a direction, dv m/s)].
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(kin["v_in"] * kin["v_out"], axis=1) / (kin["speed_in"] * kin["speed_out"])
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        moving = (kin["speed_in"] >= TURN_SPEED) & (kin["speed_out"] >= TURN_SPEED)
    angle = np.where(moving & np.isfinite(angle), angle, -1.0)
    score = np.maximum(angle / TURN_ANGLE, np.nan_to_num(kin["dv"], nan=0.0) / TURN_DV)
    return [(int(t), float(angle[t]), float(kin["dv"][t])) for t in _peaks(score, 1.0, TURN_NMS)]


def dwell_runs(near_slot, near_dist, present, team):
    """(start, stop, team) where one team stays at the ball for CONTROL_FRAMES or more."""
    rows = np.arange(len(near_slot))
    close = present & (near_dist <= CONTROL_RADIUS)
    who = np.where(close, team[rows, near_slot], -1)
    return [(s, e, v) for s, e, v in value_runs(who) if e - s >= CONTROL_FRAMES]


def carries(kin, rec, speed, near_slot, near_dist):
    """Rule 2: (loss_frame, stop, slot) where the ball was lost at a player's feet and that
    player carried on at a similar pace."""
    out = []
    present = kin["present"]
    for start, stop in runs(~present):
        if stop - start < CARRY_MIN_GAP or start < CARRY_SEEN:
            continue
        before = np.arange(start - CARRY_SEEN, start)
        slots = near_slot[before]
        if not (slots == slots[0]).all() or (near_dist[before] > CARRY_RADIUS).any():
            continue
        slot = int(slots[0])
        v0 = speed[start - 1, slot]
        if not np.isfinite(v0):
            continue
        end = start
        for t in range(start, min(stop, start + CARRY_MAX)):
            v = speed[t, slot]
            if np.isnan(rec.players[t, slot, 0]) or not np.isfinite(v):
                break
            similar = (v <= 2.5) if v0 < CARRY_SLOW else (CARRY_SPEED[0] * v0 <= v <= CARRY_SPEED[1] * v0)
            if not similar:
                break
            end = t + 1
        if end - start >= CARRY_MIN_GAP:
            out.append((start, end, slot))
    return out


# -- dead ball -----------------------------------------------------------------------------------

def local_stillness(ctx, ball_still):
    """Rule 11: [(start, stop)] where the players near the last seen ball -- at least one from
    each team -- all stand still while the ball is untracked or still.

    A foul: the ball is lost in the challenge and the players involved stop, while the rest of
    the pitch may not, which is why the global stillness rule misses it. Needing both teams keeps
    out a goalkeeper holding the ball (live). Starts where the ball went quiet, up to
    LOCAL_BACKDATE before the players' stillness is confirmed: the whistle comes first.
    """
    n = ctx.tac.n
    frames = np.arange(n)
    last = np.maximum.accumulate(np.where(ctx.kin["present"], frames, -1))
    anchored = (last >= 0) & (frames - last <= LOCAL_MAX_AGE)
    anchor = ctx.kin["pos"][np.maximum(last, 0)].astype(np.float32)
    with np.errstate(invalid="ignore"):
        near = np.linalg.norm(ctx.tac.players - anchor[:, None, :], axis=2) <= LOCAL_RADIUS
        moving = near & ~(ctx.speed < LOCAL_SPEED)
    team = ctx.tac.team[None, :]
    both = (near & (team == HOME)).any(axis=1) & (near & (team == AWAY)).any(axis=1)
    calm = anchored & ball_still & both & ~moving.any(axis=1)
    out = []
    for s, e in runs(calm):
        if e - s < LOCAL_MIN:
            continue
        start = s
        while start > max(0, s - LOCAL_BACKDATE) and ball_still[start - 1]:
            start -= 1
        out.append((start, e))
    return out


def quiet_players(ctx):
    """Rule 12: (n,) True where most tracked players stand still, smoothed over QUIET_WINDOW."""
    tracked = np.isfinite(ctx.speed)
    count = tracked.sum(axis=1)
    with np.errstate(invalid="ignore"):
        still = (ctx.speed < QUIET_SPEED).sum(axis=1)
    usable = count >= QUIET_MIN_PLAYERS
    share = np.where(usable, still / np.maximum(count, 1), 0.0)
    frames = window_sum(usable, QUIET_WINDOW)
    return (frames > 0) & (window_sum(share, QUIET_WINDOW) >= QUIET_SHARE * np.maximum(frames, 1))


def _merge(intervals):
    """Sorted (start, stop, reason) with overlapping or touching stretches joined; the earlier
    stretch's reason is kept."""
    merged = []
    for s, e, reason in sorted(intervals):
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e, merged[-1][2])
        elif e > s:
            merged.append((s, e, reason))
    return merged


def extend_dead(merged, quiet, touches, sides):
    """Rule 12: grow each dead stretch over the still players either side of it.

    Before: back while most players stand still, then to the last touch before the stoppage
    (at most EXTEND_SLACK before the stillness began), or to where the stillness began when
    there is none -- the stoppage happened at that touch, the players stopped after it. After:
    forward while they stand still, then to the touch where movement begins -- the restart --
    or to where the stillness ended. sides: "both", "before" or "after". Each side at most
    EXTEND_MAX, inside the half. Returns the new merged list and the (start, stop) added.
    """
    bounds = [(first, last + 1) for _, first, last in halves()]
    out, added = [], []
    for s, e, reason in merged:
        lo, hi = next(((a, b) for a, b in bounds if a <= s < b), (s, e))
        new_s, new_e = s, e
        if sides in ("both", "before"):
            b = s
            while b > max(lo, s - EXTEND_MAX) and quiet[b - 1]:
                b -= 1
            if b < s:
                before = touches[(touches >= b - EXTEND_SLACK) & (touches < s)]
                new_s = max(lo, int(before[-1]) if len(before) else b)
        if sides in ("both", "after"):
            f = e
            while f < min(hi, e + EXTEND_MAX) and quiet[f]:
                f += 1
            if f > e:
                after = touches[(touches >= e) & (touches <= f + EXTEND_SLACK)]
                new_e = min(hi, max(e, int(after[-1]) if len(after) else f))
        if new_s < s:
            added.append((new_s, s))
        if new_e > e:
            added.append((e, new_e))
        out.append((new_s, new_e, reason))
    return _merge(out), added


def dead_intervals(ctx, ref_delay, broadcast_mode, local=True, extend="both"):
    """[(start, stop, reason)] of dead-ball stretches, merged, plus the raw evidence runs.

    broadcast_mode: False, "no_homography" (every frame the broadcast has no homography is
    dead, exactly) or "corroborated" (a cut-away of 4 s or more with a quiet ball / players).
    local: rule 11, local stillness around a lost ball.
    extend: rule 12, grow each stretch over still players -- False, "both", "before", "after".
    """
    if broadcast_mode is True:
        broadcast_mode = "corroborated"
    n = ctx.tac.n
    length, width = PITCH
    pos, present = ctx.kin["pos"], ctx.kin["present"]
    with np.errstate(invalid="ignore"):
        inside = present & (pos[:, 0] >= -OOB_MARGIN) & (pos[:, 0] <= length + OOB_MARGIN) & \
                 (pos[:, 1] >= -OOB_MARGIN) & (pos[:, 1] <= width + OOB_MARGIN)
        ball_still = ~present | (np.nan_to_num(ctx.kin["speed"], nan=0.0) < STILL_BALL)
        slow = np.nan_to_num(ctx.mean_speed, nan=0.0) < STILL_SPEED
    outside = present & ~inside

    evidence = {"out of bounds": [r for r in runs(outside) if r[1] - r[0] >= OOB_FRAMES],
                "stillness": [r for r in runs(slow & ball_still) if r[1] - r[0] >= STILL_MIN],
                "local stillness": local_stillness(ctx, ball_still),
                "broadcast away": [],
                "broadcast no homography": [r for r in runs(ctx.alignment["state_t"] == align.STATE_CLOSEUP)
                                            if r[1] - r[0] >= BROADCAST_DEAD_MIN],
                "POSSIBLE dead (ball accel ~0)": possible_dead(ctx.kin)}
    # (start, evidence end or None, reason). Out of bounds has no "end" of its own: the ball
    # stays dead until a restart.
    triggers = [(s + ref_delay, None, "out of bounds") for s, _ in evidence["out of bounds"]]
    triggers += [(s, e, "stillness") for s, e in evidence["stillness"]]
    if local:
        triggers += [(s, e, "local stillness") for s, e in evidence["local stillness"]]
    exact = []
    if broadcast_mode == "no_homography":
        # The simplest dead-ball rule: dead exactly where the broadcast has no homography.
        exact = [(s, e, "broadcast no homography") for s, e in evidence["broadcast no homography"]]
    elif broadcast_mode == "corroborated":
        away = np.isin(ctx.alignment["state_t"], (align.STATE_REPLAY, align.STATE_CLOSEUP))
        quiet = ~present | slow
        for s, e in runs(away):
            if e - s >= AWAY_MIN and np.mean(quiet[s:e]) >= AWAY_CORROBORATE:
                evidence["broadcast away"].append((s, e))
                triggers.append((s, e, "broadcast away"))

    # Restarts: a touch that kicks the ball off from (near) rest, or brings it back in.
    touches = ctx.touch_frames
    out_recent = window_sum(outside, 2 * THROW_IN_LOOKBACK) > 0  # approximate lookback
    kin = ctx.kin
    restarts = np.array([t for t in touches if inside[t] and kin["speed_out"][t] >= RESTART_OUT_SPEED
                         and (kin["speed_in"][t] < RESTART_IN_SPEED or out_recent[t])], dtype=np.int64)
    moving = inside & (np.nan_to_num(kin["speed"], nan=0.0) >= RESUME_SPEED)
    resumed = window_sum(moving, RESUME_FRAMES) >= RESUME_FRAMES

    intervals = []
    for half, first, last in halves():
        stop_half = last + 1
        if half == 2 and KICKOFFS[2] > first:
            intervals.append((first, KICKOFFS[2], "before kick-off"))
        for start, end, reason in sorted(t for t in triggers if first <= t[0] <= last):
            if end is None:
                cap = min(stop_half, start + RESTART_MAX)
                found = restarts[(restarts >= start + MIN_DEAD) & (restarts < cap)]
                back = np.flatnonzero(resumed[start + MIN_DEAD:cap])
                options = ([int(found[0])] if len(found) else []) + \
                          ([start + MIN_DEAD + int(back[0])] if len(back) else [])
                stop = min(options) if options else cap
            else:
                near = restarts[(restarts >= max(start + MIN_DEAD, end - RESTART_NEAR)) &
                                (restarts <= end + RESTART_NEAR)]
                stop = int(near[0]) if len(near) else end
            intervals.append((start, min(stop, stop_half), reason))
        intervals += [(max(s, first), min(e, stop_half), reason) for s, e, reason in exact
                      if s <= last and e > first]

    merged = _merge(intervals)
    quiet = quiet_players(ctx)
    evidence["players mostly still"] = [r for r in runs(quiet) if r[1] - r[0] >= QUIET_WINDOW]
    evidence["dead extended (12)"] = []
    if extend:
        merged, evidence["dead extended (12)"] = extend_dead(merged, quiet, touches, extend)
    dead = np.zeros(n, dtype=bool)
    for s, e, _ in merged:
        dead[s:e] = True
    return dead, merged, evidence


# -- context and prediction ----------------------------------------------------------------------

class Context(object):
    """Everything that does not depend on the rule switches, computed once."""

    def __init__(self, tac, bro, alignment, fuse=True, kalman=False, identity=True, sticky=True, kin=None):
        """kin: ball kinematics computed elsewhere (rule 18, the detector's ball track), used in
        place of every ball the ATD would give."""
        bro = copy.copy(bro)  # oriented and team-swapped below; the caller's stays as loaded
        # Rule 15: pair the cameras' players by identity, not by nearest neighbour per frame.
        # Rule 16: cut tactical tracks where the person in them changes, and move each piece to its
        # person's column. Everything below runs on the re-slotted recording and reads the pairing
        # from here (3.a, 3.b, 3.c across cameras, 6).
        b_oriented = align.orient(bro.players, tuple(alignment["flip"]))
        pairs = sticky_pairs(tac.players, b_oriented, alignment) if sticky else alignment["pairs_t"]
        self.raw_tac, self.cuts, self.moves = tac, [], []
        if identity:
            import identity as idn
            self.cuts = idn.find_cuts(tac.players, pairs, b_oriented, alignment, bro.team, tac.team)
            tac, pairs, self.moves = idn.reslot(tac, pairs, self.cuts)
        alignment = dict(alignment, pairs_nearest=alignment["pairs_t"], pairs_t=pairs)
        self.tac, self.bro, self.alignment = tac, bro, alignment

        # Rule 8: the broadcast ball on the tactical timeline, on live frames only (replays and
        # close-ups map elsewhere or have no homography).
        live = np.flatnonzero(alignment["state_t"] == align.STATE_LIVE)
        self.ball_b = np.full(tac.ball.shape, np.nan, dtype=np.float32)
        self.ball_b[live] = align.orient(bro.ball[alignment["b_of_t"][live]], tuple(alignment["flip"]))
        if fuse:
            self.ball, self.ball_source = fuse_ball(tac.ball, self.ball_b)
        else:
            self.ball = tac.ball
            self.ball_source = np.where(np.isnan(tac.ball[:, 0]), 0, 1).astype(np.int8)
        # Rule 13: a Kalman smoother instead of the moving average (kalman.py) -- on rule 8's
        # fused ball (kalman=True), or 13.b, on both cameras' balls directly (kalman="both").
        if kin is not None:
            self.kin = kin
        elif kalman == "3d":
            # Rule 17: the ball in 3D from both cameras' lines of sight and ballistic flights.
            import ball3d
            self.kin = ball3d.kinematics(tac, self.ball_b, bro, alignment)
        elif kalman == "both":
            import kalman as kf
            self.kin = kf.ball_kinematics_two(tac.ball, self.ball_b)
        elif kalman:
            import kalman as kf
            self.kin = kf.ball_kinematics(self.ball, self.ball_source)
        else:
            self.kin = ball_kinematics(self.ball)
        self.speed = player_speed(tac)
        with np.errstate(all="ignore"):
            self.mean_speed = np.nanmean(self.speed, axis=1)
        self.half_of = half_of_frames(tac.n)
        self.sides = defending_sides(tac)
        self.touch_frames = touch_times(self.kin)
        self.turns = direction_changes(self.kin)

        # Rule 3 exception: frames near a dead ball, by the default dead-ball rules (which use
        # only the ball, the touches, player speeds and the broadcast -- not team labels).
        dead_default, _, _ = dead_intervals(self, REF_DELAY, V1["broadcast_dead"], V1["local_still"],
                                            V1["dead_extend"])
        self.gk_exempt = window_sum(dead_default, 2 * GK_DEAD_MARGIN + 1) > 0

        base = np.broadcast_to(tac.team, tac.players.shape[:2]).copy()
        fixed, self.gk_fixes, keepers = goalkeeper_fix(tac, self.half_of, self.sides, self.gk_exempt)
        self.team = {False: base, True: fixed}

        # The broadcast, in tactical orientation and with its own goalkeeper fix, for rule 3.a.
        b_frames_t = np.arange(bro.n, dtype=np.int64) + alignment["offset_b"].astype(np.int64)
        b_frames_t[alignment["offset_b"] == align.MISSING] = -1
        bro.players = b_oriented
        b_half = half_of_frames(bro.n, b_frames_t)
        b_base = np.broadcast_to(bro.team, bro.players.shape[:2]).copy()
        _, self.agreement, self.swapped = crosscheck(base, b_base, alignment)
        if self.swapped:
            bro.team = (1 - bro.team).astype(np.int8)
        b_exempt = np.zeros(bro.n, dtype=bool)
        mapped = b_frames_t >= 0
        b_exempt[mapped] = self.gk_exempt[np.clip(b_frames_t[mapped], 0, tac.n - 1)]
        # One keeper per team across cameras: the tactical keeper tracks, on broadcast frames.
        # The broadcast object paired with the tactical keeper (rule 3.a pairing) is that keeper.
        b_keepers = {team: np.full(bro.n, NO_KEEPER, dtype=np.int64) for team in keepers}
        t_of = np.clip(b_frames_t, 0, tac.n - 1)
        for team, slot in keepers.items():
            k = np.where(mapped, slot[t_of], NO_KEEPER)
            pair = alignment["pairs_t"][t_of, np.maximum(k, 0)].astype(np.int64)
            b_keepers[team] = np.where(k >= 0, np.where(pair >= 0, pair, KEEPER_UNSEEN), NO_KEEPER)
        b_fixed, _, _ = goalkeeper_fix(bro, b_half, self.sides, b_exempt, b_keepers)
        b_base = np.broadcast_to(bro.team, bro.players.shape[:2]).copy()
        self.b_team = b_fixed
        self.uncertain = {
            False: crosscheck(base, b_base, alignment)[0],
            True: crosscheck(fixed, b_fixed, alignment)[0],
        }
        # Rule 3.b, on top of rule 3: the RESOLVED team and doubt sets.
        resolved, left, self.camera_fixes, self.resolve_outcome = resolve_doubt(
            tac.players, fixed, bro.players, b_fixed, alignment, self.uncertain[True])
        self.team[RESOLVED], self.uncertain[RESOLVED] = resolved, left

        # Rule 9: trackers standing still, dropped from the player set for that time.
        self.static = static_trackers(tac.players)
        self.players_moving = np.where(self.static[..., None], np.nan, tac.players)

        # Player sets the nearest-player rules can run on, keyed (rule 6, rule 9): the tactical
        # slots as tracked or corrected by rule 6 (see add_rule6), with or without the trackers
        # standing still.
        self.sets = {(False, False): self._lookups(tac.players, self.team, self.uncertain),
                     (False, True): self._lookups(self.players_moving, self.team, self.uncertain)}
        base_set = self.sets[(False, False)]
        self.near_slot, self.near_dist = base_set["near_slot"], base_set["near_dist"]
        self.touch_slot, self.touch_dist = base_set["touch_slot"], base_set["touch_dist"]
        self.carries = carries(self.kin, tac, self.speed, self.near_slot, self.near_dist)
        self.rule6 = None
        self.b_near_dist, self.b_near_team = broadcast_nearest(
            self.kin["pos"], bro.players, self.b_team, alignment)
        self.crosscam_overrides = []

    def _lookups(self, players, team, uncertain):
        near_slot, near_dist = nearest_players(players, self.kin["pos"])
        # The toucher: nearest player at the closest moment within TOUCH_SLACK of the peak.
        touch_slot = np.zeros(len(self.touch_frames), dtype=np.int64)
        touch_dist = np.full(len(self.touch_frames), np.inf, dtype=np.float32)
        for i, t in enumerate(self.touch_frames):
            lo, hi = max(0, t - TOUCH_SLACK), min(len(near_slot), t + TOUCH_SLACK + 1)
            k = lo + int(np.argmin(near_dist[lo:hi]))
            touch_slot[i], touch_dist[i] = near_slot[k], near_dist[k]
        return {"players": players, "near_slot": near_slot, "near_dist": near_dist, "touch_slot": touch_slot,
                "touch_dist": touch_dist, "team": team, "uncertain": uncertain}

    def add_rule6(self, H):
        """Rule 6: add broadcast extras and ghosts for dropped players as extra slots."""
        import dropouts
        r6 = self.rule6 = dropouts.build(self, H)
        extra_team = np.where(r6["team"] >= 0, r6["team"], 0).astype(np.int8)
        pad = np.zeros(r6["team"].shape, dtype=bool)
        team = {g: np.concatenate([t, extra_team], axis=1) for g, t in self.team.items()}
        uncertain = {g: np.concatenate([u, pad], axis=1) for g, u in self.uncertain.items()}
        for static, base in ((False, self.tac.players), (True, self.players_moving)):
            self.sets[(True, static)] = self._lookups(
                np.concatenate([base, r6["pos"]], axis=1), team, uncertain)

    def _set(self, cfg):
        r6 = bool(cfg.get("dropouts")) and (True, False) in self.sets
        return self.sets[(r6, bool(cfg.get("static")))]

    @staticmethod
    def _key(cfg):
        """Which team / doubt set the switches select; rule 3.b needs rules 3 and 3.a."""
        return RESOLVED if cfg["goalie"] and cfg["crosscheck"] and cfg.get("resolve") else cfg["goalie"]

    def events(self, cfg):
        """Sorted [(frame, team, kind, slot)] for the rule switches in cfg; team -1 = nobody."""
        players = self._set(cfg)
        team = players["team"][self._key(cfg)]
        uncertain = players["uncertain"][self._key(cfg)] if cfg["crosscheck"] else None
        out = []
        radius = cfg.get("touch_radius", TOUCH_RADIUS)
        for t, slot, dist in zip(self.touch_frames, players["touch_slot"], players["touch_dist"]):
            t, slot = int(t), int(slot)
            kind = "touch" if dist <= TOUCH_CLOSE else "far touch"
            if dist > radius:
                out.append((int(t), -1, "touch", slot))
            elif uncertain is not None and uncertain[t, slot]:
                out.append((int(t), -2, kind, slot))  # rule 3.a: team in doubt
            else:
                out.append((int(t), int(team[t, slot]), kind, slot))
        if cfg["control"]:
            near_slot = players["near_slot"]
            for s, _, who in dwell_runs(near_slot, players["near_dist"], self.kin["present"], team):
                slot = int(near_slot[s])
                if uncertain is None or not uncertain[s, slot]:
                    out.append((s, who, "control", slot))
        if cfg["carry"]:
            # Carries need a continuous track and its speed, so they run on the tactical slots.
            for loss, _, slot in self.carries:
                at = loss - CARRY_SEEN
                if uncertain is None or not uncertain[at, slot]:
                    out.append((at, int(team[loss - 1, slot]), "carry", slot))
        if cfg.get("crosscam"):
            out = self._crosscam(out, players)
        if "z" in self.kin and cfg.get("height", True):
            out = [e for e in out if self._reachable(*e)]
        out.sort()
        return out

    def _reachable(self, t, who, kind, slot):
        """Rule 17: nobody touches a ball above a header's height, controls one above the chest,
        or carries one that was in flight when it was lost (1H 12785: a high ball's ground point
        passed over home defenders -- a home control, touch and carry against an away pass)."""
        import ball3d
        z = self.kin["z"]
        if kind in ("touch", "far touch"):
            return z[t] <= ball3d.REACH
        if kind == "control":
            return z[t] <= ball3d.CONTROL_REACH
        if kind == "carry":
            return not self.kin["flight"][t:t + CARRY_SEEN + 1].any()
        return True

    def _crosscam(self, events, players):
        """Rule 10: a touch or control the broadcast sees differently. Both cameras are measured
        against the same fused ball (rule 8); where the broadcast is live, has its nearest player
        within the event's radius and CROSS_MARGIN closer than the tactical one, and on the
        other team, the broadcast's team wins (1H 11415: the tactical camera has an away player
        ~1 m from the ball and no home player within 4 m; the broadcast has a home player at
        0.1-0.5 m). Overrides are kept in self.crosscam_overrides as (frame, kind, was, now)."""
        touch_dist = dict(zip(self.touch_frames.tolist(), players["touch_dist"].tolist()))
        out, self.crosscam_overrides = [], []
        for t, who, kind, slot in events:
            if kind in ("touch", "far touch", "control"):
                lo, hi = max(0, t - TOUCH_SLACK), t + TOUCH_SLACK + 1
                k = lo + int(np.argmin(self.b_near_dist[lo:hi]))
                d_b, team_b = float(self.b_near_dist[k]), int(self.b_near_team[k])
                d_t = float(players["near_dist"][t]) if kind == "control" else touch_dist.get(t, np.inf)
                radius = CONTROL_RADIUS if kind == "control" else TOUCH_CLOSE
                if team_b >= 0 and team_b != who and d_b <= radius and d_b + CROSS_MARGIN < d_t:
                    self.crosscam_overrides.append((t, kind, who, team_b))
                    who = team_b
            out.append((t, who, kind, slot))
        return out

    def predict(self, cfg):
        """Labels (n,) of 'H' 'A' 'D' 'X', plus what the viewer draws."""
        n = self.tac.n
        dead, dead_list, evidence = dead_intervals(self, cfg["ref_delay"], cfg["broadcast_dead"],
                                                   cfg.get("local_still", False),
                                                   cfg.get("dead_extend", False))
        events = self.events(cfg)
        frames = np.array([e[0] for e in events], dtype=np.int64)
        players = self._set(cfg)
        team = players["team"][self._key(cfg)]
        rows = np.arange(n)
        near_team = np.where(self.kin["present"] & (players["near_dist"] <= FALLBACK_RADIUS),
                             team[rows, players["near_slot"]], -1)
        # Rule 6 guard: just after the team in possession lost a player near the ball, a single
        # control by the other team is treated like a touch -- it needs confirming.
        guard = self.rule6["guard"] if cfg.get("dropouts") and self.rule6 else None

        def guarded(t, holder):
            if guard is None or holder not in guard:
                return False
            times = guard[holder]
            i = np.searchsorted(times, t, side="right")
            return i > 0 and t - times[i - 1] <= GUARD

        labels = np.full(n, "X", dtype="<U1")
        previous = HOME
        for half, first, last in halves():
            labels[first:last + 1] = np.where(dead[first:last + 1], "D", "X")
            for s, e in runs(~dead[first:last + 1]):
                s, e = s + first, e + first
                lo, hi = np.searchsorted(frames, s), np.searchsorted(frames, e)
                stretch = [ev for ev in events[lo:hi] if ev[1] >= 0]
                changes = []
                if not stretch:
                    votes = near_team[s:e]
                    h, a = int((votes == HOME).sum()), int((votes == AWAY).sum())
                    changes.append((s, previous if h == a == 0 else (HOME if h >= a else AWAY)))
                else:
                    # The first event after a restart sets possession: a restart always has a
                    # clear possessor (CHALLENGE.md).
                    state, pending = stretch[0][1], None
                    changes.append((s, state))
                    for t, who, kind, _ in stretch[1:]:
                        if who == state:
                            pending = None
                            continue
                        strong = kind in ("control", "carry") and not guarded(t, state)
                        if not cfg["confirm"] or strong:
                            at = pending[0] if pending and pending[1] == who else t
                            changes.append((at, who))
                            state, pending = who, None
                        elif pending and pending[1] == who:
                            changes.append((pending[0], who))
                            state, pending = who, None
                        else:
                            pending = (t, who)
                for i, (at, who) in enumerate(changes):
                    stop = changes[i + 1][0] if i + 1 < len(changes) else e
                    labels[at:stop] = LABELS[who]
                previous = changes[-1][1]
        return labels, {"events": events, "dead": dead_list, "evidence": evidence}
