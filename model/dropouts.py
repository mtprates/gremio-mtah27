"""Rule 6: tracked players don't vanish mid-view. See docs/TECHNICAL.md.

A tactical track that ends while the player is well inside the camera's view is a tracking
failure (a *dropout*), not an exit. The player is kept on the pitch until the tracker picks
them up again:

    - from the broadcast, wherever it is live and has a consistently tracked object of the
      same team near where the player should be -- the camera that stayed consistent wins
    - in a known gap (the same slot comes back), the broadcast takes priority over the straight
      line between the last and next known position (rule 6.b), with every instance logged
    - otherwise as a *ghost*: last position plus last velocity, decaying to a standstill

Broadcast objects are only ever used to stand in for a tactical player who just dropped out.
Broadcast objects with no tactical object near them are common in general -- ~5% of them, about
one per live frame, and they pass the "tracked continuously" test as often as matched ones do --
which says they are mostly people the tactical export does not track at all (the referee),
not missed players. Adding them all would put a phantom "player" next to the ball.

The corrected set of players -- tactical slots + dropped players' bridged positions -- is what
the nearest-player lookups of rules 1 and 1.a run on. Extra objects get slot numbers after the
tactical ones.
"""

import numpy as np

import align
import rules
from data import FPS, PITCH

# Params
DROP_MIN_GAP = 5            # frames untracked before a track end counts (shorter = flicker)
IMAGE_SIZE = (1920, 1080)   # tactical video (DATA.md); the homography maps to these pixels
VIEW_MARGIN_PX = 60         # last seen this far inside the image border = mid-view
INTERP_MAX = 20 * FPS       # same slot back within this long: interpolate the whole gap
GHOST_MAX = 3 * FPS         # otherwise a ghost lasts this long past its last anchor ...
GHOST_DECAY = 1.0           # s: ... with its last velocity decaying on this time constant
RELINK_RADIUS = 2.5         # m: a new track of the same team this close re-acquires the player
FILL_RADIUS = 3.0           # m: a broadcast object this close stands in for the player
FILL_MAX = 10 * FPS         # longest a dropout is bridged, broadcast fills included
CONSISTENT = FPS            # broadcast object tracked continuously this long to be trusted
EXTRA_UNMATCHED = 2.0       # m: broadcast object with no tactical object *of its own team*
                            # this close is unaccounted for. Same team only: at 1H 35:56 the
                            # dropped carrier stands within 2 m of the opponent marking him.
PITCH_MARGIN = 1.0          # m: extras must stand on the pitch, give or take
MAX_EXTRA = 8               # broadcast-extra columns per frame
MAX_GHOST = 10              # bridged-player columns per frame
NEAR_BALL = 5.0             # m: a dropout this close to the ball guards the state machine
                            # (3 m missed 1H 35:56, lost at 3.5 m; the guard only asks for a
                            # second piece of evidence, so erring wide is cheap)
GUARD = FPS                 # frames after such a dropout during which a switch needs more
SPRINT = 8.0                # m/s: rule 6.b broadcast relink, how far the player can have run
RELINK_MIN_AGE = 5          # frames: a new broadcast track must be this old (not a flicker)
CHUNK = 2000

KIND_BROADCAST, KIND_GHOST, KIND_INTERP = 1, 2, 3


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _pixels(H, pos_m):
    """Pitch metres -> pixels through one homography; None if behind the camera / no matrix."""
    x, y = pos_m[0] / PITCH[0], pos_m[1] / PITCH[1]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if not np.isfinite(w) or w <= 1e-9:
        return None
    return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w)


def _mid_view(H, pos_m):
    p = _pixels(H, pos_m)
    if p is None:
        return False
    return VIEW_MARGIN_PX <= p[0] <= IMAGE_SIZE[0] - VIEW_MARGIN_PX and \
        VIEW_MARGIN_PX <= p[1] <= IMAGE_SIZE[1] - VIEW_MARGIN_PX


def _streaks(tracked):
    """Per frame and slot: how many consecutive frames the slot has been tracked, up to now."""
    out = np.zeros(tracked.shape, dtype=np.int32)
    run = np.zeros(tracked.shape[1], dtype=np.int32)
    for b in range(len(tracked)):
        run = (run + 1) * tracked[b]
        out[b] = run
    return out


def broadcast_extras(ctx, streak):
    """Per tactical frame: broadcast objects the tactical tracker is missing.

    Returns (pos (n, MAX_EXTRA, 2), team (n, MAX_EXTRA) with -1 for empty, overflow count).
    """
    tac, bro, al = ctx.tac, ctx.bro, ctx.alignment
    n = tac.n
    pos = np.full((n, MAX_EXTRA, 2), np.nan, dtype=np.float32)
    team = np.full((n, MAX_EXTRA), -1, dtype=np.int8)
    tpos = np.where(np.isnan(tac.players), 1e4, tac.players).astype(np.float32)
    tteam = ctx.team[rules.RESOLVED]
    live = np.flatnonzero(al["state_t"] == align.STATE_LIVE)
    overflow = 0
    for start in range(0, len(live), CHUNK):
        t = live[start:start + CHUNK]
        b = al["b_of_t"][t]
        bp = bro.players[b]                                               # (B, Sb, 2)
        dist = np.sqrt(((bp[:, :, None, :] - tpos[t][:, None, :, :]) ** 2).sum(-1))  # (B, Sb, S)
        same = ctx.b_team[b][:, :, None] == tteam[t][:, None, :]
        d = np.where(same, dist, 1e4).min(-1)
        with np.errstate(invalid="ignore"):
            on_pitch = (bp[..., 0] >= -PITCH_MARGIN) & (bp[..., 0] <= PITCH[0] + PITCH_MARGIN) & \
                       (bp[..., 1] >= -PITCH_MARGIN) & (bp[..., 1] <= PITCH[1] + PITCH_MARGIN)
            ok = ~np.isnan(bp[..., 0]) & (d > EXTRA_UNMATCHED) & on_pitch & (streak[b] >= CONSISTENT)
        rows, cols = np.nonzero(ok)
        for r in np.unique(rows):
            slots = cols[rows == r]
            if len(slots) > MAX_EXTRA:
                overflow += 1
                slots = slots[:MAX_EXTRA]
            frame, bframe = t[r], b[r]
            pos[frame, :len(slots)] = bp[r, slots]
            team[frame, :len(slots)] = ctx.b_team[bframe, slots]
    return pos, team, overflow


def track_segments(players):
    """Per slot: [(start, stop)] of tracked stretches, gaps shorter than DROP_MIN_GAP closed."""
    tracked = ~np.isnan(players[..., 0])
    n = len(tracked)
    out = []
    for slot in range(tracked.shape[1]):
        solid = tracked[:, slot].copy()
        for s, e in _runs(~solid):
            if s > 0 and e < n and e - s < DROP_MIN_GAP:
                solid[s:e] = True
        out.append(_runs(solid))
    return out


def _record(stop, path, slot, team, p0, ball_dist, resolution, relinked, filled, half, priority=0,
            relinks=0):
    return {
        "lost": stop, "end": path[-1][0] + 1 if path else stop, "slot": slot, "team": team,
        "x": float(p0[0]), "y": float(p0[1]), "ball_dist": ball_dist, "resolution": resolution,
        "relinked_slot": relinked, "bridged": len(path), "broadcast_frames": filled,
        "priority_frames": priority, "broadcast_relinks": relinks, "half": int(half),
    }


def _paired_object(ctx, slot, last):
    """Rule 6.b: the broadcast object paired with `slot` in its last CONSISTENT tracked frames
    -- the player's identity on the broadcast when tactical lost them -- or -1."""
    pairs = ctx.alignment["pairs_t"]
    for t in range(last, max(-1, last - CONSISTENT), -1):
        if pairs[t, slot] >= 0:
            return int(pairs[t, slot])
    return -1


def _gap_objects(ctx, segments, t, slot, who, lost):
    """Rule 6.b: mask over the broadcast objects on tactical frame t (live) that may be the
    player dropped from `slot` at `lost`.

    During a known gap the player is on the pitch, so any tracked broadcast object of their team
    qualifies unless it is the counterpart of an *established* teammate -- a same-team tactical
    track that was already running when the player dropped. A counterpart of a track born during
    the gap stays in: that is often the dropped player re-acquired under another slot, possibly
    of the wrong team (1H 11409).
    """
    al = ctx.alignment
    b = al["b_of_t"][t]
    ok = ~np.isnan(ctx.bro.players[b, :, 0]) & (ctx.b_team[b] == who)
    team = ctx.team[rules.RESOLVED]
    for k in np.flatnonzero(al["pairs_t"][t] >= 0):
        obj = int(al["pairs_t"][t, k])
        if k == slot or not ok[obj] or team[t, k] != who:
            continue
        if any(s0 < lost and s0 <= t < e0 for s0, e0 in segments[k]):
            ok[obj] = False
    return ok


def _bridge_gap(ctx, streak, segments, extras_pos, extras_team, slot, who, last, p0, p1, stop,
                next_same):
    """Rule 6 interpolation with rule 6.b broadcast priority. Per frame of the gap, in order:
      1. the broadcast object the player was paired with when lost, followed by identity
      2. once the broadcast loses that object: a broadcast track of the player's team born since
         then, within reach of where the player was last placed (sprint speed x time, capped) --
         the broadcast re-acquiring them under a new id -- then followed by identity
      3. a consistently tracked candidate (extras, or step-2 objects older than CONSISTENT)
         within FILL_RADIUS of the last broadcast position, or of the line
      4. the straight line between the last and next known positions
    Returns (path, broadcast frames, priority frames, relinks): priority frames are broadcast
    frames the extras-only search (before rule 6.b) would have left to the line; relinks counts
    the step-2 adoptions, the riskiest part (positional inference, see the NOTE below).
    """
    al, bro = ctx.alignment, ctx.bro
    span = next_same - last
    path, filled, priority, relinks = [], 0, 0, 0
    same = _paired_object(ctx, slot, last)
    anchor, anchor_t, follow = p0, last, None
    for t in range(stop, next_same):
        line = p0 + (p1 - p0) * (t - last) / span
        ref = follow if follow is not None else line
        extras = extras_pos[t, extras_team[t] == who].astype(np.float64)
        kind, point, follow = KIND_INTERP, line, None
        b = al["b_of_t"][t]
        live = al["state_t"][t] == align.STATE_LIVE
        if same >= 0 and not (live and not np.isnan(bro.players[b, same, 0])):
            same = -1  # the broadcast lost it; its index may now be someone else
        if live and same < 0:
            ok = _gap_objects(ctx, segments, t, slot, who, stop)
            age = streak[b]
            newborn = ok & (age >= RELINK_MIN_AGE) & (age <= t - anchor_t + RELINK_MIN_AGE)
            if newborn.any():
                reach = FILL_RADIUS + SPRINT * (t - anchor_t) / FPS
                objs = np.flatnonzero(newborn)
                dist = np.linalg.norm(bro.players[b, objs] - anchor, axis=1)
                if dist.min() <= reach:
                    same = int(objs[np.argmin(dist)])
                    relinks += 1
        if same >= 0:
            point = bro.players[b, same].astype(np.float64)
            kind = KIND_BROADCAST
        else:
            cand = extras
            if live:
                ok = _gap_objects(ctx, segments, t, slot, who, stop) & (streak[b] >= CONSISTENT)
                cand = np.concatenate([extras, bro.players[b, ok].astype(np.float64)])
            if len(cand):
                dist = np.linalg.norm(cand - ref, axis=1)
                j = int(np.argmin(dist))
                if dist[j] <= FILL_RADIUS:
                    point, kind = cand[j], KIND_BROADCAST
        if kind == KIND_BROADCAST:
            follow, anchor, anchor_t = point, point, t
            filled += 1
            if not len(extras) or np.linalg.norm(extras - ref, axis=1).min() > FILL_RADIUS:
                priority += 1
        path.append((t, float(point[0]), float(point[1]), kind))
    return path, filled, priority, relinks


def find_dropouts(ctx, H, extras_pos, extras_team, streak):
    """Mid-view track ends, each bridged by broadcast fills or a ghost. Returns
    (records, tracks): one record dict per dropout, and per dropout its bridged path
    [(frame, x, y, kind)]."""
    tac = ctx.tac
    n = tac.n
    team = ctx.team[rules.RESOLVED]
    segments = track_segments(tac.players)
    starts = {}
    for slot, segs in enumerate(segments):
        for s, _ in segs:
            if s > 0:
                starts.setdefault(s, []).append(slot)

    records, tracks = [], []
    for slot, segs in enumerate(segments):
        for i, (_, stop) in enumerate(segs):
            last = stop - 1
            if stop >= n or ctx.half_of[last] == 0 or not _mid_view(H[last], tac.players[last, slot]):
                continue
            next_same = segs[i + 1][0] if i + 1 < len(segs) else n
            who = int(team[last, slot])
            p0 = tac.players[last, slot].astype(np.float64)
            back = max(0, last - 5)
            v0 = (p0 - tac.players[back, slot]) / max(1, last - back) * FPS
            if not np.isfinite(v0).all():
                v0 = np.zeros(2)
            ball = ctx.kin["pos"][last]
            ball_dist = float(np.linalg.norm(ball - p0)) if np.isfinite(ball).all() else float("nan")

            half = ctx.half_of[last]
            path, end, relinked, filled = [], "timeout", -1, 0

            # Known gap: the same slot comes back, in the same half, within INTERP_MAX. The
            # player was there all along (1H 38:51: hidden behind an away player), so fill the
            # whole gap -- from the broadcast where it is live and has them, otherwise on the
            # straight line between the last and the next known position. Rule 6.b: the
            # broadcast takes priority over the line (_bridge_gap). Frames it fills that the
            # extras alone would have left to the line are counted as priority frames, the MVP
            # status tracker for this rule.
            #
            # NOTE: "the same player" is decided by slot id here, which is only a starting
            # point. The tracker also re-acquires people under a *new* slot, and occasionally
            # hands a slot to someone else. The next step is positional inference: match the
            # dropped player to whichever same-team track appears where they could have run to
            # in the time elapsed (reachable distance ~ max sprint speed x gap), and check the
            # re-appearing slot is plausibly the same person before trusting it.
            if next_same - stop <= INTERP_MAX and next_same < n and ctx.half_of[next_same] == half:
                p1 = tac.players[next_same, slot].astype(np.float64)
                path, filled, priority, relinks = _bridge_gap(ctx, streak, segments, extras_pos,
                                                     extras_team, slot, who, last, p0, p1, stop,
                                                     next_same)
                records.append(_record(stop, path, slot, who, p0, ball_dist, "interpolated",
                                       -1, filled, half, priority, relinks))
                tracks.append(path)
                continue

            anchor, anchor_t, vel = p0.copy(), last, v0.copy()
            half_last = max(f for f in range(last, min(n, last + FILL_MAX + 1))
                            if ctx.half_of[f] == half)
            for t in range(stop, min(n, stop + FILL_MAX)):
                if t > half_last:
                    end = "half end"
                    break
                if t >= next_same:
                    end = "same slot back"
                    break
                tau = (t - anchor_t) / FPS
                pred = anchor + vel * GHOST_DECAY * (1 - np.exp(-tau / GHOST_DECAY))
                hit = [k for k in starts.get(t, []) if k != slot and team[t, k] == who and
                       np.linalg.norm(tac.players[t, k] - pred) <= RELINK_RADIUS]
                if hit:
                    end, relinked = "new slot", hit[0]
                    break
                cand = np.flatnonzero(extras_team[t] == who)
                if len(cand):
                    dist = np.linalg.norm(extras_pos[t, cand] - pred, axis=1)
                    j = int(np.argmin(dist))
                    if dist[j] <= FILL_RADIUS:
                        anchor, anchor_t, vel = extras_pos[t, cand[j]].astype(np.float64), t, np.zeros(2)
                        path.append((t, float(anchor[0]), float(anchor[1]), KIND_BROADCAST))
                        filled += 1
                        continue
                if t - anchor_t > GHOST_MAX:
                    end = "timeout"
                    break
                path.append((t, float(pred[0]), float(pred[1]), KIND_GHOST))
            records.append(_record(stop, path, slot, who, p0, ball_dist, end, relinked, filled, half))
            tracks.append(path)
    order = np.argsort([r["lost"] for r in records], kind="stable")
    return [records[i] for i in order], [tracks[i] for i in order]


def ghost_columns(n, tracks):
    """Pack each dropout's bridged path -- broadcast fills and ghost guesses alike -- into
    MAX_GHOST columns per frame. Returns (pos, owner index, kind, dropped)."""
    pos = np.full((n, MAX_GHOST, 2), np.nan, dtype=np.float32)
    owner = np.full((n, MAX_GHOST), -1, dtype=np.int32)
    kind = np.zeros((n, MAX_GHOST), dtype=np.int8)
    dropped = 0
    for index, path in enumerate(tracks):
        if not path:
            continue
        frames = np.array([f for f, _, _, _ in path], dtype=np.int64)
        free = [c for c in range(MAX_GHOST) if (owner[frames, c] < 0).all()]
        if not free:
            dropped += 1
            continue
        c = free[0]
        pos[frames, c] = np.array([(x, y) for _, x, y, _ in path], dtype=np.float32)
        owner[frames, c] = index
        kind[frames, c] = [k for _, _, _, k in path]
    return pos, owner, kind, dropped


def build(ctx, H):
    """Everything rule 6 adds: extra objects, their teams, the dropout records, guard windows."""
    n = ctx.tac.n
    streak = _streaks(~np.isnan(ctx.bro.players[..., 0]))
    extras_pos, extras_team, overflow = broadcast_extras(ctx, streak)
    records, tracks = find_dropouts(ctx, H, extras_pos, extras_team, streak)
    ghost_pos, owner, kind, dropped = ghost_columns(n, tracks)
    ghost_team = np.where(owner >= 0, np.array([r["team"] for r in records] + [-1])[owner], -1)
    guard = {0: [], 1: []}
    for r in records:
        if r["ball_dist"] == r["ball_dist"] and r["ball_dist"] <= NEAR_BALL:
            guard[r["team"]].append(r["lost"])
    return {
        "pos": ghost_pos,
        "team": ghost_team.astype(np.int8),
        "kind": kind,
        # Kept for the diagnostics only: how often the broadcast has someone tactical lacks.
        "unmatched_broadcast": (extras_team >= 0).any(axis=1),
        "records": records,
        "tracks": tracks,
        "guard": {team: np.array(sorted(times), dtype=np.int64) for team, times in guard.items()},
        "overflow": overflow,
        "ghosts_dropped": dropped,
    }
