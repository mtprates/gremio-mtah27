"""A three-state hidden Markov model (H, A, D) in place of rules.Context.predict's state machine.

The rules stay; what changes is how they are combined. Each rule now gives evidence -- a
log-likelihood ratio, in nats -- instead of a yes / no, and a Viterbi pass picks the labels that
best explain all of it at once, paying a price for every change of state:

    possession   which team the ball is with: the players' distances to the ball, every frame,
                 plus a bonus at each touch / control / carry (rules 1, 1.a, 2, 10 decide who)
    dead ball    each dead-ball signal's likelihood ratio, from how often it fires on dead and
                 on live frames (rules 4, 5, 7, 11, 12 and the out-of-bounds / stillness rules)
    transitions  priced from how long possessions, live and dead stretches last

The state machine's confirmation rule (1.a) is not coded: one opposing touch is worth less than
the price of switching, so it does not flip possession unless more evidence follows -- a second
touch, a dwell or a carry. Forward-backward then gives every frame a probability of its label,
which is where the model is unsure.

Every number is set from football, from reasoning, or on external matches (IDSSE, Metrica held out);
none by a score on BAR-ATH -- docs/VALUES.md, "Re-derivation off BAR-ATH".
"""

import math

import numpy as np

import align
import rules
from data import AWAY, FPS, HOME, halves

# Params
CORR_FRAMES = FPS // 2      # frame-by-frame evidence is counted once per this many frames: next
                            # frames are not independent looks (tracking is smooth over ~0.5 s)
POSSESSION_S = 7.2          # s: mean possession on IDSSE's labels (docs/VALUES.md, re-derivation off BAR-ATH)
LIVE_S = 31.2               # s: mean live stretch on IDSSE
DEAD_S = 25.3               # s: mean dead stretch on IDSSE
CONTROL_STD = 1.0           # m: how far the player in possession is from the ball, typically
PROX_CLIP = 3.0             # nats: one frame's distance evidence is capped here (one look)
BALL_STD = 0.35             # m: ball position error on the moving-average ball (no pos_std)
EVENT_LLR = {"touch": 3.0, "far touch": 1.5, "control": 4.0, "carry": 4.0}
TOUCH_PRECISION = False     # rule 53 (docs/VALUES.md): a touch x the weaker side's detection precision. --touch-precision
PRECISION_KINDS = ("touch", "far touch")    # rule 53b: the event kinds it weighs; --event-precision adds control, carry
TOUCH_SEEN_POWER = 1.0      # rule 26 touch_seen: a touch's nats x (seen share around it) ** this;
                            # 1 is rule 26 as first built (docs/VALUES.md, external tuning stage)
# Rules 55-57 (docs/VALUES.md), each behind its cfg switch, off by default.
KICK_SPEED = 8.0            # m/s: rule 55 (cfg kick), a kick sends the ball off at least this fast ...
KICK_MARGIN = 1.0           # nats: ... and is worth the switch price, log(POSSESSION_S * FPS), plus this
LIVE_KICK_LLR = 9.0         # nats: rule 56 (cfg restart_live), live evidence of a restart or a kick ...
LIVE_SPREAD = FPS           # ... spread over the frames after it
WHISTLE_BACK = 5 * FPS      # frames: rule 57 (cfg whistle_back), a broadcast whistle this soon before a dead stretch ...
WHISTLE_LLR = 1.0           # nats per frame of dead evidence from the whistle to the stretch ...
SHORT_S = 5.0               # s: rule 63 (reasoning; IDSSE: 4-6 s within 0.0015) (cfg short_spells), a spell this short between two of the opposite state is suspect
CONFIRM_S = 6.0             # s: rule 61 (cfg restart_confirm), a kick from rest with the ball back at rest this soon is no restart
STILL_SEEN = 1 / 3            # rule 64 (cfg still_cutaway): the ball seen on this share of the rules.STILL_MIN window
FAR_MARGIN = 1.0            # m: rule 60 (cfg far_clear), a far touch needs the other team this much farther
WHISTLE_RESTART = 2 * FPS   # ... unless it comes this soon after the previous stretch (a restart whistle)
PASS_LLR = 4.0              # nats: rule 23, a completed pass -- as much as a control
PASS_MAX = 3 * FPS          # frames: the longest pass counted
                            # nats for the credited team. Mirrors rule 1.a: a touch (3) is less
                            # than a switch costs (log(POSSESSION_S * FPS) ~ 5.2), two touches or a
                            # control with the frames of its dwell are more. A control under the
                            # rule 6 guard counts as a touch.
BASE_DEAD = 0.448           # dead share on IDSSE: the base rate for a signal given as (precision, recall)
SIGNALS = {                 # dead-ball signal: (present, absent) LLR in nats, or (precision, recall)
                            # for those in MEASURED. docs/VALUES.md, re-derivation off BAR-ATH.
    "broadcast no homography": (2.0, -1.3),     # reasoning: a cut-away in ~3/4 of a stoppage, ~1/10 of live play
    "players mostly still": (0.966, 0.384),     # (precision, recall) measured on IDSSE's labels (clean copies)
    "POSSIBLE dead (ball accel ~0)": (1.75, 0.0),     # present: IDSSE's (0.823, 0.947); absent: no evidence, the ball
                                                      # is missing on many of our frames (docs/VALUES.md)
    "out of bounds": (3.0, 0.0),        # the ball beyond a line, until the restart
    "out of bounds, long": (6.0, 0.0),  # rule 27 (cfg oob_grade): out for OOB_LONG or more
    "out of bounds, alone": (1.0, 0.0), # ... a short run with no other dead-ball sign after it
    "local stillness": (1.5, 0.0),      # reasoning: a lost ball with both teams standing round it is a stoppage
    "stillness": (0.992, 0.214),        # (precision, recall) measured on IDSSE's labels (clean copies)
    "ball still in a cut-away": (3.0, 0.0),   # rule 64: as sure as the ball out of bounds; its absence says nothing
}
OOB_LONG = FPS              # frames: rule 27, a ball out this long is out (docs/VALUES.md)
OOB_SUPPORT = 5 * FPS       # frames after a short run's start to look for another dead-ball sign
MEASURED = {"players mostly still", "stillness"}   # given as (precision, recall)
LOW_CONFIDENCE = 0.8        # viewer row: frames whose label has less than this probability
# Rule 25 (cfg pass_prob, flight_pc; off): pitch control and pass probability (model/pitch_control.py,
# published parameters) as the switch price at passes and landings, under each hypothesis.
PASS_MIN_S = FPS // 5       # frames: two credited touches closer than this are one touch
PASS_MAX_FRAMES = 3 * FPS   # ... farther apart than this, not one pass
PASS_MIN_M = 5.0            # m: the ball must travel this far
REL_MAX = 4.0               # a ball heading where its team controls less than usual makes a switch at
                            # most this many times as likely as usual, per frame; a safer one at most
                            # this many times less. Relative, not absolute: as a raw switch chance the
                            # model's under-confidence (~0.68 predicted vs ~89% completed) would double
                            # the switches. Rule 25 is off in the submitted model.
REL_MARGIN = FPS // 2       # frames after the reception the adjusted price still holds

H_, A_, D_ = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def signal_llr(precision, recall, base=None):
    """(present, absent) log-likelihood ratios of a binary dead-ball signal, dead vs live. base: the dead
    share the precision was measured at, BASE_DEAD by default (read at call time, so a changed BASE_DEAD counts)."""
    base = BASE_DEAD if base is None else base
    false_rate = recall * base * (1 - precision) / (precision * (1 - base))
    return math.log(recall / false_rate), math.log((1 - recall) / (1 - false_rate))


def transitions():
    """3 x 3 log transition matrix, from the mean stretch lengths."""
    switch, die, revive = (1.0 / (s * FPS) for s in (POSSESSION_S, LIVE_S, DEAD_S))
    p = np.array([[1 - switch - die, switch, die],
                  [switch, 1 - switch - die, die],
                  [revive / 2, revive / 2, 1 - revive]])
    return np.log(p)


def _masks(ctx, cfg):
    """{signal name: (n,) bool} for the dead-ball signals the switches in cfg allow, and the
    frames that must be dead (the second half before its kick-off)."""
    n = ctx.tac.n
    _, merged, evidence = rules.dead_intervals(ctx, cfg["ref_delay"], False,
                                               cfg.get("local_still", False), False)
    out = {}
    for name in ("out of bounds", "stillness", "local stillness"):
        out[name] = np.zeros(n, dtype=bool)
    forced = np.zeros(n, dtype=bool)
    for s, e, reason in merged:
        (forced if reason == "before kick-off" else out[reason])[s:e] = True
    state = ctx.alignment["state_t"]
    if cfg.get("oob_grade"):
        # Rule 27: out of bounds graded by its own run -- a ball out for OOB_LONG or more is sure; a
        # short run with no other dead-ball sign after it (no broadcast homography, players still)
        # is mostly the ball's ground point outside while in the air, or noise.
        _, plain, raw = rules.dead_intervals(ctx, cfg["ref_delay"], False, False, False)
        quiet = rules.quiet_players(ctx)
        nohom = state == align.STATE_CLOSEUP
        out["out of bounds, long"] = np.zeros(n, dtype=bool)
        out["out of bounds, alone"] = np.zeros(n, dtype=bool)
        stop_of = {s: e for s, e, reason in plain if reason == "out of bounds"}
        for s, e in raw["out of bounds"]:
            start = s + cfg["ref_delay"]
            stop = stop_of.get(start)
            if stop is None:
                continue
            after = slice(s, s + OOB_SUPPORT)
            if e - s >= OOB_LONG:
                out["out of bounds, long"][start:stop] = True
            elif not (nohom[after].any() or quiet[after].any()):
                out["out of bounds, alone"][start:stop] = True
        # The graded kinds take their frames out of the plain signal.
        out["out of bounds"] &= ~(out["out of bounds, long"] | out["out of bounds, alone"])
    if cfg["broadcast_dead"]:
        out["broadcast no homography"] = state == align.STATE_CLOSEUP
    if cfg.get("still_cutaway"):
        out["ball still in a cut-away"] = (state == align.STATE_CLOSEUP) & ball_still_seen(ctx.kin)    # rule 64
    if cfg.get("dead_extend"):
        out["players mostly still"] = rules.quiet_players(ctx)
    mask = np.zeros(n, dtype=bool)
    for s, e in evidence["POSSIBLE dead (ball accel ~0)"]:
        mask[s:e] = True
    out["POSSIBLE dead (ball accel ~0)"] = mask
    # A signal's absence is evidence only where it could have fired: the homography's where the
    # broadcast covers the moment, the acceleration's where the ball is seen.
    covered = {"broadcast no homography": np.isin(state, (align.STATE_LIVE,) if cfg.get("replay_neutral")    # rule 62
                                                  else (align.STATE_LIVE, align.STATE_REPLAY)),
               "POSSIBLE dead (ball accel ~0)": ctx.kin["present"]}
    return out, covered, forced


def ball_still_seen(kin):
    """Rule 64: (n,) the ball still whenever it is seen -- over the rules.STILL_MIN window centred on each frame, the median
    speed of its seen frames under rules.STILL_BALL, with at least STILL_SEEN of the window seen."""
    import warnings
    from numpy.lib.stride_tricks import sliding_window_view
    w = rules.STILL_MIN
    sp = np.where(kin["seen"], kin["speed"], np.nan)
    pad = np.r_[np.full(w // 2, np.nan), sp, np.full(w - 1 - w // 2, np.nan)]
    win = sliding_window_view(pad, w)
    seen = np.isfinite(win).mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(win, axis=1)
    with np.errstate(invalid="ignore"):
        return (seen >= STILL_SEEN) & (med < rules.STILL_BALL)


def dead_evidence(ctx, cfg, events=None):
    """(n,) dead-vs-live log-likelihood ratio per frame, already divided by CORR_FRAMES; and the
    frames forced dead. With cfg restart_live and the events, rule 56's live evidence too."""
    masks, covered, forced = _masks(ctx, cfg)
    llr = np.zeros(ctx.tac.n)
    for name, mask in masks.items():
        a, b = cfg.get("signals", {}).get(name, SIGNALS[name])
        present, absent = signal_llr(a, b) if name in MEASURED else (a, b)
        llr += np.where(mask, present, np.where(covered.get(name, False), absent, 0.0))
    llr /= CORR_FRAMES
    if cfg.get("restart_live") and events is not None:
        for t in live_kicks(ctx, cfg, events):
            llr[t:t + LIVE_SPREAD] -= LIVE_KICK_LLR / LIVE_SPREAD
    return llr, forced


def _inside(pos):
    length, width = rules.PITCH
    with np.errstate(invalid="ignore"):
        return (pos[..., 0] >= -rules.OOB_MARGIN) & (pos[..., 0] <= length + rules.OOB_MARGIN) & \
               (pos[..., 1] >= -rules.OOB_MARGIN) & (pos[..., 1] <= width + rules.OOB_MARGIN)


def live_kicks(ctx, cfg, events):
    """Rule 56: frames of credited touches that are restarts (a kick from rest inside the pitch,
    rules.dead_intervals' test) or rule 55 kicks (with cfg kick)."""
    kin = ctx.kin
    s_in = np.nan_to_num(kin["speed_in"], nan=0.0)
    s_out = np.nan_to_num(kin["speed_out"], nan=0.0)
    out = []
    for t, who, kind, _, _ in credited(ctx, cfg, events):
        if kind == "kick" or (kind in ("touch", "far touch") and s_in[t] < rules.RESTART_IN_SPEED and
                              s_out[t] >= rules.RESTART_OUT_SPEED and kin["present"][t] and _inside(kin["pos"][t])):
            out.append(t)
    if cfg.get("restart_confirm"):
        # Rule 61: a kick from rest after which the ball is back at rest within CONFIRM_S was a ball being set up.
        with np.errstate(invalid="ignore"):
            rest = kin["present"] & (np.nan_to_num(kin["speed"], nan=np.inf) < rules.STILL_BALL)
        run = np.zeros(len(rest), dtype=np.int32)       # frames the ball has been at rest, up to each frame
        for i in np.flatnonzero(rest):
            run[i] = run[i - 1] + 1 if i else 1
        settled = run >= FPS
        window = int(CONFIRM_S * FPS)
        out = [t for t in out if not (s_in[t] < rules.RESTART_IN_SPEED and settled[t + rules.DV_WINDOW:t + window].any())]
    return out


def whistle_frames(ctx):
    """Rule 57: tactical frames where a broadcast whistle starts (whistles_broadcast.csv, v3), through
    each half's main offset. Empty without the file or the alignment's half offsets."""
    import csv
    import os
    import data
    path = os.path.join(data.CACHE_DIR, "whistles_broadcast.csv")
    offsets = ctx.alignment.get("half_offsets") if isinstance(ctx.alignment, dict) else None
    if not os.path.isfile(path) or offsets is None or len(offsets) != 2:
        return np.zeros(0, dtype=np.int64)
    with open(path, newline="") as handle:
        b = np.array([int(r["start_frame"]) for r in csv.DictReader(handle)], dtype=np.int64)
    split = halves()[1][1]
    t = np.where(b + int(offsets[1]) >= split, b + int(offsets[1]), b + int(offsets[0]))
    return np.sort(t[(t >= 0) & (t < ctx.tac.n)])


def whistle_evidence(ctx, labels, whistles):
    """Rule 57: (n,) extra dead evidence per frame -- from the earliest whistle within WHISTLE_BACK
    before each dead stretch of `labels` to that stretch, unless the ball is out of bounds from the
    whistle to FPS into the stretch or the whistle follows the previous stretch within
    WHISTLE_RESTART. Also returns the [(whistle, stretch start)] moved."""
    n = ctx.tac.n
    extra = np.zeros(n)
    moved = []
    if not len(whistles):
        return extra, moved
    pos, present = ctx.kin["pos"], ctx.kin["present"]
    outside = present & ~_inside(pos)
    prev_end = -10 ** 9
    for s, e in rules.runs(labels == "D"):
        lo = np.searchsorted(whistles, s - WHISTLE_BACK)
        hi = np.searchsorted(whistles, s)
        cands = [int(w) for w in whistles[lo:hi] if w - prev_end > WHISTLE_RESTART]
        prev_end = e
        if not cands:
            continue
        w = cands[0]
        if outside[w:min(n, s + FPS)].any():
            continue
        extra[w:s] += WHISTLE_LLR
        moved.append((w, s))
    return extra, moved


def team_distances(ctx, cfg):
    """(n,) each team's nearest player to the ball, m (inf where it has none)."""
    players = ctx._set(cfg)
    key = ctx._key(cfg)
    team = players["team"][key]
    uncertain = players["uncertain"][key] if cfg["crosscheck"] else None
    pos = ctx.kin["pos"]
    n = len(pos)
    d_home, d_away = np.full(n, np.inf), np.full(n, np.inf)
    for s in range(0, n, 20000):
        e = min(n, s + 20000)
        d = np.linalg.norm(players["players"][s:e] - pos[s:e, None, :].astype(np.float32), axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        if uncertain is not None:
            d = np.where(uncertain[s:e], np.inf, d)  # rule 3.a: a team in doubt says nothing
        d_home[s:e] = np.where(team[s:e] == HOME, d, np.inf).min(axis=1)
        d_away[s:e] = np.where(team[s:e] == AWAY, d, np.inf).min(axis=1)
    return d_home, d_away


def team_evidence(ctx, cfg, events):
    """(n,) home-vs-away log-likelihood ratio per frame: the players' distances to the ball, per
    frame (divided by CORR_FRAMES), plus each event's bonus for the credited team."""
    players = ctx._set(cfg)
    key = ctx._key(cfg)
    team = players["team"][key]
    uncertain = players["uncertain"][key] if cfg["crosscheck"] else None
    pos = ctx.kin["pos"]
    ball_var = ctx.kin["pos_std"] ** 2 if "pos_std" in ctx.kin else np.full(len(pos), BALL_STD ** 2)
    n = len(pos)
    d_home, d_away = team_distances(ctx, cfg)
    near = np.minimum(d_home, d_away) <= rules.FALLBACK_RADIUS
    base = CONTROL_STD ** 2 + np.nan_to_num(ball_var, nan=BALL_STD ** 2)
    player_std = getattr(ctx.tac, "pos_std", None)
    if cfg.get("player_std", True) and player_std is not None and player_std.shape == players["players"].shape[:2]:
        # Rule 21: each player's own position uncertainty widens his Gaussian, so a bridged player
        # counts less as it grows: per team, the best -d^2 / 2s^2 - log(s / s0) with s^2 = s0^2 +
        # his variance (s0^2 = the old shared variance; with no uncertainty, the old evidence).
        ll = {}
        for side in (HOME, AWAY):
            best = np.full(n, -np.inf)
            for s in range(0, n, 20000):
                e = min(n, s + 20000)
                d = np.linalg.norm(players["players"][s:e] - pos[s:e, None, :].astype(np.float32), axis=2)
                s2 = base[s:e, None] + np.nan_to_num(player_std[s:e], nan=0.0) ** 2
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = -d ** 2 / (2 * s2) - 0.5 * np.log(s2 / base[s:e, None])
                ok = np.isfinite(d) & (team[s:e] == side)
                if uncertain is not None:
                    ok &= ~uncertain[s:e]
                best[s:e] = np.where(ok, v, -np.inf).max(axis=1)
            ll[side] = np.maximum(best, -1e6 / (2 * base))
        score = ll[HOME] - ll[AWAY]
    else:
        with np.errstate(invalid="ignore"):
            score = (np.minimum(d_away, 1e3) ** 2 - np.minimum(d_home, 1e3) ** 2) / (2 * base)
    score = np.where(near, np.clip(score, -PROX_CLIP, PROX_CLIP), 0.0) / CORR_FRAMES
    if "z" in ctx.kin and cfg.get("height", True):
        # Rule 17: a player under a ball in the air is not in possession of it -- the distance
        # evidence counts as much as the ball is within reach.
        import ball3d
        from scipy.special import ndtr
        reach = ndtr((ball3d.REACH - ctx.kin["z"]) / np.maximum(ctx.kin["z_std"], 0.2))
        score = score * reach
    if cfg.get("event_switch"):
        # Rule 59: the distances may only confirm the team of the half's latest credited event.
        last = np.full(n, -1, dtype=np.int8)
        for t, who, _, _, _ in credited(ctx, cfg, events):
            last[t] = who
        for _, first, last_frame in halves():
            seg = last[first:last_frame + 1]
            idx = np.where(seg >= 0, np.arange(len(seg)), -1)
            np.maximum.accumulate(idx, out=idx)
            last[first:last_frame + 1] = np.where(idx >= 0, seg[np.maximum(idx, 0)], -1)
        score = np.where(last == HOME, np.maximum(score, 0.0), np.where(last == AWAY, np.minimum(score, 0.0), score))
    return score + event_evidence(ctx, cfg, events, n)


def event_evidence(ctx, cfg, events, n):
    """(n,) the events' share of team_evidence: each credited event's bonus at its frame."""
    score = np.zeros(n)
    for t, who, _, _, llr in credited(ctx, cfg, events):
        score[t] += llr if who == HOME else -llr
    if cfg.get("passes", False):    # off: with Viterbi no gain, with posterior decoding too many spells
        for t, who in passes(events):
            score[t] += PASS_LLR if who == HOME else -PASS_LLR
    return score


def credited(ctx, cfg, events):
    """[(frame, team, kind, column, nats)]: each event with a team and what it is worth for it, after
    the rule 6 guard and rule 26's discounts."""
    out = []
    guard = ctx.rule6["guard"] if cfg.get("dropouts") and ctx.rule6 else None
    seen = ctx.kin["seen"].astype(float)
    support = _support(ctx.kin) if TOUCH_PRECISION and "pick_conf" in ctx.kin else None
    last = {}                   # team -> (frame, nats) of its last counted touch (cfg touch_once)
    dists = None                # rule 60: (home, away) nearest distances, computed once when needed
    for t, who, kind, slot in events:
        if who < 0:
            continue
        if kind == "control" and guard is not None and _guarded(guard, 1 - who, t):
            kind = "touch"
        if kind == "touch" and cfg.get("kick") and _kick(ctx, cfg, t, slot):
            # Rule 55: a kick alone pays the switch to the kicker's team.
            llr = math.log(POSSESSION_S * FPS) + KICK_MARGIN
            out.append((t, who, "kick", slot, llr * (support[t] if support is not None else 1.0)))
            continue
        if kind == "far touch" and cfg.get("far_clear"):
            # Rule 60: a far touch needs the other team FAR_MARGIN farther from the ball.
            if dists is None:
                dists = team_distances(ctx, cfg)
            own, other = (dists[0][t], dists[1][t]) if who == HOME else (dists[1][t], dists[0][t])
            if not other - own >= FAR_MARGIN:
                continue
        if kind == "touch" and cfg.get("air_touch") and _aerial(ctx.kin, t):
            kind = "far touch"
        if kind == "touch" and cfg.get("touch_sides") and not (
                seen[max(0, t - rules.DV_WINDOW):t].any() and seen[t + 1:t + rules.DV_WINDOW + 1].any()):
            kind = "far touch"
        llr = EVENT_LLR[kind]
        if kind in ("touch", "far touch"):
            if cfg.get("touch_seen"):
                # Rule 26: a touch is as sure as the ball measured around it -- the share of seen
                # frames over the velocity windows either side.
                llr *= seen[max(0, t - rules.DV_WINDOW):t + rules.DV_WINDOW + 1].mean() ** TOUCH_SEEN_POWER
            if support is not None:
                llr *= support[t]
            if cfg.get("touch_once"):
                # Rule 26: one team's touches within CORR_FRAMES are one look, not several.
                prev = last.get(who)
                if prev is not None and t - prev[0] <= CORR_FRAMES:
                    extra = max(0.0, llr - prev[1])
                    last[who] = (t, max(llr, prev[1]))
                    llr = extra
                else:
                    last[who] = (t, llr)
        elif support is not None and kind in PRECISION_KINDS:
            # Rule 53b: a control or carry located on weak picks is as doubtful as a touch there.
            llr *= support[t]
        out.append((t, who, kind, slot, llr))
    return out


def _support(kin):
    """Rule 53: (n,) the weight of a touch at each frame -- min over its two sides of the precision of the nearest
    used pick on that side (within kalman.RESET_GAP, the frame itself counting for both), over the top band's."""
    import ball_track
    import kalman
    conf = kin["pick_conf"]
    n = len(conf)
    have = conf > 0
    idx = np.arange(n)
    before = np.maximum.accumulate(np.where(have, idx, -1))
    after = np.minimum.accumulate(np.where(have, idx, n)[::-1])[::-1]
    floor = ball_track.PRECISION[0][1]
    top = ball_track.PRECISION[-1][1]

    def side(j, ok):
        jj = np.clip(j, 0, n - 1)
        return np.where(ok, ball_track.precision(conf[jj]), floor)
    left = side(before, (before >= 0) & (idx - before <= kalman.RESET_GAP))
    right = side(after, (after < n) & (after - idx <= kalman.RESET_GAP))
    return np.minimum(1.0, np.minimum(left, right) / top)


def _kick(ctx, cfg, t, slot):
    """Rule 55: the touch at t by the player in `slot` is a kick -- the ball seen on both sides, on
    the ground and not arriving in a flight (where the 3D ball exists), sent off at KICK_SPEED or
    more, no slower than it came, and away from the player."""
    kin = ctx.kin
    seen = kin["seen"]
    if not (seen[max(0, t - rules.DV_WINDOW):t].any() and seen[t + 1:t + rules.DV_WINDOW + 1].any()):
        return False
    s_in, s_out = float(kin["speed_in"][t]), float(kin["speed_out"][t])
    if not (np.isfinite(s_out) and s_out >= KICK_SPEED and (not np.isfinite(s_in) or s_out >= s_in)):
        return False
    if "z" in kin and "flight" in kin:
        import ball3d
        if kin["z"][t] > ball3d.CONTROL_REACH or kin["flight"][max(0, t - rules.TOUCH_NMS):t].any():
            return False
    away = kin["pos"][t] - ctx._set(cfg)["players"][t, slot]
    v = kin["v_out"][t]
    return bool(np.all(np.isfinite(away)) and np.all(np.isfinite(v)) and float(np.dot(v, away)) > 0)


def _aerial(kin, t):
    """Rule 26: a touch on a ball above control height, or within rules.TOUCH_NMS of a flight -- a
    header or a challenge for a high ball, which wins no control by itself (CHALLENGE.md: "an aerial
    duel won without subsequent control")."""
    import ball3d
    return kin["z"][t] > ball3d.CONTROL_REACH or kin["flight"][max(0, t - rules.TOUCH_NMS):t + rules.TOUCH_NMS + 1].any()


def passes(events):
    """Rule 23: [(frame, team)] where a team completes a pass -- a touch by one of its players, after a
    touch by a different player of the same team within PASS_MAX and no other team's touch in
    between. Two players of one team on the ball in turn is control (CHALLENGE.md's proxy: a team
    completing passes), where a lone touch may be a deflection. 1H 11510 -> 11553: an away pass in
    an attack that two single touches (6 nats) could not pay two switches (~11.6 nats) for."""
    out = []
    last = None     # (frame, team, column) of the last credited touch
    for t, who, kind, slot in sorted(e for e in events if e[1] >= 0):
        if last is not None and who == last[1] and slot != last[2] and t - last[0] <= PASS_MAX:
            out.append((t, who))
        last = (t, who, slot)
    return out


def _guarded(guard, holder, t):
    """Rule 6 guard: within rules.GUARD frames of `holder`'s team losing a player near the ball."""
    if holder not in guard:
        return False
    times = guard[holder]
    i = np.searchsorted(times, t, side="right")
    return i > 0 and t - times[i - 1] <= rules.GUARD


def emissions(team_llr, dead_llr, forced):
    """(n, 3) log emission scores. The team evidence decides H vs A and barely touches live vs
    dead: log(2 sigmoid(s)) is at most log 2 for the favoured team."""
    e = np.empty((len(team_llr), 3))
    e[:, H_] = math.log(2) - np.logaddexp(0, -team_llr)
    e[:, A_] = math.log(2) - np.logaddexp(0, team_llr)
    e[:, D_] = dead_llr
    e[forced, H_] = e[forced, A_] = -np.inf
    return e


def special_transitions(ctx, cfg, events):
    """Rule 25: {frame: (3, 3) log transitions into that frame} while the ball travels.

    Not a prediction of where the ball goes -- the reception's touch evidence says that -- but how
    exposed the move was: a ball sent where its team controls less than usual is more likely to be
    stolen. Each move, under each hypothesis, gets q = 1 - (that team's chance of keeping it), and
    the usual per-frame switch price is scaled by q / q_ref (clipped to 1/REL_MAX..REL_MAX) from the
    kick to REL_MARGIN after the arrival. q_ref is the match's typical q: the median over its moves
    of q for the team that made the touch -- the model's own output, no labels.

      cfg pass_prob   a pass: two credited touches PASS_MIN_S..PASS_MAX_FRAMES apart, the ball
                      moving PASS_MIN_M or more. Keeping it = pass probability from that team's
                      player nearest the ball (left out) to where the next touch took it, at its
                      real travel time.
      cfg flight_pc   a flight not already inside a pass: keeping it = the team's pitch control
                      at the landing spot, from the players at the launch.
      cfg kappa       Spearman's defender advantage within each hypothesis (1: symmetric).
    Live and dead keep their usual prices."""
    import pitch_control as pc
    players = ctx._set(cfg)
    pos = players["players"]
    team = players["team"][ctx._key(cfg)]
    if not hasattr(ctx, "_pc_vel") or ctx._pc_vel.shape != pos.shape:
        ctx._pc_vel = pc.velocities(pos)
    vel = ctx._pc_vel
    ball = ctx.kin["pos"]
    kappa = cfg.get("kappa", 1.0)
    base = np.exp(transitions())
    die, switch = base[H_, D_], base[H_, A_]
    touches = sorted(e for e in events if e[1] >= 0)
    toucher = lambda t: next((who for f, who, _, _ in reversed(touches) if f <= t), -1)

    def apply(out, spans, q_home, q_away, owner):
        """Scale the switch price over each (start, stop) span by q / q_ref."""
        q_own = np.where(owner == HOME, q_home, q_away)[owner >= 0]
        q_ref = float(np.median(q_own)) if len(q_own) else 0.5
        for (a, b), qh, qa in zip(spans, q_home, q_away):
            fh = np.clip(qh / max(q_ref, 1e-6), 1 / REL_MAX, REL_MAX)
            fa = np.clip(qa / max(q_ref, 1e-6), 1 / REL_MAX, REL_MAX)
            p = base.copy()
            p[H_] = [1 - switch * fh - die, switch * fh, die]
            p[A_] = [switch * fa, 1 - switch * fa - die, die]
            m = np.log(p)
            for t in range(a + 1, b + REL_MARGIN + 1):
                out.setdefault(int(t), m)
        return q_ref

    out = {}
    if cfg.get("pass_prob"):
        kicks = [(a[0], b[0], a[1]) for a, b in zip(touches[:-1], touches[1:])
                 if PASS_MIN_S <= b[0] - a[0] <= PASS_MAX_FRAMES and np.isfinite(ball[a[0], 0]) and
                 np.isfinite(ball[b[0], 0]) and np.linalg.norm(ball[b[0]] - ball[a[0]]) >= PASS_MIN_M]
        if kicks:
            t1 = np.array([k[0] for k in kicks])
            t2 = np.array([k[1] for k in kicks])
            keep = {}
            for side in (HOME, AWAY):
                d = np.linalg.norm(pos[t1] - ball[t1][:, None, :], axis=2)
                d = np.where((team[t1] == side) & np.isfinite(d), d, np.inf)
                passer = np.where(np.isfinite(d.min(axis=1)), d.argmin(axis=1), -1)
                ph, pa = pc.integrate(pos[t1], vel[t1], team[t1], ball[t1].astype(float), ball[t2].astype(float),
                                      (t2 - t1) / FPS, along=True, exclude=passer,
                                      attack=np.full(len(t1), side), kappa=kappa)
                keep[side] = ph if side == HOME else pa
            ctx._rule25_ref = {"pass": apply(out, list(zip(t1, t2)), 1 - keep[HOME], 1 - keep[AWAY],
                                             np.array([k[2] for k in kicks]))}
    if cfg.get("flight_pc") and ctx.kin.get("flights"):
        fl = [(s, e) for s, e, *_ in ctx.kin["flights"]
              if s + 1 not in out and np.isfinite(ball[e - 1, 0]) and e - s >= PASS_MIN_S]
        if fl:
            s = np.array([f[0] for f in fl])
            e = np.array([f[1] for f in fl])
            land = ball[e - 1].astype(float)
            ph, pa = pc.integrate(pos[s], vel[s], team[s], land, land, (e - s) / FPS)
            share = ph / np.maximum(ph + pa, 1e-9)
            apply(out, list(zip(s, e)), 1 - share, share, np.array([toucher(int(x)) for x in s]))
    return out


def viterbi(e, T, special=None):
    """Most likely state per frame for log emissions e (m, 3) and log transitions T (3, 3);
    special {local frame: (3, 3)} replaces T into those frames (rule 25)."""
    m = len(e)
    back = np.zeros((m, 3), dtype=np.int8)
    T = T.tolist()
    special = {t: s.tolist() for t, s in (special or {}).items()}
    rows = e.tolist()
    d = list(rows[0])
    for t in range(1, m):
        r, nd = rows[t], [0.0, 0.0, 0.0]
        Tt = special.get(t, T)
        for j in range(3):
            c0, c1, c2 = d[0] + Tt[0][j], d[1] + Tt[1][j], d[2] + Tt[2][j]
            if c0 >= c1 and c0 >= c2:
                nd[j], back[t, j] = c0 + r[j], 0
            elif c1 >= c2:
                nd[j], back[t, j] = c1 + r[j], 1
            else:
                nd[j], back[t, j] = c2 + r[j], 2
        d = nd
    path = np.zeros(m, dtype=np.int8)
    path[-1] = int(np.argmax(d))
    for t in range(m - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def posterior(e, T, special=None):
    """(m, 3) probability of each state per frame: scaled forward-backward. special as in
    viterbi."""
    m = len(e)
    p = np.exp(e - np.max(np.where(np.isfinite(e), e, -1e300), axis=1, keepdims=True)).tolist()
    A0 = np.exp(T).tolist()
    special = {t: np.exp(s).tolist() for t, s in (special or {}).items()}
    alpha = [[0.0] * 3 for _ in range(m)]
    a = [p[0][k] / 3 for k in range(3)]
    for t in range(m):
        if t:
            prev = alpha[t - 1]
            A = special.get(t, A0)
            a = [(prev[0] * A[0][j] + prev[1] * A[1][j] + prev[2] * A[2][j]) * p[t][j] for j in range(3)]
        s = a[0] + a[1] + a[2]
        alpha[t] = [a[0] / s, a[1] / s, a[2] / s]
    out = np.empty((m, 3))
    b = [1.0, 1.0, 1.0]
    for t in range(m - 1, -1, -1):
        g = [alpha[t][k] * b[k] for k in range(3)]
        s = g[0] + g[1] + g[2]
        out[t] = (g[0] / s, g[1] / s, g[2] / s)
        nb = [b[k] * p[t][k] for k in range(3)]
        A = special.get(t, A0)     # the transition into t
        b = [A[i][0] * nb[0] + A[i][1] * nb[1] + A[i][2] * nb[2] for i in range(3)]
        s = b[0] + b[1] + b[2]
        b = [b[0] / s, b[1] / s, b[2] / s]
    return out


EXPLAIN_BIN = FPS // 2      # frames per bin of the viewer's distance rows


def explain(ctx, cfg, labels, detail):
    """For the viewer: why each label is what it is. detail from predict(with_posterior=True).

    touches   every rule 1 peak: [frame, column, that column's team (-1 none), distance m, ball
              height m, fate: 0 credited, 1 nobody within the touch radius, 2 above reach (rule 17),
              kind code as run.KIND_CODES (touch or far touch)]
    bins      per EXPLAIN_BIN frames: the distance evidence (nats, + home) and each team's nearest
              player at the closest moment (m)
    runs      per label run: [start, stop, label, distance, events, dead (nats, + for the run's own
              label), mean confidence, then the frames inside it whose evidence for the other team
              sums highest: start, stop, nats]. The best path flips only where those nats beat
              switch_cost, the price of switching to the other team and back.
    """
    n = ctx.tac.n
    events = detail["events"]
    ev = event_evidence(ctx, cfg, events, n)
    llr = detail["team_llr"]
    prox = llr - ev
    players = ctx._set(cfg)
    team = players["team"][ctx._key(cfg)]
    z = ctx.kin.get("z") if cfg.get("height", True) else None
    num = lambda v, k: round(float(v), k) if v is not None and np.isfinite(v) else None

    credited = {(t, k) for t, _, k, _ in events}
    radius = cfg.get("touch_radius", rules.TOUCH_RADIUS)
    touches = []
    for t, slot, dist in zip(ctx.touch_frames, players["touch_slot"], players["touch_dist"]):
        t, slot = int(t), int(slot)
        kind = "touch" if dist <= rules.TOUCH_CLOSE else "far touch"
        fate = 1 if dist > radius else 0 if (t, kind) in credited else 2
        touches.append([t, slot, int(team[t, slot]) if np.isfinite(dist) else -1, num(dist, 2),
                        num(z[t], 2) if z is not None else None, fate, 0 if kind == "touch" else 3])

    T = transitions()
    return {"touches": touches, "bin": EXPLAIN_BIN, "bins": explain_bins(ctx, cfg, prox),
            "runs": explain_runs(labels, llr, prox, ev, detail["dead_llr"], detail["confidence"]),
            "switch_cost": round(float(2 * T[H_, H_] - T[H_, A_] - T[A_, H_]), 1), "event_llr": EVENT_LLR}


def explain_bins(ctx, cfg, prox):
    """explain()'s bins: prox (n,) nats summed per EXPLAIN_BIN, and each team's nearest player."""
    n = len(prox)
    num = lambda v, k: round(float(v), k) if v is not None and np.isfinite(v) else None
    nb = -(-n // EXPLAIN_BIN)
    pad = nb * EXPLAIN_BIN - n
    d_home, d_away = team_distances(ctx, cfg)

    def closest(d):
        d = np.pad(d, (0, pad), constant_values=np.inf).reshape(nb, EXPLAIN_BIN).min(axis=1)
        return [num(v, 1) for v in d]

    return {"distance": [round(float(v), 2) for v in np.pad(prox, (0, pad)).reshape(nb, EXPLAIN_BIN).sum(axis=1)],
            "home": closest(d_home), "away": closest(d_away)}


def explain_runs(labels, llr, prox, ev, dead_llr, confidence):
    """explain()'s runs, from any model's per-frame team evidence llr (nats, + home) split into
    prox + ev, its dead_llr and its confidence."""
    num = lambda v, k: round(float(v), k) if v is not None and np.isfinite(v) else None
    runs = []
    for k in "HAD":
        for s, e in rules.runs(labels == k):
            sign = -1.0 if k == "A" else 1.0
            conf = confidence[s:e]
            row = [s, e, k, round(sign * float(prox[s:e].sum()), 1), round(sign * float(ev[s:e].sum()), 1),
                   round(float(dead_llr[s:e].sum()) * (1 if k == "D" else -1), 1),
                   num(np.nanmean(conf), 2) if np.isfinite(conf).any() else None]
            if k == "D":
                runs.append(row + [None, None, None])
                continue
            c = np.concatenate([[0.0], np.cumsum(-sign * llr[s:e])])
            j = int(np.argmax(c - np.minimum.accumulate(c)))
            i = int(np.argmin(c[:j + 1]))
            runs.append(row + [s + i, s + j, round(float(c[j] - c[i]), 1)])
    runs.sort()
    return runs


def predict(ctx, cfg, with_posterior=False):
    """Labels (n,) of 'H' 'A' 'D' 'X' for the rule switches in cfg, and what run.py reports:
    events, the per-frame evidence and, with_posterior, the probability of each label."""
    n = ctx.tac.n
    events = ctx.events(cfg)
    team_llr = team_evidence(ctx, cfg, events)
    dead_llr, forced = dead_evidence(ctx, cfg, events)
    T = transitions()
    special = special_transitions(ctx, cfg, events) if cfg.get("pass_prob") or cfg.get("flight_pc") else {}
    labels, confidence = _decode(ctx, cfg, emissions(team_llr, dead_llr, forced), T, special, with_posterior)
    moved = []
    if cfg.get("whistle_back"):
        # Rule 57: a second decode with each whistled stoppage's start moved back to its whistle.
        extra, moved = whistle_evidence(ctx, labels, whistle_frames(ctx))
        if moved:
            dead_llr = dead_llr + extra
            labels, confidence = _decode(ctx, cfg, emissions(team_llr, dead_llr, forced), T, special, with_posterior)
    if cfg.get("short_spells"):
        # rule 63; cfg short_spells: True (both parts), "live" (part 1 only) or "dead" (part 2 only); cfg
        # short_broadcast: part 2 only where the broadcast's close-up is the stretch's only dead-ball signal
        only_bro = None
        if cfg.get("short_broadcast"):
            masks, _, _ = _masks(ctx, cfg)
            only_bro = np.zeros(ctx.tac.n, dtype=bool)
            for name, m in masks.items():
                if name != "broadcast no homography":
                    only_bro |= m
            only_bro = ~only_bro
        labels = short_spells(ctx, labels, forced, cfg["short_spells"], only_bro, cfg.get("short_s"))
    return labels, {"events": events, "team_llr": team_llr, "dead_llr": dead_llr,
                    "confidence": confidence, "special": special, "whistles_moved": moved}


def short_spells(ctx, labels, forced, mode=True, only_bro=None, short_s=None):
    """Rule 63: a live stretch shorter than SHORT_S between dead ones becomes dead, unless the ball leaves the pitch within
    FPS of its end; then a dead stretch shorter than SHORT_S between two of the same team becomes that team's, unless the
    ball is outside the pitch within FPS of it, the players are mostly still in it, or it is forced."""
    labels = labels.copy()
    n = len(labels)
    short_s = SHORT_S if short_s is None else short_s
    short = int(short_s * FPS)
    outside = ctx.kin["present"] & ~_inside(ctx.kin["pos"])
    quiet = rules.quiet_players(ctx)

    def near_out(s, e):
        return outside[max(0, s - FPS):min(n, e + FPS)].any()

    for _, first, last in halves():
        seg = labels[first:last + 1]
        live = np.isin(seg, ("H", "A"))
        stretches = rules.runs(live)
        for s, e in stretches if mode in (True, "live") else ():      # 1: short live between dead
            if s == 0 or e >= len(seg) or e - s >= short:
                continue
            if seg[s - 1] == "D" and seg[e] == "D" and not near_out(first + e, first + e):
                seg[s:e] = "D"
        for s, e in rules.runs(seg == "D") if mode in (True, "dead") else ():    # 2: short dead between the same team
            if s == 0 or e >= len(seg) or e - s >= short:
                continue
            a, b = first + s, first + e
            if only_bro is not None and not only_bro[a:b].all():
                continue
            if seg[s - 1] == seg[e] and seg[e] in ("H", "A") and not (near_out(a, b) or quiet[a:b].any() or forced[a:b].any()):
                seg[s:e] = seg[e]
        labels[first:last + 1] = seg
    return labels


def _decode(ctx, cfg, e, T, special, with_posterior):
    n = ctx.tac.n
    labels = np.full(n, "X", dtype="<U1")
    confidence = np.full(n, np.nan)
    for _, first, last in halves():
        local = {t - first: s for t, s in special.items() if first < t <= last}
        # cfg "decode": "viterbi", the one best path; "posterior", each frame's most probable
        # label -- a short spell the evidence favours but that cannot pay two switches on the
        # best path still shows (1H 11553).
        if cfg.get("decode", "viterbi") == "posterior":
            post = posterior(e[first:last + 1], T, local)
            path = post.argmax(axis=1)
        else:
            path = viterbi(e[first:last + 1], T, local)
            post = posterior(e[first:last + 1], T, local) if with_posterior else None
        labels[first:last + 1] = LABELS[path]
        if post is not None:
            confidence[first:last + 1] = post[np.arange(len(path)), path]
    return labels, confidence
