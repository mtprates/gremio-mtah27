"""Model T: possession changes only at touches, priced by pitch control.

A simpler model than C (model/hmm.py), on the same ball and player tracks. The ball says where
something happened to it:

    bounce   a landing (the end of a 3D arc) with no horizontal change -- no rule 1 peak near it,
             or one that keeps the ball's line and only loses speed. Nobody touched it.
    touch    every other rule 1 peak (a change of the horizontal velocity) within reach.

Between touches possession stays, bar a rare switch (BACKGROUND_S) for touches never seen. At a
touch two things decide it:

    pitch control  who would win a ball sent to the touch spot, from the players where they were
                   when it set off (the previous touch): the prior on which team touches it
    proximity      each team's nearest player at the touch: the likelihood of who did, against a
                   false peak ("nobody", at NOBODY_M)

A touch by the other team wins possession with CONTROL_WIN; otherwise it was a deflection. Worked
example: home on the ball, an away player alone at a touch where control is even -- P(switch) 0.46;
then a home player alone at the next one: home all along (0.54) beats away and back (0.46 x 0.47).
Two away touches in a row switch it. docs/VALUES.md, model T.

Dead-ball evidence and the live / dead prices are C's (hmm.dead_evidence, hmm.transitions). Each
touch's probabilities become that frame's transitions (hmm.viterbi's special), so C's decoder runs
unchanged.
"""

import math

import numpy as np

import ball3d
import hmm
import pitch_control as pc
import rules
from data import FPS, halves

# Params
BOUNCE_WINDOW = rules.TOUCH_NMS     # frames: a peak this close to a landing is the landing's
BOUNCE_ANGLE = rules.TURN_ANGLE     # degrees: ... and a bounce if it turns less than this
BOUNCE_KEEP = 0.5                   # ... keeping this share of its speed, at most all of it
LANDING_Z = ball3d.AIR_MIN          # m: an arc ending this low ends on the grass
TOUCH_NOISE = 0.25                  # share of rule 1 peaks with nobody within 4 m (a false peak)
UNSURE_NOISE = 0.5                  # ... where the ball was unseen on a side of the peak
CONTROL_WIN = 0.5                   # an opposing touch wins possession (else a deflection)
NOBODY_M = rules.TOUCH_CLOSE        # m: a false peak is as likely as a player this far away
BACKGROUND_S = 120.0                # s: mean live time between switches with no touch
PC_CLIP = 0.02                      # pitch control kept within this of 0 and 1
LAUNCH_MAX = int(pc.PASS_MAX_S * FPS)   # frames: the players at the previous touch, at most this early

H_, A_, D_ = hmm.H_, hmm.A_, hmm.D_
FATE_BOUNCE = 3                     # hmm.explain()'s fates, plus this one


def transitions():
    """C's 3 x 3 log transitions with the touchless switch price."""
    T = np.exp(hmm.transitions())
    die = T[H_, D_]
    switch = 1.0 / (BACKGROUND_S * FPS)
    T[H_, :2] = [1 - switch - die, switch]
    T[A_, :2] = [switch, 1 - switch - die]
    return np.log(T)


def landings(kin):
    """Frames where a fitted arc comes down on the grass."""
    z = kin["z"]
    return np.array([e - 1 for s, e, *_ in kin["flights"] if z[e - 1] <= LANDING_Z], dtype=np.int64)


def bounces(kin, peaks):
    """(bounce frames, (len(peaks),) bool: the peak is a bounce). A landing with no peak within
    BOUNCE_WINDOW is a bounce at the landing; one whose nearest peak keeps the ball's line (under
    BOUNCE_ANGLE, BOUNCE_KEEP..1 of its speed) is a bounce at that peak."""
    is_bounce = np.zeros(len(peaks), dtype=bool)
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(kin["v_in"] * kin["v_out"], axis=1) / (kin["speed_in"] * kin["speed_out"])
        keep = kin["speed_out"] / kin["speed_in"]
    straight = (cos >= math.cos(math.radians(BOUNCE_ANGLE))) & (keep >= BOUNCE_KEEP) & (keep <= 1.0)
    for land in landings(kin):
        near = np.flatnonzero(np.abs(peaks - land) <= BOUNCE_WINDOW)
        if not len(near):
            out.append(int(land))
            continue
        i = near[np.argmin(np.abs(peaks[near] - land))]
        if straight[peaks[i]]:
            is_bounce[i] = True
            out.append(int(peaks[i]))
    return np.array(sorted(set(out)), dtype=np.int64), is_bounce


def touches(ctx, cfg):
    """Every rule 1 peak and what the model makes of it: a dict of (m,) arrays -- frame, column,
    team (of the nearest player), dist, fate (hmm.explain()'s, plus FATE_BOUNCE), each team's
    nearest distance, the noise and win shares, and for the kept ones (fate 0) pitch control at the
    spot (P home) from the players at launch."""
    kin = ctx.kin
    players = ctx._set(cfg)
    team = players["team"][ctx._key(cfg)]
    peaks = np.asarray(ctx.touch_frames, dtype=np.int64)
    slot = np.asarray(players["touch_slot"], dtype=np.int64)
    dist = np.asarray(players["touch_dist"], dtype=float)
    radius = cfg.get("touch_radius", rules.TOUCH_RADIUS)
    d_home, d_away = hmm.team_distances(ctx, cfg)
    n = len(d_home)
    lo = np.maximum(peaks - rules.TOUCH_SLACK, 0)
    near_h = np.array([d_home[a:t + rules.TOUCH_SLACK + 1].min() for a, t in zip(lo, peaks)]) if len(peaks) else np.zeros(0)
    near_a = np.array([d_away[a:t + rules.TOUCH_SLACK + 1].min() for a, t in zip(lo, peaks)]) if len(peaks) else np.zeros(0)

    bounce_frames, is_bounce = bounces(kin, peaks)
    fate = np.zeros(len(peaks), dtype=np.int8)
    fate[dist > radius] = 1
    if cfg.get("height", True):
        fate[kin["z"][peaks] > ball3d.REACH] = 2       # rule 17
    fate[is_bounce] = FATE_BOUNCE

    seen = kin["seen"]
    sides = np.array([seen[max(0, t - rules.DV_WINDOW):t].any() and seen[t + 1:t + rules.DV_WINDOW + 1].any()
                      for t in peaks], dtype=bool)
    noise = np.where(sides, TOUCH_NOISE, UNSURE_NOISE)
    win = np.array([0.0 if hmm._aerial(kin, int(t)) else CONTROL_WIN for t in peaks])

    # Pitch control at the spot, from the players when the ball set off: the previous kept touch,
    # at most LAUNCH_MAX earlier.
    kept = np.flatnonzero(fate == 0)
    pc_home = np.full(len(peaks), 0.5)
    if len(kept):
        t = peaks[kept]
        prev = np.r_[t[0] - LAUNCH_MAX, t[:-1]]
        launch = np.maximum(prev, t - LAUNCH_MAX).clip(0, n - 1)
        pos = players["players"]
        if not hasattr(ctx, "_pc_vel") or ctx._pc_vel.shape != pos.shape:
            ctx._pc_vel = pc.velocities(pos)
        spot = kin["pos"][t].astype(float)
        ok = np.isfinite(spot[:, 0])
        ph, pa = pc.integrate(pos[launch[ok]], ctx._pc_vel[launch[ok]], team[launch[ok]], spot[ok], spot[ok],
                              (t[ok] - launch[ok]) / FPS)
        with np.errstate(invalid="ignore"):
            share = np.where(ph + pa > 0, ph / np.maximum(ph + pa, 1e-12), 0.5)
        pc_home[kept[ok]] = np.clip(share, PC_CLIP, 1 - PC_CLIP)
    who = np.where(np.isfinite(dist), team[peaks, slot.clip(0)], -1)
    return {"frame": peaks, "slot": slot, "team": who, "dist": dist, "fate": fate, "near_h": near_h,
            "near_a": near_a, "noise": noise, "win": win, "pc_home": pc_home, "bounces": bounce_frames}


def touch_transitions(ctx, tch):
    """{frame: (3, 3) log transitions into it} at each kept touch, the touch's observation folded in
    (rows normalised), and per touch (P home -> away, P away -> home, proximity nats for home)."""
    base = np.exp(transitions())
    die = base[H_, D_]
    ball_var = ctx.kin["pos_std"] ** 2 if "pos_std" in ctx.kin else np.full(len(ctx.kin["pos"]), hmm.BALL_STD ** 2)
    out, p_ha, p_ah, nats = {}, {}, {}, {}
    for i in np.flatnonzero(tch["fate"] == 0):
        t = int(tch["frame"][i])
        s2 = hmm.CONTROL_STD ** 2 + float(np.nan_to_num(ball_var[t], nan=hmm.BALL_STD ** 2))
        lh, la = (math.exp(-min(d, 1e3) ** 2 / (2 * s2)) for d in (tch["near_h"][i], tch["near_a"][i]))
        l0 = math.exp(-NOBODY_M ** 2 / (2 * s2))
        nu, win, ph = tch["noise"][i], tch["win"][i], tch["pc_home"][i]
        pa = 1 - ph
        z_h = (1 - nu) * (ph * lh + pa * la) + nu * l0
        q_ha = (1 - nu) * pa * win * la / z_h            # home on the ball, away wins it here
        q_ah = (1 - nu) * ph * win * lh / z_h
        p = base.copy()
        p[H_] = [(1 - q_ha) * (1 - die), q_ha * (1 - die), die]
        p[A_] = [q_ah * (1 - die), (1 - q_ah) * (1 - die), die]
        with np.errstate(divide="ignore"):
            out[t] = np.log(p)
        p_ha[i], p_ah[i] = q_ha, q_ah
        nats[i] = math.log((lh + l0) / (la + l0))
    return out, p_ha, p_ah, nats


def predict(ctx, cfg, with_posterior=False):
    """As hmm.predict: labels (n,) of 'H' 'A' 'D' 'X' and the detail the viewer explains."""
    n = ctx.tac.n
    tch = touches(ctx, cfg)
    special, p_ha, p_ah, nats = touch_transitions(ctx, tch)
    dead_llr, forced = hmm.dead_evidence(ctx, cfg)
    e = hmm.emissions(np.zeros(n), dead_llr, forced)
    T = transitions()
    labels = np.full(n, "X", dtype="<U1")
    confidence = np.full(n, np.nan)
    for _, first, last in halves():
        local = {t - first: s for t, s in special.items() if first < t <= last}
        path = hmm.viterbi(e[first:last + 1], T, local)
        labels[first:last + 1] = hmm.LABELS[path]
        if with_posterior:
            post = hmm.posterior(e[first:last + 1], T, local)
            confidence[first:last + 1] = post[np.arange(len(path)), path]
    ev = np.zeros(n)
    events = []
    for i, v in nats.items():
        t = int(tch["frame"][i])
        ev[t] += v
        events.append((t, int(tch["team"][i]), "touch" if tch["dist"][i] <= rules.TOUCH_CLOSE else "far touch",
                       int(tch["slot"][i])))
    return labels, {"events": sorted(events), "team_llr": ev, "dead_llr": dead_llr, "confidence": confidence,
                    "special": special, "touches": tch, "p_ha": p_ha, "p_ah": p_ah, "nats": nats}


def explain(ctx, cfg, labels, detail):
    """hmm.explain()'s payload for model T. touches carry, after hmm's seven fields, the
    proximity nats for home, pitch control (P home) at the spot and the touch's switch
    probabilities home -> away and away -> home (None where the peak was dropped); bounces is
    every bounce frame. No fixed switch price: each touch has its own."""
    tch = detail["touches"]
    num = lambda v, k: round(float(v), k) if v is not None and np.isfinite(v) else None
    z = ctx.kin["z"]
    rows = []
    for i, t in enumerate(tch["frame"]):
        t = int(t)
        kind = 0 if tch["dist"][i] <= rules.TOUCH_CLOSE else 3
        kept = i in detail["nats"]
        rows.append([t, int(tch["slot"][i]), int(tch["team"][i]), num(tch["dist"][i], 2), num(z[t], 2),
                     int(tch["fate"][i]), kind, num(detail["nats"][i], 2) if kept else None,
                     num(tch["pc_home"][i], 2) if kept else None, num(detail["p_ha"][i], 3) if kept else None,
                     num(detail["p_ah"][i], 3) if kept else None])
    ev = detail["team_llr"]
    return {"touches": rows, "bounces": [int(t) for t in tch["bounces"]], "bin": hmm.EXPLAIN_BIN,
            "bins": hmm.explain_bins(ctx, cfg, ev),
            "runs": hmm.explain_runs(labels, ev, np.zeros_like(ev), ev, detail["dead_llr"], detail["confidence"]),
            "switch_cost": None, "event_llr": None}


def summary(detail):
    """Counts for the printout: peaks by fate, bounces, mean switch probability at kept touches."""
    tch = detail["touches"]
    fate = tch["fate"]
    q = [max(detail["p_ha"][i], detail["p_ah"][i]) for i in detail["nats"]]
    return ("{} peaks: {} touches kept, {} nobody near, {} above reach, {} bounces (peaks); {} bounces in all; "
            "switch probability at a kept touch, the likelier way: median {:.2f}, over 0.5 on {:.0%}").format(
        len(fate), int((fate == 0).sum()), int((fate == 1).sum()), int((fate == 2).sum()),
        int((fate == FATE_BOUNCE).sum()), len(tch["bounces"]), float(np.median(q)) if q else 0.0,
        float(np.mean(np.array(q) > 0.5)) if q else 0.0)
