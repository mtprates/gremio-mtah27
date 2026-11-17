"""Model P: C's HMM (model/hmm.py) tracking which player has the ball, not only which team.

The states are every player column (model/detector_model.py's detected players) plus dead. The
evidence is C's, per player instead of per team:

    proximity   each column's -d^2 / 2s^2 against the best column's on that frame (C keeps only
                the nearest of each team), capped at hmm.PROX_CLIP, once per hmm.CORR_FRAMES,
                times the height gate (rule 17); an absent column scores ABSENT_LLR
    events      each credited touch / control / carry (hmm.credited, rule 26's discounts) to the
                column it was credited to, not to its whole team
    dead ball   hmm.dead_evidence, unchanged

and the transitions add a pass: the ball goes to a teammate once per HOLDER_S, to the other team
once per hmm.POSSESSION_S, each split over the ON_PITCH players who could get it. What it changes:
in C, a home player's touch is home evidence however far the ball then goes; here the ball has to
reach a teammate for home to keep it, and a pass is cheaper than a turnover.

The label is the holder's team. Viterbi and forward-backward run over the structured matrix
(stay / teammate / opponent / dead) in O(states) per frame.
"""

import math

import numpy as np

import hmm
import rules
from data import FPS, HOME, halves

# Params
HOLDER_S = 3.0              # s: mean time on the ball before a teammate gets it (~500 passes a match)
ON_PITCH = 11               # a switch is split over the players on the pitch, not over our columns
ABSENT_LLR = -hmm.PROX_CLIP  # nats per CORR_FRAMES: a column with no position on a frame


def log_rates():
    """(stay, to one teammate, to one opponent, die, revive into one player, stay dead) log probs."""
    T = np.exp(hmm.transitions())
    die, revive = T[hmm.H_, hmm.D_], 2 * T[hmm.D_, hmm.H_]
    passing, turnover = 1.0 / (HOLDER_S * FPS), 1.0 / (hmm.POSSESSION_S * FPS)
    return tuple(math.log(v) for v in (1 - passing - turnover - die, passing / (ON_PITCH - 1),
                                       turnover / ON_PITCH, die, revive / (2 * ON_PITCH), 1 - revive))


def column_teams(ctx, cfg):
    """(k,) each column's team: C's columns keep one team (identity rules off)."""
    team = ctx._set(cfg)["team"][ctx._key(cfg)]
    col = np.asarray(team[0])
    if not (team == col[None, :]).all():
        raise ValueError("model P needs columns with one team each (C's detected players)")
    return col


def evidence(ctx, cfg, events):
    """(n, k) float32 log evidence per column per frame (already per CORR_FRAMES), events included."""
    players = ctx._set(cfg)
    P = players["players"]
    n, k = P.shape[:2]
    pos = ctx.kin["pos"]
    ball_var = ctx.kin["pos_std"] ** 2 if "pos_std" in ctx.kin else np.full(n, hmm.BALL_STD ** 2)
    base = hmm.CONTROL_STD ** 2 + np.nan_to_num(ball_var, nan=hmm.BALL_STD ** 2)
    player_std = getattr(ctx.tac, "pos_std", None)
    use_std = cfg.get("player_std", True) and player_std is not None and player_std.shape == P.shape[:2]
    reach = np.ones(n)
    if "z" in ctx.kin and cfg.get("height", True):
        from scipy.special import ndtr
        import ball3d
        reach = ndtr((ball3d.REACH - ctx.kin["z"]) / np.maximum(ctx.kin["z_std"], 0.2))
    out = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, 20000):
        e = min(n, s + 20000)
        d = np.linalg.norm(P[s:e] - pos[s:e, None, :].astype(np.float32), axis=2)
        s2 = base[s:e, None] + (np.nan_to_num(player_std[s:e], nan=0.0) ** 2 if use_std else 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ll = -d ** 2 / (2 * s2) - 0.5 * np.log(s2 / base[s:e, None])
        ok = np.isfinite(ll)
        best = np.where(ok, ll, -np.inf).max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            near = (np.where(np.isfinite(d), d, np.inf).min(axis=1) <= rules.FALLBACK_RADIUS)[:, None]
            rel = np.clip(ll - best, -hmm.PROX_CLIP, 0.0)
        # Nobody near (the ball in the air, or far from every column): no evidence between the
        # present players, as in C; an absent column always pays ABSENT_LLR.
        v = np.where(ok, np.where(near, rel, 0.0), ABSENT_LLR)
        out[s:e] = v * (reach[s:e, None] / hmm.CORR_FRAMES)
    for t, who, _, slot, llr in hmm.credited(ctx, cfg, events):
        if 0 <= slot < k:
            out[t, slot] += llr
    return out


def viterbi(e_live, e_dead, home, rates):
    """Best state path (m,) for live log emissions (m, k), dead (m,), home (k,) bool; state k is
    dead. rates from log_rates()."""
    stay, mate, opp, die, revive, dd = rates
    m, k = e_live.shape
    teams = [np.flatnonzero(home), np.flatnonzero(~home)]
    back = np.zeros((m, k + 1), dtype=np.int16)
    d = np.append(e_live[0].astype(float), e_dead[0])
    for t in range(1, m):
        live = d[:k]
        top, second = {}, {}
        prev = np.empty(k + 1, dtype=np.int16)
        for side, idx in enumerate(teams):
            v = live[idx]
            order = np.argsort(v)[::-1][:2]
            top[side] = (idx[order[0]], v[order[0]])
            second[side] = (idx[order[1]], v[order[1]]) if len(order) > 1 else (-1, -np.inf)
        new = np.empty(k + 1)
        for side, idx in enumerate(teams):
            (i1, v1), (i2, v2) = top[side], second[side]
            o1, ov = top[1 - side]
            same = np.where(idx == i1, v2, v1) + mate
            same_i = np.where(idx == i1, i2, i1)
            cand = np.stack([live[idx] + stay, same, np.full(len(idx), ov + opp), np.full(len(idx), d[k] + revive)])
            pick = cand.argmax(axis=0)
            new[idx] = cand[pick, np.arange(len(idx))]
            prev[idx] = np.choose(pick, [idx, same_i, np.full(len(idx), o1), np.full(len(idx), k)])
        j = int(live.argmax())
        if live[j] + die >= d[k] + dd:
            new[k], prev[k] = live[j] + die, j
        else:
            new[k], prev[k] = d[k] + dd, k
        new[:k] += e_live[t]
        new[k] += e_dead[t]
        back[t] = prev
        d = new
    path = np.zeros(m, dtype=np.int64)
    path[-1] = int(np.argmax(d))
    for t in range(m - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def team_posterior(e_live, e_dead, home, rates):
    """(m, 3) P(home holds, away holds, dead) per frame: scaled forward-backward."""
    stay, mate, opp, die, revive, dd = (math.exp(r) for r in rates)
    m, k = e_live.shape
    e = np.column_stack([e_live.astype(float), e_dead])
    e = np.exp(e - np.max(np.where(np.isfinite(e), e, -1e300), axis=1, keepdims=True))
    hm = home.astype(float)
    am = 1.0 - hm

    def step(a):
        """Row vector a (k+1,) times the transition matrix."""
        live = a[:k]
        sh, sa = live @ hm, live @ am
        same = np.where(home, sh, sa) - live
        other = np.where(home, sa, sh)
        out = np.empty(k + 1)
        out[:k] = live * stay + same * mate + other * opp + a[k] * revive
        out[k] = live.sum() * die + a[k] * dd
        return out

    alpha = np.empty((m, k + 1))
    a = e[0] / (k + 1)
    a /= a.sum()
    alpha[0] = a
    for t in range(1, m):
        a = step(alpha[t - 1]) * e[t]
        alpha[t] = a / a.sum()
    # The matrix is symmetric in its structure but not its values (die vs revive): backward with
    # the transpose -- live i <- dead is die, dead <- live j is revive.
    out = np.empty((m, 3))
    b = np.ones(k + 1)
    for t in range(m - 1, -1, -1):
        g = alpha[t] * b
        g /= g.sum()
        out[t] = (g[:k] @ hm, g[:k] @ am, g[k])
        nb = b * e[t]
        live = nb[:k]
        sh, sa = live @ hm, live @ am
        same = np.where(home, sh, sa) - live
        other = np.where(home, sa, sh)
        b = np.empty(k + 1)
        b[:k] = live * stay + same * mate + other * opp + nb[k] * die
        b[k] = live.sum() * revive + nb[k] * dd
        b /= b.sum()
    return out


def predict(ctx, cfg, with_posterior=False):
    """As hmm.predict: labels (n,) of 'H' 'A' 'D' 'X'; detail adds holder (n,) column (-1 dead or
    not graded) and C's team evidence (team_llr) for the viewer's account of each label."""
    n = ctx.tac.n
    events = ctx.events(cfg)
    home = column_teams(ctx, cfg) == HOME
    e_live = evidence(ctx, cfg, events)
    dead_llr, forced = hmm.dead_evidence(ctx, cfg)
    e_live[forced] = -np.inf
    rates = log_rates()
    labels = np.full(n, "X", dtype="<U1")
    holder = np.full(n, -1, dtype=np.int64)
    confidence = np.full(n, np.nan)
    k = e_live.shape[1]
    for _, first, last in halves():
        w = slice(first, last + 1)
        path = viterbi(e_live[w], dead_llr[w], home, rates)
        dead = path == k
        labels[w] = np.where(dead, "D", np.where(home[np.minimum(path, k - 1)], "H", "A"))
        holder[w] = np.where(dead, -1, path)
        if with_posterior:
            post = team_posterior(e_live[w], dead_llr[w], home, rates)
            idx = np.where(dead, hmm.D_, np.where(home[np.minimum(path, k - 1)], hmm.H_, hmm.A_))
            confidence[w] = post[np.arange(len(path)), idx]
    return labels, {"events": events, "team_llr": hmm.team_evidence(ctx, cfg, events), "dead_llr": dead_llr,
                    "confidence": confidence, "special": {}, "holder": holder}


def explain(ctx, cfg, labels, detail):
    """hmm.explain()'s payload (on C's team evidence, which P shares), plus holder: runs [start,
    stop, column, team] of who had the ball."""
    out = hmm.explain(ctx, cfg, labels, detail)
    holder = detail["holder"]
    team = column_teams(ctx, cfg)
    runs = []
    starts = np.flatnonzero(np.r_[True, holder[1:] != holder[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(holder)]):
        if holder[s] >= 0:
            runs.append([int(s), int(e), int(holder[s]), int(team[holder[s]])])
    out["holder"] = runs
    rates = log_rates()
    out["switch_cost"] = round(-2 * rates[2] + 2 * rates[0], 1)    # to an opponent and back
    out["pass_cost"] = round(-rates[1] + rates[0], 1)
    return out


def summary(labels, detail):
    """Counts for the printout: holder changes within a team and across."""
    h = detail["holder"]
    ch = np.flatnonzero((h[1:] != h[:-1]) & (h[1:] >= 0) & (h[:-1] >= 0)) + 1
    same = labels[ch] == labels[ch - 1]
    return "{} holder changes between live states: {} to a teammate, {} to the other team".format(
        len(ch), int(same.sum()), int((~same).sum()))
