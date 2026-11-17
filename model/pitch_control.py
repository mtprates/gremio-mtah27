"""Rule 24: pitch control and pass probability on our own detections -- an inspection, not yet in
the possession model.

    .venv/Scripts/python model/pitch_control.py 11185 3000
    .venv/Scripts/python model/pitch_control.py match     # every chunk, for the viewer (match())

Needs the window's detector outputs: teams-tactical-<first>-<count>.csv (model/teams.py) and what
model/ball_track.py builds from the ball detections. Players are model/detector_model.py's
detected players (variant C: our tracks, our shirt-colour teams, Kalman-smoothed).

  pitch control     who would win the ball at a point (Spearman 2018, "Beyond Expected Goals";
                    the integration follows Laurie Shaw's Friends of Tracking version). A player
                    keeps his velocity for REACTION_S, then runs straight at VMAX; his chance of
                    having arrived by t is a logistic of t minus that time (SIGMA_S). From when the
                    ball gets there, control flows to the players who have arrived, at LAMBDA each.
                    Symmetric: no attacking / defending parameters -- who is attacking is what the
                    possession model has to find out.
  pass probability  the same integration while the ball travels from the passer to the target
                    (Spearman, Basye, Dick, Hotovy & Pop 2017): an opponent who reaches the line
                    before the ball cuts the pass out. P = the passing team's share of control.

Every parameter is from those papers, none fitted here. Checked against the possession labels it
can be compared with (aggregates only):

  at the ball       how often the team in possession controls the ball's spot, and whether the
                    other team's control there says a turnover is coming (within TURNOVER_S)
  passes            touch to touch, PASS_MIN_M or longer, owners from the nearest detected player:
                    predicted completion vs whether the next touch was a teammate's

The references are the public ground truth where the window overlaps it, and C's HMM labels
(evaluation/results/detector_hmm_<first>_<count>.csv) where model/detector_model.py has written
them. Writes data/.model/pitchcontrol/players-tactical-<first>-<count>.csv (frame, team, x, y, vx,
vy) and frames-tactical-<first>-<count>.csv (frame, whether the ball was seen / bridged by the
smoother / not there, ball x, y, control at the ball, team of the last touch) for the viewer, which draws both as surfaces over the pitch itself -- control at
every spot, and the chance a pass from the ball to every spot is completed (app.js mirrors these
parameters).
"""

import csv
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import data  # noqa: E402
from data import AWAY, FPS, HOME  # noqa: E402

# Params
REACTION_S = 0.7            # s a player carries on along his velocity before changing it
VMAX = 5.0                  # m/s: top running speed
SIGMA_S = 0.45              # s: spread of a player's arrival time
LAMBDA = 4.3                # 1/s: rate at which a player who has arrived takes control
DT = 0.04                   # s: integration step
MAX_S = 10.0                # s: integration horizon past the start
CONVERGED = 0.99            # stop once this much control has been handed out everywhere
BALL_SPEED = 15.0           # m/s: a pass's average speed, where the real travel time is unknown
VEL_FRAMES = 2              # frames either side for a player's velocity (positions are smoothed)
VMAX_SEEN = 10.0            # m/s: faster than this is a tracking jump, not a sprint
SAMPLE = 5                  # frames between checks at the ball
TURNOVER_S = 3.0            # s: "a turnover is coming" means the other team has it this soon
PASS_MIN_M = 5.0            # m: shorter ball moves between two touches are dribbles
PASS_MAX_S = 3.0            # s: longer ones are not one pass
OUT_DIR = os.path.join(data.CACHE_DIR, "pitchcontrol")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SCALE = np.pi / np.sqrt(3) / SIGMA_S


def velocities(players):
    """(n, k, 2) m/s from (n, k, 2) positions: central differences VEL_FRAMES either side, one-sided
    at a track's ends, 0 for a player seen once; capped at VMAX_SEEN."""
    n = len(players)
    vel = np.full(players.shape, np.nan, dtype=np.float32)
    w = VEL_FRAMES
    vel[w:n - w] = (players[2 * w:] - players[:n - 2 * w]) * FPS / (2 * w)
    ahead = np.full(players.shape, np.nan, dtype=np.float32)
    ahead[:n - w] = (players[w:] - players[:n - w]) * FPS / w
    behind = np.full(players.shape, np.nan, dtype=np.float32)
    behind[w:] = (players[w:] - players[:n - w]) * FPS / w
    for fill in (ahead, behind, np.zeros_like(vel)):
        vel = np.where(np.isnan(vel), fill, vel)
    vel = np.where(np.isnan(players), np.nan, vel)
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(speed > VMAX_SEEN, vel * VMAX_SEEN / speed, vel)


def integrate(pos, vel, team, origin, target, travel, along=False, exclude=None, attack=None, kappa=1.0):
    """(P home, P away) of winning the ball, (m,) each.

    pos, vel (m, k, 2): the players when the ball sets off (NaN: not there); team (k,) or (m, k)
    HOME / AWAY (other values: nobody's); origin, target (m, 2): where the ball sets off and where
    it goes, reaching it after travel (m,) s. along=False is pitch control at the target: counting
    starts when the ball arrives. along=True is a pass: counting starts at the kick, with the ball
    moving, so a player who reaches the line first wins it there. exclude (m,): a column left out
    (the passer), -1 for none. attack (m,): the side on the ball, whose opponents take control
    kappa times as fast (Spearman 2018's defender advantage, 1.72 there); None: symmetric."""
    m, k = pos.shape[:2]
    reach = pos + vel * REACTION_S
    valid = np.isfinite(reach[..., 0])
    if exclude is not None:
        valid[np.arange(m)[exclude >= 0], exclude[exclude >= 0]] = False
    team = team if team.ndim == 2 else np.broadcast_to(team[None, :], (m, k))
    home = valid & (team == HOME)
    away = valid & (team == AWAY)
    w_home = np.ones(m) if attack is None else np.where(attack == AWAY, kappa, 1.0)
    w_away = np.ones(m) if attack is None else np.where(attack == HOME, kappa, 1.0)
    reach = np.nan_to_num(reach)
    ph, pa = np.zeros(m), np.zeros(m)
    t = np.zeros(m) if along else travel.astype(float).copy()
    span = np.maximum(travel, 1e-9)
    for _ in range(int(MAX_S / DT)):
        left = 1.0 - ph - pa
        if (left < 1.0 - CONVERGED).all():
            break
        ball = origin + (target - origin) * np.clip(t / span, 0.0, 1.0)[:, None]
        tau = REACTION_S + np.linalg.norm(ball[:, None, :] - reach, axis=2) / VMAX
        arrived = 1.0 / (1.0 + np.exp(-SCALE * (t[:, None] - tau)))
        rate_h = LAMBDA * w_home * np.where(home, arrived, 0.0).sum(axis=1)
        rate_a = LAMBDA * w_away * np.where(away, arrived, 0.0).sum(axis=1)
        total = rate_h + rate_a
        # This step's share of what is still loose, solved exactly over DT (no Euler overshoot).
        got = left * -np.expm1(-total * DT)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total > 0, rate_h / total, 0.0)
        ph += got * share
        pa += got * (1.0 - share)
        t += DT
    return ph, pa


def control_at_ball(players, vel, team, ball, frames):
    """P home of winning the ball where it is, at each of frames."""
    b = ball[frames].astype(float)
    ph, pa = integrate(players[frames], vel[frames], team, b, b, np.zeros(len(frames)))
    with np.errstate(invalid="ignore"):
        return ph / (ph + pa)


def auc(score, positive):
    """Area under the ROC curve: the chance a positive outranks a negative (ties count half)."""
    score, positive = np.asarray(score), np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):              # average ranks over ties
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    n_pos = positive.sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (~positive).sum()))


def read_labels(path, n):
    if not os.path.isfile(path):
        return None
    labels = np.array([r["label"].strip().upper() for r in csv.DictReader(open(path, newline=""))])
    return labels if len(labels) == n else None


def turnover_coming(labels, frames):
    """For each frame (labelled H or A): does the other team have the ball within TURNOVER_S,
    before any dead ball? None where the frame has no team."""
    out = []
    horizon = int(TURNOVER_S * FPS)
    for f in frames:
        own = labels[f]
        if own not in ("H", "A"):
            out.append(None)
            continue
        ahead = labels[f + 1:f + 1 + horizon]
        stop = np.flatnonzero(ahead != own)
        out.append(bool(len(stop)) and ahead[stop[0]] in ("H", "A"))
    return out


def report_at_ball(name, labels, frames, p_home):
    """How control at the ball lines up with one reference's labels."""
    rows = [(f, p) for f, p in zip(frames, p_home) if labels[f] in ("H", "A") and np.isfinite(p)]
    if not rows:
        print("  {}: no frames with a team and a ball".format(name))
        return
    f = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    own = np.where(labels[f] == "H", p, 1 - p)          # control of the team in possession
    coming = np.array([bool(c) for c in turnover_coming(labels, f)])
    print("  {} ({} frames, every {}): team in possession controls the ball's spot (> 0.5) on {:.0%}; "
          "mean {:.2f}".format(name, len(f), SAMPLE, (own > 0.5).mean(), own.mean()))
    print("    turnover within {:.0f} s: {:.0%} of frames; AUC of the other team's control {:.2f}".format(
        TURNOVER_S, coming.mean(), auc(1 - own, coming)))
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        sel = (1 - own >= lo) & (1 - own < hi)
        if sel.any():
            print("      other team's control {:.1f}-{:.1f}: {:>4} frames, turnover {:.0%}".format(
                lo, min(hi, 1.0), int(sel.sum()), coming[sel].mean()))


def touches_with_owners(kin, players, team, window):
    """[(frame, column, team)] touches inside the window, owned the way the possession model owns
    them (rules.Context._lookups, rule 1): the nearest detected player at the closest moment within
    rules.TOUCH_SLACK frames of the peak, if within rules.TOUCH_RADIUS. The velocity change peaks a
    few frames after the contact, when the kicker is already moving off (1H 12649: away 1.8 m from
    the ball four frames before the peak, 2.9 m at it). Touches with nobody in reach are dropped."""
    import rules
    out = []
    for t in rules.touch_times(kin):
        if not window.start <= t < window.stop or not kin["present"][t]:
            continue
        best = (np.inf, -1)
        for k in range(max(window.start, t - rules.TOUCH_SLACK), min(window.stop, t + rules.TOUCH_SLACK + 1)):
            if not np.isfinite(kin["pos"][k][0]):
                continue
            d = np.linalg.norm(players[k] - kin["pos"][k][None, :], axis=1)
            if np.isfinite(d).any() and np.nanmin(d) < best[0]:
                best = (float(np.nanmin(d)), int(np.nanargmin(d)))
        if best[0] <= rules.TOUCH_RADIUS:
            out.append((int(t), best[1], int(team[best[1]])))
    return out


def report_passes(touches, kin, players, vel, team, refs):
    """Pass probability at each kick vs whether the next touch was a teammate's."""
    kicks = []
    for (t1, c1, s1), (t2, c2, s2) in zip(touches[:-1], touches[1:]):
        dist = np.linalg.norm(kin["pos"][t2] - kin["pos"][t1])
        if (t2 - t1) / FPS > PASS_MAX_S or dist < PASS_MIN_M:
            continue
        # A dead ball in between (any reference) is a restart, not a pass.
        if any(labels is not None and (labels[t1:t2 + 1] == "D").any() for labels in refs.values()):
            continue
        kicks.append((t1, t2, c1, s1, s2))
    if not kicks:
        print("  passes: none found")
        return
    t1 = np.array([k[0] for k in kicks])
    t2 = np.array([k[1] for k in kicks])
    origin, target = kin["pos"][t1].astype(float), kin["pos"][t2].astype(float)
    ph, pa = integrate(players[t1], vel[t1], team, origin, target, (t2 - t1) / FPS, along=True,
                       exclude=np.array([k[2] for k in kicks]))
    side = np.array([k[3] for k in kicks])
    p = np.where(side == HOME, ph, pa)
    done = np.array([k[3] == k[4] for k in kicks])
    brier = np.mean((p - done) ** 2)
    base = np.mean((done.mean() - done) ** 2)
    print("  passes (touch to touch, >= {:.0f} m, <= {:.0f} s, owners from detected players): {}, "
          "{:.0%} completed".format(PASS_MIN_M, PASS_MAX_S, len(kicks), done.mean()))
    print("    predicted completion: mean {:.2f}; Brier {:.3f} (a constant at the completion rate: {:.3f}); "
          "AUC {:.2f}".format(p.mean(), brier, base, auc(p, done)))
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            print("      predicted {:.2f}-{:.2f}: {:>3} passes, {:.0%} completed".format(
                lo, min(hi, 1.0), int(sel.sum()), done[sel].mean()))


def write_viewer(first, count, players, vel, team, kin, p_home, touches):
    os.makedirs(OUT_DIR, exist_ok=True)
    w = range(first, first + count)
    with open(os.path.join(OUT_DIR, "players-tactical-{}-{}.csv".format(first, count)), "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["frame", "team", "x", "y", "vx", "vy"])
        for f in w:
            for c in np.flatnonzero(np.isfinite(players[f, :, 0])):
                out.writerow([f, int(team[c])] + [round(float(v), 2) for v in (*players[f, c], *vel[f, c])])
    owner = np.full(len(players), -1)
    for t, _, s in touches:
        owner[t:] = s
    with open(os.path.join(OUT_DIR, "frames-tactical-{}-{}.csv".format(first, count)), "w", newline="") as handle:
        out = csv.writer(handle)
        # ball: 2 detected, 1 bridged by the smoother (an estimate), 0 none (bx, by, pc_home empty).
        out.writerow(["frame", "ball", "bx", "by", "pc_home", "owner"])
        for f in w:
            if not np.isfinite(players[f, :, 0]).any():
                continue
            p = p_home[f - first]
            if np.isfinite(kin["pos"][f][0]) and np.isfinite(p):
                out.writerow([f, 2 if kin["present"][f] else 1, round(float(kin["pos"][f][0]), 2),
                              round(float(kin["pos"][f][1]), 2), round(float(p), 3), int(owner[f])])
            else:
                out.writerow([f, 0, "", "", "", int(owner[f])])


def match():
    """The viewer's export over the whole match: the match ball track (ball_track.build_match) and
    our detected players stitched over every chunk (detector_model.match_players), then each
    tactical chunk's pitch control at the ball, touches and passes written as in main(). For
    visualization only: the model does not read it (rule 25 is off)."""
    import ball_track
    import detect_all
    import detector_model
    chunks = list(detect_all.chunks())
    tac_chunks = [(s, c) for src, s, c in chunks if src == "tactical"]
    with redirect_stdout(io.StringIO()):
        kin, _, _ = ball_track.build_match(tac_chunks, [(s, c) for src, s, c in chunks if src == "broadcast"])
        det, _ = detector_model.match_players(tac_chunks, len(kin["pos"]))
    players, team = det.players, det.team.astype(int)
    vel = velocities(players)
    for first, count in tac_chunks:
        every = np.arange(first, first + count)
        p_home = np.full(count, np.nan)
        ok = np.isfinite(kin["pos"][every, 0])
        for s in range(0, count, 2000):
            sel = every[s:s + 2000][ok[s:s + 2000]]
            if len(sel):
                p_home[sel - first] = control_at_ball(players, vel, team, kin["pos"], sel)
        touches = touches_with_owners(kin, players, team, range(first, first + count))
        write_viewer(first, count, players, vel, team, kin, p_home, touches)
        print("  tactical {}-{}: control at the ball on {} frames, {} touches with an owner".format(
            first, first + count - 1, int(np.isfinite(p_home).sum()), len(touches)), flush=True)
    print("  wrote {}".format(os.path.relpath(OUT_DIR, ROOT)))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "match":
        return match()
    import ball_track
    import detector_model
    first, count = int(sys.argv[1]), int(sys.argv[2])
    window = range(first, first + count)
    with redirect_stdout(io.StringIO()):
        kin = ball_track.build(first, count)["kin"]
        det, _ = detector_model.detected_players(first, count, len(kin["pos"]))
    players, team = det.players, det.team.astype(int)
    vel = velocities(players)
    n = len(players)
    print("window {}-{}: {} detected player columns; ball present on {:.0%} of frames".format(
        first, first + count - 1, players.shape[1], kin["present"][window.start:window.stop].mean()))

    # Pitch control at the ball, every frame of the window with a ball position -- seen, or bridged
    # by the smoother across a gap (the viewer reads them all).
    every = np.arange(first, first + count)
    p_home = np.full(count, np.nan)
    ok = np.isfinite(kin["pos"][every, 0])
    for s in range(0, count, 2000):
        sel = every[s:s + 2000][ok[s:s + 2000]]
        if len(sel):
            p_home[sel - first] = control_at_ball(players, vel, team, kin["pos"], sel)
    seen = kin["present"][every]
    print("  control at the ball computed on {} frames ({} with the ball seen, {} bridged)".format(
        int(np.isfinite(p_home).sum()), int((np.isfinite(p_home) & seen).sum()),
        int((np.isfinite(p_home) & ~seen).sum())))

    refs = {"public GT": read_labels(PUBLIC_GT, n),
            "C's HMM": read_labels(os.path.join(ROOT, "evaluation", "results",
                                                "detector_hmm_{}_{}.csv".format(first, count)), n)}
    sample = every[::SAMPLE]
    # The checks use frames with the ball seen only: a bridged ball is the smoother's guess.
    p_seen = np.where(seen, p_home, np.nan)
    for name, labels in refs.items():
        if labels is None:
            print("  {}: not available for this window".format(name))
            continue
        report_at_ball(name, labels, sample, p_seen[sample - first])

    touches = touches_with_owners(kin, players, team, window)
    import rules
    print("  {} touches with an owner (rule 1: within {:.0f} m, +/- {} frames)".format(
        len(touches), rules.TOUCH_RADIUS, rules.TOUCH_SLACK))
    report_passes(touches, kin, players, vel, team, refs)
    write_viewer(first, count, players, vel, team, kin, p_home, touches)
    print("  wrote {}".format(os.path.relpath(OUT_DIR, ROOT)))


if __name__ == "__main__":
    main()
