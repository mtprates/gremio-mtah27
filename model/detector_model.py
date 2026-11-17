"""Rule 20: the possession models on our own detections, scored where they exist.

    .venv/Scripts/python model/detector_model.py 11185 3000
    .venv/Scripts/python model/detector_model.py match     # the whole match (match())

The detector covers a window of the match (model/detect_ball.py), so the models are scored on
that window's frames of the public ground truth only -- and the ATD-based models on exactly the
same frames, for comparison. Variants:

  B  ball: the detector's track (rule 18, with heights -- rule 17); players: the ATD
  C  ball: the detector's track; players: our detections with their shirt-colour teams
     (model/teams.py; keepers with their team, officials and unknown left out)

For C the detected tracks become the model's player columns: each track is cut where its label
changes, and pieces of one team that do not overlap in time share a column (keeping the arrays
small). The ATD's identity rules (3.a, 3.b and 3.c across cameras, 6 dropouts, 9 static, 10
broadcast nearest, 16 cuts) are off for C: our tracks already carry their team and keeper. Prints
aggregates only.
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
import kalman  # noqa: E402

# Params
COLUMN_GAP = 5              # frames between two people sharing a player column
RELINK_MAX = data.FPS       # frames: a track that ends is continued by one of its team
VIEW_MARGIN = 2.0           # m: ... and only where both ends are this far inside the camera's view.
                            # An occlusion, not an exit: an occlusion lasts under ~1 s; 2 m is a body
                            # width, the feet's projection error and a few frames at sprint pace from
                            # the edge (docs/VALUES.md, re-derivation off BAR-ATH).
RELINK_M, RELINK_SPEED = 2.0, 8.0       # starting within this long, within 2 m + 8 m/s of it
# Rule 37 (docs/VALUES.md): a longer gap, at a jog. RELINK_LONG = RELINK_MAX is off; iterate.py --relink: 75.
RELINK_LONG, LONG_SPEED = RELINK_MAX, 3.0
# Rule 21: player tracks through kalman.smooth_track, not straight lines across their gaps.
FEET_STD, FEET_REF_PX = 0.15, 60.0      # m: a player's feet on the pitch for a box this tall ...
FEET_STD_MIN, FEET_STD_MAX = 0.08, 0.8  # ... scaled by 60 px / box height: distant players are less
                                        # precise
TEAM_OF = {0: 0, 1: 1, 2: 0, 3: 1}   # teams.py labels: home, away, home keeper, away keeper
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
ATD_MODELS = ("rules_v1", "rules_hmm", "rules_hmm_2cam", "rules_hmm_3d")
# Rule 26 for C: a touch the ball was not seen on both sides of, or on a high ball, counts as a far
# touch (touch_seen / touch_once are the stricter options, off). Rule 27: out of bounds graded by its
# run, dead from the crossing (ref_delay 0). docs/VALUES.md.
C_TOUCHES = dict(touch_sides=True, air_touch=True, oob_grade=True, ref_delay=0)


def detected_players(first, count, n, fill="kalman"):
    """A tactical Recording whose players are our detected, team-labelled tracks."""
    import teams
    path = os.path.join(data.CACHE_DIR, "detections", "teams-tactical-{}-{}.csv".format(first, count))
    # Boxes only (kind 0, or 1 a merged box): predicted rows (kind 2, 3) are no measurements.
    rows = [r for r in csv.DictReader(open(path, newline="")) if int(r.get("kind") or 0) <= 1]
    frames = np.array([int(r["frame"]) for r in rows])
    boxes = np.array([[float(r[k]) for k in ("x", "y", "w", "h")] for r in rows])
    label = np.array([int(r["label"]) for r in rows])
    track = np.array([int(r["track"]) for r in rows])
    H = data.homography_matrices("tactical")
    feet = teams.to_pitch(H, frames, np.column_stack([boxes[:, 0], boxes[:, 1] + boxes[:, 3] / 2]))
    team = np.array([TEAM_OF.get(int(l), -1) for l in label])
    keep = (team >= 0) & np.isfinite(feet[:, 0])
    frames, feet, team, track = frames[keep], feet[keep], team[keep], track[keep]
    box_h = boxes[keep, 3]

    # Pieces: a track cut where its team changes.
    pieces = []          # (team, rows sorted by frame)
    for tid in np.unique(track):
        rows = np.flatnonzero(track == tid)
        rows = rows[np.argsort(frames[rows])]
        cut = np.flatnonzero(np.diff(team[rows])) + 1
        for part in np.split(rows, cut):
            pieces.append((int(team[part[0]]), part))
    pieces.sort(key=lambda p: frames[p[1][0]])

    # Players do not vanish: a piece that ends is followed by the nearest piece of its team that
    # starts within RELINK_MAX frames, within reach (1H 13651: a home player behind an away one
    # lost his box -- the detector keeps one of two people lined up -- and came back as a new track).
    views = {}

    def in_view(f, p):
        if f not in views:
            views[f] = teams.view(H, f)
        poly = views[f]
        if poly is None or not teams.inside(poly, p[None])[0]:
            return False
        # Distance to the polygon's edges.
        a, b = poly, np.roll(poly, -1, axis=0)
        ab = b - a
        u = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
        return np.linalg.norm(a + ab * u[:, None] - p, axis=1).min() >= VIEW_MARGIN

    nxt, taken = {}, set()
    links = []
    for i, (t, part) in enumerate(pieces):
        end_f, end_p = frames[part[-1]], feet[part[-1]]
        if not in_view(end_f, end_p):
            continue
        for j, (t2, part2) in enumerate(pieces):
            gap = frames[part2[0]] - end_f
            if t2 != t or gap <= 0 or gap > RELINK_LONG or not in_view(frames[part2[0]], feet[part2[0]]):
                continue
            d = np.linalg.norm(feet[part2[0]] - end_p)
            reach = RELINK_M + (RELINK_SPEED if gap <= RELINK_MAX else LONG_SPEED) * gap / data.FPS
            if d <= reach:
                links.append((d / reach, i, j))
    for _, i, j in sorted(links):
        if i not in nxt and j not in taken:
            nxt[i] = j
            taken.add(j)
    persons = []         # (team, [piece indices]) chained
    for i in range(len(pieces)):
        if i in taken:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        persons.append((pieces[i][0], chain))

    # Columns: persons of one team that do not overlap (COLUMN_GAP apart) share one.
    columns = []         # (team, last frame used, [rows of each person])
    for t, chain in sorted(persons, key=lambda p: frames[pieces[p[1][0]][1][0]]):
        rows = np.concatenate([pieces[k][1] for k in chain])
        start, stop = frames[rows[0]], frames[rows[-1]]
        for col in columns:
            if col[0] == t and col[1] + COLUMN_GAP < start:
                col[1] = stop
                col[2].append(rows)
                break
        else:
            columns.append([t, stop, [rows]])
    players = np.full((n, len(columns), 2), np.nan, dtype=np.float32)
    speed = np.full((n, len(columns)), np.nan, dtype=np.float32)
    std = np.full((n, len(columns)), np.nan, dtype=np.float32)
    filled = 0
    speed_rows = []      # [(frame, track, speed m/s)] per person, Kalman fill only
    for c, (_, _, people) in enumerate(columns):
        for rows in people:
            f, p = frames[rows], feet[rows]
            if fill == "kalman":
                # Rule 21: the person through the Kalman smoother; gaps up to RELINK_MAX bridged
                # along his last and next velocities, with each position's uncertainty.
                span = np.arange(f[0], f[-1] + 1)
                z = np.full((len(span), 2), np.nan)
                z[f - f[0]] = p
                r2 = np.full(len(span), np.nan)
                r2[f - f[0]] = np.clip(FEET_STD * FEET_REF_PX / box_h[rows], FEET_STD_MIN, FEET_STD_MAX) ** 2
                pos, vel, sd, br = kalman.smooth_track(z, r2, kalman.PLAYER_ACCEL, RELINK_LONG)
                ok = np.isfinite(pos[:, 0])
                players[span[ok], c] = pos[ok]
                speed[span[ok], c] = np.linalg.norm(vel[ok], axis=1)
                std[span[ok], c] = sd[ok]
                filled += int(br.sum())
                # For the viewer: each frame's speed under the track last seen (a bridged frame
                # under the track that ended -- the one teams.py draws a predicted box for).
                seen = np.full(len(span), -1)
                seen[f - f[0]] = np.arange(len(rows))
                last = np.maximum.accumulate(seen)
                speed_rows.append(np.column_stack([span[ok], track[rows][last[ok]],
                                                   np.linalg.norm(vel[ok], axis=1)]))
                continue
            players[f, c] = p
            std[f, c] = 0.0
            # Every gap inside a person up to RELINK_MAX, on the straight line between where he was
            # last seen and where he is seen again.
            for a, b in zip(f[:-1], f[1:]):
                if 1 < b - a <= RELINK_LONG + 1:
                    miss = np.arange(a + 1, b)
                    for k in (0, 1):
                        players[miss, c, k] = np.interp(miss, f, p[:, k])
                    std[miss, c] = 0.0
                    filled += len(miss)
    tac = data.load("tactical")
    rec = data.Recording("tactical", players, speed, tac.ball,
                         np.array([c[0] for c in columns], dtype=np.int8), tac.valid_h)
    rec.pos_std = std            # m per player per frame: the HMM trusts uncertain positions less
    rec.speed_rows = np.vstack(speed_rows) if speed_rows else np.zeros((0, 3))
    return rec, {"pieces": len(pieces), "persons": len(persons), "links": len(nxt), "filled": filled,
                 "present": int(np.isfinite(players[..., 0]).sum())}


def write_speeds(speed_rows, first, count):
    """For the viewer: the speeds the stillness signals read (rule 12: still below rules.QUIET_SPEED)."""
    path = os.path.join(data.CACHE_DIR, "detections", "playerspeed-tactical-{}-{}.csv".format(first, count))
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["frame", "track", "speed"])
        # In frame order: the viewer reads a window of it by byte offset.
        for f, t, v in sorted(speed_rows, key=lambda row: (row[0], row[1])):
            out.writerow([int(f), int(t), round(float(v), 2)])


def match_players(chunks, n):
    """detected_players() over every tactical chunk, on one timeline. The chunks do not overlap in
    time, so chunk k's columns of a team take the same column numbers as chunk j's; a person is not
    carried across a chunk's edge (a new column there, as for a player who leaves the view)."""
    parts = []
    for first, count in chunks:
        rec, stats = detected_players(first, count, n)
        w = slice(first, first + count)
        parts.append((w, rec.players[w], rec.speed[w], rec.pos_std[w], rec.team, stats))
        write_speeds(rec.speed_rows, first, count)
        del rec
    width = {t: max(int((p[4] == t).sum()) for p in parts) for t in (data.HOME, data.AWAY)}
    team = np.repeat([data.HOME, data.AWAY], [width[data.HOME], width[data.AWAY]]).astype(np.int8)
    players = np.full((n, len(team), 2), np.nan, dtype=np.float32)
    speed = np.full((n, len(team)), np.nan, dtype=np.float32)
    std = np.full((n, len(team)), np.nan, dtype=np.float32)
    for w, p, v, s, t, _ in parts:
        for k in (data.HOME, data.AWAY):
            src = np.flatnonzero(t == k)
            dst = np.flatnonzero(team == k)[:len(src)]
            players[w, dst], speed[w, dst], std[w, dst] = p[:, src], v[:, src], s[:, src]
    tac = data.load("tactical")
    rec = data.Recording("tactical", players, speed, tac.ball, team, tac.valid_h)
    rec.pos_std = std
    totals = {k: sum(p[5][k] for p in parts) for k in ("pieces", "persons", "links", "filled", "present")}
    return rec, totals


# The possession models on the same ball and player tracks, for the viewer's model picker:
# key (data/.model/<key>_debug.json), submission name (evaluation/results/), title, module.
MODELS = (("detector", "detector_hmm", "C: team HMM (current)", "hmm"),
          ("detector_touchpc", "detector_touchpc", "T: touches + pitch control", "touch_model"),
          ("detector_player", "detector_player_hmm", "P: player HMM", "player_hmm"))


def write_debug(ctx, cfg, key="detector"):
    """For the viewer: what one model sees over the match, in data/.model/<key>_debug.json -- its
    labels as runs, the events it credits ([frame, team, kind code, column]), each dead-ball
    signal's stretches, where its label is less than hmm.LOW_CONFIDENCE sure (forward-backward),
    and the model's explain() account of each label. Returns the labels (Viterbi)."""
    import importlib
    import json
    import hmm
    import rules
    import run
    _, _, title, module = next(m for m in MODELS if m[0] == key)
    model = importlib.import_module(module)
    labels, detail = model.predict(ctx, cfg, with_posterior=True)
    masks, _, _ = hmm._masks(ctx, cfg)
    with np.errstate(invalid="ignore"):
        unsure = detail["confidence"] < hmm.LOW_CONFIDENCE
    payload = {
        "name": key,
        "title": title,
        "labels": [[int(s), int(e), k] for k in "HAD" for s, e in rules.runs(labels == k)],
        "events": [[int(t), int(who), run.KIND_CODES[kind], int(slot)] for t, who, kind, slot in detail["events"]],
        "evidence": {name: [[int(s), int(e)] for s, e in rules.runs(mask)] for name, mask in masks.items()},
        "unsure": [[int(s), int(e)] for s, e in rules.runs(unsure)],
        "low_confidence": hmm.LOW_CONFIDENCE,
    }
    payload["labels"].sort()
    payload.update(model.explain(ctx, cfg, labels, detail))   # why each label: the viewer's "why" rows
    path = os.path.join(data.CACHE_DIR, key + "_debug.json")
    with open(path, "w") as handle:
        json.dump(payload, handle)
    extra = ""
    if module == "player_hmm":
        extra = model.summary(labels, detail)
    elif module == "touch_model":
        extra = model.summary(detail)
    print("  viewer {}: {} events, {} label stretches under {:.0%} sure -> {}{}".format(
        key, len(payload["events"]), len(payload["unsure"]), hmm.LOW_CONFIDENCE, os.path.relpath(path, ROOT),
        "\n    " + extra if extra else ""))
    return labels


def match():
    """The detector models over the whole match: every chunk model/detect_all.py ran, after
    model/teams.py on each. Writes evaluation/results/detector_hmm.csv (C),
    detector_touchpc.csv (T), detector_player_hmm.csv (P) and detector_ball_hmm.csv (B), full-match
    submissions, plus the viewer's ball track and player
    speeds per chunk. Prints aggregates only; public-window scores only with --score."""
    import possession_metrics as pm
    import ball_track
    import detect_all
    import hmm
    import rules
    import run
    chunks = list(detect_all.chunks())
    tac_chunks = [(s, c) for src, s, c in chunks if src == "tactical"]
    bro_chunks = [(s, c) for src, s, c in chunks if src == "broadcast"]
    print("whole match: {} tactical and {} broadcast chunks".format(len(tac_chunks), len(bro_chunks)))
    with redirect_stdout(io.StringIO()):
        kin, full_t, full_b = ball_track.build_match(tac_chunks, bro_chunks)
        tac, bro = data.load("tactical"), data.load("broadcast")
        al = run.load_alignment(tac, bro)
    for first, count in tac_chunks:
        ball_track.write(ball_track.window(kin, full_t, full_b, first, count), first, count)
    graded = np.zeros(tac.n, dtype=bool)
    for _, first, last in data.halves():
        graded[first:last + 1] = True
    t_on, b_on = np.isfinite(full_t[graded, 0]), np.isfinite(full_b[graded, 0])
    print("  ball picked on {:.1%} of the halves' frames (tactical {:.1%}, broadcast {:.1%}); after "
          "smoothing known on {:.1%}; {} flights".format(
              (t_on | b_on).mean(), t_on.mean(), b_on.mean(), np.isfinite(kin["pos"][graded, 0]).mean(),
              len(kin["flights"])))
    print("  prefilter dropped: " + ", ".join("{} {} {}".format(n, src, name) for (src, name), n in sorted(ball_track.PREFILTER_DROPPED.items())))

    det, stats = match_players(tac_chunks, tac.n)
    print("  C: {pieces} track pieces, re-linked {links} times into {persons} players; {filled} player-frames "
          "bridged; {} player columns ({} home, {} away)".format(
              det.players.shape[1], int((det.team == data.HOME).sum()), int((det.team == data.AWAY).sum()), **stats))
    with redirect_stdout(io.StringIO()):
        ctx_c = rules.Context(det, bro, al, kin=kin, identity=False)
        ctx_b = rules.Context(tac, bro, al, kin=kin)
        ctx_b.add_rule6(data.homography_matrices("tactical"))
    # As main(): Viterbi for C and B. Posterior decoding was better on the two windows (rule 23) but
    # fragments the whole match: 426 possessions of 9.0 s against 290 with Viterbi, and 2 s flips
    # like 1H 38306-38354 (BRAINSTORM.md rule 20, whole match).
    cfg_c = dict(rules.V1, goalie=False, crosscheck=False, resolve=False, dropouts=False, static=False,
                 crosscam=False, **C_TOUCHES)
    # The three models on C's tracks (MODELS), each with its viewer payload; B for comparison.
    rows = [(name, desc, write_debug(ctx_c, cfg_c, key)) for key, name, desc, _ in MODELS]
    rows.append(("detector_ball_hmm", "B: detector ball + ATD players, HMM", hmm.predict(ctx_b, rules.V1)[0]))
    for name, _, labels in rows:
        run.write_submission(os.path.join(ROOT, "evaluation", "results", name + ".csv"), labels)
    rows.append((None, "C: ... HMM without the height gate", hmm.predict(ctx_c, dict(cfg_c, height=False))[0]))
    rows.append((None, "C: ... HMM, posterior decoding", hmm.predict(ctx_c, dict(cfg_c, decode="posterior"))[0]))
    for name in ATD_MODELS:
        path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
        if os.path.isfile(path):
            rows.append((None, "ATD: " + name, np.array(
                [r["label"].strip().upper() for r in csv.DictReader(open(path, newline=""))])))

    gt = np.array([r["label"].strip().upper() for r in csv.DictReader(open(PUBLIC_GT, newline=""))])
    public = slice(0, len(gt))
    gt_codes = pm.label_codes(gt[public])
    res = {}
    score = "--score" in sys.argv       # public-window scores only on request
    print("\n  {:<44} {:>8} {:>8} {:>6}   {:>10} {:>7} {:>6}".format(
        "", "macroF1" if score else "", "spellF1" if score else "", "poss" if score else "",
        "match poss", "mean s", "dead"))
    for _, desc, labels in rows:
        lab = labels[graded]
        spells = [(e - s) / data.FPS for k in "HA" for s, e in rules.runs(lab == k)]
        if score:
            res = pm.evaluate(gt_codes, pm.label_codes(labels[public]))
            pub = "{:>8.4f} {:>8.4f} {:>6}".format(res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"])
        else:
            pub = "{:>8} {:>8} {:>6}".format("", "", "")
        print("  {:<44} {}   {:>10} {:>7.1f} {:>6.1%}".format(
            desc, pub, len(spells), float(np.mean(spells)) if spells else 0.0, (lab == "D").mean()))
    if score:
        print("  (public window: GT {} possessions; the match columns are over both halves, where there is "
              "no ground truth)".format(res["n_gt_spells"]))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "match":
        return match()
    import possession_metrics as pm
    import ball_track
    import hmm
    import rules
    import run
    first, count = int(sys.argv[1]), int(sys.argv[2])
    w = slice(first, first + count)
    with redirect_stdout(io.StringIO()):
        tr = ball_track.build(first, count)
        ball_track.write(tr, first, count)      # the viewer shows the track that was scored
        tac, bro = data.load("tactical"), data.load("broadcast")
        al = run.load_alignment(tac, bro)
    kin = tr["kin"]
    gt = np.array([r["label"].strip().upper() for r in csv.DictReader(open(PUBLIC_GT, newline=""))])
    graded = np.zeros(len(gt), dtype=bool)
    graded[w] = gt[w] != "X"
    print("\nwindow {}-{}: {} graded frames, {} GT spells".format(
        first, first + count - 1, int(graded.sum()), 1 + int(np.sum(gt[w][1:] != gt[w][:-1]))))

    rows = []
    for name in ATD_MODELS:
        path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
        if os.path.isfile(path):
            labels = np.array([r["label"].strip().upper() for r in csv.DictReader(open(path, newline=""))])
            rows.append(("ATD: " + name, labels))

    with redirect_stdout(io.StringIO()):
        ctx_b = rules.Context(tac, bro, al, kin=kin)
        ctx_b.add_rule6(data.homography_matrices("tactical"))
    rows.append(("B: detector ball + ATD players, v1", ctx_b.predict(rules.V1)[0]))
    rows.append(("B: detector ball + ATD players, HMM", hmm.predict(ctx_b, rules.V1)[0]))

    # Viterbi, as for B. Posterior decoding (each frame's most probable label) was better on both
    # windows for C (BRAINSTORM.md rule 23) but fragments the whole match (rule 20, whole match).
    cfg_c = dict(rules.V1, goalie=False, crosscheck=False, resolve=False, dropouts=False, static=False,
                 crosscam=False, **C_TOUCHES)
    # Rule 21 ablation: (a) straight lines across player gaps, (b) Kalman without the HMM's
    # uncertainty weighting, (c) the full version (the one written for the viewer).
    variants = {}
    for fill in ("line", "kalman"):
        det, stats = detected_players(first, count, tac.n, fill=fill)
        with redirect_stdout(io.StringIO()):
            ctx_c = rules.Context(det, bro, al, kin=kin, identity=False)
        print("  C ({}): {pieces} track pieces, re-linked {links} times into {persons} players; {filled} "
              "player-frames bridged ({:.1%} of the {present} present); {} player columns".format(
                  fill, stats["filled"] / max(stats["present"], 1), det.players.shape[1], **stats))
        variants[fill] = ctx_c
        if fill == "kalman":
            speed_rows = det.speed_rows
    labels_c = {
        "(a) straight line": hmm.predict(variants["line"], dict(cfg_c, player_std=False))[0],
        "(b) Kalman, no uncertainty weighting": hmm.predict(variants["kalman"], dict(cfg_c, player_std=False))[0],
        "(c) Kalman, full": hmm.predict(variants["kalman"], cfg_c)[0],
    }
    ctx_c = variants["kalman"]
    write_speeds(speed_rows, first, count)
    rows.append(("C: detector ball + detected players, v1", ctx_c.predict(cfg_c)[0]))
    for name, lab in labels_c.items():
        rows.append(("C HMM " + name, lab))
    det_hmm = labels_c["(c) Kalman, full"]
    # For the viewer: C's HMM labels on the detector's window, X elsewhere (it has no data there).
    # Not a valid submission -- the scorer will skip it.
    out = np.full(len(det_hmm), "X", dtype="<U1")
    out[w] = det_hmm[w]
    run.write_submission(os.path.join(ROOT, "evaluation", "results", "detector_hmm_{}_{}.csv".format(first, count)), out)
    rows.append(("C: ... HMM without the height gate", hmm.predict(ctx_c, dict(cfg_c, height=False))[0]))
    rows.append(("C: ... HMM, posterior decoding", hmm.predict(ctx_c, dict(cfg_c, decode="posterior"))[0]))
    rows.append(("C: ... HMM with passes (rule 23)", hmm.predict(ctx_c, dict(cfg_c, passes=True))[0]))
    # Rule 25 (off by default): pass probability / pitch control scaling the switch price, relative
    # to the match's typical pass; the count is of frames with an adjusted price.
    for name, extra in (("+ pass probability", dict(pass_prob=True)),
                        ("+ pitch control at landings", dict(flight_pc=True)),
                        ("+ both", dict(pass_prob=True, flight_pc=True)),
                        ("+ both, kappa 1.72", dict(pass_prob=True, flight_pc=True, kappa=1.72)),
                        ("+ both, Viterbi", dict(pass_prob=True, flight_pc=True, decode="viterbi"))):
        lab, detail = hmm.predict(ctx_c, dict(cfg_c, **extra))
        rows.append(("C: ... HMM {} (25: {} frames)".format(name, len(detail["special"])), lab))
    rows.append(("B: ... HMM + both (rule 25)", hmm.predict(ctx_b, dict(rules.V1, pass_prob=True, flight_pc=True))[0]))
    rows.append(("B: ... HMM, posterior decoding", hmm.predict(ctx_b, dict(rules.V1, decode="posterior"))[0]))

    gt_codes = pm.label_codes(gt[w])
    print("  {:<46} {:>8} {:>8} {:>6}".format("", "macroF1", "spellF1", "poss"))
    for name, labels in rows:
        res = pm.evaluate(gt_codes, pm.label_codes(labels[w]))
        print("  {:<46} {:>8.4f} {:>8.4f} {:>6}".format(name, res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"]))
    print("  (GT: {} possessions in the window; one possession is a large share of spellF1 here)".format(
        len([1 for k in range(count) if gt[first + k] in "HA" and (k == 0 or gt[first + k - 1] != gt[first + k])])))

    # Where the rule 21 variants differ from (a), and what the ground truth says there.
    import rules as rl
    base = labels_c["(a) straight line"][w]
    for name in ("(b) Kalman, no uncertainty weighting", "(c) Kalman, full"):
        lab = labels_c[name][w]
        print("\n  {} vs (a):".format(name))
        diff = lab != base
        if not diff.any():
            print("    no label changes")
        for s, e in rl.runs(diff):
            g = gt[w][s:e]
            print("    {}-{} ({}, {:.1f} s): (a) {} -> {}; GT {}".format(
                first + s, first + e - 1, run.match_clock(first + s), (e - s) / data.FPS,
                "".join(sorted(set(base[s:e]))), "".join(sorted(set(lab[s:e]))),
                " ".join("{} {:.1f}s".format(k, (g == k).sum() / data.FPS) for k in "HAD" if (g == k).any())))


if __name__ == "__main__":
    main()
