"""Fast iteration on C (detector ball + detected players, HMM) over the whole match.

    .venv/Scripts/python model/iterate.py [--repick] [--persons] [--refs] [--feet] [--smooth] [variant ...]
        --feet     rule 32, weak candidates at a player's feet far from the last confident ball
        --out      rule 33, follow the ball out of bounds    --flights  rule 34, drag and joined arcs in the 3D ball
        --arc-tol  rule 35, arcs may miss by 1.0 + 0.45 T m    --keeper  rule 36, a keeper holds the ball within 3 m
        --traces   rule 42, each camera's own trace fills its short gaps before fusion
        --image    rule 44, picks linked and the filter gated by how they move in each camera's image
        --restart  rule 45, no ball appears from nowhere (reach from the last confident ball, feet, exits, isolated picks)
        --heights  rule 46, arcs grown per end and a vertical pace limit on the height (no jumps from triangulation)
        --solo     rule 47, flights one camera alone shows (an arc through its lines of sight)
        --gaps     rule 48, no bridge no ball could travel, keeper holds released at ball pace
        --side     rule 41, the broadcast is dropped where it contradicts the tactical camera for long
        --tri      rule 40, a ball held up keeps its triangulated height
        --relink   rule 37, a player is relinked across up to 3 s   --cover  rule 38, an unseen ball is with the player who reaches it
        --persons  rule 30, lonely from our detected people     --refs    rule 29, the other camera's picks
        --smooth   rule 31, the smoother with confidences       (picks are cached per option set)
        --kick     rule 55, a kick alone hands possession over   --restart-live  rule 56, restarts and kicks say live
        --whistle-back  rule 57, a broadcast whistle just before a stoppage moves its start back
        --touch-echo  rule 58, a gap's kink gives way to a stronger touch beside it
        --event-switch  rule 59, possession changes only at an event   --far-clear  rule 60, a far touch needs a clear nearest team
        --restart-confirm  rule 61 (off)   --replay-neutral  rule 62, broadcast replays are no evidence about the live ball
        --short-spells  rule 63, short spells between two of the opposite state   --short-broadcast  ... its false-dead half only on broadcast-only stoppages
    .venv/Scripts/python model/iterate.py [--persons] [--refs] [--smooth] models
        # C, T and P and the ball track for the viewer (models()); no flags = the committed pipeline

detector_model.match() takes ~15 min, most of it in the per-chunk candidate picks and the detected
players, which do not change while the smoother, the HMM or the dead-ball rules are worked on. Both
are cached under data/.model/iterate/ (--repick rebuilds the picks after a change to
ball_track.candidates / associate). Each variant then rebuilds the kinematics and the labels, and
prints aggregates only: public-window scores, match spell counts, ball artefact counts, dead-ball
signal counts, and per watched frame (Matias's cases) only what kind of spell the model put there.
"""

import io
import os
import pickle
import sys
import time
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import data  # noqa: E402
from data import FPS  # noqa: E402

CACHE = os.path.join(data.CACHE_DIR, "iterate")
WATCH = (154865, 148562, 136899)    # Matias, 2026-10-24: very short possessions
HMM_CFG = {}                        # HMM cfg switches from the command line (rules 55-57), on every variant
OOB_MATCH = 3 * FPS                 # frames: our out-of-bounds run and the ATD's are the same event
# name -> (ball_track.finish options, extra HMM cfg)
VARIANTS = {
    "baseline": ({}, {}),           # before rule 26
    "rule26": ({}, {"touch_sides": True, "air_touch": True}),
    "default": ({}, {"touch_sides": True, "air_touch": True, "oob_grade": True, "ref_delay": 0}),  # C_TOUCHES
    "grade_6_1": ({}, {"touch_sides": True, "air_touch": True, "oob_grade": True}),
    "grade_6_0": ({}, {"touch_sides": True, "air_touch": True, "oob_grade": True,
                       "signals": {"out of bounds, alone": (0.0, 0.0)}}),
    "grade_12_1": ({}, {"touch_sides": True, "air_touch": True, "oob_grade": True,
                        "signals": {"out of bounds, long": (12.0, 0.0)}}),
    "grade_3_1": ({}, {"touch_sides": True, "air_touch": True, "oob_grade": True,
                       "signals": {"out of bounds, long": (3.0, 0.0)}}),
    "grade_delay0": ({}, {"touch_sides": True, "air_touch": True, "oob_grade": True, "ref_delay": 0}),
    "default_delay0": ({}, {"touch_sides": True, "air_touch": True, "ref_delay": 0}),
    "oob_off": ({},{"touch_sides": True, "air_touch": True, "signals": {"out of bounds": (0.0, 0.0)}}),
    "oob_x2": ({}, {"touch_sides": True, "air_touch": True, "signals": {"out of bounds": (6.0, 0.0)}}),
    "oob_x4": ({}, {"touch_sides": True, "air_touch": True, "signals": {"out of bounds": (12.0, 0.0)}}),
    "touch_seen": ({}, {"touch_seen": True}),
    "touch_once": ({}, {"touch_once": True}),
    "touch_both": ({}, {"touch_seen": True, "touch_once": True}),
    "air_touch": ({}, {"air_touch": True}),
    "seen_air": ({}, {"touch_seen": True, "air_touch": True}),
    "sides": ({}, {"touch_sides": True}),
    "sides_air": ({}, {"touch_sides": True, "air_touch": True}),
    "sides_air_once": ({}, {"touch_sides": True, "air_touch": True, "touch_once": True}),
    "seen_air_once": ({}, {"touch_seen": True, "air_touch": True, "touch_once": True}),
}


def _cached(name, build):
    path = os.path.join(CACHE, name + ".pkl")
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = build()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(value, handle, protocol=4)
    return value


def setup(repick=False, turn=0.0, persons=False, refs=False, feet=False, out=False, relink=False, image=False, restart=False):
    """The cached inputs: the players, the candidates of every chunk (the slow part, once), and the
    picks priced under the options (rule 28 turn cost, rule 30 persons, rule 29 refs), each cached
    under its own name. env["conf"]: the picks' confidences, for the smoother (rule 31)."""
    import ball_track
    import ball3d
    import detect_all
    import detector_model
    import run
    chunks = list(detect_all.chunks())
    tac_chunks = [(s, c) for src, s, c in chunks if src == "tactical"]
    bro_chunks = [(s, c) for src, s, c in chunks if src == "broadcast"]
    with redirect_stdout(io.StringIO()):
        tac, bro = data.load("tactical"), data.load("broadcast")
        al = run.load_alignment(tac, bro)
    c_t, c_b, _ = ball3d.camera_geometry(tac, bro, al)
    flip = tuple(bool(v) for v in al["flip"])
    c_b_own = np.array([data.PITCH[0] - c_b[0] if flip[0] else c_b[0],
                        data.PITCH[1] - c_b[1] if flip[1] else c_b[1], c_b[2]])

    def cands():
        with redirect_stdout(io.StringIO()):
            return ball_track.base_candidates(tac_chunks, bro_chunks, c_t, c_b_own)

    def picks():
        ball_track.TURN_PEN = turn      # rule 28; 0 is off
        cand_t, cand_b = _cached("cands2", cands)       # rule 33's off-the-pitch candidates are flagged in it

        def build(**opts):
            with redirect_stdout(io.StringIO()):
                return ball_track.pick_match(cand_t, cand_b, tac_chunks, bro_chunks, al, bro, tac.n, opts,
                                             centres=(c_t, c_b_own))

        first = None
        if refs or feet or restart:     # the second passes start from the first pass, cached by itself
            first = _cached("picks2" + ("_turn{:g}".format(turn) if turn else "") + ("_persons" if persons else "") +
                            ("_out" if out else "") + ("_image" if image else "") + ("_prec" if ball_track.PRICE_PRECISION else ""),
                            lambda: build(persons=persons, refs=False, feet=False, out=out, image=image))
        with redirect_stdout(io.StringIO()):
            return ball_track.pick_match(cand_t, cand_b, tac_chunks, bro_chunks, al, bro, tac.n,
                                         dict(persons=persons, refs=refs, feet=feet, out=out, image=image, restart=restart),
                                         first=first,
                                         centres=(c_t, c_b_own))

    name = ("picks2" + ("_turn{:g}".format(turn) if turn else "") + ("_persons" if persons else "") +
            ("_refs" if refs else "") + ("_feet" if feet else "") + ("_out" if out else "") + ("_image" if image else "") + ("_prec" if ball_track.PRICE_PRECISION else "") +
            ("_restart" if restart else ""))
    if repick and os.path.isfile(os.path.join(CACHE, name + ".pkl")):
        os.remove(os.path.join(CACHE, name + ".pkl"))
    full_t, full_b, conf_t, conf_b = _cached(name, picks)
    det = _cached("players_relink" if relink else "players", lambda: detector_model.match_players(tac_chunks, tac.n)[0])
    img = None
    if image or ball_track.IMAGE:
        import image_track
        with redirect_stdout(io.StringIO()):
            img = image_track.context(al, bro.n, ball_track.main_broadcast(al, bro), c_t, c_b_own)
    return dict(tac=tac, bro=bro, al=al, c_t=c_t, c_b=c_b, full_t=full_t, full_b=full_b, det=det,
                tac_chunks=tac_chunks, conf=(conf_t, conf_b), image=img)


def kinematics(env, **opts):
    """build_match()'s kinematics from the cached picks; opts reach ball_track.finish()."""
    import ball_track
    with redirect_stdout(io.StringIO()):
        return ball_track.finish(env["full_t"], env["full_b"], env["c_t"], env["c_b"], env["tac_chunks"],
                                 conf=env["conf"], players=env["det"], image=env.get("image"), **opts)


def labels_of(env, kin, **cfg_extra):
    import hmm
    import rules
    with redirect_stdout(io.StringIO()):
        ctx = rules.Context(env["det"], env["bro"], env["al"], kin=kin, identity=False)
    cfg = dict(rules.V1, goalie=False, crosscheck=False, resolve=False, dropouts=False, static=False,
               crosscam=False, **cfg_extra)
    cfg.update(HMM_CFG)
    labels, detail = hmm.predict(ctx, cfg)
    return ctx, cfg, labels, detail


def report(name, env, kin, ctx, cfg, labels, detail):
    import csv
    moved = detail.get("whistles_moved") or []
    if cfg.get("whistle_back"):
        secs = np.array([s - w for w, s in moved]) / FPS
        print("  rule 57: {} dead stretches moved back to a broadcast whistle, by {:.1f} s median ({:.0f} s in all)".format(
            len(moved), float(np.median(secs)) if len(secs) else 0.0, secs.sum()))
    if cfg.get("kick"):
        import hmm
        kicks = [c for c in hmm.credited(ctx, cfg, detail["events"]) if c[2] == "kick"]
        print("  rule 55: {} kicks credited".format(len(kicks)))
    import possession_metrics as pm
    import rules
    import detector_model
    gt = np.array([r["label"].strip().upper() for r in csv.DictReader(open(detector_model.PUBLIC_GT, newline=""))])
    res = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels[:len(gt)]))
    graded = np.zeros(env["tac"].n, dtype=bool)
    for _, first, last in data.halves():
        graded[first:last + 1] = True
    lab = np.where(graded, labels, "X")
    spells = [(s, e) for k in "HA" for s, e in rules.runs(lab == k)]
    lens = np.array([e - s for s, e in spells]) / FPS
    dead = [(s, e) for s, e in rules.runs(lab == "D")]

    # Ball artefacts: rule 1 peaks with nobody within the touch radius; sharp reversals.
    players = ctx._set(cfg)
    nobody = int((players["touch_dist"] > rules.TOUCH_RADIUS).sum())
    seen = kin["seen"][graded]
    with np.errstate(invalid="ignore"):
        cos = np.sum(kin["v_in"] * kin["v_out"], axis=1) / (kin["speed_in"] * kin["speed_out"])
        rev = (cos < -0.5) & (kin["speed_in"] > 3) & (kin["speed_out"] > 3)
    reversals = len(rules._peaks(rev.astype(float), 0.5, rules.TOUCH_NMS))
    masks, _, _ = __import__("hmm")._masks(ctx, cfg)
    oob = rules.runs(masks["out of bounds"])
    # Out-of-bounds runs against the provider's ball as a yardstick only (not in the model): a run
    # of ours "matches" when the ATD ball is also out within OOB_MATCH of its start.
    _, _, ev = rules.dead_intervals(ctx, cfg["ref_delay"], False, False, False)
    ours = [s for s, _ in ev["out of bounds"] if graded[s]]
    atd = env["tac"].ball
    L, W = data.PITCH
    with np.errstate(invalid="ignore"):
        out_atd = (atd[:, 0] < -rules.OOB_MARGIN) | (atd[:, 0] > L + rules.OOB_MARGIN) | \
                  (atd[:, 1] < -rules.OOB_MARGIN) | (atd[:, 1] > W + rules.OOB_MARGIN)
    ref = np.array([s for s, e in rules.runs(out_atd & graded) if e - s >= rules.OOB_FRAMES])
    hit = lambda t, arr: len(arr) and np.abs(np.asarray(arr) - t).min() <= OOB_MATCH
    oob_p = np.mean([bool(hit(t, ref)) for t in ours]) if ours else float("nan")
    oob_r = np.mean([bool(hit(t, ours)) for t in ref]) if len(ref) else float("nan")

    # Short spells and what was going on: a flight / air hint, an unseen ball, a nobody-peak.
    flight = kin["flight"]
    nob_frames = np.zeros(len(flight), bool)
    for t, d in zip(ctx.touch_frames, players["touch_dist"]):
        if d > rules.TOUCH_RADIUS:
            nob_frames[max(0, t - FPS):t + FPS] = True
    short = [(s, e) for s, e in spells if e - s < 2 * FPS]
    tags = {"flight": 0, "mostly unseen": 0, "nobody-peak": 0}
    for s, e in short:
        tags["flight"] += bool(flight[max(0, s - FPS // 2):e + FPS // 2].any())
        tags["mostly unseen"] += bool(kin["seen"][s:e].mean() < 0.5)
        tags["nobody-peak"] += bool(nob_frames[s:e].any())
    # Cross-camera consistency where both cameras picked: the miss across the lines of sight (no
    # height explains it) and the triangulated height.
    import ball3d
    g_t, g_b = env["full_t"].astype(float), env["full_b"].astype(float) - kin["bias"]
    miss = ball3d.ray_miss(g_t, g_b, env["c_t"], env["c_b"])
    both = np.isfinite(miss) & graded
    print("\n== {} ==".format(name))
    print("  both cameras picked on {:.1%} of frames; lines of sight miss by > 1 m on {:.1%} of those, "
          "> 3 m on {:.1%}; triangulated above reach (no flight) on {:.1%}".format(
              both.mean(), np.mean(miss[both] > 1), np.mean(miss[both] > 3),
              np.mean((kin["z_tri"][both] > ball3d.REACH) & ~kin["flight"][both])))
    print("  public: macroF1 {:.4f} spellF1 {:.4f} poss {} (GT {})".format(
        res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"], res["n_gt_spells"]))
    print("  match: {} poss, mean {:.1f} s; under 1 s {}, under 2 s {} (flight {flight}, mostly unseen "
          "{mostly unseen}, nobody-peak {nobody-peak}); dead {:.1%} in {} stretches".format(
              len(spells), lens.mean(), int((lens < 1).sum()), len(short), (lab[graded] == "D").mean(), len(dead), **tags))
    print("  ball: seen {:.1%}, touch peaks {} ({} nobody within {} m), reversals {}, flights {}".format(
        seen.mean(), len(ctx.touch_frames), nobody, rules.TOUCH_RADIUS, reversals, len(kin["flights"])))
    # Reversals by context: nobody near, next to a change of which camera picked, next to a gap.
    rv = rules._peaks(rev.astype(float), 0.5, rules.TOUCH_NMS)
    src = np.isfinite(env["full_t"][:, 0]) * 1 + np.isfinite(env["full_b"][:, 0]) * 2
    single = np.where(src == 3, 0, src)
    switch = np.zeros(len(src), bool)
    nz = np.flatnonzero(single)
    chg = nz[1:][(single[nz[1:]] != single[nz[:-1]]) & (nz[1:] - nz[:-1] <= 3)]
    switch[chg] = True
    edge = np.zeros(len(src), bool)
    edge[np.flatnonzero(np.diff(kin["seen"].astype(np.int8)) != 0)] = True
    near = lambda m, t: m[max(0, t - 3):t + 4].any()
    nd = players["near_dist"]
    print("  reversals: nobody within 4 m {}, at a camera switch {}, at a gap edge {}, flight {}".format(
        sum(nd[t] > 4 for t in rv), sum(near(switch, t) for t in rv), sum(near(edge, t) for t in rv),
        sum(bool(kin["flight"][t]) for t in rv)))
    # What paid for each short spell: the spell team's event and distance evidence, by event kind.
    import hmm
    ev_team = {}
    llr = detail["team_llr"]
    evs = hmm.event_evidence(ctx, cfg, detail["events"], len(labels))
    kinds = {}
    for s, e in short:
        sign = 1 if labels[s] == "H" else -1
        for t, who, kind, _ in detail["events"]:
            if s - 6 <= t < e and who == (data.HOME if sign > 0 else data.AWAY):
                air = kin["z"][t] > ball3d.CONTROL_REACH or kin["flight"][max(0, t - 6):t + 7].any()
                kinds[kind + (" air" if air else "")] = kinds.get(kind + (" air" if air else ""), 0) + 1
        ev_team.setdefault("events", []).append(sign * evs[s:e].sum())
        ev_team.setdefault("distance", []).append(sign * (llr[s:e] - evs[s:e]).sum())
    if short:
        print("  short spells: median event evidence {:+.1f}, distance {:+.1f} nats; own-team events {}".format(
            np.median(ev_team["events"]), np.median(ev_team["distance"]), kinds))
    # Dead frames against the ATD model's (rules_hmm_3d: provider ball and players) over both
    # halves -- a yardstick with its own errors, not ground truth.
    ref_path = os.path.join(ROOT, "evaluation", "results", "rules_hmm_3d.csv")
    if os.path.isfile(ref_path):
        atd_lab = np.array([r["label"].strip().upper() for r in csv.DictReader(open(ref_path, newline=""))])
        a, b = (lab == "D") & graded, (atd_lab == "D") & graded
        new = [(s, e) for s, e in dead if not b[s:e].any()]
        print("  dead vs the ATD model over the match: precision {:.1%}, recall {:.1%}; our stretches with no ATD "
              "dead frame: {}".format(a[b].mean() if False else (a & b).sum() / max(a.sum(), 1),
                                     (a & b).sum() / max(b.sum(), 1), len(new)))
    print("  out of bounds: {} raw runs ({:.0%} with the ATD ball out within 3 s; ATD {} runs, {:.0%} "
          "found); {} HMM mask stretches".format(len(ours), oob_p, len(ref), oob_r, len(oob)))
    for f in WATCH:
        # The labels within +-5 s as runs: label and length (s), the watched frame's run in [].
        lo, hi = f - 5 * FPS, f + 5 * FPS
        seq = []
        starts = np.flatnonzero(np.r_[True, labels[lo + 1:hi] != labels[lo:hi - 1]]) + lo
        for s, e in zip(starts, np.r_[starts[1:], hi]):
            txt = "{}{:.1f}".format(labels[s], (e - s) / FPS)
            seq.append("[{}]".format(txt) if s <= f < e else txt)
        # Explain the shortest team spell within +-3 s (the watched frame's own, or its neighbour).
        cand = [(b - a, a, b) for a, b in zip(starts, np.r_[starts[1:], hi])
                if labels[a] in "HA" and a < f + 3 * FPS and b > f - 3 * FPS and a > lo]
        _, s, e = min(cand) if cand else (0, f, f + 1)
        sign = 1 if labels[s] == "H" else -1
        own = data.HOME if sign > 0 else data.AWAY
        ends = np.array([fl[1] for fl in kin["flights"]])

        def since_landing(t):
            before = ends[ends <= t]
            return " {:+.2f}s after landing".format((t - before.max()) / FPS) if len(before) and t - before.max() < 2 * FPS else ""
        kinds = ["{}{}{}{}".format(k, " air" if hmm._aerial(kin, t) else "", "" if w == own else " (other)", since_landing(t))
                 for t, w, k, _ in detail["events"] if s - 6 <= t < e and w >= 0]
        print("  watch {}: {}; flight within 1 s {}; ball seen {:.0%} of +-1 s; its spell: events {:+.1f}, "
              "distance {:+.1f} nats, height-gated frames {:.0%}; events {}".format(
                  f, " ".join(seq), bool(flight[f - FPS:f + FPS].any()), kin["seen"][f - FPS:f + FPS].mean(),
                  sign * evs[s:e].sum(), sign * (llr[s:e] - evs[s:e]).sum(),
                  np.mean(kin["z"][s:e] > ball3d.REACH), kinds))
    disagreement(env, kin, graded)
    if "--angles" in sys.argv:
        angles(kin, ctx, players)
    if "--oob" in sys.argv:
        oob_breakdown(env, kin, ev["out of bounds"], ref, graded, ctx)
    if "--edges" in sys.argv:
        edges = np.flatnonzero(np.diff(kin["seen"].astype(np.int8)) != 0)
        t = ctx.touch_frames
        off = np.abs(t[:, None] - edges[None, :]).min(axis=1) if len(edges) else np.full(len(t), 99)
        nob = players["touch_dist"] > rules.TOUCH_RADIUS
        for lo, hi in ((0, 3), (3, 5), (5, 9), (9, 13), (13, 99999)):
            m = (off >= lo) & (off < hi)
            print("  touch peaks {}-{} frames from a gap edge: {:>5}, nobody within 4 m {:.0%}".format(
                lo, hi - 1, int(m.sum()), nob[m].mean() if m.any() else float("nan")))
    if "--exits" in sys.argv:
        exits(env, kin, ctx, ref, graded)
    if "--excursions" in sys.argv:
        for cam in ("full_t", "full_b"):
            counts = {k: 0 for k in range(1, 7)}
            p = env[cam].astype(float)
            idx = np.flatnonzero(np.isfinite(p[:, 0]) & graded)
            # Consecutive picks (in pick order): a jump out > SPIKE, then within k picks a jump back
            # to within SPIKE of where it left.
            q = p[idx]
            step = np.linalg.norm(np.diff(q, axis=0), axis=1)
            for i in np.flatnonzero(step > rules.SPIKE):
                for k in range(1, 7):
                    j = i + k + 1
                    if j < len(q) and np.linalg.norm(q[j] - q[i]) < rules.SPIKE and step[j - 1] > rules.SPIKE \
                            and idx[j] - idx[i] <= k + 3:
                        counts[k] += 1
                        break
            print("  {} out-and-back excursions by length (picks): {}".format(cam, counts))
    return res


def disagreement(env, kin, graded):
    """How often the forward and backward smoother passes differ on a pick (one took it, the other
    rejected it), in runs of consecutive picks; and the goalkeeper's kick at 1H 78887 (Matias,
    2026-10-25): how far the fused track strays from the broadcast's picks of the ball at rest
    over 78880-78920 (rule 31's target; only distances are printed)."""
    import rules
    pick = (np.isfinite(env["full_t"][:, 0]) | np.isfinite(env["full_b"][:, 0])) & graded
    tf, tb = (kin["used"] & 3) > 0, (kin["used_b"] & 3) > 0
    dis = pick & (tf != tb)
    idx = np.flatnonzero(pick)
    runs = [e - s for s, e in rules.runs(dis[idx])]
    held = kin.get("held")
    if held is not None:
        print("  ball held (a keeper, or with --cover a player): {:.0f} s over the halves".format((held & graded).sum() / FPS))
    print("  smoother passes disagree on {} of {} pick frames ({:.1%}); runs of consecutive picks >= 10: {}, >= 20: {}".format(
        int(dis.sum()), int(pick.sum()), dis.sum() / max(pick.sum(), 1), sum(r >= 10 for r in runs), sum(r >= 20 for r in runs)))
    w = slice(78880, 78921)
    b = env["full_b"][w].astype(float)
    if np.isfinite(b[:, 0]).sum() >= 3:
        rest = np.nanmedian(b, axis=0)
        d = np.linalg.norm(kin["pos"][w] - rest, axis=1)
        d = d[np.isfinite(d)]
        print("  1H 78880-78920 (ball at rest, goalkeeper): fused track strays up to {:.1f} m from it ({} of {} frames over 5 m)".format(
            d.max(), int((d > 5).sum()), len(d)))


def exits(env, kin, ctx, ref, graded):
    """Every ball gap of 0.5 s or more, by where it was last seen (metres inside the nearest line,
    negative outside), its speed toward that line, and what follows; share matching the ATD's
    out-of-bounds runs (yardstick)."""
    import align
    import rules
    L, W = data.PITCH
    seen, pos, vel = kin["seen"], kin["pos"], kin["v_in"]
    nohom = ctx.alignment["state_t"] == align.STATE_CLOSEUP
    rows = []
    for s, e in rules.runs(~seen):
        f = s - 1
        if f < 0 or not graded[f] or e - s < FPS // 2 or not np.isfinite(pos[f, 0]):
            continue
        x, y = pos[f]
        lines = [(y, -vel[f, 1]), (W - y, vel[f, 1]), (x, -vel[f, 0]), (L - x, vel[f, 0])]  # (inside m, speed toward)
        d, v = min(lines)
        match = bool(len(ref) and np.abs(ref - f).min() <= OOB_MATCH)
        cut = bool(nohom[f:f + 5 * FPS].any())
        rows.append((d, v, (e - s) / FPS, cut, match))
    rows = np.array(rows, dtype=float)
    print("  gaps >= 0.5 s: {}; ATD out-of-bounds runs {}".format(len(rows), len(ref)))
    for lo, hi in ((-99, 0), (0, 1), (1, 2), (2, 4), (4, 99)):
        for vlo in (-99, 0, 3, 8):
            m = (rows[:, 0] >= lo) & (rows[:, 0] < hi) & (rows[:, 1] >= vlo)
            if m.sum():
                print("    last seen {:>3}..{:<3} m inside, toward the line >= {:>3} m/s: {:>4} gaps, {:.0%} ATD out;"
                      " of those with a broadcast cut within 5 s: {:>3}, {:.0%}".format(
                          lo, hi, vlo, int(m.sum()), rows[m, 4].mean(), int((m & (rows[:, 3] > 0)).sum()),
                          rows[m & (rows[:, 3] > 0), 4].mean() if (m & (rows[:, 3] > 0)).any() else float("nan")))


def oob_breakdown(env, kin, runs_, ref, graded, ctx):
    """Our out-of-bounds runs by route and shape, each with its share matching the ATD's (yardstick)."""
    L, W = data.PITCH
    import align
    import ball_track
    import rules
    nohom = ctx.alignment["state_t"] == align.STATE_CLOSEUP
    quiet = rules.quiet_players(ctx)
    # Yardstick: the ATD model (rules_hmm_3d) dead from 1 s before to 5 s after the run's start.
    import csv
    ref_path = os.path.join(ROOT, "evaluation", "results", "rules_hmm_3d.csv")
    atd_dead = np.array([r["label"].strip().upper() == "D" for r in csv.DictReader(open(ref_path, newline=""))]) \
        if os.path.isfile(ref_path) else None
    print("  (yardstick: {})".format("ATD model dead within -1..+5 s" if atd_dead is not None else "ATD ball out"))
    out_t = ball_track.outside(np.nan_to_num(env["full_t"].astype(float), nan=-1e9)) * np.isfinite(env["full_t"][:, 0])
    out_b = ball_track.outside(np.nan_to_num(env["full_b"].astype(float), nan=-1e9)) * np.isfinite(env["full_b"][:, 0])
    pos = kin["pos"]
    rows = []
    for s, e in runs_:
        if not graded[s]:
            continue
        p = pos[s:e]
        far = float(ball_track.outside(p).max())
        side = "near touchline" if np.nanmin(p[:, 1]) < 0 else "far touchline" if np.nanmax(p[:, 1]) > W else "goal line"
        route = ("tactical" if (out_t[s:e] > 0.5).any() else "") + ("+broadcast" if (out_b[s:e] > 0.5).any() else "")
        route = route or "smoother only"
        # Moving out: the ball's velocity on the last frames before the run points across the line.
        v = kin["v_in"][s]
        toward = (v[1] < 0) if side == "near touchline" else (v[1] > 0) if side == "far touchline" else \
                 (v[0] < 0 if p[0, 0] < L / 2 else v[0] > 0)
        match = bool(atd_dead[max(0, s - FPS):s + 5 * FPS].any()) if atd_dead is not None else \
            bool(len(ref) and np.abs(ref - s).min() <= OOB_MATCH)
        support = ("cut" if nohom[s:s + 5 * FPS].any() else "") + ("+still" if quiet[s:s + 5 * FPS].any() else "")
        rows.append((route, side, far, (e - s) / FPS, bool(toward), bool(kin["flight"][max(0, s - FPS):e].any()),
                     support or "none", match))
    def show(label, key):
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r[-1])
        print("  {}: ".format(label) + "; ".join("{} {} ({:.0%})".format(k, len(v), np.mean(v)) for k, v in sorted(groups.items())))
    show("route", lambda r: r[0])
    show("side", lambda r: r[1])
    show("metres out", lambda r: "<1" if r[2] < 1 else "1-3" if r[2] < 3 else ">=3")
    show("length", lambda r: "<0.5s" if r[3] < 0.5 else "0.5-1s" if r[3] < 1 else ">=1s")
    show("moving toward the line", lambda r: r[4])
    show("flight within 1 s", lambda r: r[5])
    show("followed within 5 s by", lambda r: r[6])


def angles(kin, ctx, players):
    """Direction change and speed kept at rule 1 peaks: within 2 s of a flight's landing vs elsewhere."""
    import rules
    t = ctx.touch_frames
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(kin["v_in"][t] * kin["v_out"][t], axis=1) / (kin["speed_in"][t] * kin["speed_out"][t])
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        keep = kin["speed_out"][t] / kin["speed_in"][t]
    ends = np.array(sorted(fl[1] for fl in kin["flights"]))
    i = np.searchsorted(ends, t, side="right") - 1
    after = (i >= 0) & (t - ends[np.maximum(i, 0)] <= 2 * FPS)
    near = players["touch_dist"] <= rules.TOUCH_RADIUS
    fast = kin["speed_in"][t] >= 3
    for name, m in (("after landing", after & near & fast), ("elsewhere", ~after & near & fast)):
        a, k = ang[m], keep[m]
        hist = np.histogram(a[np.isfinite(a)], bins=[0, 10, 20, 30, 45, 90, 180])[0]
        both = np.isfinite(a) & np.isfinite(k)
        print("  peaks {} (player within reach, ball >= 3 m/s in): {}; angle bins 0/10/20/30/45/90/180: {}; "
              "under 20 deg keeping >= 50% speed: {:.0%}".format(
                  name, int(m.sum()), hist.tolist(), np.mean((a[both] < 20) & (k[both] >= 0.5))))


def models(env):
    """The three possession models (detector_model.MODELS) on the default kinematics and C's cfg:
    the viewer's payloads and the submissions, as detector_model.match() writes them, and a
    comparison -- public scores, match spells, short spells, dead share -- aggregates only."""
    import csv
    import possession_metrics as pm
    import detector_model
    import rules
    import run
    kin = kinematics(env)
    import ball_track
    with redirect_stdout(io.StringIO()):
        for first, count in env["tac_chunks"]:      # the viewer's ball track, as detector_model.match() writes it
            ball_track.write(ball_track.window(kin, env["full_t"], env["full_b"], first, count), first, count)
    ctx, cfg, _, _ = labels_of(env, kin, **VARIANTS["default"][1])
    gt = np.array([r["label"].strip().upper() for r in csv.DictReader(open(detector_model.PUBLIC_GT, newline=""))])
    graded = np.zeros(env["tac"].n, dtype=bool)
    for _, first, last in data.halves():
        graded[first:last + 1] = True
    rows = []
    for key, name, title, _ in detector_model.MODELS:
        t0 = time.time()
        labels = detector_model.write_debug(ctx, cfg, key)
        run.write_submission(os.path.join(ROOT, "evaluation", "results", name + ".csv"), labels)
        rows.append((title, labels, time.time() - t0))
    # Public-window scores only with --score; reproduce.py never passes it, so building the submission shows none.
    score = "--score" in sys.argv
    print("\n  {:<30} {:>8} {:>8} {:>5}   {:>10} {:>7} {:>9} {:>6} {:>6}".format(
        "", "macroF1" if score else "", "spellF1" if score else "", "poss" if score else "",
        "match poss", "mean s", "under 2 s", "dead", "time"))
    for title, labels, took in rows:
        lab = np.where(graded, labels, "X")
        lens = np.array([e - s for k in "HA" for s, e in rules.runs(lab == k)]) / FPS
        if score:
            res = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels[:len(gt)]))
            pub = "{:>8.4f} {:>8.4f} {:>5}".format(res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"])
        else:
            pub = "{:>8} {:>8} {:>5}".format("", "", "")
        print("  {:<30} {}   {:>10} {:>7.1f} {:>9} {:>6.1%} {:>5.0f}s".format(
            title, pub, len(lens), lens.mean(), int((lens < 2).sum()), (lab[graded] == "D").mean(), took))
    if score:
        print("  (public window: GT {} possessions)".format(res["n_gt_spells"]))
    # Where the models disagree over the match, as a share of live graded frames.
    base = rows[0][1]
    for title, labels, _ in rows[1:]:
        live = graded & (base != "D") & (labels != "D")
        print("  {} vs C: other team on {:.1%} of frames both call live; live / dead differ on {:.1%}".format(
            title, (labels[live] != base[live]).mean(), ((labels == "D") != (base == "D"))[graded].mean()))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    t0 = time.time()
    turn = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--turn=")), 0.0)
    import kalman
    kalman.FIX = "--smooth" in sys.argv        # rule 31
    import ball3d
    ball3d.DRAG = ball3d.JOIN = flights_fix = "--flights" in sys.argv       # rule 34
    ball3d.TRI_HOLD = "--tri" in sys.argv                                    # rule 40
    ball3d.TRACES = "--traces" in sys.argv                                   # rule 42
    import detector_model
    if "--relink" in sys.argv:                                               # rule 37
        detector_model.RELINK_LONG = 75
    if "--arc-tol" in sys.argv:                                              # rule 35
        ball3d.FIT_RMS_BASE, ball3d.FIT_RMS_PER_S = 1.0, 0.45
        ball3d.COVER = True
    import ball_track
    if "--keeper" in sys.argv:                                               # rule 36
        ball_track.KEEPER_HOLD_RADIUS = 3.0
    ball_track.HOLD_PLAYER = "--cover" in sys.argv                           # rule 38
    ball_track.DROP_SIDE = "--side" in sys.argv                              # rule 41
    ball_track.IMAGE = "--image" in sys.argv                                 # rule 44
    ball3d.CONTINUITY = "--heights" in sys.argv                              # rule 46
    ball3d.SOLO = "--solo" in sys.argv                                       # rule 47
    kalman.GAPS = ball_track.HOLD_RELEASE = "--gaps" in sys.argv              # rule 48
    import rules
    rules.TOUCH_CAP = "--touch-cap" in sys.argv                              # rule 49
    rules.TOUCH_LOCAL = "--touch-local" in sys.argv                          # rule 54
    kalman.CONFIRM = "--confirm" in sys.argv                                 # rule 50
    kalman.FINAL_PACE = "--bridge-pace" in sys.argv                          # rule 51
    ball_track.PRICE_PRECISION = "--precision" in sys.argv                    # rule 52
    import hmm
    hmm.TOUCH_PRECISION = "--touch-precision" in sys.argv                     # rule 53
    if "--event-precision" in sys.argv:                                      # rule 53b
        hmm.PRECISION_KINDS = ("touch", "far touch", "control", "carry")
    rules.TOUCH_ECHO = "--touch-echo" in sys.argv                            # rule 58
    for flag, key in (("--kick", "kick"), ("--restart-live", "restart_live"), ("--whistle-back", "whistle_back"),
                      ("--event-switch", "event_switch"), ("--far-clear", "far_clear"),
                      ("--restart-confirm", "restart_confirm"), ("--replay-neutral", "replay_neutral"),
                      ("--short-spells", "short_spells"), ("--short-broadcast", "short_broadcast"),
                      ("--still-cutaway", "still_cutaway")):
        if flag in sys.argv:                                                 # rules 55, 56, 57, 59, 60
            HMM_CFG[key] = True
    env = setup(repick="--repick" in sys.argv, turn=turn, persons="--persons" in sys.argv, refs="--refs" in sys.argv,
                feet="--feet" in sys.argv, out="--out" in sys.argv, relink="--relink" in sys.argv,
                image="--image" in sys.argv, restart="--restart" in sys.argv)
    print("setup {:.0f} s (turn cost {:g}, persons {}, refs {}, feet {}, out {}, smoother fix {}, flights fix {})".format(
        time.time() - t0, turn, "--persons" in sys.argv, "--refs" in sys.argv, "--feet" in sys.argv,
        "--out" in sys.argv, kalman.FIX, flights_fix))
    if args == ["models"]:
        return models(env)
    for name in args or ["baseline"]:
        t0 = time.time()
        kin_opts, cfg_extra = VARIANTS[name]
        kin = kinematics(env, **kin_opts)
        ctx, cfg, labels, detail = labels_of(env, kin, **cfg_extra)
        report(name, env, kin, ctx, cfg, labels, detail)
        print("  ({:.0f} s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
