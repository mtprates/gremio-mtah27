"""Build the rules-based possession model and write it where the scorer and the viewer find it.

    py model/run.py

Writes
    evaluation/results/rules_r1.csv     rule 1 alone: every touch hands possession over
    evaluation/results/rules_v1.csv     every rule in docs/TECHNICAL.md
    evaluation/results/rules_hmm.csv    the same rules as evidence for an HMM (rule 14), on the
                                        Kalman-smoothed ball (rule 13)
    evaluation/results/rules_hmm_2cam.csv  the HMM on the two-camera Kalman ball (rule 13.b)
    evaluation/results/rules_hmm_3d.csv    the HMM on the 3D ball (rule 17)
    data/.model/rules_v1_debug.json     touches, dead-ball evidence, broadcast state and
                                        goalkeeper fixes, drawn by the viewer

and prints aggregate diagnostics only -- counts and shares, never tracking values -- then scores
both on the public window, plus rules_v1 with each rule switched off in turn. With 17 public
possessions that table is for spotting a rule that is broken, not for tuning one
(EVALUATION.md: "The public window is not a leaderboard").

The first run parses both ATDs and aligns the broadcast (a few minutes); both are cached under
data/.model/ and later runs take seconds.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import align  # noqa: E402
import cameras  # noqa: E402
import data  # noqa: E402
import hmm  # noqa: E402
import kalman  # noqa: E402
import rules  # noqa: E402
from games import asset_paths  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
VARIANTS = (("rules_r1", rules.R1), ("rules_v1", rules.V1))
ABLATIONS = (
    ("v1 without 1.a confirmation", {"confirm": False}),
    ("v1 without 1.a dwell control", {"control": False}),
    ("v1 without rule 2 carries", {"carry": False}),
    ("v1 without rule 3 goalkeeper", {"goalie": False}),
    ("v1 without rule 3.a cross-check", {"crosscheck": False}),
    ("v1 without rule 3.b doubt resolution", {"resolve": False}),
    ("v1 without rule 4 broadcast", {"broadcast_dead": False}),
    ("v1 rule 4 corroborated (previous)", {"broadcast_dead": "corroborated"}),
    ("v1 rule 5 delay 0 s", {"ref_delay": 0}),
    ("v1 rule 5 delay 2 s", {"ref_delay": 2 * data.FPS}),
    ("v1 without rule 6 dropouts", {"dropouts": False}),
    ("v1 without rule 9 static drop", {"static": False}),
    ("v1 without rule 10 broadcast nearest", {"crosscam": False}),
    ("v1 without rule 11 local stillness", {"local_still": False}),
    ("v1 without rule 12 dead extension", {"dead_extend": False}),
    ("v1 rule 12 before only", {"dead_extend": "before"}),
    ("v1 rule 12 after only", {"dead_extend": "after"}),
    ("v1 touch radius 2 m (previous)", {"touch_radius": rules.TOUCH_CLOSE}),
)
HMM_ABLATIONS = (
    ("hmm on the moving-average ball", "ctx", {}),
    ("hmm without rule 4 broadcast", "kctx", {"broadcast_dead": False}),
    ("hmm without rule 10 broadcast nearest", "kctx", {"crosscam": False}),
    ("hmm without 1.a dwell control", "kctx", {"control": False}),
)
DROPOUTS_CSV = os.path.join(data.CACHE_DIR, "dropouts.csv")
KIND_CODES = {"touch": 0, "control": 1, "carry": 2, "far touch": 3}


def load_alignment(tac, bro):
    paths = [asset_paths(source=s)["atd_raw"] for s in ("tactical", "broadcast")]
    print("  aligning broadcast to tactical (cached after the first run) ...")
    key = "alignment_v2_halves" if align.HALF_OFFSETS else "alignment_v2"
    return data.cached(key, paths, lambda: align.align(tac, bro))


def write_submission(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def share(mask, graded):
    return 100.0 * float(np.mean(mask[graded])) if graded.any() else float("nan")


def diagnostics(ctx, graded):
    tac, al = ctx.tac, ctx.alignment
    print("\nball (rule 8: fused across cameras)")
    for code, name in enumerate(rules.BALL_SOURCES):
        print("  {:<32}  {:5.1f}% of graded frames".format(name, share(ctx.ball_source == code, graded)))
    print("\ntactical")
    print("  ball present after interpolation  {:5.1f}% of graded frames".format(
        share(ctx.kin["present"], graded)))
    tracked = ~np.isnan(tac.players[..., 0])
    print("  static trackers dropped (rule 9)  {:.2f}% of tracked object-frames; "
          "check slot 24 at 1H frame 11951: {}".format(
              100.0 * ctx.static[graded].sum() / max(1, tracked[graded].sum()),
              "dropped" if ctx.static[11951, 23] else "kept"))
    with_player = int(np.sum(ctx.touch_dist <= rules.TOUCH_RADIUS))
    far = int(np.sum((ctx.touch_dist > rules.TOUCH_CLOSE) & (ctx.touch_dist <= rules.TOUCH_RADIUS)))
    print("  touches (rule 1)                  {:>6}, {:.0f}% with a player within {} m "
          "({} of them {}-{} m: far touches)".format(
              len(ctx.touch_frames), 100.0 * with_player / max(1, len(ctx.touch_frames)),
              rules.TOUCH_RADIUS, far, rules.TOUCH_CLOSE, rules.TOUCH_RADIUS))
    print("  carries (rule 2)                  {:>6}".format(len(ctx.carries)))
    turned = [t for t, angle, _ in ctx.turns if angle >= 0]
    print("  ball direction changes (marker)   {:>6}, {} of them turns of {:.0f} deg or more".format(
        len(ctx.turns), len(turned), rules.TURN_ANGLE))
    gk_frames = sum(e - s for _, s, e, _ in ctx.gk_fixes)
    print("  goalkeeper fixes (rule 3)         {:>6} runs, {:.0f} s in total".format(
        len(ctx.gk_fixes), gk_frames / data.FPS))
    team = ctx.team[True]
    print("  check rule 3.c: 1H 19820 slot 46 {} -> {} -> {} (keeper re-acquired under an away slot); "
          "1H 15655 slot 24 {}, slot 3 at 15720 {} (attackers on the away goal line)".format(
              *("HA"[team[f, s]] for f, s in ((19810, 45), (19900, 45), (20100, 45), (15655, 23), (15720, 2)))))
    print("  defending sides                   " + "  ".join(
        "half {}: home goal at x={:.0f} m".format(h, s[data.HOME]) for h, s in sorted(ctx.sides.items())))

    print("\nbroadcast alignment")
    print("  orientation (flip x, flip y)      {}".format(tuple(bool(v) for v in al["flip"])))
    seeds = al["seeds"]
    confident = (seeds[:, 2] < align.GOOD_COST) & (seeds[:, 3] >= align.UNIQUE_RATIO * seeds[:, 2])
    print("  seeds                             {} searched, {} confident".format(
        len(seeds), int(confident.sum())))
    print("  stretches                         {} found, {} on the main timeline".format(
        len(al["stretches"]), len(al["chain"])))
    for b0, b1, offset, n in al["chain"]:
        print("    broadcast {:>6}-{:<6}  broadcast = tactical {:+d}   ({} seeds)".format(
            b0, b1, -offset, n))
    if len(al["half_offsets"]):
        print("  per half{}  ".format(" (used, HALF_OFFSETS)" if align.HALF_OFFSETS else " (not used)  ") + "  ".join(
            "half {}: broadcast = tactical {:+d}".format(h, -int(o)) for h, o in zip((1, 2), al["half_offsets"])))
    for code, name in align.STATE_NAMES.items():
        print("  broadcast state {:<16}  {:5.1f}% of graded frames".format(
            name, share(al["state_t"] == code, graded)))
    print("  team agreement across cameras     {:.1f}%{}".format(
        100 * ctx.agreement, "  (broadcast team order swapped)" if ctx.swapped else ""))
    print("  uncertain objects (rule 3.a)      {:.2f}% of tracked object-frames".format(
        100.0 * ctx.uncertain[True].sum() / max(1, np.sum(~np.isnan(tac.players[..., 0])))))
    print("  rule 3.b doubts settled           {}; still uncertain {:.2f}%".format(
        ", ".join("{} {}".format(k, v) for k, v in ctx.resolve_outcome.items()),
        100.0 * ctx.uncertain[rules.RESOLVED].sum() / max(1, np.sum(~np.isnan(tac.players[..., 0])))))
    nt = ctx.raw_tac.players.shape[1]
    print("  pairing switches (rule 15)        {} nearest-per-frame -> {} sticky one-to-one".format(
        rules.pairing_switches(al["pairs_nearest"][:, :nt]), rules.pairing_switches(al["pairs_t"][:, :nt])))
    moved = [m for m in ctx.moves]
    print("  identity cuts (rule 16)           {} pieces, {} back into the slot that dropped the person, "
          "{} new columns; {} change team, {:.0f} min of tracks in all".format(
              len(moved), sum(m[3] < nt for m in moved), ctx.tac.players.shape[1] - nt,
              sum(ctx.tac.team[m[3]] != ctx.raw_tac.team[m[0]] for m in moved),
              sum(m[2] - m[1] for m in moved) / data.FPS / 60))
    print("  check 1H 11410: {}".format(", ".join("slot {} {}-{} -> {}".format(m[0] + 1, m[1], m[2],
          "slot {}".format(m[3] + 1) if m[3] < nt else "new column") for m in moved if m[1] <= 11410 < m[2]) or "no cut"))
    at = 11409
    print("  check 1H frame {}: doubts before 3.b {}, after {}, camera fixes covering it {}".format(
        at, int(ctx.uncertain[True][at].sum()), int(ctx.uncertain[rules.RESOLVED][at].sum()),
        sum(1 for _, s, e, _ in ctx.camera_fixes if s <= at < e)))


def kalman_report(ctx, kctx, graded):
    """Rule 13: what the smoother changes about the ball and the touches found on it."""
    print("\nrule 13: Kalman-smoothed ball (moving average -> Kalman)")
    print("  ball present (rules' meaning)     {:5.1f}% -> {:5.1f}% of graded frames".format(
        share(ctx.kin["present"], graded), share(kctx.kin["present"], graded)))
    print("  ball position known               {:5.1f}% -> {:5.1f}%  (bridged while within {} m)".format(
        share(~np.isnan(ctx.kin["pos"][:, 0]), graded), share(~np.isnan(kctx.kin["pos"][:, 0]), graded),
        kalman.POS_STD_MAX))
    std = kctx.kin["pos_std"][graded]
    print("  position std, median / 90th pct   {:5.2f} / {:.2f} m".format(
        float(np.nanmedian(std)), float(np.nanpercentile(std, 90))))
    for name, c in (("moving average", ctx), ("Kalman", kctx)):
        t = c.touch_frames
        print("  touches, {:<15}         {:>6}, {:.0f}% with a player within {} m, {} inside a ball gap".format(
            name, len(t), 100.0 * np.mean(c.touch_dist <= rules.TOUCH_CLOSE) if len(t) else 0.0,
            rules.TOUCH_CLOSE, int(np.sum(~c.kin["seen"][t]))))


def two_camera_report(kctx, k2, graded):
    """Rule 13.b: what the two-camera filter took from each camera, and what changed."""
    kin = k2.kin
    used = kin["used"]
    print("\nrule 13.b: Kalman on both cameras' balls (vs rule 8's fused ball)")
    print("  camera noise read off the data    tactical {tactical:.2f} m, broadcast {broadcast:.2f} m".format(**kin["noise"]))
    for name, mask in (("both cameras taken", (used & 3) == 3), ("tactical only", (used & 3) == 1),
                       ("broadcast only", (used & 3) == 2), ("tactical ball rejected", (used & 4) > 0),
                       ("broadcast ball rejected", (used & 8) > 0)):
        print("  {:<32}  {:5.1f}% of graded frames".format(name, share(mask, graded)))
    off = np.linalg.norm(kin["offset"], axis=1)[graded]
    print("  broadcast offset, median / 90th   {:.2f} / {:.2f} m".format(float(np.nanmedian(off)), float(np.nanpercentile(off, 90))))
    print("  ball position known               {:5.1f}% -> {:5.1f}%".format(
        share(~np.isnan(kctx.kin["pos"][:, 0]), graded), share(~np.isnan(kin["pos"][:, 0]), graded)))
    for name, c in (("fused ball", kctx), ("two cameras", k2)):
        t = c.touch_frames
        print("  touches, {:<15}         {:>6}, {:.0f}% with a player within {} m, {} inside a ball gap".format(
            name, len(t), 100.0 * np.mean(c.touch_dist <= rules.TOUCH_CLOSE) if len(t) else 0.0,
            rules.TOUCH_CLOSE, int(np.sum(~c.kin["seen"][t]))))


def three_d_report(k3, graded):
    """Rule 17: the cameras, the flights found, and what the height took out."""
    kin, geo = k3.kin, k3.kin["geometry"]
    print("\nrule 17: the ball in 3D")
    print("  cameras (tactical coordinates)    tactical ({:.1f}, {:.1f}, {:.1f}) m, broadcast ({:.1f}, {:.1f}, {:.1f}) m".format(
        *geo["tactical"], *geo["broadcast"]))
    print("  broadcast main camera on          {:5.1f}% of graded frames; calibrated offset ({:+.2f}, {:+.2f}) m".format(
        share(geo["main"], graded), *geo["bias"]))
    ground = graded & ~kin["flight"] & np.isfinite(kin["z_tri"])
    print("  per-frame height off flights      median {:+.2f} m, |z| 90th pct {:.2f} m (the noise floor)".format(
        float(np.median(kin["z_tri"][ground])), float(np.percentile(np.abs(kin["z_tri"][ground]), 90))))
    for kind in ("two cameras", "tactical"):
        f = [x for x in kin["flights"] if x[4] == kind]
        if f:
            peaks = np.array([x[2] for x in f])
            print("  flights, {:<12}             {:>5}, {:.0f} s in all; peak median {:.1f} m, 90th pct {:.1f} m; fit RMS median {:.2f} m".format(
                kind, len(f), sum(x[1] - x[0] for x in f) / data.FPS, float(np.median(peaks)),
                float(np.percentile(peaks, 90)), float(np.median([x[3] for x in f]))))
    print("  ball in flight                    {:5.1f}% of graded frames".format(share(kin["flight"], graded)))
    on, off = k3.events(rules.V1), k3.events(dict(rules.V1, height=False))
    for kind in ("touch", "far touch", "control", "carry"):
        a, b = sum(e[2] == kind for e in off), sum(e[2] == kind for e in on)
        print("  {:<10} dropped as out of reach    {:>4} of {:>5}".format(kind, a - b, a))
    at = 12793
    fl = [x for x in kin["flights"] if x[0] <= at < x[1]]
    print("  check 1H 12793 (1:04 airball): height {:.1f} m{}".format(
        kin["z"][at], "; in a {} flight {:+.1f}..{:+.1f} s, peak {:.1f} m".format(
            fl[0][4], (fl[0][0] - 12785) / data.FPS, (fl[0][1] - 12785) / data.FPS, fl[0][2]) if fl else "; no flight"))


def hmm_report(labels, detail, graded):
    """Rule 14: how sure the HMM is of its labels; the least sure minutes are the ones to label."""
    conf = detail["confidence"]
    codes = np.where(np.isin(labels, ("H", "A")), (labels == "A").astype(int), -1)
    print("\nrule 14: HMM (Viterbi labels, forward-backward confidence)")
    print("  dead {:.1f}% of graded frames; {} possessions".format(
        share(labels == "D", graded), len(rules.value_runs(codes))))
    print("  label probability < {:.0%} on {:.1f}% of graded frames; mean {:.3f}".format(
        hmm.LOW_CONFIDENCE, share(conf < hmm.LOW_CONFIDENCE, graded), float(np.nanmean(conf[graded]))))
    minute = 60 * data.FPS
    worst = []
    for _, first, last in data.halves():
        for s in range(first, last + 1 - minute, minute):
            worst.append((float(np.mean(conf[s:s + minute])), s))
    worst.sort()
    print("  least sure minutes (label these first): " + ", ".join(
        "{} ({:.2f})".format(match_clock(s), c) for c, s in worst[:6]))


def match_clock(frame):
    """'1H 35:56' style clock for a tactical frame, as the viewer's jump box reads it."""
    k1, k2 = data.KICKOFFS[1], data.KICKOFFS[2]
    if frame >= k2:
        seconds, half = 45 * 60 + (frame - k2) / data.FPS, 2
    else:
        seconds, half = (frame - k1) / data.FPS, 1
    seconds = int(max(0, seconds))
    return "{}H {:02d}:{:02d}".format(half, seconds // 60, seconds % 60)


def rule6_report(ctx):
    """Aggregates for rule 6, and every mid-view dropout written to data/.model/dropouts.csv."""
    r6 = ctx.rule6
    records = r6["records"]
    print("\nrule 6: players dropped mid-view (tactical)")
    print("  dropouts                          {:>6}  (half 1: {}, half 2: {})".format(
        len(records), sum(r["half"] == 1 for r in records), sum(r["half"] == 2 for r in records)))
    near = [r for r in records if r["ball_dist"] == r["ball_dist"] and r["ball_dist"] <= 3.0]
    print("  of which within 3 m of the ball   {:>6}".format(len(near)))
    for name in ("interpolated", "same slot back", "new slot", "timeout", "half end"):
        print("  ended: {:<26} {:>6}".format(name, sum(r["resolution"] == name for r in records)))
    print("  bridged with broadcast frames     {:>6}  ({} frames filled from the broadcast)".format(
        sum(r["broadcast_frames"] > 0 for r in records), sum(r["broadcast_frames"] for r in records)))
    prio = [r for r in records if r["priority_frames"]]
    print("  rule 6.b broadcast over the line  {:>6}  of {} interpolated gaps ({} frames; "
          "{} gaps with a broadcast relink)".format(
              len(prio), sum(r["resolution"] == "interpolated" for r in records),
              sum(r["priority_frames"] for r in prio), sum(r["broadcast_relinks"] > 0 for r in records)))
    near = [r for r in records if r["lost"] <= 11409 < r["end"]]
    print("  check 1H frame 11409: gaps spanning it {}, rule 6.b frames in them {}".format(
        len(near), sum(r["priority_frames"] for r in near)))
    live = ctx.alignment["state_t"] == align.STATE_LIVE
    print("  live frames where the broadcast   {:5.1f}%  has an object tactical lacks "
          "(mostly not players -- used only to fill dropouts)".format(
              100.0 * r6["unmatched_broadcast"][live].mean() if live.any() else float("nan")))
    if r6["overflow"] or r6["ghosts_dropped"]:
        print("  capacity: {} frames over the extra-object cap, {} ghosts with no free column".format(
            r6["overflow"], r6["ghosts_dropped"]))

    columns = ["lost_frame", "match_clock", "half", "slot", "team", "x_m", "y_m", "ball_dist_m",
               "resolution", "relinked_slot", "bridged_frames", "broadcast_frames",
               "broadcast_priority_frames", "broadcast_relinks"]
    with open(DROPOUTS_CSV, "w", newline="\n") as handle:
        handle.write(",".join(columns) + "\n")
        for r in records:
            handle.write(",".join(str(v) for v in (
                r["lost"], match_clock(r["lost"]), r["half"], r["slot"] + 1, "HA"[r["team"]],
                "{:.1f}".format(r["x"]), "{:.1f}".format(r["y"]),
                "" if r["ball_dist"] != r["ball_dist"] else "{:.1f}".format(r["ball_dist"]),
                r["resolution"], r["relinked_slot"] + 1 if r["relinked_slot"] >= 0 else "",
                r["bridged"], r["broadcast_frames"], r["priority_frames"],
                r["broadcast_relinks"])) +"\n")
    print("  instances -> {}".format(os.path.relpath(DROPOUTS_CSV, ROOT)))


def far_touch_report(events, gt_labels):
    """Far touches (credited player 2-4 m from the ball): how many, and on the public window how
    often the credited team matches the GT label at that frame, next to plain touches -- if far
    touches agree much less often, the wider radius is adding noise."""
    print("\nfar touches (rule 1, {}-{} m)".format(rules.TOUCH_CLOSE, rules.TOUCH_RADIUS))
    graded = gt_labels != "X"
    for kind in ("touch", "far touch"):
        evs = [(t, who) for t, who, k, _ in events if k == kind and who >= 0]
        public = [(t, who) for t, who in evs if t < len(gt_labels) and gt_labels[t] in ("H", "A")]
        agree = sum(gt_labels[t] == "HA"[who] for t, who in public)
        print("  {:<10} {:>5} in the match; public window {:>3} on live GT frames, team agrees "
              "with GT {:>3} ({:.0%})".format(kind, len(evs), len(public), agree,
                                               agree / len(public) if public else float("nan")))
    near = [(t, who, k) for t, who, k, _ in events if abs(t - 11551) <= 15 and "touch" in k]
    print("  check 1H 11551: " + (", ".join("{:+d} {} {}".format(t - 11551, k, "HA"[who] if who >= 0 else
                                            {-1: "nobody", -2: "in doubt"}[who]) for t, who, k in near) or "none"))


def ref_delay_lag(ctx, gt_labels):
    """Rule 5 diagnostic: GT dead-ball starts minus our out-of-bounds onsets, public window."""
    _, _, evidence = rules.dead_intervals(ctx, 0, False)
    onsets = np.array([s for s, _ in evidence["out of bounds"]])
    graded = gt_labels != "X"
    dead = (gt_labels == "D") & graded
    starts = [t for t in np.flatnonzero(dead[1:] & ~dead[:-1]) + 1 if graded[t - 1]]
    lags = []
    for t in starts:
        near = onsets[(onsets >= t - 10 * data.FPS) & (onsets <= t + 2 * data.FPS)]
        if len(near):
            lags.append((t - near[np.argmin(np.abs(near - t))]) / data.FPS)
    print("\nrule 5: GT dead-ball start minus ball-out onset, public window: {}".format(
        "n={}, median {:+.2f} s, range {:+.2f} .. {:+.2f} s".format(
            len(lags), float(np.median(lags)), min(lags), max(lags)) if lags else "no pairs"))


def dead_evidence_vs_gt(ctx, gt_labels, evidence):
    """How each dead-ball signal lines up with the public GT's D frames, and its full-match share.

    precision = share of flagged graded frames the GT calls D; recall = share of GT D frames
    flagged. Five minutes, ~1,500 D frames: a direction, not a measurement.
    """
    graded_gt = gt_labels != "X"
    gt_dead = graded_gt & (gt_labels == "D")
    graded = ctx.half_of > 0
    print("\ndead-ball signals vs public GT D frames (precision / recall), and full-match share")
    for name, items in evidence.items():
        mask = np.zeros(ctx.tac.n, dtype=bool)
        for s, e in items:
            mask[s:e] = True
        flagged = mask & graded_gt
        precision = gt_dead[flagged].mean() if flagged.any() else float("nan")
        recall = mask[gt_dead].mean() if gt_dead.any() else float("nan")
        print("  {:<32} {:>6.0%} / {:<6.0%}   {:5.1f}% of graded frames, {} stretches".format(
            name, precision, recall, share(mask, graded), len(items)))


def score_table(rows, gt_labels):
    try:
        import possession_metrics as pm
    except ImportError as error:
        print("\nscoring skipped ({}); pip install -r requirements.txt".format(error))
        return
    gt_codes = pm.label_codes(gt_labels)
    print("\npublic window (17 possessions; one possession is ~0.06 spellF1)")
    print("  {:<34} {:>8} {:>8} {:>6}".format("", "macroF1", "spellF1", "poss"))
    for name, labels in rows:
        result = pm.evaluate(gt_codes, pm.label_codes(labels))
        print("  {:<34} {:>8.4f} {:>8.4f} {:>6}".format(
            name, result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"]))


def debug_payload(name, ctx, detail):
    al = ctx.alignment
    uncertain = []
    left = ctx.uncertain[rules.RESOLVED]
    for slot in range(left.shape[1]):
        uncertain += [[slot, s, e] for s, e in rules.runs(left[:, slot])]
    return {
        "name": name,
        "events": [[t, who, KIND_CODES[kind], slot] for t, who, kind, slot in detail["events"]],
        "dead": [[s, e, reason] for s, e, reason in detail["dead"]],
        "evidence": {k: [[s, e] for s, e in v] for k, v in detail["evidence"].items()},
        "broadcast_state": [[s, e, v] for s, e, v in rules.value_runs(al["state_t"], ignore=0)],
        "broadcast_states": {str(k): v for k, v in align.STATE_NAMES.items()},
        "gk_fixes": [list(f) for f in ctx.gk_fixes],
        "camera_fixes": [list(f) for f in ctx.camera_fixes],
        "crosscam": [list(o) for o in ctx.crosscam_overrides],
        "static": [[slot, s, e] for slot in range(ctx.static.shape[1])
                   for s, e in rules.runs(ctx.static[:, slot])],
        "carries": [[s, e, int(ctx.team[rules.RESOLVED][s - 1, slot])] for s, e, slot in ctx.carries],
        # Ball direction changes, player or not: [frame, angle deg (-1: too slow to say), dv m/s,
        # x, y in metres of the smoothed ball].
        "turns": [[t, round(angle), round(dv, 1), round(float(ctx.kin["pos"][t, 0]), 1),
                   round(float(ctx.kin["pos"][t, 1]), 1)] for t, angle, dv in ctx.turns
                  if np.isfinite(ctx.kin["pos"][t]).all() and np.isfinite(dv)],
        "uncertain": uncertain,
        "alignment": {"flip": [bool(v) for v in al["flip"]],
                      "chain": [[int(b0), int(b1), int(o)] for b0, b1, o, _ in al["chain"]],
                      # broadcast - tactical per half, what the viewer links the feeds with
                      "half_offsets": [-int(o) for o in al["half_offsets"]] if align.HALF_OFFSETS else None},
        "n_slots": int(ctx.tac.players.shape[1]),
        # Rule 16: [raw tactical slot, start, stop, column] -- the piece of that slot's track from
        # start to stop belongs to the person in `column` (a raw slot, or a new column after them);
        # and the team of every column.
        "moves": [list(m) for m in ctx.moves],
        "column_team": [int(t) for t in ctx.tac.team],
        # Rule 6: one row per dropout, and its bridged path as [start, team, slot, [x, y, kind,
        # ...]] with kind 1 = broadcast fill, 2 = ghost, 3 = interpolated between the last and
        # next known position. Paths are contiguous from start.
        "dropouts": [[r["lost"], r["end"], r["slot"], r["team"], r["resolution"],
                      None if r["ball_dist"] != r["ball_dist"] else round(r["ball_dist"], 1),
                      r["broadcast_frames"], r["priority_frames"], r["broadcast_relinks"]] for r in (ctx.rule6["records"] if ctx.rule6 else [])],
        "ghost_tracks": [[path[0][0], r["team"], r["slot"],
                          [v for _, x, y, k in path for v in (round(x, 1), round(y, 1), k)]]
                         for r, path in (zip(ctx.rule6["records"], ctx.rule6["tracks"])
                                         if ctx.rule6 else []) if path],
    }


def kalman_payload(kctx, events):
    """Rule 13 for the viewer: the smoothed ball per tactical frame as decimetres (-1: none) and
    its std in centimetres, negative where the ball was not seen (bridged across a gap); the
    touches found on it; and every ball gap with the kick found in it, if any."""
    kin = kctx.kin
    known = ~np.isnan(kin["pos"][:, 0])
    dm = np.where(known[:, None], np.round(np.nan_to_num(kin["pos"]) * 10), -1).astype(int)
    std = np.where(known, np.round(np.nan_to_num(kin["pos_std"]) * 100), 0).astype(int)
    std = np.where(kin["seen"], std, -np.maximum(std, 1))
    return {
        "x": dm[:, 0].tolist(), "y": dm[:, 1].tolist(), "std": std.tolist(),
        "touches": [[t, who, KIND_CODES[kind], slot, int(not kin["seen"][t])]
                    for t, who, kind, slot in events if kind in ("touch", "far touch")],
        "gaps": [[lo, hi + 1, kink, int(kicked)] for lo, hi, kink, kicked in kin["gaps"]],
    }


def camera_balls_payload(kctx2):
    """Rule 13.b's inputs for the viewer, per tactical frame in decimetres (-1: none): the tactical
    ball as exported and the broadcast ball on the tactical pitch (live frames only), plus what
    the two-camera filter's forward pass did with them (kalman._run2's bit mask: 1 tactical, 2
    broadcast taken; 4 tactical, 8 broadcast rejected)."""
    def dm(ball):
        known = ~np.isnan(ball[:, 0])
        return np.where(known[:, None], np.round(np.nan_to_num(ball) * 10), -1).astype(int)
    t, b = dm(kctx2.tac.ball), dm(kctx2.ball_b)
    return {"tx": t[:, 0].tolist(), "ty": t[:, 1].tolist(), "bx": b[:, 0].tolist(), "by": b[:, 1].tolist(),
            "used": kctx2.kin["used"].astype(int).tolist()}


def broadcast_camera_payload(alignment, bro):
    """Step 1 (model/cameras.py) for the viewer's broadcast panel: runs of broadcast frames as
    [start, stop, class]: 0 the main camera, 1 another camera (live and precise, but looking from
    elsewhere), 2 replay or not matched to the tactical players, 3 imprecise homography."""
    H = data.homography_matrices("broadcast")
    cams = cameras.recover(H)
    err = data.homography_errors("broadcast")
    live = cameras.broadcast_live(alignment, bro.n)
    main, _, _ = cameras.main_camera(H, cams, err, live)
    valid = bro.valid_h
    cls = np.full(bro.n, -1, dtype=np.int64)
    cls[valid] = 3
    cls[valid & ~live] = 2
    with np.errstate(invalid="ignore"):
        cls[valid & live & cams["ok"] & (err <= cameras.ERR_MAX)] = 1
    cls[main] = 0
    return [[s, e, v] for s, e, v in rules.value_runs(cls)]


def main():
    print("data: {}".format(data.DATA_DIR))
    tac = data.load("tactical")
    bro = data.load("broadcast")
    alignment = load_alignment(tac, bro)
    H = data.homography_matrices("tactical")
    print("  building rule context ...")
    ctx = rules.Context(tac, bro, alignment)
    print("  rule 6: finding dropouts ...")
    ctx.add_rule6(H)

    graded = ctx.half_of > 0
    diagnostics(ctx, graded)
    rule6_report(ctx)
    print("  context on the Kalman-smoothed ball (rule 13) ...")
    kctx = rules.Context(tac, bro, alignment, kalman=True)
    kctx.add_rule6(H)
    kalman_report(ctx, kctx, graded)
    print("  context on the two-camera Kalman ball (rule 13.b) ...")
    kctx2 = rules.Context(tac, bro, alignment, kalman="both")
    kctx2.add_rule6(H)
    two_camera_report(kctx, kctx2, graded)
    print("  context on the 3D ball (rule 17) ...")
    kctx3 = rules.Context(tac, bro, alignment, kalman="3d")
    kctx3.add_rule6(H)
    three_d_report(kctx3, graded)
    print("  rule 14: HMM ...")
    hmm_labels, hmm_detail = hmm.predict(kctx, rules.V1, with_posterior=True)
    hmm_report(hmm_labels, hmm_detail, graded)
    low = hmm_detail["confidence"] < hmm.LOW_CONFIDENCE

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for name, cfg in VARIANTS:
        labels, detail = ctx.predict(cfg)
        write_submission(os.path.join(RESULTS_DIR, name + ".csv"), labels)
        reasons = {}
        for s, e, reason in detail["dead"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        print("\n{}: dead {:.1f}% of graded frames; dead stretches by trigger: {}".format(
            name, share(labels == "D", graded),
            ", ".join("{} {}".format(k, v) for k, v in sorted(reasons.items()))))
        rows.append((name, labels))
        if name == "rules_v1":
            detail["evidence"]["HMM label < {:.0%} sure (14)".format(hmm.LOW_CONFIDENCE)] = rules.runs(low)
            v1_events = detail["events"]
            local = detail["evidence"]["local stillness"]
            mask = np.zeros(len(labels), dtype=bool)
            for s, e in local:
                mask[s:e] = True
            quiet = np.zeros(len(labels), dtype=bool)
            for s, e in detail["evidence"]["players mostly still"]:
                quiet[s:e] = True
            added = detail["evidence"]["dead extended (12)"]
            print("  rule 12 players mostly still {:.1f}% of graded frames; {} extensions adding {:.0f} s "
                  "(median {:.1f} s)".format(share(quiet, graded), len(added),
                                             sum(e - s for s, e in added) / data.FPS,
                                             float(np.median([(e - s) / data.FPS for s, e in added])) if added else 0.0))
            at = 118553
            print("  rule 11 local stillness {} stretches, {:.1f}% of graded frames; check 2H frame {}: {}".format(
                len(local), share(mask, graded), at,", ".join("{:+.1f} s .. {:+.1f} s".format((s - at) / data.FPS, (e - at) / data.FPS)
                              for s, e in local if s - 10 * data.FPS <= at < e + 10 * data.FPS) or "none within 10 s"))
            ov = ctx.crosscam_overrides
            print("  rule 10 broadcast overruled {} events ({} touches, {} far touches, {} controls); "
                  "check 1H 11415: {}".format(
                      len(ov), sum(o[1] == "touch" for o in ov), sum(o[1] == "far touch" for o in ov),
                      sum(o[1] == "control" for o in ov),
                      ", ".join("{:+d} {} {}->{}".format(o[0] - 11415, o[1], "HA"[o[2]] if o[2] >= 0 else o[2],
                                                         "HA"[o[3]]) for o in ov if abs(o[0] - 11415) <= 40) or "none"))
            path = os.path.join(data.CACHE_DIR, name + "_debug.json")
            with open(path, "w") as handle:
                payload = debug_payload(name, ctx, detail)
                payload["kalman"] = kalman_payload(kctx, hmm_detail["events"])
                payload["kalman2"] = kalman_payload(kctx2, kctx2.events(rules.V1))
                payload["camera_balls"] = camera_balls_payload(kctx2)
                payload["kalman3"] = kalman_payload(kctx3, kctx3.events(rules.V1))
                payload["kalman3"]["z"] = np.round(kctx3.kin["z"] * 100).astype(int).tolist()
                payload["kalman3"]["flights"] = [[int(s), int(e), round(p, 1), round(r, 2), k]
                                                 for s, e, p, r, k in kctx3.kin["flights"]]
                payload["broadcast_camera"] = broadcast_camera_payload(alignment, bro)
                json.dump(payload, handle, separators=(",", ":"))
            print("  viewer debug -> {}".format(os.path.relpath(path, ROOT)))
    write_submission(os.path.join(RESULTS_DIR, "rules_hmm.csv"), hmm_labels)
    rows.append(("rules_hmm", hmm_labels))
    rows.append(("v1 on the Kalman ball (13)", kctx.predict(rules.V1)[0]))
    hmm2_labels = hmm.predict(kctx2, rules.V1)[0]
    write_submission(os.path.join(RESULTS_DIR, "rules_hmm_2cam.csv"), hmm2_labels)
    rows.append(("rules_hmm_2cam (13.b)", hmm2_labels))
    rows.append(("v1 on the two-camera ball (13.b)", kctx2.predict(rules.V1)[0]))
    hmm3_labels, hmm3_detail = hmm.predict(kctx3, rules.V1)
    write_submission(os.path.join(RESULTS_DIR, "rules_hmm_3d.csv"), hmm3_labels)
    rows.append(("rules_hmm_3d (17)", hmm3_labels))
    rows.append(("hmm_3d without the height gate", hmm.predict(kctx3, dict(rules.V1, height=False))[0]))
    rows.append(("v1 on the 3D ball (17)", kctx3.predict(rules.V1)[0]))
    window = hmm3_labels[12635:12935]
    changes = [0] + [i for i in range(1, len(window)) if window[i] != window[i - 1]]
    print("\nrule 17 check 1H 12785 (1:04 airball, offside called 1:06): rules_hmm_3d " + "  ".join(
        "{:+.1f}s {}".format((12635 + i - 12785) / data.FPS, window[i]) for i in changes))
    contexts = {"ctx": ctx, "kctx": kctx}
    for name, which, change in HMM_ABLATIONS:
        rows.append((name, hmm.predict(contexts[which], dict(rules.V1, **change))[0]))
    for name, change in ABLATIONS:
        rows.append((name, ctx.predict(dict(rules.V1, **change))[0]))
    print("  ablation: context without rule 8 ball fusion ...")
    raw = rules.Context(tac, bro, alignment, fuse=False)
    raw.add_rule6(H)
    rows.append(("v1 without rule 8 ball fusion", raw.predict(rules.V1)[0]))

    import csv
    with open(PUBLIC_GT, newline="") as handle:
        gt_labels = np.array([row["label"].strip().upper() for row in csv.DictReader(handle)])
    far_touch_report(v1_events, gt_labels)
    ref_delay_lag(ctx, gt_labels)
    dead_evidence_vs_gt(ctx, gt_labels, rules.dead_intervals(ctx, rules.REF_DELAY, "no_homography")[2])
    score_table(rows, gt_labels)
    print("\nwrote {} -- score with: py evaluation/run_evaluation.py".format(
        ", ".join("evaluation/results/{}.csv".format(n) for n in [v for v, _ in VARIANTS] + ["rules_hmm", "rules_hmm_2cam", "rules_hmm_3d"])))


if __name__ == "__main__":
    main()
