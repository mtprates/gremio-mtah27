"""The HMM's tuning stage on external matches: clean and noisy copies of public games, a parallel
scorer, and the searches (docs/VALUES.md, "HMM tuning stage on external matches").

    .venv/Scripts/python model/tune_external.py baseline            # today's values, every copy
    .venv/Scripts/python model/tune_external.py search [--set tune]  # coordinate search on IDSSE
    .venv/Scripts/python model/tune_external.py measure              # dead-ball signals on IDSSE
    .venv/Scripts/python model/tune_external.py holdout '<json>'     # one candidate on Metrica too
    .venv/Scripts/python model/tune_external.py export               # TRIED's labels for the viewer

A candidate is a JSON dict: hmm module constants by name (e.g. {"POSSESSION_S": 6,
"EVENT_LLR": {"touch": 5, ...}}), plus "cfg": {...} for extra rule switches on top of hmm_sides
(model/external_eval.py: iterate.py's default HMM switches less air_touch).

Sets: "tune" is the 7 IDSSE matches, each clean and noisy (14 copies); "holdout" is the 3 Metrica
games, clean and noisy. The objective is the mean spellF1 over the tune copies. Every result is
appended to validation/tuning.csv (aggregates only; public data only, never BAR-ATH).

The noise (noisy()) mimics our detector's input on BAR-ATH as measured in aggregate: the ball
disappears in runs whose start depends on how close the nearest player is, with a heavy-tailed
length; small jitter on the ball and the players; players missing 1/22 of the time.
"""

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import external_eval as ee  # noqa: E402
import rules  # noqa: E402

# Rule switches on module constants that the cached contexts are built with (e.g. rules.TOUCH_ECHO):
# TUNE_RULES='{"TOUCH_ECHO": true}', read here so the worker processes get them too.
for _k, _v in json.loads(os.environ.get("TUNE_RULES", "{}")).items():
    setattr(rules, _k, _v)

# Params (docs/VALUES.md, "HMM tuning stage on external matches")
WORKERS = 8
# Our unseen-run lengths on BAR-ATH, frames: (lo, hi inclusive, count)
UNSEEN_RUNS = ((1, 2, 2683), (3, 6, 949), (7, 12, 446), (13, 24, 458), (25, 49, 395), (50, 124, 270),
               (125, 249, 62), (250, 500, 16))
# Nearest player's distance (m, upper edges) -> our unseen share of frames there
UNSEEN_EDGES = (1.0, 2.0, 4.0, 8.0, np.inf)
UNSEEN_SHARE = (0.30, 0.25, 0.29, 0.26, 0.43)
BALL_JITTER = 0.10          # m per axis
PLAYER_OFFSET = 0.10        # m per axis, AR(1) ...
PLAYER_TAU = 25             # ... with this time constant, frames
PLAYER_MISSING = 1 / 22     # share of player-frames missing ...
PLAYER_GAP = 25             # ... in runs of this mean length, frames (a guess)
SETS = {"tune": [g for g in ee.all_games() if g.startswith("idsse_")],
        "holdout": [g for g in ee.all_games() if g.startswith("metrica_")]}
BASE_CFG = ee.VARIANTS["hmm_sides"][2]
LOG = os.path.join(ROOT, "validation", "tuning.csv")
# What the searches tried, by name, for the viewer (export): validation/predictions/<copy>/<name>.csv
ACCEL = "POSSIBLE dead (ball accel ~0)"
SUBMITTED_CFG = dict(kick=True, restart_live=True, event_switch=True, far_clear=True, replay_neutral=True,
                     short_spells=True, short_broadcast=True)     # reproduce.MODEL_FLAGS' HMM switches (rule 58: TUNE_RULES)
BEFORE = {"POSSESSION_S": 13.4, "LIVE_S": 30.0, "DEAD_S": 20.0, "BASE_DEAD": 0.20,   # model C's values on 2026-11-16
          "MEASURED": ["broadcast no homography", "players mostly still", ACCEL],
          "SIGNALS": {"broadcast no homography": [0.85, 0.76], "players mostly still": [0.58, 0.12], ACCEL: [0.39, 0.30],
                      "stillness": [0.5, 0.0]}}
TRIED = {
    "submitted": {"cfg": SUBMITTED_CFG},   # the frozen submission; export it with TUNE_RULES='{"TOUCH_ECHO": true}'
    "before": BEFORE,                       # model C of 2026-11-16 (export without TUNE_RULES)
    "today": {},
    "stage1": {"CORR_FRAMES": 6, "EVENT_LLR": {"far touch": 2.0, "touch": 4}, "POSSESSION_S": 10},
    "stage2": {"CORR_FRAMES": 6, "EVENT_LLR": {"far touch": 2.0, "touch": 4}, "POSSESSION_S": 10,
               "cfg": {"signals": {ACCEL: [0.5, 0.5]}}},
    "r1": {"variant": "rules_r1"},
}


# -- noise ---------------------------------------------------------------------------------------

def _renewal(rng, n, hazard, lengths, weights):
    """(n,) bool, True where off: off-runs start with per-frame probability hazard (n,) and last a
    length drawn from (lengths, weights)."""
    off = np.zeros(n, dtype=bool)
    candidates = np.flatnonzero(rng.random(n) < hazard)
    ks = rng.choice(len(lengths), size=len(candidates), p=weights)
    u = rng.random(len(candidates))
    free = 0                    # first frame after the last run (and one seen frame)
    for t, k, v in zip(candidates.tolist(), ks.tolist(), u.tolist()):
        if t < free:
            continue
        lo, hi = lengths[k]
        length = lo + int(v * (hi - lo + 1))
        off[t:t + length] = True
        free = t + length + 1
    return off


def noisy(game, seed):
    """A copy of game with our detector's kind of noise (module docstring)."""
    rng = np.random.default_rng(seed)
    tac = game.tac
    players, ball = tac.players.copy(), tac.ball.copy()
    n, slots = players.shape[:2]
    # Ball: runs of unseen frames, more likely to start where the share is higher.
    with np.errstate(invalid="ignore"):
        d = np.nanmin(np.linalg.norm(players - ball[:, None, :], axis=2), axis=1)
    d = np.where(np.isfinite(d), d, np.inf)
    lengths = [(lo, hi) for lo, hi, _ in UNSEEN_RUNS]
    counts = np.array([c for _, _, c in UNSEEN_RUNS], dtype=float)
    weights = counts / counts.sum()
    mean_len = float(sum(w * (lo + hi) / 2 for (lo, hi), w in zip(lengths, weights)))
    share = np.array(UNSEEN_SHARE)[np.searchsorted(UNSEEN_EDGES, d)]
    hazard = share / ((1 - share) * mean_len)
    off = _renewal(rng, n, hazard, lengths, weights)
    ball = ball + rng.normal(0, BALL_JITTER, ball.shape).astype(np.float32)
    ball[off] = np.nan
    # Players: a slow offset each, and runs missing.
    from scipy.signal import lfilter
    a = np.exp(-1.0 / PLAYER_TAU)
    eps = rng.normal(0, PLAYER_OFFSET * np.sqrt(1 - a * a), (n, slots, 2))
    players = players + lfilter([1.0], [1.0, -a], eps, axis=0).astype(np.float32)
    h = PLAYER_MISSING / ((1 - PLAYER_MISSING) * PLAYER_GAP)
    for s in range(slots):
        gone = _renewal(rng, n, np.full(n, h), [(1, 2 * PLAYER_GAP - 1)], [1.0])
        players[gone, s] = np.nan
    return ee.Game(game.name + "~noisy", ee._recording(players, ball, tac.team), game.halves, game.kickoffs,
                   game.pitch, game.gt)


def copy_names(set_name):
    return [g + suffix for g in SETS[set_name] for suffix in ("", "~noisy")]


def load_copy(name):
    base = name.split("~")[0]
    game = ee.load_game(base)
    if name.endswith("~noisy"):
        game = noisy(game, seed=sum(map(ord, base)))
    return game


# -- scoring -------------------------------------------------------------------------------------

def _apply(params):
    """Set hmm's module constants from params; return what to restore."""
    import hmm
    saved = {}
    for k, v in params.items():
        if k in ("cfg", "variant"):
            continue
        saved[k] = getattr(hmm, k)
        setattr(hmm, k, dict(getattr(hmm, k), **v) if isinstance(v, dict) else v)
    return saved


def score_copy(task):
    """(copy name, [candidates]) -> [result dict] for each candidate, on that copy. With a third
    item, a folder, each candidate's labels are also written there as <name>.csv (export)."""
    import contextlib
    import io
    import hmm
    import possession_metrics as pm
    import rules
    name, candidates = task[:2]
    folder = task[2] if len(task) > 2 else None
    game = load_copy(name)
    out = []
    with ee.use_game(game), contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        bro, al = ee.no_broadcast(game.tac)
        contexts = {}

        def context(kalman):
            if kalman not in contexts:
                contexts[kalman] = rules.Context(game.tac, bro, al, kalman=kalman, identity=False, sticky=False)
            return contexts[kalman]

        gt = pm.label_codes(game.gt)
        for params in candidates:
            saved = _apply({k: v for k, v in params.items() if k != "name"})
            try:
                if "variant" in params:     # a reference: external_eval's variant as it is
                    use_kalman, decoder, cfg = ee.VARIANTS[params["variant"]]
                    ctx = context(use_kalman)
                    labels = (ctx.predict(cfg) if decoder == "rules" else hmm.predict(ctx, cfg))[0]
                else:
                    cfg = dict(BASE_CFG, **params.get("cfg", {}))
                    labels = hmm.predict(context(True), cfg)[0]
            finally:
                for k, v in saved.items():
                    setattr(hmm, k, v)
            if folder:
                os.makedirs(os.path.join(folder, name), exist_ok=True)
                ee.write_labels(os.path.join(folder, name, params["name"] + ".csv"), labels)
            r = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)
            out.append({k: r[k] for k in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive",
                                          "n_pred_spells", "n_gt_spells")})
    return name, out


def score(candidates, set_name="tune", pool=None):
    """[summary dict] per candidate: mean over the set's copies, clean / noisy / both."""
    names = copy_names(set_name)
    own = pool is None
    pool = pool or ProcessPoolExecutor(WORKERS)
    try:
        results = dict(pool.map(score_copy, [(n, candidates) for n in names]))
    finally:
        if own:
            pool.shutdown()
    out = []
    for i, params in enumerate(candidates):
        row = {"set": set_name, "params": json.dumps(params, sort_keys=True)}
        for part, keep in (("clean", lambda n: "~" not in n), ("noisy", lambda n: n.endswith("~noisy")),
                           ("all", lambda n: True)):
            rs = [results[n][i] for n in names if keep(n)]
            for k in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive"):
                row["{}_{}".format(part, k)] = float(np.mean([r[k] for r in rs]))
            row[part + "_spells_ratio"] = float(np.sum([r["n_pred_spells"] for r in rs]) /
                                                np.sum([r["n_gt_spells"] for r in rs]))
        out.append(row)
    _log(out)
    return out


FIELDS = ["time", "set", "params"] + ["{}_{}".format(p, k) for p in ("all", "clean", "noisy") for k in (
    "spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive", "spells_ratio")]


def _log(rows):
    new = not os.path.isfile(LOG)
    with open(LOG, "a", newline="") as handle:
        w = csv.DictWriter(handle, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        stamp = time.strftime("%Y-%m-%d %H:%M")
        for r in rows:
            w.writerow(dict({k: "{:.4f}".format(v) if isinstance(v, float) else v for k, v in r.items()}, time=stamp))


def show(rows, label=None):
    for r in rows:
        print("  spellF1 {all_spell_f1_mean:.4f} (clean {clean_spell_f1_mean:.4f} noisy {noisy_spell_f1_mean:.4f})"
              "  macroF1 {all_macro_f1:.4f}  alive/dead {all_alive_dead_f1:.4f}  team {all_team_acc_both_alive:.4f}"
              "  spells x{all_spells_ratio:.2f} (clean x{clean_spells_ratio:.2f} noisy x{noisy_spells_ratio:.2f})"
              .format(**r) + "  " + (label or r["params"]))


# -- searches ------------------------------------------------------------------------------------

def merge(base, change):
    out = json.loads(json.dumps(base))
    for k, v in change.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = dict(out[k], **v)
        else:
            out[k] = v
    return out


# (name, [changes]): one coordinate of the search, each change merged into the current best.
def coordinates():
    touch_scales = [{"EVENT_LLR": {"touch": t, "far touch": t / 2}} for t in (2, 3, 4, 5, 6, 8)]
    return [
        ("POSSESSION_S", [{"POSSESSION_S": v} for v in (3, 5, 7, 10, 13.4, 20)]),
        ("touch", touch_scales),
        ("control", [{"EVENT_LLR": {"control": v}} for v in (2, 3, 4, 6, 8)]),
        ("carry", [{"EVENT_LLR": {"carry": v}} for v in (2, 4, 6, 8)]),
        ("touch_seen", [{"cfg": {"touch_seen": False}}] +
         [{"cfg": {"touch_seen": True}, "TOUCH_SEEN_POWER": p} for p in (0.5, 1, 2, 4)]),
        ("touch_once", [{"cfg": {"touch_once": v}} for v in (False, True)]),
        ("passes", [{"cfg": {"passes": False}}] + [{"cfg": {"passes": True}, "PASS_LLR": v} for v in (2, 4, 6)]),
        ("CORR_FRAMES", [{"CORR_FRAMES": v} for v in (6, 12, 25)]),
        ("PROX_CLIP", [{"PROX_CLIP": v} for v in (1.5, 3, 5)]),
    ]


def coordinates_dead():
    """Stage 2: the price of a dead stretch and the weight of the measured dead-ball signals.
    'accel' is (precision, recall) as hmm.SIGNALS: the earlier value (0.39, 0.30), IDSSE's
    own (0.82, 0.95; probably flattered by the provider parking the ball at stoppages, see
    docs/VALUES.md) and points between; 'still' likewise for players mostly still."""
    accel = "POSSIBLE dead (ball accel ~0)"
    still = "players mostly still"
    return [
        ("LIVE_S", [{"LIVE_S": v} for v in (15, 30, 60, 120, 240)]),
        ("DEAD_S", [{"DEAD_S": v} for v in (10, 20, 40)]),
        ("accel", [{"cfg": {"signals": {accel: v}}} for v in ([0.39, 0.30], [0.39, 0.15], [0.3, 0.3],
                                                             [0.5, 0.5], [0.82, 0.95])]),
        ("still", [{"cfg": {"signals": {still: v}}} for v in ([0.58, 0.12], [0.8, 0.2], [0.97, 0.38])]),
    ] + coordinates()[:3]


def search(start=None, passes=2, min_gain=0.002, coords=None):
    coords = coords or coordinates
    best = start or {}
    with ProcessPoolExecutor(WORKERS) as pool:
        best_score = score([best], pool=pool)[0]
        print("start"); show([best_score])
        for p in range(passes):
            changed = False
            for name, changes in coords():
                cands = [merge(best, c) for c in changes]
                t0 = time.time()
                rows = score(cands, pool=pool)
                print("\npass {} {} ({:.0f} s)".format(p + 1, name, time.time() - t0))
                show(rows)
                top = max(rows, key=lambda r: r["all_spell_f1_mean"])
                if top["all_spell_f1_mean"] > best_score["all_spell_f1_mean"] + min_gain:
                    best, best_score, changed = json.loads(top["params"]), top, True
                print("  best so far {:.4f}: {}".format(best_score["all_spell_f1_mean"], json.dumps(best, sort_keys=True)))
            if not changed:
                break
    return best, best_score


def measure():
    """Dead-ball signals' (precision, recall) on the tune set's labels, as hmm.SIGNALS defines them,
    and the dead share and mean live / dead / possession stretch lengths."""
    import contextlib
    import io
    import hmm
    import rules
    tot = {}
    stretch = {"live": [], "dead": [], "poss": []}
    for name in copy_names("tune"):
        game = load_copy(name)
        with ee.use_game(game), contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            bro, al = ee.no_broadcast(game.tac)
            ctx = rules.Context(game.tac, bro, al, kalman=True, identity=False, sticky=False)
            masks, covered, _ = hmm._masks(ctx, BASE_CFG)
        graded = game.gt != "X"
        dead = game.gt == "D"
        part = "noisy" if "~" in name else "clean"
        for sig, m in masks.items():
            cov = covered.get(sig, np.ones_like(m)) & graded
            t = tot.setdefault((part, sig), np.zeros(4))
            t += [(m & dead & cov).sum(), (m & cov).sum(), (dead & cov).sum(), cov.sum()]
        if part == "clean":
            for s, e in rules.runs(graded & ~dead):
                stretch["live"].append(e - s)
            for s, e in rules.runs(dead & graded):
                stretch["dead"].append(e - s)
            for k in "HA":
                stretch["poss"] += [e - s for s, e in rules.runs(game.gt == k)]
    for (part, sig), (tp, fired, dead_n, cov) in sorted(tot.items()):
        print("  {:6} {:32} precision {:.3f} recall {:.3f}  (fires on {:.1%}; dead share {:.3f})".format(
            part, sig, tp / max(fired, 1), tp / max(dead_n, 1), fired / max(cov, 1), dead_n / max(cov, 1)))
    for k, v in stretch.items():
        print("  mean {} stretch {:.1f} s (n {})".format(k, np.mean(v) / 25, len(v)))


def main(argv):
    if not argv or argv[0] == "baseline":
        rows = score([{}], "tune") + score([{}], "holdout")
        show(rows[:1], "baseline, tune (IDSSE)")
        show(rows[1:], "baseline, holdout (Metrica)")
    elif argv[0] == "search":
        best, best_score = search()
        print("\nbest on tune:"); show([best_score])
        print(json.dumps(best, sort_keys=True))
    elif argv[0] == "search2":
        best, best_score = search(start=json.loads(argv[1]), coords=coordinates_dead)
        print("\nbest on tune:"); show([best_score])
        print(json.dumps(best, sort_keys=True))
    elif argv[0] == "measure":
        measure()
    elif argv[0] == "export":
        folder = os.path.join(ee.OUT_DIR, "predictions")
        keep = argv[1:] or list(TRIED)        # export [name ...]: only those
        cands = [dict(v, name="tuned_" + k) for k, v in TRIED.items() if k in keep]
        names = copy_names("tune") + copy_names("holdout")
        with ProcessPoolExecutor(WORKERS) as pool:
            for name, rows in pool.map(score_copy, [(n, cands, folder) for n in names]):
                print("  {:24}".format(name) + "  ".join("{} {:.3f} ({}/{})".format(
                    c["name"][6:], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]) for c, r in zip(cands, rows)))
    elif argv[0] == "holdout":
        cands = [{}] + [json.loads(a) for a in argv[1:]]
        for set_name in ("tune", "holdout"):
            print(set_name)
            show(score(cands, set_name))


if __name__ == "__main__":
    main(sys.argv[1:])
