"""The submitted model on the external matches, against references, as committed aggregates for the
documentation and its figures (public data only).

    .venv/Scripts/python model/readiness.py

Reads validation/predictions/<copy>/ (tune_external.py export: tuned_submitted, tuned_before, tuned_r1) and
validation/gt/<game>.csv, and writes
  validation/readiness_sets.csv    per set (IDSSE / Metrica x clean / noisy) and model: spellF1, macroF1,
                                   possessions predicted / true, alive/dead F1, team accuracy, and the share of
                                   true possessions matched (same team, IoU >= 0.5) by their length
  validation/readiness_games.csv   per clean match and model: spellF1, macroF1
"truth N s late" rows are the ground truth itself delayed by N s: a perfect labeller running late, for scale.
"""

import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
BUCKETS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9))
BUCKET_NAMES = ("<1s", "1-2s", "2-4s", "4-8s", ">8s")
MODELS = {"tuned_submitted": "submitted", "tuned_before": "before (2026-11-16)", "tuned_r1": "R1"}
LATE_S = (0.5, 1.0, 2.0)


def load(path):
    return np.array([r["label"].strip().upper() for r in csv.DictReader(open(path, newline=""))])


def spells(lab):
    out = []
    for k in "HA":
        m = np.r_[False, lab == k, False]
        d = np.flatnonzero(np.diff(m.astype(int)))
        out += [(s, e, k) for s, e in zip(d[::2], d[1::2])]
    return out


def matched_by_length(gt, pred):
    """[matched, total] per length bucket: true possessions with a predicted one of the same team at IoU >= 0.5."""
    by_team = {k: sorted((s, e) for s, e, kk in spells(pred) if kk == k) for k in "HA"}
    hit = [[0, 0] for _ in BUCKETS]
    for s, e, k in spells(gt):
        i = next(j for j, (lo, hi) in enumerate(BUCKETS) if lo <= (e - s) / FPS < hi)
        hit[i][1] += 1
        if any(min(e, pe) - max(s, ps) > 0 and (min(e, pe) - max(s, ps)) / (max(e, pe) - min(s, ps)) >= 0.5
               for ps, pe in by_team[k]):
            hit[i][0] += 1
    return hit


def late(gt, k):
    out = gt.copy()
    out[k:] = gt[:-k]
    out[:k] = gt[0]
    return out


def main():
    sets, games = {}, []
    for folder in sorted(glob.glob(os.path.join(ROOT, "validation", "predictions", "*"))):
        copy = os.path.basename(folder)
        game = copy.split("~")[0]
        gt_path = os.path.join(ROOT, "validation", "gt", game + ".csv")
        if not os.path.isfile(gt_path):
            continue
        gt = load(gt_path)
        noisy = "~" in copy
        group = ("IDSSE" if game.startswith("idsse") else "Metrica", "noisy" if noisy else "clean")
        preds = {name: load(os.path.join(folder, f + ".csv")) for f, name in MODELS.items()
                 if os.path.isfile(os.path.join(folder, f + ".csv"))}
        if not noisy:
            preds.update({"truth {:g} s late".format(s): late(gt, int(s * FPS)) for s in LATE_S})
        for name, lab in preds.items():
            g, p = gt, lab[:len(gt)]
            r = pm.evaluate(pm.label_codes(g), pm.label_codes(p), bootstrap_n=0)
            hit = matched_by_length(np.where(g == "X", "D", g), np.where(p == "X", "D", p))
            sets.setdefault(group + (name,), []).append((r, hit))
            if not noisy:
                games.append({"game": game, "model": name, "spell_f1": round(r["spell_f1_mean"], 4),
                              "macro_f1": round(r["macro_f1"], 4)})
    with open(os.path.join(ROOT, "validation", "readiness_sets.csv"), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["dataset", "copy", "model", "spell_f1", "macro_f1", "possessions_ratio", "alive_dead_f1",
                    "team_acc"] + ["matched_" + b for b in BUCKET_NAMES] + ["n_" + b for b in BUCKET_NAMES])
        for (ds, cp, name), vals in sorted(sets.items()):
            rs = [v[0] for v in vals]
            hit = np.sum([v[1] for v in vals], axis=0)
            w.writerow([ds, cp, name] + ["{:.4f}".format(np.mean([r[k] for r in rs])) for k in
                                         ("spell_f1_mean", "macro_f1")] +
                       ["{:.3f}".format(sum(r["n_pred_spells"] for r in rs) / sum(r["n_gt_spells"] for r in rs))] +
                       ["{:.4f}".format(np.mean([r[k] for r in rs])) for k in ("alive_dead_f1", "team_acc_both_alive")] +
                       ["{:.3f}".format(a / max(n, 1)) for a, n in hit] + [int(n) for _, n in hit])
    with open(os.path.join(ROOT, "validation", "readiness_games.csv"), "w", newline="") as handle:
        w = csv.DictWriter(handle, ["game", "model", "spell_f1", "macro_f1"])
        w.writeheader()
        w.writerows(games)
    print("wrote validation/readiness_sets.csv ({} rows), validation/readiness_games.csv ({} rows)".format(
        len(sets), len(games)))


if __name__ == "__main__":
    main()
