"""The tracking-only possession model on public matches that come with their own possession labels.

    .venv/Scripts/python model/external_eval.py              # every game
    .venv/Scripts/python model/external_eval.py idsse        # or one dataset / some games:
    .venv/Scripts/python model/external_eval.py metrica_1 idsse_J03WPY

Is the model right when the data is good? Clean optical tracking of every player and the ball,
no detector, no broadcast, no noise added. Two public datasets (validation/README.md):

  idsse    7 Bundesliga matches (DFL / Sportec, CC BY 4.0, Bassek et al. 2025), 25 fps, metres.
           Labels: the provider's per-frame BallStatus (0 -> D) and BallPossession (1 home, 2 away).
           Downloaded into external/idsse/raw/ (validation/README.md has the commands).
  metrica  Metrica's 3 anonymised sample games, 25 fps, 0-1 units. Labels: derived from the events
           by model/metrica_sample_gt.py (an approximation, see its docstring).

Neither labeller is CHALLENGE.md's: DFL's possession and ball status are its own definitions,
Metrica's come from events. Scores are a check that the model works on other matches, not a
measure of its BAR-ATH score.

What changes for a game that is not BAR-ATH, and nothing else:
  - the match constants the model reads (halves, kick-offs, pitch size) point at the game;
  - there is no broadcast camera: an empty one, never live, and every broadcast rule switched off
    (4, 3.a, 3.b, 6, 10, 15, 16) -- the model as it runs on the tactical tracking alone;
  - speeds are derived from positions (rules.player_speed's fallback) rather than read, so no
    provider's speed unit is assumed.

Writes validation/predictions/<game>/<variant>.csv, validation/gt/<game>.csv, validation/reports/
<game>_<variant>.json (possession_metrics.evaluate) and validation/scores.csv, and prints the
scores. Parsed tracking is cached under external/<dataset>/cache/.
"""

import contextlib
import csv
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import align  # noqa: E402
import hmm  # noqa: E402
import possession_metrics as pm  # noqa: E402
import rules  # noqa: E402
from data import AWAY, HOME, Recording  # noqa: E402

# Params
EXTERNAL = os.path.join(ROOT, "external")
IDSSE_DIR = os.path.join(EXTERNAL, "idsse")
METRICA_DIR = os.path.join(EXTERNAL, "metrica-sample")
OUT_DIR = os.path.join(ROOT, "validation")
CACHE_VERSION = 1
IDSSE_MATCHES = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
                 "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
                 "J03WR9": "DFL-COM-000002"}
METRICA_PITCH = (105.0, 68.0)   # the sample's FieldSize (game 3 metadata); games 1 and 2 are the same
# Every rule that needs the broadcast camera, off: there is none.
NO_BROADCAST = dict(broadcast_dead=False, crosscam=False, crosscheck=False, resolve=False, dropouts=False)
# name -> (Kalman ball, decoder, cfg). hmm_sides is iterate.py's "default" HMM switches less
# air_touch, which needs the detector's 3D ball.
VARIANTS = {
    "rules_r1": (False, "rules", rules.R1),
    "rules_v1": (False, "rules", dict(rules.V1, **NO_BROADCAST)),
    "rules_hmm": (True, "hmm", dict(rules.V1, **NO_BROADCAST)),
    "hmm_sides": (True, "hmm", dict(rules.V1, touch_sides=True, oob_grade=True, ref_delay=0, **NO_BROADCAST)),
}
SCORE_COLUMNS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive", "accuracy",
                 "n_gt_spells", "n_pred_spells", "gt_mean_spell_duration_s", "mean_spell_duration_s")


class Game(object):
    """One match on one timeline: tac (data.Recording, metres), halves [(half, first, last)],
    kickoffs {half: frame}, pitch (length, width) and gt (n,) labels 'H' 'A' 'D' 'X'."""

    def __init__(self, name, tac, halves, kickoffs, pitch, gt):
        self.name, self.tac, self.halves, self.kickoffs, self.pitch, self.gt = \
            name, tac, halves, kickoffs, pitch, gt


def _recording(players, ball, team):
    n, slots = players.shape[:2]
    return Recording("tactical", players.astype(np.float32), np.full((n, slots), np.nan, dtype=np.float32),
                     ball.astype(np.float32), team.astype(np.int8), np.ones(n, dtype=bool))


def _cached(path, sources, build):
    """build() -> dict of arrays, cached at path while the sources' sizes and mtimes hold."""
    stamp = json.dumps([[os.path.getsize(p), os.path.getmtime(p)] for p in sources] + [CACHE_VERSION])
    try:
        with np.load(path) as stored:
            if str(stored["stamp"]) == stamp:
                return {k: stored[k] for k in stored.files}
    except (OSError, KeyError, ValueError):
        pass
    out = build()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, stamp=np.array(stamp), **out)
    return out


# -- IDSSE (DFL) ---------------------------------------------------------------------------------

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*? TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
SECTIONS = ("firstHalf", "secondHalf")


def _idsse_paths(match):
    c = IDSSE_MATCHES[match]
    raw = os.path.join(IDSSE_DIR, "raw")
    return (os.path.join(raw, "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml".format(c, match)),
            os.path.join(raw, "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(c, match)))


def _parse_idsse(meta_path, pos_path):
    root = ET.parse(meta_path).getroot()
    home_id = next(t.get("TeamId") for t in root.iter("Team") if t.get("Role") == "home")
    env = next(root.iter("Environment"))
    length, width = float(env.get("PitchX")), float(env.get("PitchY"))
    tracks, ball = {}, {s: [] for s in SECTIONS}   # (section, person) -> (team, [(N, x, y)])
    current, is_ball = None, False
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = FRAME.match(line)
                row = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                if is_ball:
                    flags = BALL_FLAGS.search(line)
                    ball[current[0]].append(row + ((int(flags.group(1)), int(flags.group(2))) if flags else (0, -1)))
                else:
                    current[1].append(row)
            elif line.startswith("<FrameSet "):
                section, team, person = FRAMESET.match(line).groups()
                is_ball = team == "BALL"
                if section not in SECTIONS:
                    current, is_ball = (section, []), False   # extra time: none in IDSSE, skipped
                elif not is_ball:
                    current = tracks.setdefault((section, person), (HOME if team == home_id else AWAY, []))
                else:
                    current = (section,)
    # One timeline: the first half's frames, then the second's. The ball is in every frame.
    first_n = {s: min(r[0] for r in ball[s]) for s in SECTIONS}
    lengths = [max(r[0] for r in ball[s]) - first_n[s] + 1 for s in SECTIONS]
    offset = {SECTIONS[0]: 0, SECTIONS[1]: lengths[0]}
    n = sum(lengths)
    people = sorted({p for _, p in tracks})
    slot = {p: i for i, p in enumerate(people)}
    team = np.zeros(len(people), dtype=np.int8)
    players = np.full((n, len(people), 2), np.nan, dtype=np.float32)
    for (section, person), (side, rows) in tracks.items():
        team[slot[person]] = side
        a = np.array(rows, dtype=np.float64)
        players[a[:, 0].astype(np.int64) - first_n[section] + offset[section], slot[person]] = \
            np.stack([a[:, 1] + length / 2, a[:, 2] + width / 2], axis=1)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    gt = np.full(n, "X", dtype="<U1")
    for s in SECTIONS:
        a = np.array(ball[s], dtype=np.float64)
        idx = a[:, 0].astype(np.int64) - first_n[s] + offset[s]
        ball_xy[idx] = np.stack([a[:, 1] + length / 2, a[:, 2] + width / 2], axis=1)
        status, poss = a[:, 4], a[:, 3]
        gt[idx] = np.where(status == 0, "D", np.where(poss == 1, "H", np.where(poss == 2, "A", "X")))
    return {"players": players, "ball": ball_xy, "team": team, "gt": gt,
            "lengths": np.array(lengths), "pitch": np.array([length, width])}


def idsse_game(match):
    meta, pos = _idsse_paths(match)
    a = _cached(os.path.join(IDSSE_DIR, "cache", match + ".npz"), [meta, pos], lambda: _parse_idsse(meta, pos))
    n1, n2 = (int(v) for v in a["lengths"])
    return Game("idsse_" + match, _recording(a["players"], a["ball"], a["team"]),
                [(1, 0, n1 - 1), (2, n1, n1 + n2 - 1)], {1: 0, 2: n1},
                tuple(float(v) for v in a["pitch"]), a["gt"])


# -- Metrica sample ------------------------------------------------------------------------------

def _parse_metrica_csv(n):
    import pandas as pd
    folder = os.path.join(METRICA_DIR, "games", "Sample_Game_{}".format(n))
    tables = [pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(n, side)),
                          skiprows=3, header=None).to_numpy(dtype=np.float64) for side in ("Home", "Away")]
    frames = tables[0][:, 1].astype(np.int64)
    size = int(frames.max()) + 1
    xy = [t[:, 3:-2].reshape(len(t), -1, 2) for t in tables]
    players = np.full((size, xy[0].shape[1] + xy[1].shape[1], 2), np.nan, dtype=np.float32)
    players[frames] = np.concatenate(xy, axis=1)
    ball = np.full((size, 2), np.nan, dtype=np.float32)
    ball[frames] = tables[0][:, -2:]
    team = np.array([HOME] * xy[0].shape[1] + [AWAY] * xy[1].shape[1], dtype=np.int8)
    period = np.zeros(size, dtype=np.int8)
    period[frames] = tables[0][:, 0]
    return players, ball, team, period


def _parse_metrica_epts(n):
    folder = os.path.join(METRICA_DIR, "games", "Sample_Game_{}".format(n))
    root = ET.parse(os.path.join(folder, "Sample_Game_{}_metadata.xml".format(n))).getroot()
    home_id = next(root.iter("Team")).get("id")    # FIFATMA, first: home (metrica_sample_gt.py)
    side_of = {p.get("id"): HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")}
    people = []
    for c in root.iter("PlayerChannel"):
        if c.get("playerId") not in people:
            people.append(c.get("playerId"))
    slot_of_channel = {c.get("id"): people.index(c.get("playerId")) for c in root.iter("PlayerChannel")}
    specs = []   # (first frame, last frame, slot of each ;-group)
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      np.array([slot_of_channel[r] for r in refs if r.endswith("_x")])))
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    path = os.path.join(folder, "Sample_Game_{}_tracking.txt".format(n))
    with open(path) as handle:
        lines = handle.readlines()
    size = int(lines[-1].split(":")[0]) + 1
    players = np.full((size, len(people), 2), np.nan, dtype=np.float32)
    ball = np.full((size, 2), np.nan, dtype=np.float32)
    k = 0
    for line in lines:
        frame, groups, b = line.rstrip("\n").split(":")
        frame = int(frame)
        while not specs[k][0] <= frame <= specs[k][1]:
            k += 1
        values = np.array(groups.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 2)
        players[frame, specs[k][2]] = values
        ball[frame] = np.array(b.split(","), dtype=np.float32)
    period = np.zeros(size, dtype=np.int8)
    for half, key in ((1, "first_half"), (2, "second_half")):
        period[int(params[key + "_start"]):int(params[key + "_end"]) + 1] = half
    team = np.array([side_of[p] for p in people], dtype=np.int8)
    return players, ball, team, period


def metrica_game(n):
    folder = os.path.join(METRICA_DIR, "games", "Sample_Game_{}".format(n))
    sources = sorted(os.path.join(folder, f) for f in os.listdir(folder))

    def build():
        players, ball, team, period = (_parse_metrica_csv if n < 3 else _parse_metrica_epts)(n)
        scale = np.array(METRICA_PITCH, dtype=np.float32)
        return {"players": players * scale, "ball": ball * scale, "team": team, "period": period}

    a = _cached(os.path.join(METRICA_DIR, "cache", "game_{}.npz".format(n)), sources, build)
    halves = [(h, int(np.flatnonzero(a["period"] == h)[0]), int(np.flatnonzero(a["period"] == h)[-1]))
              for h in (1, 2)]
    with open(os.path.join(METRICA_DIR, "gt", "sample_game_{}.csv".format(n)), newline="") as handle:
        gt = np.array([r["label"] for r in csv.DictReader(handle)], dtype="<U1")
    tac = _recording(a["players"], a["ball"], a["team"])
    assert len(gt) == tac.n, "metrica gt and tracking lengths differ: run model/metrica_sample_gt.py"
    return Game("metrica_{}".format(n), tac, halves, {h: first for h, first, _ in halves}, METRICA_PITCH, gt)


# -- the model on another game -------------------------------------------------------------------

@contextlib.contextmanager
def use_game(game):
    """Point the match constants the model reads at module level (data.halves, KICKOFFS, PITCH,
    imported by name into rules, hmm and align) at `game` for the duration."""
    saved = []

    def put(module, name, value):
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    for module in (rules, hmm, align):
        put(module, "halves", lambda: list(game.halves))
    for module in (rules, align):
        put(module, "PITCH", game.pitch)
    put(rules, "KICKOFFS", dict(game.kickoffs))
    try:
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def no_broadcast(tac):
    """An empty broadcast camera and an alignment that never has it live."""
    nan = np.float32(np.nan)
    bro = Recording("broadcast", np.full((1, 1, 2), nan), np.full((1, 1), nan), np.full((1, 2), nan),
                    np.zeros(1, dtype=np.int8), np.zeros(1, dtype=bool))
    alignment = {"state_t": np.full(tac.n, align.STATE_NONE, dtype=np.int8),
                 "b_of_t": np.zeros(tac.n, dtype=np.int64),
                 "pairs_t": np.full(tac.players.shape[:2], -1, dtype=np.int32),
                 "flip": np.array([False, False]),
                 "offset_b": np.array([align.MISSING], dtype=np.int64)}
    return bro, alignment


def predict(game, variants):
    """{variant: labels (n,)} for one game."""
    out, contexts = {}, {}
    with use_game(game):
        bro, alignment = no_broadcast(game.tac)
        for name in variants:
            use_kalman, decoder, cfg = VARIANTS[name]
            if use_kalman not in contexts:
                contexts[use_kalman] = rules.Context(game.tac, bro, alignment, kalman=use_kalman,
                                                     identity=False, sticky=False)
            ctx = contexts[use_kalman]
            out[name] = (ctx.predict(cfg) if decoder == "rules" else hmm.predict(ctx, cfg))[0]
    return out


def write_labels(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def all_games():
    return ["metrica_{}".format(n) for n in (1, 2, 3)] + ["idsse_" + m for m in IDSSE_MATCHES]


def load_game(name):
    kind, key = name.split("_", 1)
    return metrica_game(int(key)) if kind == "metrica" else idsse_game(key)


def main(argv):
    wanted = argv or ["metrica", "idsse"]
    games = [g for g in all_games() if any(g == w or g.startswith(w + "_") for w in wanted)]
    rows = []
    for name in games:
        t0 = time.time()
        game = load_game(name)
        graded = game.gt != "X"
        shares = {k: float(np.mean(game.gt[graded] == k)) for k in "HAD"}
        print("{}: {} frames, {:.0f} graded min; gt H {H:.0%} A {A:.0%} D {D:.0%}".format(
            name, game.tac.n, graded.sum() / 25 / 60, **shares))
        write_labels(os.path.join(OUT_DIR, "gt", name + ".csv"), game.gt)
        gt_codes = pm.label_codes(game.gt)
        for variant, labels in predict(game, list(VARIANTS)).items():
            write_labels(os.path.join(OUT_DIR, "predictions", name, variant + ".csv"), labels)
            result = pm.evaluate(gt_codes, pm.label_codes(labels), bootstrap_n=0)
            os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)
            with open(os.path.join(OUT_DIR, "reports", "{}_{}.json".format(name, variant)), "w") as handle:
                json.dump(result, handle, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
            rows.append([name, variant] + [result[c] for c in SCORE_COLUMNS])
            print("  {:<10} spellF1 {:.3f}  macroF1 {:.3f}  alive/dead F1 {:.3f}  team acc {:.3f}  "
                  "spells {} vs gt {}".format(variant, result["spell_f1_mean"], result["macro_f1"],
                                              result["alive_dead_f1"], result["team_acc_both_alive"],
                                              result["n_pred_spells"], result["n_gt_spells"]))
        print("  ({:.0f} s)".format(time.time() - t0))
    path = os.path.join(OUT_DIR, "scores.csv")
    old = []
    if os.path.isfile(path):   # keep the games not rerun this time
        with open(path, newline="") as handle:
            old = [r for r in csv.reader(handle)][1:]
        old = [r for r in old if r[0] not in games]
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(("game", "variant") + SCORE_COLUMNS)
        w.writerows(sorted(old + [[r[0], r[1]] + ["{:.4f}".format(v) if isinstance(v, float) else v
                                                   for v in r[2:]] for r in rows]))


if __name__ == "__main__":
    main(sys.argv[1:])
