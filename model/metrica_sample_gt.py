"""Possession labels for Metrica's public sample games, derived from their event data.

    .venv/Scripts/python model/metrica_sample_gt.py

A validation set that is not BAR-ATH (no hand labels on the competition match, not even for
validation -- docs/TECHNICAL.md, "Before submission"). The three games are Metrica's own open
sample (github.com/metrica-sports/sample-data, cloned into external/metrica-sample/ with its data/
folder renamed games/; acknowledge the source if used publicly). Their events and tracking are
synchronised, so event frames are tracking frames.

The labels follow CHALLENGE.md's definition as closely as events allow:

  alive        from a restart (SET PIECE, or a goal kick) to the next stop
  stop         BALL OUT and a shot out / goal at the event's end frame; a foul, handball at its
               start frame; offside at its end frame (the ball reaching the offside player).
               Dead until the next restart; events tagged during the stoppage do not end it.
  team         the team of the last pass, recovery, shot, set piece or ball lost: a ball lost
               stays with the losing team until the other side's recovery or pass (a loose ball
               keeps its last team). Challenges, aerial duels included, do not change it.
  X            before each period's kick-off, after its END HALF (or its last event)

Three repairs for gaps in the tagging, counted in the output: a restart in live play puts the
missing stop at the previous event's end; a stop whose restart was not tagged ends at the first
pass after it (dead_intervals); a foul that play carried on through was played on.

It is an approximation of the challenge's ground truth, which comes from corrected tracking flags,
not events: stoppages here start at the event, not at the referee's whistle, and RECOVERY is
Metrica's tagger's idea of control, not necessarily CHALLENGE.md's.

Writes external/metrica-sample/gt/sample_game_<n>.csv (frame,label; frame = tracking frame, row 0
is X) and its _meta.json, for evaluation/run_evaluation.py --gt <path>. Prints aggregates only.
"""

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
SAMPLE_DIR = os.path.join(ROOT, "external", "metrica-sample")
GAMES_DIR = os.path.join(SAMPLE_DIR, "games")
OUT_DIR = os.path.join(SAMPLE_DIR, "gt")
FPS = 25
PLAY = {"PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST", "BALL OUT"}
ADVANTAGE_S = 3             # a pass or shot this long after a foul, before its free kick: played on
UNTAGGED_S = 2              # an untagged restart is the first pass at least this long after the stop
LONG_DEAD_S = 90            # dead stretches longer than this are counted: a missed restart?


def load_csv_game(n):
    """Games 1 and 2: [event dict], last tracking frame. Teams are already Home / Away."""
    folder = os.path.join(GAMES_DIR, "Sample_Game_{}".format(n))
    events = []
    with open(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(n)), newline="") as handle:
        for r in csv.DictReader(handle):
            events.append({"side": "H" if r["Team"] == "Home" else "A", "type": r["Type"],
                           "sub": r["Subtype"], "period": int(r["Period"]),
                           "start": int(r["Start Frame"]), "end": int(r["End Frame"])})
    with open(os.path.join(folder, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(n))) as handle:
        last = int(handle.readlines()[-1].split(",")[1])
    return events, last


def load_json_game(n):
    """Game 3: EPTS tracking plus JSON events. Team A (FIFATMA, first in the metadata) is home."""
    folder = os.path.join(GAMES_DIR, "Sample_Game_{}".format(n))
    with open(os.path.join(folder, "Sample_Game_{}_events.json".format(n))) as handle:
        raw = json.load(handle)["data"]
    events = []
    for e in raw:
        sub = e.get("subtypes")
        sub = "-".join(s["name"] for s in sub) if isinstance(sub, list) else (sub["name"] if sub else "")
        events.append({"side": "H" if e["team"]["id"] == "FIFATMA" else "A", "type": e["type"]["name"],
                       "sub": sub, "period": int(e["period"]),
                       "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"])})
    with open(os.path.join(folder, "Sample_Game_{}_tracking.txt".format(n))) as handle:
        last = int(handle.readlines()[-1].split(":")[0])
    return events, last


def is_restart(e):
    return e["type"] == "SET PIECE" or "GOAL KICK" in e["sub"]


def stop(e):
    """(frame the ball goes dead at, certain) or None if the event does not stop play. A foul is
    not certain: the referee may play advantage."""
    if e["type"] == "BALL OUT":
        return e["end"], True
    if e["type"] == "SHOT" and (e["sub"].endswith("OUT") or "GOAL" in e["sub"]):
        return e["end"], True
    if "OFFSIDE" in e["sub"]:
        return e["end"], True
    if "HAND BALL" in e["sub"]:
        return e["start"], True
    if e["type"] == "FAULT RECEIVED" or e["type"] == "CHALLENGE" and "FAULT" in e["sub"] and "ADVANTAGE" not in e["sub"]:
        return e["start"], False
    return None


def dead_intervals(events, fixes):
    """[(first, stop)] dead frames of one period's events (sorted, kick-off first). Events tagged
    inside a stoppage (the keeper fetching the ball, a tackle just after it went out) do not end it:
    only the next restart does. When that restart was not tagged -- another certain stop comes
    first -- play resumes at the first pass or ball lost UNTAGGED_S after the stop. A foul was
    played on (advantage) if a pass or shot comes ADVANTAGE_S or more after it, before its free
    kick, or if no free kick comes before the next stop. Time alone says nothing: an injury can
    hold a free kick up for minutes."""
    out, i = [], 0
    while i < len(events):
        s = stop(events[i])
        if s is None:
            i += 1
            continue
        frame, certain = s
        later = events[i + 1:]
        restart = next((e for e in later if is_restart(e) and e["start"] >= frame), None)
        nxt = next((e for e in later if stop(e) and stop(e)[1] and e["start"] > frame), None)
        tagged = restart is not None and (nxt is None or restart["start"] <= nxt["start"])
        played_on = any(e["type"] in ("PASS", "SHOT") and frame + ADVANTAGE_S * FPS <= e["start"]
                        < (restart["start"] if restart else frame + 1e9) for e in later)
        if not certain and (not tagged or played_on):
            fixes["foul played on"] += 1
            i += 1
            continue
        if tagged:
            until = restart["start"]
        else:
            fixes["untagged restart"] += 1
            resume = next((e for e in later if e["type"] in ("PASS", "BALL LOST")
                           and e["start"] >= frame + UNTAGGED_S * FPS), None)
            until = resume["start"] if resume else events[-1]["end"] + 1
        out.append((frame, until))
        while i < len(events) and events[i]["start"] < until:
            i += 1
    return out


def period_points(events):
    """[(frame, label)] change points of one period, its first and last labelled frame, and the
    repair counts."""
    # A restart goes first on its frame: a goal kick comes with a RECOVERY on the same frame.
    events = sorted(events, key=lambda e: (e["start"], not is_restart(e)))
    kickoff = next((e for e in events if e["type"] == "SET PIECE" and "KICK OFF" in e["sub"]), events[0])
    end_half = [e["start"] for e in events if "END HALF" in e["sub"]]
    last = end_half[0] - 1 if end_half else max(e["end"] for e in events)
    events = [e for e in events if kickoff["start"] <= e["start"] <= last and "END HALF" not in e["sub"]]
    fixes = {"missing stop": 0, "untagged restart": 0, "foul played on": 0}
    dead = dead_intervals(events, fixes)
    is_dead = lambda f: any(a <= f < b for a, b in dead)
    points = [(a, "D") for a, _ in dead]
    state, last_end = None, kickoff["start"]
    for e in events:
        if is_dead(e["start"]):
            continue
        if is_restart(e) and state is not None and not any(b == e["start"] for _, b in dead):
            # A restart in live play: the stop before it was not tagged (a throw-in with no BALL OUT).
            fixes["missing stop"] += 1
            points.append((last_end, "D"))
        if e["type"] in PLAY:
            back = any(b == e["start"] for _, b in dead)
            if e["side"] != state or back or is_restart(e):
                state = e["side"]
                points.append((e["start"], state))
            last_end = max(last_end, e["end"])
    return sorted(points, key=lambda p: p[0]), kickoff["start"], last, fixes


def labels_for(events, n_frames):
    labels = np.full(n_frames, "X", dtype="<U1")
    fixes = {}
    for period in sorted({e["period"] for e in events}):
        points, first, last, f = period_points([e for e in events if e["period"] == period])
        for k, v in f.items():
            fixes[k] = fixes.get(k, 0) + v
        for i, (frame, label) in enumerate(points):
            stop = points[i + 1][0] if i + 1 < len(points) else last + 1
            labels[max(frame, first):min(stop, last + 1)] = label
    return labels, fixes


def runs(labels):
    """[(label, start, stop)] maximal runs."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(labels)]])
    return [(labels[s], s, e) for s, e in zip(starts, stops)]


def report(n, labels, fixes):
    graded = labels != "X"
    r = runs(labels)
    possessions = [x for x in r if x[0] in "HA"]
    dead = [(e - s) / FPS for lab, s, e in r if lab == "D"]
    share = {k: (labels == k).sum() / graded.sum() for k in "HAD"}
    print("game {}: {} frames, {} labelled ({:.0f} min); H {:.0%} A {:.0%} D {:.0%}".format(
        n, len(labels), graded.sum(), graded.sum() / FPS / 60, share["H"], share["A"], share["D"]))
    print("  {} possessions (mean {:.1f} s); {} dead stretches (median {:.1f} s, longest {:.0f} s)".format(
        len(possessions), np.mean([(e - s) / FPS for _, s, e in possessions]),
        len(dead), np.median(dead), max(dead)))
    print("  repairs: {} missing stops, {} untagged restarts, {} fouls played on; dead > {} s: {}".format(
        fixes["missing stop"], fixes["untagged restart"], fixes["foul played on"], LONG_DEAD_S,
        sum(d > LONG_DEAD_S for d in dead)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for n, loader in ((1, load_csv_game), (2, load_csv_game), (3, load_json_game)):
        events, last = loader(n)
        labels, fixes = labels_for(events, last + 1)
        name = "sample_game_{}".format(n)
        with open(os.path.join(OUT_DIR, name + ".csv"), "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["frame", "label"])
            w.writerows(enumerate(labels))
        with open(os.path.join(OUT_DIR, name + "_meta.json"), "w") as handle:
            json.dump({"game": "Metrica Sample Game {}".format(n), "n_video_frames": int(len(labels)),
                       "fps": FPS, "source": "github.com/metrica-sports/sample-data, derived from events "
                       "by model/metrica_sample_gt.py"}, handle, indent=2)
        report(n, labels, fixes)


if __name__ == "__main__":
    main()
