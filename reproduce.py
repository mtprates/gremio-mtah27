"""From the match data to our submission file, in one command.

    python reproduce.py --team <team_name>              # every step still missing, then the file
    python reproduce.py --team <team_name> --from teams # re-run from a step on (and all after it)
    python reproduce.py --team <team_name> --fresh      # also drop the possession model's caches
    python reproduce.py --list                          # the steps and whether their outputs exist

Needs the match data laid out as DATA.md describes (./data, or $MTAH_DATA_DIR), and
requirements-detect.txt installed (torch for your GPU first). Every step writes under
data/.model/ (gitignored) and is skipped when its outputs already exist, so the run can be
stopped and started again; --from forces a step and the ones after it.

Steps, in order (timings on a GTX 1660 Ti; each script's docstring has details):

  detect   model/detect_all.py         YOLO balls and people over both videos, in chunks (~13 h)
  shots    model/shots.py features     the broadcast's shot features (~10 min)
  cuts     model/cameras.py cuts       the broadcast's cuts and cut-aways
  teams    model/teams.py, per chunk   tracker and team labels: tactical, broadcast, tactical again,
                                       so each camera gets the other's shirt-colour votes
  model    model/iterate.py ... models the ball track and the possession models (C is submitted)
  submit   copy C to submission/<team>.csv and run evaluation/validate_submission.py on it

The broadcast-tactical alignment and the provider caches are built on first use by the steps that
need them (~15 min). The whistles and spectrograms are for the viewer only and are not run.

Reproducibility: everything after detection is deterministic (the kit mixture is seeded). YOLO on a
different GPU or driver can differ in the last digits of a box, which can move a pick; with the
same detections the file is reproduced exactly.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "model")
sys.path.insert(0, MODEL)
import data  # noqa: E402

PY = sys.executable
DET = os.path.join(data.CACHE_DIR, "detections")
# The rules behind switches that the submission uses (docs/VALUES.md, docs/TECHNICAL.md).
# Changing this list changes the submission.
MODEL_FLAGS = ["--persons", "--refs", "--smooth", "--out", "--flights", "--arc-tol", "--side", "--traces",
               "--restart", "--heights", "--solo", "--gaps",
               # possession model (docs/VALUES.md, "Submission configuration"): rules 53, 53b, 55, 56, 58-60, 62-64
               "--touch-precision", "--event-precision", "--kick", "--restart-live", "--touch-echo", "--event-switch",
               "--far-clear", "--replay-neutral", "--short-spells", "--short-broadcast", "--still-cutaway"]
SUBMITTED = os.path.join(ROOT, "evaluation", "results", "detector_hmm.csv")   # model C
OUT_DIR = os.path.join(ROOT, "submission")    # gitignored: a whole match of labels
STEPS = ("detect", "shots", "cuts", "teams", "model", "submit")


_CHUNKS = []


def chunks():
    if not _CHUNKS:
        import detect_all
        _CHUNKS.extend(detect_all.chunks())
    return _CHUNKS


def run(args):
    print("\n$ " + " ".join(os.path.relpath(a, ROOT) if os.path.isabs(a) else a for a in args), flush=True)
    t0 = time.time()
    subprocess.run(args, cwd=ROOT, check=True)
    print("  ({:.1f} min)".format((time.time() - t0) / 60), flush=True)


def teams_path(source, first, count):
    return os.path.join(DET, "teams-{}-{}-{}.csv".format(source, first, count))


def done(step, team):
    """Whether the step's outputs exist (not whether they are current: use --from after a change)."""
    if step == "detect":
        import detect_all
        return all(detect_all.done(*c) for c in chunks())
    if step == "shots":
        return os.path.isfile(os.path.join(data.CACHE_DIR, "shots_broadcast.npz"))
    if step == "cuts":
        return all(os.path.isfile(os.path.join(data.CACHE_DIR, f)) for f in ("broadcast_cuts.csv", "broadcast_usable.npz"))
    if step == "teams":
        return all(os.path.isfile(teams_path(*c)) for c in chunks())
    if step == "model":
        return False        # minutes from its caches; always re-run so the file matches the code
    if step == "submit":
        return False
    raise ValueError(step)


def do(step, team, fresh):
    if step == "detect":
        run([PY, os.path.join(MODEL, "detect_all.py")])
    elif step == "shots":
        run([PY, os.path.join(MODEL, "shots.py"), "features"])
    elif step == "cuts":
        run([PY, os.path.join(MODEL, "cameras.py"), "cuts"])
    elif step == "teams":
        cs = chunks()
        order = [c for c in cs if c[0] == "tactical"] + [c for c in cs if c[0] == "broadcast"] + \
                [c for c in cs if c[0] == "tactical"]
        for source, first, count in order:
            run([PY, os.path.join(MODEL, "teams.py"), source, str(first), str(count)])
    elif step == "model":
        if fresh:
            cache = os.path.join(data.CACHE_DIR, "iterate")
            for path in glob.glob(os.path.join(cache, "*.pkl")):
                os.remove(path)
            print("  dropped the possession model's caches in {}".format(os.path.relpath(cache, ROOT)))
        run([PY, os.path.join(MODEL, "iterate.py")] + MODEL_FLAGS + ["models"])
    elif step == "submit":
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "{}.csv".format(team))
        shutil.copyfile(SUBMITTED, out)
        print("\nwrote {}".format(os.path.relpath(out, ROOT)))
        run([PY, os.path.join(ROOT, "evaluation", "validate_submission.py"), out])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--team", help="team name: the file is written as submission/<team>.csv")
    ap.add_argument("--from", dest="start", choices=STEPS, help="force this step and every step after it")
    ap.add_argument("--fresh", action="store_true", help="drop data/.model/iterate/ caches before the model step "
                                                         "(needed after a change to the ball picks; 10-60 min)")
    ap.add_argument("--list", action="store_true", help="list the steps and whether their outputs exist")
    args = ap.parse_args()
    if args.list:
        for step in STEPS:
            print("  {:<7} {}".format(step, "done" if done(step, args.team) else "to run"))
        return
    if not args.team:
        ap.error("--team is required (the file is named after it, SUBMISSION.md)")
    forced = STEPS.index(args.start) if args.start else len(STEPS)
    t0 = time.time()
    for i, step in enumerate(STEPS):
        if i < forced and done(step, args.team):
            print("[{}] outputs exist, skipped".format(step))
            continue
        print("[{}]".format(step), flush=True)
        do(step, args.team, args.fresh)
    print("\nall steps in {:.1f} min".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
