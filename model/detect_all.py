"""Run model/detect_ball.py over the whole match, in chunks, resumable. For an overnight run.

    .venv/Scripts/python model/detect_all.py              # everything still missing
    .venv/Scripts/python model/detect_all.py --dry-run    # list the chunks and what is done

Tactical: both graded halves (data.halves()). Broadcast: the broadcast frames the alignment maps
into them. Each chunk is CHUNK frames and writes the usual files in data/.model/detections/
(<source>_<start>_<count>.csv and persons-...); a chunk whose files exist and are complete is
skipped, so the run can be stopped and started again. Progress and an estimate of the time left
are printed after every chunk. At the measured speeds on a GTX 1660 Ti (tactical 4.5 fps tiled,
broadcast ~14 fps) the whole match is about 10 h tactical + 3.5 h broadcast.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import align  # noqa: E402
import data  # noqa: E402

# Params
CHUNK = 15000               # frames per chunk: 10 min of match, ~55 min tactical
OUT_DIR = os.path.join(data.CACHE_DIR, "detections")


def ranges():
    """[(source, first, stop)] to cover: tactical graded halves, broadcast frames mapped into them."""
    import run
    out = [("tactical", first, last + 1) for _, first, last in data.halves()]
    tac, bro = data.load("tactical"), data.load("broadcast")
    al = run.load_alignment(tac, bro)
    off = al["offset_b"]
    b = np.flatnonzero(off != align.MISSING)
    t = b + off[b].astype(np.int64)
    for _, first, last in data.halves():
        sel = b[(t >= first) & (t <= last)]
        if len(sel):
            out.append(("broadcast", int(sel.min()), int(sel.max()) + 1))
    return out


def chunks():
    for source, first, stop in ranges():
        for s in range(first, stop, CHUNK):
            yield source, s, min(CHUNK, stop - s)


def done(source, start, count):
    """The chunk's ball and persons files exist and the ball file reaches its last frame."""
    ball = os.path.join(OUT_DIR, "{}_{}_{}.csv".format(source, start, count))
    persons = os.path.join(OUT_DIR, "persons-{}-{}-{}.csv".format(source, start, count))
    if not (os.path.isfile(ball) and os.path.isfile(persons)):
        return False
    with open(ball, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 4096))
        tail = handle.read().decode("utf-8", "ignore").strip().splitlines()
    try:
        return int(tail[-1].split(",")[0]) >= start + count - 30   # a few last frames may have no ball
    except (ValueError, IndexError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    todo = [c for c in chunks() if not done(*c)]
    all_chunks = list(chunks())
    frames = {s: sum(c for src, _, c in all_chunks if src == s) for s in ("tactical", "broadcast")}
    left = {s: sum(c for src, _, c in todo if src == s) for s in ("tactical", "broadcast")}
    print("chunks: {} in all, {} to run; frames to run: tactical {} of {}, broadcast {} of {}".format(
        len(all_chunks), len(todo), left["tactical"], frames["tactical"], left["broadcast"], frames["broadcast"]))
    if args.dry_run:
        for c in all_chunks:
            print("  {} {}-{} {}".format(c[0], c[1], c[1] + c[2] - 1, "done" if c not in todo else "to run"))
        return
    fps = {"tactical": 4.5, "broadcast": 14.0}
    t0 = time.time()
    for i, (source, start, count) in enumerate(todo):
        t1 = time.time()
        cmd = [sys.executable, os.path.join(HERE, "detect_ball.py"), "--source", source,
               "--start", str(start), "--count", str(count)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        took = time.time() - t1
        if result.returncode != 0:
            print("chunk {} {}-{} FAILED:\n{}".format(source, start, start + count - 1, result.stderr[-2000:]))
            sys.exit(1)
        fps[source] = count / max(took, 1e-9)
        left[source] -= count
        eta = sum(left[s] / fps[s] for s in left) / 3600
        print("[{}/{}] {} {}-{} in {:.0f} min ({:.1f} fps); about {:.1f} h left".format(
            i + 1, len(todo), source, start, start + count - 1, took / 60, fps[source], eta), flush=True)
    print("all done in {:.1f} h".format((time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
