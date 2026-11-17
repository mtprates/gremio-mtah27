"""How our own ball detections (model/detect_ball.py) compare with the ATD ball. Aggregates only.

    py model/compare_ball.py tactical 11185 3000
    py model/compare_ball.py broadcast 7171 3000

(the detections model/detect_ball.py wrote for that source, first frame and count)

Per confidence threshold, with the best detection per frame:
    coverage     frames with a detection, and how many of them the ATD has no ball on
    agreement    pitch distance to the ATD ball where both have one
    continuity   a detection within CONTINUITY m of the previous or next frame's -- a real ball
                 moves smoothly; a head, a shoe or a spare ball by the line mostly does not
Broadcast detections count only on the main camera's frames (model/cameras.py), and are also put
on the tactical timeline and pitch through the alignment.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

# Params
THRESHOLDS = (0.1, 0.25, 0.4, 0.6)
CONTINUITY = 1.5            # m between consecutive frames (37 m/s)
CHECK = (12785, 12835)      # tactical frames: the 1:04 airball


def load(path):
    with open(path, newline="") as handle:
        rows = [(int(r["frame"]), float(r["x"]), float(r["y"]), float(r["conf"])) for r in csv.DictReader(handle)]
    return rows


def to_pitch(H, frames, px):
    """Pixels -> pitch metres through each frame's homography (normalised pitch -> pixels)."""
    out = np.full((len(frames), 2), np.nan)
    for i, (f, (u, v)) in enumerate(zip(frames, px)):
        h = H[f]
        if np.isnan(h).any():
            continue
        p = np.linalg.solve(h, np.array([u, v, 1.0]))
        out[i] = p[0] / p[2] * data.PITCH[0], p[1] / p[2] * data.PITCH[1]
    return out


def best_per_frame(rows, threshold, first, count):
    """(count,) best detection's (x px, y px) per frame at or above threshold, NaN if none."""
    best = np.full((count, 3), np.nan)
    for f, x, y, c in rows:
        i = f - first
        if c >= threshold and 0 <= i < count and not (best[i, 2] >= c):
            best[i] = x, y, c
    return best


def report(name, det, ref, extra_mask=None):
    """det, ref (n, 2) metres; prints coverage, agreement and continuity."""
    have = ~np.isnan(det[:, 0])
    ref_have = ~np.isnan(ref[:, 0])
    if extra_mask is not None:
        have &= extra_mask
    step = np.full(len(det), np.inf)
    d = np.linalg.norm(det[1:] - det[:-1], axis=1)
    step[1:] = np.where(np.isnan(d), np.inf, d)
    smooth = have & ((step <= CONTINUITY) | (np.append(step[1:], np.inf) <= CONTINUITY))
    both = have & ref_have
    dist = np.linalg.norm(det - ref, axis=1)[both]
    n = int((extra_mask if extra_mask is not None else np.ones(len(det), bool)).sum())
    print("  {:<34} detections on {:5.1%} of frames ({:.0%} continuous); where the ATD has no ball: {:5.1%} "
          "of frames".format(name, have.sum() / max(n, 1), smooth.sum() / max(have.sum(), 1),
                             (have & ~ref_have).sum() / max(n, 1)))
    if len(dist):
        print("  {:<34} vs ATD where both: median {:.2f} m; within 1 m {:.0%}, 2 m {:.0%}; beyond 5 m {:.0%}"
              " (of those, continuous {:.0%})".format("", float(np.median(dist)), np.mean(dist <= 1), np.mean(dist <= 2),
                                                      np.mean(dist > 5), np.mean(smooth[both][dist > 5]) if (dist > 5).any() else 0))


def main():
    source, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    path = os.path.join(data.CACHE_DIR, "detections", "{}_{}_{}.csv".format(source, first, count))
    rows = load(path)
    H = data.homography_matrices(source)
    rec = data.load(source)
    frames = np.arange(first, first + count)
    ref = rec.ball[frames].astype(float)
    print("\n{} frames {}-{}: {} detections at conf >= 0.05".format(source, first, first + count - 1, len(rows)))
    extra = None
    if source == "broadcast":
        import cameras
        import run
        tac = data.load("tactical")
        al = run.load_alignment(tac, rec)
        cams = cameras.recover(H)
        main_b, _, _ = cameras.main_camera(H, cams, data.homography_errors("broadcast"),
                                           cameras.broadcast_live(al, rec.n))
        extra = main_b[frames]
        print("  main-camera frames in the window: {:.0%}".format(extra.mean()))
    for th in THRESHOLDS:
        best = best_per_frame(rows, th, first, count)
        det = to_pitch(H, frames, best[:, :2])
        print(" conf >= {}".format(th))
        report("vs own ATD ball", det, ref, extra)
        if source == "broadcast":
            # Onto the tactical pitch and timeline.
            flip = tuple(bool(v) for v in al["flip"])
            det_t = det.copy()
            if flip[0]:
                det_t[:, 0] = data.PITCH[0] - det_t[:, 0]
            if flip[1]:
                det_t[:, 1] = data.PITCH[1] - det_t[:, 1]
            off = al["offset_b"][frames].astype(np.int64)
            t = frames + off
            ok = (al["offset_b"][frames] != -2147483648) & (t >= 0) & (t < tac.n)
            ref_t = np.full((count, 2), np.nan)
            ref_t[ok] = tac.ball[t[ok]]
            report("vs tactical ATD ball (aligned)", det_t, ref_t, extra & ok)
    if source == "tactical" and first <= CHECK[0] and CHECK[1] < first + count:
        best = best_per_frame(rows, 0.25, first, count)
        det = to_pitch(H, frames, best[:, :2])
        i0, i1 = CHECK[0] - first, CHECK[1] - first
        d = np.linalg.norm(det[i0:i1] - ref[i0:i1], axis=1)
        print("\n 1:04 airball ({}-{}), conf >= 0.25: detection on {} of {} frames; distance to ATD ball, every 5th frame: {}".format(
            CHECK[0], CHECK[1], int((~np.isnan(det[i0:i1, 0])).sum()), i1 - i0,
            " ".join("--" if np.isnan(v) else "{:.1f}".format(v) for v in d[::5])))


if __name__ == "__main__":
    main()
