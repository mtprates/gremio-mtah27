"""Shots of the broadcast: where the image cuts, and which shots show the live wide picture.

    .venv/Scripts/python model/shots.py features     # one pass over the broadcast video (~10 min)
    .venv/Scripts/python model/shots.py              # aggregates of the features and the cuts

A close-up can keep the wide shot's homography: the provider carries it across the cut
(broadcast 81355-81427: look-at point and focal length unchanged, error 0.27 px, the provider's
tracking down to 1 player). So whether a frame shows the live picture is not read from its
homography but from the image: cuts split the broadcast into shots, and each shot is kept or
ignored as a whole.

Per frame, from a THUMB-sized copy of the video (decoded locally; nothing leaves the machine):

  hist      distance between this frame's and the previous frame's colour histogram (HSV, hue x
            saturation; Bhattacharyya, 0 same .. 1 disjoint)
  pixel     mean absolute grey-level difference to the previous frame, 0..255
  grass     share of grass-coloured pixels (detect_ball.GRASS_HUE / GRASS_SAT)

Saved to data/.model/shots_broadcast.npz. A cut is a frame whose hist stands CUT_SPIKE above its
neighbours (cuts()); model/cameras.py cuts then keeps or drops each shot as a whole. Prints
aggregates only.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import data  # noqa: E402

# Params
THUMB = (64, 36)            # px: enough for colour and layout, cheap to compare
HIST_BINS = (16, 4)         # hue x saturation bins
FEATURES = os.path.join(data.CACHE_DIR, "shots_broadcast.npz")
CUT_SPIKE = 0.2             # a cut: the histogram distance this far above its neighbours' median ...
CUT_NEIGHBOURS = 6          # ... over this many frames either side (a pan or a dissolve raises its
                            # neighbours too; docs/VALUES.md)


def features():
    """One pass over the broadcast video: (hist, pixel, grass) per video frame."""
    import cv2
    import detect_ball
    from games import SOURCES
    path = detect_ball.video_path("broadcast")
    cap = cv2.VideoCapture(path)
    n = SOURCES["broadcast"]["n_video_frames"]
    hist = np.full(n, np.nan, dtype=np.float32)
    pixel = np.full(n, np.nan, dtype=np.float32)
    grass = np.full(n, np.nan, dtype=np.float32)
    prev_h, prev_g = None, None
    lo_h, hi_h = detect_ball.GRASS_HUE
    t0 = time.time()
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, THUMB, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
        cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_h is not None:
            hist[f] = cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA)
            pixel[f] = float(np.abs(g - prev_g).mean())
        grass[f] = float(((hsv[..., 0] >= lo_h) & (hsv[..., 0] <= hi_h) & (hsv[..., 1] >= detect_ball.GRASS_SAT)).mean())
        prev_h, prev_g = h, g
        if f and f % 20000 == 0:
            print("  {} / {} frames, {:.0f} fps".format(f, n, f / (time.time() - t0)), flush=True)
    cap.release()
    np.savez_compressed(FEATURES, hist=hist, pixel=pixel, grass=grass)
    print("features for {} frames in {:.1f} min -> {}".format(
        int(np.isfinite(grass).sum()), (time.time() - t0) / 60, os.path.relpath(FEATURES, os.path.dirname(HERE))))


def load():
    z = np.load(FEATURES)
    return {k: z[k] for k in z.files}


def cut_spike(hist):
    """(n,) the histogram distance minus its median over CUT_NEIGHBOURS frames either side (the
    frame itself left out): a cut is one frame far above its neighbours."""
    from numpy.lib.stride_tricks import sliding_window_view
    h = np.nan_to_num(hist, nan=0.0)
    k = CUT_NEIGHBOURS
    win = sliding_window_view(np.pad(h, k, mode="edge"), 2 * k + 1)
    return h - np.median(np.concatenate([win[:, :k], win[:, k + 1:]], axis=1), axis=1)


def cuts():
    """Sorted frames where a new shot starts (hist[f] compares frame f with f - 1)."""
    return np.flatnonzero(cut_spike(load()["hist"]) >= CUT_SPIKE)


def report():
    """Distributions of the cut signals, to set the cut threshold where they separate."""
    ft = load()
    for name in ("hist", "pixel"):
        v = ft[name][np.isfinite(ft[name])]
        q = np.percentile(v, [50, 90, 99, 99.5, 99.9])
        print("{}: median {:.3f}, 90th {:.3f}, 99th {:.3f}, 99.5th {:.3f}, 99.9th {:.3f}".format(name, *q))
        edges = np.quantile(v, [0, 0.5, 0.9, 0.99]).tolist() + np.linspace(q[2], v.max(), 12)[1:].tolist()
        counts, edges = np.histogram(v, np.unique(edges))
        for lo, hi, c in zip(edges[:-1], edges[1:], counts):
            print("  {:8.3f}-{:<8.3f} {:>7}".format(lo, hi, c))
    c = cuts()
    gaps = np.diff(np.r_[0, c, len(ft["hist"])])
    print("cuts (spike >= {}): {}; shots median {:.1f} s, 10th {:.1f} s".format(
        CUT_SPIKE, len(c), np.median(gaps) / data.FPS, np.percentile(gaps, 10) / data.FPS))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "features":
        features()
    else:
        report()
