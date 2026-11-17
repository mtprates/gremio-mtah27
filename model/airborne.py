"""Flag stretches where the tracked ball is probably in the air, and estimate its height there.

    py model/airborne.py

The ATD ball has no height, and the homography only describes the grass, so a lofted ball is
drawn where its pitch position is, not where it is in the image. This is the cheap heuristic for
finding those stretches: between two touches a ball in flight moves in a straight line at a
roughly constant horizontal speed. So does a hard ground pass, which is why the viewer draws each
flag as a 3D ball through a camera recovered from the homography -- where the guess is wrong, the
drawn ball floats visibly off the real one.

    segment   a greedy run of tracked ball positions that a constant-velocity line fits to within
              RMS_MAX, with no point JUMP_MAX off the line so far, at SPEED_MIN or faster, for
              MIN_FRAMES or longer. Gaps up to MAX_GAP frames are bridged by the fit.
    kind      0 probably airborne
              1 too straight to be measured (RMS under PERFECT_RMS): most likely a gap the
                tracker filled by interpolating along the ground, not a flight
    height    ballistic, from the flight time alone: launch and landing at z = 0, so
              z(t) = g/2 * t * (T - t), peaking at g T^2 / 8. Computed in the viewer.

Writes data/.model/airborne.json for the viewer (it is derived from the tracking, so it stays in
data/) and prints counts only.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

# Params
SPEED_MIN = 8.0         # m/s; a firm pass or longer
MIN_FRAMES = 15         # 0.6 s
MAX_FRAMES = 150        # 6 s; bounds the greedy search on a stationary ball
MAX_GAP = 6             # untracked frames the fit may bridge
RMS_MAX = 0.35          # m, residual to the constant-velocity line
JUMP_MAX = 0.6          # m, a new point this far off the line so far ends it: a touch
PERFECT_RMS = 0.02      # m; real measurements are never this clean
OUT_PATH = os.path.join(data.CACHE_DIR, "airborne.json")


def _rms(n, st, stt, sx, sxx, stx):
    """Residual RMS of the least-squares line x(t) from running sums. 0 below three points."""
    if n < 3:
        return 0.0
    var_t = stt - st * st / n
    ssr = sxx - sx * sx / n - ((stx - st * sx / n) ** 2 / var_t if var_t > 0 else 0.0)
    return math.sqrt(max(0.0, ssr) / n)


def _predict(ts, xs, t):
    """The least-squares line through the points so far, (n, sum t, sum t^2) and
    (sum x, sum x^2, sum t*x), evaluated at t."""
    n, st, stt = ts
    sx, _, stx = xs
    var_t = stt - st * st / n
    slope = (stx - st * sx / n) / var_t if var_t > 0 else 0.0
    return sx / n + slope * (t - st / n)


def _fit(frames, xy):
    """(x0, y0, vx, vy) at the first frame, metres and m/s."""
    t = (frames - frames[0]) / data.FPS
    vx, x0 = np.polyfit(t, xy[:, 0], 1)
    vy, y0 = np.polyfit(t, xy[:, 1], 1)
    return float(x0), float(y0), float(vx), float(vy)


def segments(ball):
    """[start, stop, speed, rms, missing_share, kind, x0, y0, vx, vy], stop exclusive."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    xs, ys = ball[:, 0].astype(float).tolist(), ball[:, 1].astype(float).tolist()
    out = []
    i = 0
    while i < n:
        if not seen[i]:
            i += 1
            continue
        # Running sums over the observed points: count, t, t^2, then x, x^2, t*x and y, y^2, t*y.
        acc = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        last, rms = i, 0.0
        for k in range(i, min(n, i + MAX_FRAMES)):
            if not seen[k]:
                if k - last > MAX_GAP:
                    break
                continue
            t, x, y = float(k - i), xs[k], ys[k]
            if acc[0] >= 3 and math.hypot(x - _predict(acc[:3], acc[3:6], t),
                                          y - _predict(acc[:3], acc[6:], t)) > JUMP_MAX:
                break
            trial =[acc[0] + 1, acc[1] + t, acc[2] + t * t, acc[3] + x, acc[4] + x * x,
                     acc[5] + t * x, acc[6] + y, acc[7] + y * y, acc[8] + t * y]
            trial_rms = math.hypot(_rms(*trial[:3], *trial[3:6]), _rms(*trial[:3], *trial[6:]))
            if trial_rms > RMS_MAX:
                break
            acc, last, rms = trial, k, trial_rms

        stop = last + 1
        if stop - i < MIN_FRAMES:
            i += 1
            continue
        frames = np.flatnonzero(seen[i:stop]) + i
        x0, y0, vx, vy = _fit(frames, ball[frames].astype(float))
        speed = math.hypot(vx, vy)
        if speed >= SPEED_MIN:
            out.append([i, stop, round(speed, 2), round(rms, 3),
                        round(1 - len(frames) / (stop - i), 3), 1 if rms < PERFECT_RMS else 0,
                        round(x0, 3), round(y0, 3), round(vx, 3), round(vy, 3)])
        # Long enough to judge: whatever broke the line (a touch, a bounce) starts the next try.
        i = stop
    return out


def main():
    rec = data.load("tactical")
    found = segments(rec.ball)

    graded = np.zeros(rec.n, dtype=bool)
    for _, first, last in data.halves():
        graded[first:last + 1] = True
    print("\nprobably airborne (tactical)")
    for kind, name in ((0, "airborne"), (1, "interpolated-looking")):
        picked = [s for s in found if s[5] == kind]
        frames = sum(s[1] - s[0] for s in picked if graded[s[0]])
        durations = [(s[1] - s[0]) / data.FPS for s in picked]
        print("  {:<22} {:>5} segments  {:>6} graded frames ({:.1f}%)  median {:.2f} s".format(
            name, len(picked), frames, 100.0 * frames / graded.sum(),
            float(np.median(durations)) if durations else float("nan")))

    os.makedirs(data.CACHE_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as handle:
        json.dump({"params": {"speed_min": SPEED_MIN, "min_frames": MIN_FRAMES,
                              "max_gap": MAX_GAP, "rms_max": RMS_MAX,
                              "perfect_rms": PERFECT_RMS},
                   "segments": found}, handle, separators=(",", ":"))
    print("  viewer -> {}".format(os.path.relpath(OUT_PATH, data.ROOT)))


if __name__ == "__main__":
    main()
