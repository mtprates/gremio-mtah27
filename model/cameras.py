"""Step 1 towards a 3D ball: where each camera is, recovered from its homographies.

    py model/cameras.py
    py model/cameras.py cuts     # the broadcast's shots and cut-aways (write_cuts), after
                                 # model/shots.py features; teams.py reads the result

Each export's ball is where its camera's line of sight to the ball meets the grass, so a ball in
the air is reported displaced away from the camera by h / (h - z) (h camera height, z ball
height). Knowing both cameras' centres turns the two reported points into two lines of sight
that cross at the ball in 3D. This module recovers the centres and says how far to trust them.

Recovery is the standard decomposition (as in our match viewer, not published), vectorised: the homography
is K [r1 r2 t] up to scale; square pixels and the principal point at the image centre leave the
focal length f, which r1 . r2 = 0 fixes ("ortho"); |r1| = |r2| gives a second, independent f
("norm"). The two agree only if the image centre is right and the view is one pinhole camera, so
their disagreement is the check on both -- a stitched panorama would fail it. The centre is then
C = -R^T t, in that recording's own pitch metres, z up.

The broadcast is not one camera: replays, close-ups and side cameras zoomed on players are cut in.
The main camera's frames are those that are
    live          the alignment matched the broadcast's players to the tactical ones at that moment
                  (align.STATE_LIVE): no replay, a homography exists
    precise       reprojection error at most ERR_MAX
    the same camera  a centre within CENTRE_RADIUS of the main camera's (the median over the frames
                  above): a side camera is another physical camera, somewhere else
Prints aggregates only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import align  # noqa: E402
import data  # noqa: E402

# Params
IMAGE = (1920, 1080)        # px; DATA.md gives the tactical video's, the broadcast is assumed alike
                            # (the focal-length agreement below would show a wrong guess)
ERR_MAX = 2.0               # px: the broadcast homographies kept (the tactical median is 0.24 in play)
CENTRE_RADIUS = 5.0         # m: a recovered centre further than this from the main camera's is
                            # another camera
F_AGREE = 0.1               # relative: ortho and norm focal lengths this close agree
# A recovered centre is noisy frame by frame, mostly along the line of sight (zoomed in, a longer
# focal length from further away looks nearly the same): 10-20% of the FIXED tactical camera's
# frames point more than 10 degrees off. So camera identity is read from the viewing direction
# over a second, not frame by frame. (Splitting the broadcast into shots at jumps of the look-at
# point or the focal length did not work: per-frame noise cut it into one-frame "shots".)
DIRECTION_WINDOW = 25  # frames: rolling median of the viewing angle
MAIN_ANGLE = 8.0            # deg: within this of the main camera's direction, it is the main camera
OTHER_ANGLE = 20.0          # deg: a stretch past MAIN_ANGLE is another camera only if it peaks above
                            # this; shorter wobbles are the main camera's noisy centre when zoomed
                            # (docs/VALUES.md)
HEIGHT_RATIO_MAX = 2.0      # a shot (between image cuts, model/shots.py) whose people are, on the median,
                            # this many times taller than its homography allows is a cut-away (live
                            # frames 1.04, 99th pct 1.67; the 81355 close-up 2.4-3.2; docs/VALUES.md)


def recover(H, image=IMAGE, pitch=data.PITCH):
    """Pinhole cameras from (n, 3, 3) homographies (normalised pitch -> pixels; NaN rows allowed).
    Returns {"f", "f_ortho", "f_norm" (n,); "centre" (n, 3) pitch metres, z up; "ok" (n,)}."""
    L, W = pitch
    cx, cy = image[0] / 2, image[1] / 2
    c1, c2, c3 = H[:, :, 0] / L, H[:, :, 1] / W, H[:, :, 2]          # (n, 3) columns in metres

    def centred(c):
        return np.stack([c[:, 0] - cx * c[:, 2], c[:, 1] - cy * c[:, 2], c[:, 2]], axis=1)

    a, b = centred(c1), centred(c2)
    with np.errstate(all="ignore"):
        ortho = -(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (a[:, 2] * b[:, 2])
        norm = -(a[:, 0] ** 2 + a[:, 1] ** 2 - b[:, 0] ** 2 - b[:, 1] ** 2) / (a[:, 2] ** 2 - b[:, 2] ** 2)
        f2 = np.where(ortho > 0, ortho, np.where(norm > 0, norm, np.nan))
        f = np.sqrt(f2)

        def unK(c):
            return np.stack([(c[:, 0] - cx * c[:, 2]) / f, (c[:, 1] - cy * c[:, 2]) / f, c[:, 2]], axis=1)

        r1, r2, t = unK(c1), unK(c2), unK(c3)
        s = 2 / (np.linalg.norm(r1, axis=1) + np.linalg.norm(r2, axis=1))
        front = r1[:, 2] * L / 2 + r2[:, 2] * W / 2 + t[:, 2]
        s = np.where(front < 0, -s, s)[:, None]
        r1, r2, t = r1 * s, r2 * s, t * s
        r3 = np.cross(r1, r2)
        r3 /= np.linalg.norm(r3, axis=1, keepdims=True)
        height = -(r3 * t).sum(axis=1)
        r3 = np.where((height < 0)[:, None], -r3, r3)
        centre = np.stack([-(r1 * t).sum(axis=1), -(r2 * t).sum(axis=1), -(r3 * t).sum(axis=1)], axis=1)
    ok = np.isfinite(f) & np.isfinite(centre).all(axis=1)
    return {"f": f, "f_ortho": np.sqrt(np.where(ortho > 0, ortho, np.nan)),
            "f_norm": np.sqrt(np.where(norm > 0, norm, np.nan)), "centre": centre, "ok": ok}


def broadcast_live(alignment, n_b):
    """(n_b,) True on broadcast frames the alignment calls live (players matched, no replay)."""
    out = np.zeros(n_b, dtype=bool)
    b = np.arange(n_b)
    off = alignment["offset_b"]
    mapped = off != align.MISSING
    t = b[mapped] + off[mapped].astype(np.int64)
    inside = (t >= 0) & (t < len(alignment["state_t"]))
    idx = np.flatnonzero(mapped)[inside]
    out[idx] = alignment["state_t"][t[inside]] == align.STATE_LIVE
    return out


def look_at(H, image=IMAGE, pitch=data.PITCH):
    """(n, 2) pitch metres under the image centre, per frame (NaN where no homography)."""
    bad = np.isnan(H).any(axis=(1, 2))
    Hi = np.linalg.inv(np.where(bad[:, None, None], np.eye(3), H))
    p = Hi @ np.array([image[0] / 2, image[1] / 2, 1.0])
    with np.errstate(all="ignore"):
        out = np.stack([p[:, 0] / p[:, 2] * pitch[0], p[:, 1] / p[:, 2] * pitch[1]], axis=1)
    out[bad] = np.nan
    return out


def direction_angle(cams, target, centre):
    """(n,) degrees between each frame's recovered camera direction and the direction to `centre`,
    both seen from the point the frame looks at."""
    p = np.concatenate([target, np.zeros((len(target), 1))], axis=1)
    est, ref = cams["centre"] - p, centre - p
    with np.errstate(all="ignore"):
        cos = (est * ref).sum(1) / np.linalg.norm(est, axis=1) / np.linalg.norm(ref, axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def rolling_median(values, valid, width=DIRECTION_WINDOW):
    """Centred rolling median of `values` within each run of `valid` frames (NaN outside)."""
    import rules
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(values), np.nan)
    half = width // 2
    for s, e in rules.runs(valid):
        v = values[s:e]
        padded = np.concatenate([np.full(half, np.nan), v, np.full(half, np.nan)])
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[s:e] = np.nanmedian(sliding_window_view(padded, width), axis=1)[:e - s]
    return out


def main_camera(H, cams, err, live):
    """(mask of the main camera's frames, its centre, smoothed angle per frame) for the broadcast.
    The centre is the median over precise live frames that point its way, found by re-centring
    twice; a frame is the main camera's where the viewing angle, as a rolling median over
    DIRECTION_WINDOW frames of the same homography run, is within MAIN_ANGLE -- or past it in a
    stretch that never goes beyond OTHER_ANGLE (broadcast 11521-11535: 8.1 deg, 0.6 s of one shot)."""
    base = cams["ok"] & live & (err <= ERR_MAX)
    target = look_at(H)
    centre = np.median(cams["centre"][base], axis=0)
    for _ in range(2):
        angle = direction_angle(cams, target, centre)
        centre = np.median(cams["centre"][base & (angle <= 5)], axis=0)
    angle = direction_angle(cams, target, centre)
    smooth = rolling_median(np.where(cams["ok"], angle, np.nan), cams["ok"])
    with np.errstate(invalid="ignore"):
        mask = base & (smooth <= MAIN_ANGLE)
    # Stretches of base frames past MAIN_ANGLE: only those peaking above OTHER_ANGLE are another camera.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (base & ~mask).astype(np.int8), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if np.nanmax(smooth[s:e]) <= OTHER_ANGLE:
            mask[s:e] = True
    return mask, centre, smooth


def expected_height(cams, frames, feet, height):
    """(k,) pixels a person `height` m tall standing at `feet` (pitch metres) spans on those
    frames, from recover()'s camera: a vertical segment at horizontal distance d and range D images
    as about f * height * d / D^2."""
    c = cams["centre"][frames]
    horiz = np.linalg.norm(c[:, :2] - feet, axis=1)
    with np.errstate(all="ignore"):
        return cams["f"][frames] * height * horiz / (horiz ** 2 + c[:, 2] ** 2)


def frame_height_ratio(cams, frames, box_h, feet, height, n):
    """(n,) per frame, the median over its boxes of box height / expected_height: about 1 where the
    homography is the image's own (live frames 1.04), far above it on a close-up under the wide
    shot's homography (81355-81427: 2.4-3.2). NaN on frames without a box or a camera."""
    with np.errstate(all="ignore"):
        ratio = box_h / expected_height(cams, frames, feet, height)
    out = np.full(n, np.nan)
    ok = np.isfinite(ratio)
    frames, ratio = frames[ok], ratio[ok]
    order = np.argsort(frames, kind="stable")
    fs, rs = frames[order], ratio[order]
    starts = np.r_[0, np.flatnonzero(np.diff(fs)) + 1] if len(fs) else np.array([], int)
    for a, e in zip(starts, np.r_[starts[1:], len(fs)]):
        out[fs[a]] = np.median(rs[a:e])
    return out


def describe(name, cams, mask):
    c = cams["centre"][mask]
    centre = np.median(c, axis=0)
    dist = np.linalg.norm(c - centre, axis=1)
    rel = np.abs(cams["f_ortho"][mask] - cams["f_norm"][mask]) / cams["f"][mask]
    print("  {:<28} {:>7} frames; centre x {:6.1f}  y {:6.1f}  height {:5.1f} m; spread around it: "
          "median {:.2f} m, 90th pct {:.2f} m".format(name, int(mask.sum()), *centre, float(np.median(dist)),
                                                     float(np.percentile(dist, 90))))
    print("  {:<28} focal length median {:.0f} px (10th-90th {:.0f}-{:.0f}); ortho vs norm agree within "
          "{:.0%} on {:.0%} of frames".format("", float(np.nanmedian(cams["f"][mask])),
                                               *np.nanpercentile(cams["f"][mask], [10, 90]), F_AGREE,
                                               float(np.nanmean(rel <= F_AGREE))))
    return centre


def main():
    import run
    tac, bro = data.load("tactical"), data.load("broadcast")
    alignment = run.load_alignment(tac, bro)

    print("\ntactical camera")
    cams_t = recover(data.homography_matrices("tactical"))
    err_t = data.homography_errors("tactical")
    for half, first, last in data.halves():
        mask = np.zeros(tac.n, dtype=bool)
        mask[first:last + 1] = True
        mask &= cams_t["ok"]
        describe("half {}".format(half), cams_t, mask)
    print("  recovered on {:.1%} of graded frames".format(
        np.mean(cams_t["ok"][np.concatenate([np.arange(a, b + 1) for _, a, b in data.halves()])])))

    print("\nbroadcast camera")
    cams_b = recover(data.homography_matrices("broadcast"))
    err_b = data.homography_errors("broadcast")
    live = broadcast_live(alignment, bro.n)
    valid = bro.valid_h
    print("  homography valid {:.1%} of frames; recovered {:.1%} of those; live {:.1%} of those".format(
        valid.mean(), cams_b["ok"][valid].mean(), live[valid].mean()))
    q = np.nanpercentile(err_b[valid & live], [50, 75, 90, 95, 99])
    print("  reprojection error on live frames, px: median {:.2f}, 75th {:.2f}, 90th {:.2f}, 95th {:.2f}, "
          "99th {:.2f}".format(*q))
    print("  reprojection error on replay / mismatch frames: median {:.2f} px".format(
        float(np.nanmedian(err_b[valid & ~live]))))
    H_b = data.homography_matrices("broadcast")
    main, centre, smooth = main_camera(H_b, cams_b, err_b, live)
    describe("main camera", cams_b, main)
    import rules
    precise = valid & live & (err_b <= ERR_MAX) & cams_b["ok"]
    other = precise & ~main
    lens = np.array([e - s for s, e in rules.runs(other)]) / data.FPS
    print("  of live frames with a precise homography the main camera keeps {:.1%}; the rest: {} stretches, "
          "{:.0f} s, {} of them 2 s or longer ({:.0f} s)".format(
              main[precise].mean(), len(lens), lens.sum(), int(np.sum(lens >= 2)), lens[lens >= 2].sum()))
    with np.errstate(invalid="ignore"):
        sm = smooth[precise]
    for lo, hi in ((0, MAIN_ANGLE), (MAIN_ANGLE, 20), (20, 45), (45, 180)):
        sel = (sm > lo) & (sm <= hi) if lo else (sm <= hi)
        print("    smoothed angle {:>3.0f}-{:<3.0f} deg: {:6.1%} of those frames".format(lo, hi, np.mean(sel)))
    notlive = valid & ~live & cams_b["ok"]
    print("  replay / mismatch frames that look like the main camera: {:.0%} (excluded anyway: not live)".format(
        np.mean(smooth[notlive] <= MAIN_ANGLE)))

    # The two cameras in the same coordinates: the broadcast's pitch is mirrored onto tactical.
    flip = tuple(bool(v) for v in alignment["flip"])
    cb = centre.copy()
    if flip[0]:
        cb[0] = data.PITCH[0] - cb[0]
    if flip[1]:
        cb[1] = data.PITCH[1] - cb[1]
    ct = np.median(cams_t["centre"][cams_t["ok"]], axis=0)
    print("\nin tactical coordinates: tactical camera at ({:.1f}, {:.1f}, {:.1f}) m, broadcast at "
          "({:.1f}, {:.1f}, {:.1f}) m; {:.1f} m apart".format(*ct, *cb, float(np.linalg.norm(ct - cb))))


CUTS = os.path.join(data.CACHE_DIR, "broadcast_cuts.csv")
USABLE = os.path.join(data.CACHE_DIR, "broadcast_usable.npz")
CUT_STATES = ("usable", "cut-away", "no homography", "imprecise homography", "no detections", "cut")


def write_cuts():
    """The broadcast's shots: split at the image cuts (model/shots.py), and a shot whose people are,
    on the median, more than HEIGHT_RATIO_MAX times taller than its homography allows is a cut-away
    as a whole -- a close-up under the wide shot's homography. Writes the per-frame mask teams.py
    reads (USABLE: a homography, error at most ERR_MAX, not in a cut-away shot) and, for the
    viewer, CUTS: runs (start, stop exclusive, state, median ratio) of CUT_STATES, then the cuts
    themselves as one-frame "cut" rows. Our detections and the broadcast's own homography only."""
    import csv
    import ball_track
    import detect_all
    import shots
    import teams
    bro = data.load("broadcast")
    H = data.homography_matrices("broadcast")
    cams = recover(H)
    err = data.homography_errors("broadcast")
    covered = np.zeros(bro.n, dtype=bool)
    ratio = np.full(bro.n, np.nan)
    for src, s, c in detect_all.chunks():
        if src != "broadcast":
            continue
        covered[s:s + c] = True
        f, b, _ = teams.load("broadcast", s, c)
        feet = teams.to_pitch(H, f, np.column_stack([b[:, 0], b[:, 1] + b[:, 3] / 2]))
        on = np.flatnonzero(np.isfinite(feet[:, 0]) & (ball_track.outside(np.nan_to_num(feet, nan=1e9)) <= teams.OFF_PITCH))
        r = frame_height_ratio(cams, f[on], b[on, 3], feet[on], teams.PLAYER_HEIGHT_M, bro.n)
        ratio[s:s + c] = r[s:s + c]
    with np.errstate(invalid="ignore"):
        has = bro.valid_h & (err <= ERR_MAX)
    cut = shots.cuts()
    away = np.zeros(bro.n, dtype=bool)
    shot_ratio = np.full(bro.n, np.nan)
    bounds = np.unique(np.r_[0, cut, bro.n])
    for a, e in zip(bounds[:-1], bounds[1:]):
        r = ratio[a:e][has[a:e]]
        r = r[np.isfinite(r)]
        if len(r):
            shot_ratio[a:e] = np.median(r)
            away[a:e] = shot_ratio[a] > HEIGHT_RATIO_MAX
    usable = has & ~away
    np.savez_compressed(USABLE, usable=usable, covered=covered)
    state = np.where(~covered, 4, np.where(~bro.valid_h, 2, np.where(~has, 3, np.where(away, 1, 0)))).astype(np.int8)
    edges = np.r_[0, np.flatnonzero(np.diff(state) | np.isin(np.arange(1, bro.n), cut)) + 1, bro.n]
    with open(CUTS, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["start", "stop", "state", "ratio"])
        for a, e in zip(edges[:-1], edges[1:]):
            r = shot_ratio[a]
            out.writerow([int(a), int(e), int(state[a]), round(float(r), 2) if np.isfinite(r) else ""])
        for f in cut:
            out.writerow([int(f), int(f) + 1, 5, ""])
    shots_away = int(sum(away[a] for a in bounds[:-1]))
    print("broadcast frames by state: " + ", ".join("{} {:.1%}".format(n, (state == k).mean()) for k, n in enumerate(CUT_STATES[:5])))
    print("{} cuts, {} shots; {} cut-away shots -> {}, {}".format(len(cut), len(bounds) - 1, shots_away,
                                                                os.path.relpath(CUTS, os.path.dirname(HERE)),
                                                                os.path.relpath(USABLE, os.path.dirname(HERE))))


def broadcast_usable():
    """(n,) the mask write_cuts() saved: a homography, error at most ERR_MAX, not in a cut-away shot."""
    if not os.path.isfile(USABLE):
        raise SystemExit("no {}: run model/shots.py features, then model/cameras.py cuts".format(USABLE))
    return np.load(USABLE)["usable"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cuts":
        write_cuts()
    else:
        main()
