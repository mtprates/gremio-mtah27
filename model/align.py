"""Map the broadcast recording onto the tactical timeline, and say what the broadcast shows.

The broadcast is not frame-aligned to the tactical video and no mapping is provided (DATA.md).
But both ATDs are in pitch coordinates, so the same moment has the same layout of players in
both. A broadcast frame is matched to a tactical frame by the median distance from each
broadcast object to its nearest tactical object -- one-directional, because the broadcast only
ever sees part of the pitch.

    1. seeds: every 3 s of broadcast, search the whole tactical video for the best match
    2. stretches: consecutive confident seeds with the same offset (t - b)
    3. the main timeline: the heaviest chain of stretches that moves forward in tactical time.
       Replays map *back* in time, so they fall out of the chain on their own.
    4. per frame: one offset per half, the seed-weighted median of the main timeline in that
       half (the same offsets the viewer links the feeds with), then score every frame. With HALF_OFFSETS off,
       each stretch keeps its own offset instead.

What comes out, per tactical frame: the broadcast frame showing it, whether that frame is a live
wide shot, a replay / mismatch or a close-up (rule 4), and for each tactical object the
broadcast object standing on the same spot (rule 3.a).
"""

import numpy as np

from data import FPS, KICKOFFS, PITCH, halves

# Params
HALF_OFFSETS = True       # one offset per half instead of one per stretch
SEED_STRIDE_B = 3 * FPS   # broadcast frames between seeds
SEED_STRIDE_T = 10        # tactical candidate spacing in the global search (0.4 s)
MIN_PLAYERS = 5           # broadcast objects needed for a seed
GOOD_COST = 1.6           # m: median nearest-object distance of a confident match
UNIQUE_RATIO = 1.4        # the best match > 5 s away must be this much worse
SAME_OFFSET = 4           # frames: seeds this close in offset belong to one stretch
MAX_SEED_GAP = 60 * FPS   # broadcast frames: a longer silence starts a new stretch
LIVE_COST = 2.0           # m: per-frame "this broadcast frame shows this tactical moment"
PAIR_RADIUS = 1.5         # m: cross-camera pairing of objects (rule 3.a)
FINE_TUNE = 4             # frames either side tried when refining a stretch's offset
CHUNK = 2000
BIG = 1e4

STATE_NONE, STATE_LIVE, STATE_REPLAY, STATE_CLOSEUP = 0, 1, 2, 3
STATE_NAMES = {STATE_NONE: "none", STATE_LIVE: "live", STATE_REPLAY: "replay/mismatch",
               STATE_CLOSEUP: "close-up"}
ORIENTATIONS = ((False, False), (True, False), (False, True), (True, True))
MISSING = np.iinfo(np.int32).min


def orient(points, flip):
    """Mirror pitch coordinates in x and/or y, so the broadcast export matches the tactical one."""
    out = np.array(points, dtype=np.float32, copy=True)
    if flip[0]:
        out[..., 0] = PITCH[0] - out[..., 0]
    if flip[1]:
        out[..., 1] = PITCH[1] - out[..., 1]
    return out


def _filled(players):
    """NaN -> far away, so a missing slot is never anybody's nearest neighbour."""
    return np.where(np.isnan(players), BIG, players).astype(np.float32)


def _costs(tac_frames, points):
    """Median distance from each broadcast point to its nearest tactical object, per candidate.

    tac_frames (C, S, 2), points (m, 2) -> (C,)
    """
    out = np.empty(len(tac_frames), dtype=np.float32)
    for start in range(0, len(tac_frames), CHUNK):
        block = tac_frames[start:start + CHUNK]
        diff = block[:, None, :, :] - points[None, :, None, :]
        nearest = np.sqrt((diff ** 2).sum(-1)).min(-1)
        out[start:start + CHUNK] = np.median(nearest, axis=1)
    return out


def _seed_points(bro, b, flip):
    points = bro.players[b]
    points = points[~np.isnan(points[:, 0])]
    return orient(points, flip) if len(points) >= MIN_PLAYERS else None


def _search(tpos, candidates, coarse, points, n_tac):
    """Best tactical frame for one broadcast frame: coarse global pass, then frame-exact."""
    costs = _costs(coarse, points)
    t0 = int(candidates[int(np.argmin(costs))])
    fine = np.arange(max(0, t0 - SEED_STRIDE_T), min(n_tac, t0 + SEED_STRIDE_T + 1))
    fine_costs = _costs(tpos[fine], points)
    best = int(np.argmin(fine_costs))
    t, cost = int(fine[best]), float(fine_costs[best])
    far = np.abs(candidates - t) > 5 * FPS
    second = float(costs[far].min()) if far.any() else BIG
    return t, cost, second


def choose_orientation(tac, bro, tpos, candidates, coarse, n_probe=24):
    """Try the four mirrorings on a spread of broadcast frames; keep the one that matches best."""
    usable = [b for b in range(0, bro.n, SEED_STRIDE_B)
              if bro.valid_h[b] and np.sum(~np.isnan(bro.players[b, :, 0])) >= 8]
    probes = usable[::max(1, len(usable) // n_probe)][:n_probe]
    scores = []
    for flip in ORIENTATIONS:
        best = [_search(tpos, candidates, coarse, _seed_points(bro, b, flip), tac.n)[1]
                for b in probes]
        scores.append(float(np.median(best)) if best else BIG)
    return ORIENTATIONS[int(np.argmin(scores))], scores


def find_seeds(tac, bro, flip, tpos, candidates, coarse):
    seeds = []
    for b in range(0, bro.n, SEED_STRIDE_B):
        if not bro.valid_h[b]:
            continue
        points = _seed_points(bro, b, flip)
        if points is None:
            continue
        t, cost, second = _search(tpos, candidates, coarse, points, tac.n)
        seeds.append((b, t, cost, second))
    return np.array(seeds, dtype=np.float64).reshape(-1, 4)


def stretches(seeds):
    """Confident seeds grouped into stretches of one offset: [(b0, b1, offset, n_seeds)]."""
    good = (seeds[:, 2] < GOOD_COST) & (seeds[:, 3] >= UNIQUE_RATIO * seeds[:, 2])
    out = []
    for b, t, _, _ in seeds[good]:
        offset = t - b
        last = out[-1] if out else None
        if last and abs(offset - last["offsets"][-1]) <= SAME_OFFSET and b - last["b1"] <= MAX_SEED_GAP:
            last["b1"] = b
            last["offsets"].append(offset)
        else:
            out.append({"b0": b, "b1": b, "offsets": [offset]})
    return [(int(s["b0"]), int(s["b1"]), int(round(np.median(s["offsets"]))), len(s["offsets"]))
            for s in out]


def main_chain(found):
    """Heaviest chain of stretches moving forward in both broadcast and tactical time."""
    found = [s for s in found if s[3] >= 2]
    if not found:
        return []
    weight = [s[3] for s in found]
    best, back = list(weight), [-1] * len(found)
    for i, (b0, _, o, n) in enumerate(found):
        for j in range(i):
            pb0, pb1, po, _ = found[j]
            if b0 > pb1 and b0 + o > pb1 + po and best[j] + n > best[i]:
                best[i], back[i] = best[j] + n, j
    i = int(np.argmax(best))
    chain = []
    while i >= 0:
        chain.append(found[i])
        i = back[i]
    return chain[::-1]


def per_frame_offsets(chain, n_bro):
    """Offset for every broadcast frame: each stretch's own, and across a gap between two stretches
    either the shared offset or a split at the midpoint."""
    offsets = np.full(n_bro, MISSING, dtype=np.int32)
    for i, (b0, b1, o, _) in enumerate(chain):
        lo = 0 if i == 0 else None
        if lo is None:
            pb1, po = chain[i - 1][1], chain[i - 1][2]
            lo = pb1 + 1 if abs(po - o) <= SAME_OFFSET else (pb1 + b0) // 2 + 1
        hi = n_bro - 1 if i == len(chain) - 1 else None
        if hi is None:
            nb0, no = chain[i + 1][0], chain[i + 1][2]
            hi = nb0 - 1 if abs(no - o) <= SAME_OFFSET else (b1 + nb0) // 2
        offsets[lo:hi + 1] = o
    return offsets


def offsets_by_half(chain):
    """[o1, o2]: each half's offset (t - b), the seed-weighted median of the main timeline's
    stretches that start in it. A median, not the stretch nearest the kick-off: short stretches
    are a frame or a few off (one 3-seed stretch right after the 2H kick-off is 2 off)."""
    split = halves()[1][1]
    out = []
    for in_half in (lambda t: t < split, lambda t: t >= split):
        picked = [(o, n) for b0, _, o, n in chain if in_half(b0 + o)]
        if not picked:
            return None
        values = np.repeat([o for o, _ in picked], [n for _, n in picked])
        out.append(int(np.median(values)))
    return out


def half_offsets(by_half, n_bro):
    """Offset for every broadcast frame from [o1, o2]. The second half's applies from the start of
    its graded block, so half-time keeps the first's."""
    offsets = np.full(n_bro, MISSING, dtype=np.int32)
    if by_half is None:
        return offsets
    first, second = by_half
    b = np.arange(n_bro)
    offsets[:] = np.where(b + second >= halves()[1][1], second, first)
    return offsets


def frame_costs(tpos, bro, flip, offsets, n_tac, frames=None):
    """Per broadcast frame: match cost at its offset, and for each tactical object the paired
    broadcast slot (-1 none). frames limits the work to a subset (used when fine-tuning)."""
    frames = np.arange(bro.n) if frames is None else np.asarray(frames)
    cost = np.full(len(frames), np.nan, dtype=np.float32)
    pairs = np.full((len(frames), tpos.shape[1]), -1, dtype=np.int16)
    t_all = frames.astype(np.int64) + offsets[frames]
    usable = (offsets[frames] != MISSING) & bro.valid_h[frames] & (t_all >= 0) & (t_all < n_tac)
    for start in range(0, len(frames), CHUNK):
        rows = np.arange(start, min(len(frames), start + CHUNK))
        rows = rows[usable[rows]]
        if not len(rows):
            continue
        bp = orient(bro.players[frames[rows]], flip)                 # (B, Sb, 2), NaN missing
        tp = tpos[t_all[rows]]                                       # (B, S, 2), far missing
        missing = np.isnan(bp[..., 0])
        bp = np.where(np.isnan(bp), -BIG, bp)
        dist = np.sqrt(((bp[:, :, None, :] - tp[:, None, :, :]) ** 2).sum(-1))  # (B, Sb, S)
        nearest = dist.min(-1)
        nearest[missing] = np.nan
        enough = (~missing).sum(1) >= 3
        with np.errstate(all="ignore"):
            cost[rows[enough]] = np.nanmedian(nearest[enough], axis=1)
        dist[missing] = BIG
        partner = dist.argmin(1)                                     # (B, S)
        close = np.take_along_axis(dist, partner[:, None, :], 1)[:, 0, :] <= PAIR_RADIUS
        pairs[rows] = np.where(close, partner, -1)
    return cost, pairs


def fine_tune(tpos, bro, flip, chain, n_tac):
    """Refine each stretch's offset to the frame on a sample of its own frames."""
    tuned = []
    for b0, b1, o, n in chain:
        sample = np.arange(b0, b1 + 1, 5)
        sample = sample[bro.valid_h[sample]]
        best_o, best_cost = o, np.inf
        for candidate in range(o - FINE_TUNE, o + FINE_TUNE + 1):
            offsets = np.full(bro.n, MISSING, dtype=np.int32)
            offsets[sample] = candidate
            cost, _ = frame_costs(tpos, bro, flip, offsets, n_tac, frames=sample)
            score = np.nanmean(np.minimum(cost, 5.0)) if np.isfinite(cost).any() else np.inf
            if score < best_cost:
                best_o, best_cost = candidate, score
        tuned.append((b0, b1, best_o, n))
    return tuned


def align(tac, bro):
    """Everything the rules need from the broadcast, as a dict of arrays."""
    tpos = _filled(tac.players)
    candidates = np.arange(0, tac.n, SEED_STRIDE_T)
    coarse = tpos[candidates]

    flip, orientation_scores = choose_orientation(tac, bro, tpos, candidates, coarse)
    print("    orientation {} (median match cost per mirroring: {})".format(
        flip, ", ".join("{:.2f}".format(s) for s in orientation_scores)))
    seeds = find_seeds(tac, bro, flip, tpos, candidates, coarse)
    found = stretches(seeds)
    chain = fine_tune(tpos, bro, flip, main_chain(found), tac.n)
    by_half = offsets_by_half(chain)
    offsets = half_offsets(by_half, bro.n) if HALF_OFFSETS else per_frame_offsets(chain, bro.n)
    cost_b, pairs_b = frame_costs(tpos, bro, flip, offsets, tac.n)

    # Back onto the tactical timeline.
    b_of_t = np.full(tac.n, -1, dtype=np.int32)
    have = np.flatnonzero(offsets != MISSING)
    t = have.astype(np.int64) + offsets[have]
    keep = (t >= 0) & (t < tac.n)
    b_of_t[t[keep]] = have[keep]

    state = np.full(tac.n, STATE_NONE, dtype=np.int8)
    mapped = np.flatnonzero(b_of_t >= 0)
    b = b_of_t[mapped]
    live = bro.valid_h[b] & (cost_b[b] < LIVE_COST)
    state[mapped] = np.where(~bro.valid_h[b], STATE_CLOSEUP, np.where(live, STATE_LIVE, STATE_REPLAY))

    pairs_t = np.full((tac.n, tpos.shape[1]), -1, dtype=np.int16)
    live_t = mapped[live]
    pairs_t[live_t] = pairs_b[b_of_t[live_t]]

    return {
        "flip": np.array(flip, dtype=bool),
        "seeds": seeds,
        "stretches": np.array(found, dtype=np.int64).reshape(-1, 4),
        "chain": np.array(chain, dtype=np.int64).reshape(-1, 4),
        "half_offsets": np.array(by_half or [], dtype=np.int64),
        "offset_b": offsets,
        "cost_b": cost_b,
        "b_of_t": b_of_t,
        "state_t": state,
        "pairs_t": pairs_t,
    }
