"""Rule 16: a tactical track is cut where the person in it changes, and the pieces are re-slotted.

Terms (docs/TECHNICAL.md): a *slot* is a column of an ATD export; an *object* is one slot's
position on one frame; a *track* is a run of frames one slot is tracked without a break; a
*person* is who is really there. The rules treat a tactical slot's track as one person (rules 3.b,
3.c, 6). The tracker does not always keep to that: at 1H 11400 an away player (tactical slot 31)
went behind a home player (slot 18); slot 31's track ended and slot 18's track slid onto the away
player without a jump, while the broadcast kept both persons in their own slots (2: home, 34:
away) throughout.

The broadcast says when that happens. A tactical track's partner (rule 15's pairing) changes from
one broadcast slot `a` to another `b`, each held for CUT_HOLD frames, with `a` still tracked after
the change (its person did not vanish) and `b` tracked before it (nor did this one). Then the
person in the tactical track changed. It is cut there, and the piece after the cut goes to:

  - the tactical slot that had `b` as partner just before, if that slot's track has ended -- the
    tracker dropped the person and kept following them under another slot (slot 31 at 11400);
  - otherwise a new column, on the team of the tactical slot unless `a` and `b` are on different
    broadcast teams.

If that previous slot is still tracked, the swap may be the broadcast's, or both tactical tracks
traded persons: ambiguous, so no cut. The rules then run on the re-slotted recording unchanged:
the slot the person left ends at the cut, and rule 6 bridges the person from their broadcast slot
(`a`) as any other dropout.
"""

import numpy as np

import dropouts
import rules
from data import FPS, Recording

# Params
CUT_HOLD = FPS // 2         # frames: each partner held this long, either side of the change
CUT_LOOKBACK = 2 * FPS      # frames: how far back `b`'s previous tactical slot is looked for
CUT_RETURN = 0.1            # the piece after a cut is paired with `a` again on at most this share
                            # of its paired frames -- else the person did not change
CUT_TEAM_SHARE = 0.6        # the piece's team is its partners' team on at least this share of its
                            # paired frames (rule 3.b's whole-track test), else no cut
REUSE_GAP = dropouts.INTERP_MAX + FPS  # frames between two persons' pieces in one new column


def find_cuts(players_t, pairs, players_b, alignment, bro_team, tac_team):
    """[(slot, frame, stop, a, b, previous slot of b or -1, team)]: a track cut at `frame`, its
    piece running to `stop` (the next cut or the track's end) on `team`. players_b in tactical
    orientation, on broadcast frames; pairs on tactical frames (rule 15); bro_team per broadcast
    slot, tac_team per tactical slot.

    Only where the partner before the change, `a`, was on the tactical slot's own team: else the
    pairing before was the error, not the person after. The first version read those as cuts and
    relabelled 101 pieces, a minute long at the median, against 99% of their partners."""
    # The broadcast's teams in tactical order: the exports may list the two teams either way.
    rows, cols = np.nonzero(pairs[:, :len(tac_team)] >= 0)
    if len(rows) and np.mean(bro_team[pairs[rows, cols]] == tac_team[cols]) < 0.5:
        bro_team = 1 - bro_team
    tracked_t = ~np.isnan(players_t[..., 0])
    tracked_b = ~np.isnan(players_b[..., 0])
    b_of_t = alignment["b_of_t"]

    def b_tracked(obj, t):
        b = b_of_t[t] if 0 <= t < len(b_of_t) else -1
        return b >= 0 and bool(tracked_b[b, obj])

    cuts = []
    for slot in range(players_t.shape[1]):
        for s0, e0 in rules.runs(tracked_t[:, slot]):
            held = []  # (start, stop, partner), gaps without a partner bridged over
            for s, e, obj in rules.value_runs(pairs[s0:e0, slot].astype(np.int64)):
                if held and held[-1][2] == obj:
                    held[-1] = (held[-1][0], e + s0, obj)
                else:
                    held.append((s + s0, e + s0, obj))
            found = []
            for (sa, ea, a), (sb, eb, b) in zip(held, held[1:]):
                if ea - sa < CUT_HOLD or eb - sb < CUT_HOLD:
                    continue
                t = sb
                if bro_team[a] != tac_team[slot]:
                    continue  # the pairing before the change was the error
                if not (b_tracked(a, t) and b_tracked(a, t + CUT_HOLD - 1) and b_tracked(b, t - 1)):
                    continue
                window = pairs[max(0, t - CUT_LOOKBACK):t]
                rows, cols = np.nonzero(window == b)
                before = [(r, c) for r, c in zip(rows, cols) if c != slot]
                prev = int(max(before)[1]) if before else -1
                if prev >= 0 and tracked_t[t, prev]:
                    continue  # both tactical tracks go on: whose swap it is, is not clear
                found.append((t, int(a), int(b), prev))
            # Each cut must hold over its whole piece, up to the next candidate or the track's end.
            kept = []
            for i, (t, a, b, prev) in enumerate(found):
                stop = found[i + 1][0] if i + 1 < len(found) else e0
                partners = pairs[t:stop, slot].astype(np.int64)
                partners = partners[partners >= 0]
                if not len(partners) or np.mean(partners == a) > CUT_RETURN:
                    continue
                home = float(np.mean(bro_team[partners] == 0))
                if max(home, 1 - home) < CUT_TEAM_SHARE:
                    continue
                kept.append((t, a, b, prev, 0 if home >= 0.5 else 1))
            found = kept
            for i, (t, a, b, prev, team) in enumerate(found):
                stop = found[i + 1][0] if i + 1 < len(found) else e0
                cuts.append((slot, t, stop, a, b, prev, team))
    return cuts


def reslot(tac, pairs, cuts):
    """The tactical recording and pairing with every cut piece moved to its person's column.
    Returns (recording, pairs, moves [(slot, start, stop, column)]); new columns are appended
    after the export's slots."""
    players, speed, team = tac.players.copy(), tac.speed.copy(), list(tac.team)
    pairs = pairs.copy()
    extra_players, extra_speed, extra_pairs = [], [], []
    moves = []
    taken = {}  # column -> [(start, stop)] filled by moved pieces
    for slot, start, stop, a, b, prev, want in sorted(cuts, key=lambda c: c[1]):
        piece = slice(start, stop)
        free = prev >= 0 and team[prev] == want and np.isnan(players[piece, prev, 0]).all() and \
            not any(s < stop and start < e for s, e in taken.get(prev, []))
        if free:
            column = prev
        else:
            # A new column on the piece's team -- an earlier one, where it is free.
            # Pieces of different persons in one column must not read as one track (rules 3.c, 6)
            # or as one person coming back (rule 6 interpolates a slot back within INTERP_MAX).
            reuse = [c for c in range(players.shape[1], len(team)) if team[c] == want and
                     not any(s < stop + REUSE_GAP and start < e + REUSE_GAP for s, e in taken.get(c, []))]
            if reuse:
                column = reuse[0]
            else:
                column = players.shape[1] + len(extra_players)
                team.append(want)
                extra_players.append(np.full(players.shape[::2], np.nan, dtype=players.dtype))
                extra_speed.append(np.full(len(speed), np.nan, dtype=speed.dtype))
                extra_pairs.append(np.full(len(pairs), -1, dtype=pairs.dtype))
        if column < players.shape[1]:
            players[piece, column], speed[piece, column], pairs[piece, column] = \
                players[piece, slot], speed[piece, slot], pairs[piece, slot]
        else:
            k = column - players.shape[1]
            extra_players[k][piece], extra_speed[k][piece], extra_pairs[k][piece] = \
                players[piece, slot], speed[piece, slot], pairs[piece, slot]
        players[piece, slot], speed[piece, slot], pairs[piece, slot] = np.nan, np.nan, -1
        taken.setdefault(column, []).append((start, stop))
        moves.append((int(slot), int(start), int(stop), int(column)))
    if extra_players:
        players = np.concatenate([players, np.stack(extra_players, axis=1)], axis=1)
        speed = np.concatenate([speed, np.stack(extra_speed, axis=1)], axis=1)
        pairs = np.concatenate([pairs, np.stack(extra_pairs, axis=1)], axis=1)
    out = Recording(tac.source, players, speed, tac.ball, np.array(team, dtype=np.int8), tac.valid_h)
    return out, pairs, moves
