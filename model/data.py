"""The provided tracking as numpy arrays, parsed once and cached.

Parsing the raw ATD text costs about a minute per recording, so the arrays are cached as .npz
under data/.model/ -- inside data/, so the same gitignore that keeps the match data out of git
keeps these out too. The cache is keyed on the source file's size and mtime.

Everything here is in metres, on each recording's own video-frame index (ATD frame_id - 1).
"""

import csv
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, SOURCES, asset_paths, half_video_blocks  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, ".model")
CACHE_VERSION = 1
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]  # metres; the ATD is normalised 0-1
HOME, AWAY = 0, 1


class Recording(object):
    """One camera's tracking.

    players  (n, slots, 2)  x, y in metres, NaN where the slot is not tracked
    speed    (n, slots)     m/s as exported, NaN where missing
    ball     (n, 2)         x, y in metres, NaN where the ball is not tracked
    team     (slots,)       HOME / AWAY per track slot, from the metadata
    valid_h  (n,)           homography exists on this frame
    """

    def __init__(self, source, players, speed, ball, team, valid_h):
        self.source = source
        self.players = players
        self.speed = speed
        self.ball = ball
        self.team = team
        self.valid_h = valid_h
        self.n = len(ball)


def _stamp(*paths):
    return [[os.path.getsize(p), os.path.getmtime(p)] for p in paths] + [CACHE_VERSION]


def cached(key, paths, build):
    """build() -> dict of arrays, cached as data/.model/<key>.npz while `paths` are unchanged."""
    npz_path = os.path.join(CACHE_DIR, key + ".npz")
    meta_path = os.path.join(CACHE_DIR, key + ".json")
    stamp = _stamp(*paths)
    try:
        with open(meta_path) as handle:
            if json.load(handle) == stamp:
                with np.load(npz_path) as stored:
                    return {name: stored[name] for name in stored.files}
    except (OSError, ValueError):
        pass
    out = build()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(npz_path, **out)
    with open(meta_path, "w") as handle:
        json.dump(stamp, handle)
    return out


def slot_teams(meta_path):
    """HOME / AWAY per track slot. The first team in the metadata is home (see baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def _parse_atd(raw_path, n_frames, n_slots):
    """Raw ATD -> (players x,y,speed in 0-1 units, ball x,y). One streaming pass."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    width = 3 * n_slots
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            fields = parts[1].replace(";", ",").split(",")
            if len(fields) == width:
                # Fast path: every group is exactly x,y,speed, NaN included.
                players[frame] = np.array(fields, dtype=np.float32).reshape(n_slots, 3)
            else:
                for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                    values = group.split(",")
                    for k in range(min(3, len(values))):
                        try:
                            players[frame, slot, k] = float(values[k])
                        except ValueError:
                            pass
            values = parts[2].split(",")
            if len(values) >= 2:
                try:
                    ball[frame] = (float(values[0]), float(values[1]))
                except ValueError:
                    pass
    return players, ball


def _homography_valid(path, n_frames):
    valid = np.zeros(n_frames, dtype=bool)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = int(row["frame"])
            if 0 <= frame < n_frames:
                valid[frame] = row["valid"].strip() in ("1", "1.0", "True", "true")
    return valid


def load(source="tactical", game=GAME):
    """A Recording for one source, from the cache when possible."""
    paths = asset_paths(game, source=source)
    n_frames = SOURCES[source]["n_video_frames"]
    team = slot_teams(paths["atd_meta"])
    length, width = PITCH

    def build():
        players, ball = _parse_atd(paths["atd_raw"], n_frames, len(team))
        return {
            "players": np.stack([players[..., 0] * length, players[..., 1] * width], axis=-1),
            "speed": players[..., 2],
            "ball": np.stack([ball[:, 0] * length, ball[:, 1] * width], axis=-1),
            "valid_h": _homography_valid(paths["homography"], n_frames),
        }

    print("  loading {} tracking ...".format(source))
    arrays = cached(source + "_tracking", [paths["atd_raw"], paths["atd_meta"],
                                           paths["homography"]], build)
    return Recording(source, arrays["players"], arrays["speed"], arrays["ball"], team,
                     arrays["valid_h"])


def homography_matrices(source="tactical", game=GAME):
    """(n, 3, 3) float32 pitch (0-1) -> pixel matrices, NaN where no homography exists."""
    path = asset_paths(game, source=source)["homography"]
    n_frames = SOURCES[source]["n_video_frames"]

    def build():
        out = np.full((n_frames, 9), np.nan, dtype=np.float32)
        names = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                frame = int(row["frame"])
                if 0 <= frame < n_frames and row["valid"].strip() in ("1", "1.0", "True", "true"):
                    try:
                        out[frame] = [float(row[name]) for name in names]
                    except ValueError:
                        pass
        return {"H": out.reshape(n_frames, 3, 3)}

    print("  loading {} homography ...".format(source))
    return cached(source + "_homography", [path], build)["H"]


def homography_errors(source="tactical", game=GAME):
    """(n,) float32 reprojection error of each frame's homography (pixels), NaN where none."""
    path = asset_paths(game, source=source)["homography"]
    n_frames = SOURCES[source]["n_video_frames"]

    def build():
        out = np.full(n_frames, np.nan, dtype=np.float32)
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                frame = int(row["frame"])
                if 0 <= frame < n_frames:
                    try:
                        out[frame] = float(row["reprojection_error"])
                    except ValueError:
                        pass
        return {"err": out}

    return cached(source + "_homography_err", [path], build)["err"]


def smart_tagging():
    return load_smart_tagging.load()


def halves(game=GAME):
    """[(half, first_video_frame, last_video_frame)] of the graded halves."""
    return half_video_blocks(game)


# Published in DATA.md / FAQ.md: the first half is graded from its kick-off; the second half is
# graded from 110808 but kicks off at 113742.
KICKOFFS = dict(zip((1, 2), load_smart_tagging.KICKOFF_VIDEO_FRAMES.values()))
