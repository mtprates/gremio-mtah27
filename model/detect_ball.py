"""Our own ball and player detections from the video, to compare with (and later replace or fuse
with) the ATD.

    .venv/Scripts/python model/detect_ball.py --source tactical --start 11185 --count 3000
    .venv/Scripts/python model/detect_ball.py --source broadcast --start 7171 --count 3000

A COCO-pretrained YOLO ("sports ball", class 32) as a baseline before any fine-tuning. The tactical
video is a wide shot where the ball is a few pixels, so each frame is run as overlapping tiles at
native resolution; the broadcast is zoomed in and runs whole at 1280 px. Every detection above
CONF_MIN goes to data/.model/detections/<source>_<start>_<count>.csv (frame, x, y, w, h, conf in
pixels) -- inside data/, so gitignored with the rest. Players ("person") come from the same pass,
above PERSON_CONF, to data/.model/detections/persons-<source>-<start>-<count>.csv (same columns,
box centre). Prints counts only.

--ball-model takes the balls from a model fine-tuned on public data (model/tune_ball.py) in a
second pass, the players still from --model; --ball-imgsz runs the ball pass at another input size.
Either writes the same files, over the COCO ones.
"""

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import data  # noqa: E402
from games import SOURCES, asset_paths  # noqa: E402

# Params
MODEL = "yolo11m.pt"        # COCO-pretrained; downloaded by ultralytics on first use, kept in
WEIGHTS_DIR = os.path.join(os.path.dirname(HERE), "weights")  # ... here (gitignored)
PERSON_CLASS = 0           # COCO "person"
PERSON_CONF = 0.25
PERSON_OVERLAP = 0.6        # a person box this much inside another (of the smaller's area) is the
                            # same person seen from two tiles, or a part of them
CONF_MIN = 0.05             # keep weak detections too; the comparison sweeps the threshold
TILE = 640                  # px, tactical tiles at native resolution ...
TILE_X = (0, 427, 853, 1280)  # ... 4 x 2 over 1920 x 1080, ~210 px overlap
TILE_Y = (0, 440)
MERGE_PX = 8                # detections of one ball from overlapping tiles
# Shirt colour per player box (model/teams.py clusters these into teams).
TORSO = (0.2, 0.5)          # share of the box height from its top: below the head, above the shorts
TORSO_WIDTH = 0.5           # share of the box width, centred: the body, not the arms or what is beside
GRASS_HUE = (35, 85)        # OpenCV hue (0-180) of grass ...
GRASS_SAT = 40              # ... at this saturation or more: dropped from the torso
SHIRT_MIN_PX = 6            # fewer shirt pixels than this: no colour
HUE_BINS = 6
SHIRT_COLUMNS = ["shirt_px", "L", "a", "b"] + ["hue{}".format(k) for k in range(HUE_BINS)]
BROADCAST_IMGSZ = 1280
OUT_DIR = os.path.join(data.CACHE_DIR, "detections")


def video_path(source):
    """The video games.py names, or else the one .mp4 in its folder starting with the stem -- the
    bundle has been delivered with a numbered suffix."""
    import glob
    path = asset_paths(source=source)["video"]
    if os.path.isfile(path):
        return path
    found = sorted(glob.glob(os.path.join(os.path.dirname(path), SOURCES[source]["stem"] + "*.mp4")))
    if not found:
        raise SystemExit("no {} video next to {}".format(source, path))
    return found[0]


def weights(name):
    """A model file: a path as given, else in WEIGHTS_DIR, downloaded there on first use."""
    import shutil
    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # no analytics/crash reports to Ultralytics: match data stays local
    if os.path.isfile(name):
        return YOLO(name)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        YOLO(os.path.basename(name))  # downloads into the working directory
        shutil.move(os.path.basename(name), path)
    return YOLO(path)


def ball_class(model):
    """The ball's class id: "ball" in a model of our own (model/tune_ball.py), else COCO's
    "sports ball"."""
    ids = {n: k for k, n in model.names.items()}
    k = ids.get("ball", ids.get("sports ball"))
    if k is None:
        raise SystemExit("no ball class in {}".format(model.names))
    return int(k)


def passes(model, ball_model=None, imgsz=TILE, ball_imgsz=None):
    """[(model, classes, imgsz, ball class)], one per predict call on a frame: one when the
    players and the ball come from the same model at the same size, else one each."""
    ball_model, ball_imgsz = ball_model or model, ball_imgsz or imgsz
    ball = ball_class(ball_model)
    if ball_model is model and ball_imgsz == imgsz:
        return [(model, [ball, PERSON_CLASS], imgsz, ball)]
    return [(model, [PERSON_CLASS], imgsz, None), (ball_model, [ball], ball_imgsz, ball)]


def _split(boxes, ball, ox=0, oy=0):
    """(balls, persons) as [(cx, cy, w, h, conf)] from one ultralytics result, shifted by a tile's
    origin."""
    balls, persons = [], []
    for (cx, cy, w, h), c, k in zip(boxes.xywh.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        d = (cx + ox, cy + oy, w, h, c)
        if int(k) == ball:
            balls.append(d)
        elif c >= PERSON_CONF:
            persons.append(d)
    return balls, persons


def _overlap(a, b):
    """Intersection over the smaller box's area, for (cx, cy, w, h, ...) boxes."""
    ix = min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2)
    iy = min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / max(1e-9, min(a[2] * a[3], b[2] * b[3]))


def detect_tiles(runs, frame):
    """runs: passes(..., imgsz=TILE)."""
    tiles = [frame[y:y + TILE, x:x + TILE] for y in TILE_Y for x in TILE_X]
    origins = [(x, y) for y in TILE_Y for x in TILE_X]
    balls, persons = [], []
    for model, classes, imgsz, ball in runs:
        for result, (ox, oy) in zip(model.predict(tiles, imgsz=imgsz, classes=classes, conf=CONF_MIN,
                                                  verbose=False), origins):
            b, p = _split(result.boxes, ball, ox, oy)
            balls += b
            persons += p
    balls.sort(key=lambda d: -d[4])
    merged = []
    for d in balls:
        if all(abs(d[0] - m[0]) > MERGE_PX or abs(d[1] - m[1]) > MERGE_PX for m in merged):
            merged.append(d)
    # A person cut by a tile border is found twice, whole in one tile and in part in the other:
    # keep the bigger box.
    persons.sort(key=lambda d: -d[2] * d[3])
    people = []
    for d in persons:
        if all(_overlap(d, m) < PERSON_OVERLAP for m in people):
            people.append(d)
    return merged, people


def detect_whole(runs, frame):
    """runs: passes(..., imgsz=BROADCAST_IMGSZ)."""
    balls, persons = [], []
    for model, classes, imgsz, ball in runs:
        b, p = _split(model.predict(frame, imgsz=imgsz, classes=classes, conf=CONF_MIN, verbose=False)[0].boxes, ball)
        balls += b
        persons += p
    return balls, persons


def shirt(lab, hsv, cx, cy, w, h):
    """A player's shirt colour from the frame (OpenCV Lab and HSV images): the torso -- TORSO of
    the box height from the top, the middle TORSO_WIDTH of its width -- without grass-green
    pixels. [n pixels, median L, a, b, then the hue histogram (HUE_BINS shares)]; NaNs and zeros
    when too few pixels remain."""
    H, W = lab.shape[:2]
    top = cy - h / 2
    y0, y1 = int(round(top + TORSO[0] * h)), int(round(top + TORSO[1] * h))
    x0, x1 = int(round(cx - TORSO_WIDTH * w / 2)), int(round(cx + TORSO_WIDTH * w / 2))
    y0, y1, x0, x1 = max(0, y0), min(H, max(y1, y0 + 1)), max(0, x0), min(W, max(x1, x0 + 1))
    crop_hsv = hsv[y0:y1, x0:x1].reshape(-1, 3)
    crop_lab = lab[y0:y1, x0:x1].reshape(-1, 3)
    grass = (crop_hsv[:, 0] >= GRASS_HUE[0]) & (crop_hsv[:, 0] <= GRASS_HUE[1]) & (crop_hsv[:, 1] >= GRASS_SAT)
    keep = ~grass
    n = int(keep.sum())
    if n < SHIRT_MIN_PX:
        return [n, "", "", ""] + [0] * HUE_BINS
    import numpy as np
    med = np.median(crop_lab[keep], axis=0)
    # Hue only where there is colour to speak of; grey / white / black pixels count in no bin.
    coloured = keep & (crop_hsv[:, 1] >= 50) & (crop_hsv[:, 2] >= 40)
    hist = np.histogram(crop_hsv[coloured, 0], bins=HUE_BINS, range=(0, 180))[0] / max(1, n)
    return [n] + [int(v) for v in med] + [round(float(v), 3) for v in hist]


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("tactical", "broadcast"), required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ball-model", help="balls from this model instead (model/tune_ball.py promote); "
                                         "players still from --model")
    ap.add_argument("--ball-imgsz", type=int, help="input size of the ball pass (default: the players')")
    args = ap.parse_args()

    model = weights(args.model)
    ball_model = weights(args.ball_model) if args.ball_model else None
    video = cv2.VideoCapture(video_path(args.source))
    if not video.isOpened():
        raise SystemExit("cannot open the {} video".format(args.source))
    video.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    detect = detect_tiles if args.source == "tactical" else detect_whole
    runs = passes(model, ball_model, TILE if args.source == "tactical" else BROADCAST_IMGSZ, args.ball_imgsz)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "{}_{}_{}.csv".format(args.source, args.start, args.count))
    path_p = os.path.join(OUT_DIR, "persons-{}-{}-{}.csv".format(args.source, args.start, args.count))
    t0, with_ball, total, people = time.time(), 0, 0, 0
    with open(path, "w", newline="") as handle, open(path_p, "w", newline="") as handle_p:
        out, out_p = csv.writer(handle), csv.writer(handle_p)
        out.writerow(["frame", "x", "y", "w", "h", "conf"])
        out_p.writerow(["frame", "x", "y", "w", "h", "conf"] + SHIRT_COLUMNS)
        for i in range(args.count):
            ok, frame = video.read()
            if not ok:
                break
            balls, persons = detect(runs, frame)
            with_ball += bool(balls)
            total += 1
            people += len(persons)
            for x, y, w, h, c in balls:
                out.writerow([args.start + i, round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(c, 3)])
            lab, hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for x, y, w, h, c in persons:
                out_p.writerow([args.start + i, round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(c, 3)] +
                               shirt(lab, hsv, x, y, w, h))
    dt = time.time() - t0
    print("{}: {} frames in {:.0f} s ({:.1f} fps); frames with a detection (conf >= {}): {} ({:.0%})".format(
        args.source, total, dt, total / max(dt, 1e-9), CONF_MIN, with_ball, with_ball / max(total, 1)))
    print("persons (conf >= {}): {:.1f} per frame".format(PERSON_CONF, people / max(total, 1)))
    print("detections -> {}".format(os.path.relpath(path, os.path.dirname(HERE))))


if __name__ == "__main__":
    main()
