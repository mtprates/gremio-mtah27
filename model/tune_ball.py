"""Fine-tuning the ball detector on public data, so model/detect_ball.py can run a ball model of
our own instead of COCO's "sports ball". Nothing here reads BAR-ATH: training, validation and the
score are on the public dataset's own splits (docs/VALUES.md, "Ball detector fine-tuning").

    .venv/Scripts/python model/tune_ball.py stats                     # what the export holds
    .venv/Scripts/python model/tune_ball.py prepare                   # tiles at tactical ball size
    .venv/Scripts/python model/tune_ball.py eval --model yolo11m.pt   # baseline: COCO sports ball
    .venv/Scripts/python model/tune_ball.py train --name m640
    .venv/Scripts/python model/tune_ball.py kaggle --user <you> --name m640   # or on Kaggle's GPU
    .venv/Scripts/python model/tune_ball.py eval --model external/ball-tune/runs/m640/weights/best.pt
    .venv/Scripts/python model/tune_ball.py mine --name m640          # hard negatives: boots, socks
    .venv/Scripts/python model/tune_ball.py train --name m640-hard --hard m640
    .venv/Scripts/python model/tune_ball.py promote --name m640       # -> weights/ball-m640.pt
    .venv/Scripts/python model/detect_ball.py --source tactical --start 11185 --count 3000 --ball-model ball-m640.pt

Input: a Roboflow YOLO export (train/, valid/, test/, each with images/ and labels/, and
data.yaml) in external/football-ball-detection.v4i.yolov11/ (external/README.md); --src for another
one in the same layout. `prepare` re-splits by whole source clip (SPLIT_CLIPS, --fold picks which),
undoes a stretch to square (the ball's boxes should be round), rescales each labelled image so its
ball is BALL_PX wide, as on the tactical camera, and cuts TILE x TILE tiles the way detect_ball.py
tiles a frame: one around each ball, plus NEG_SHARE tiles away from it. Unlabelled frames are left
out: the export labels a quarter of each clip's frames, so they may hold an unlabelled ball. The
tiles go, one class "ball", to external/ball-tune/<dataset>-fold<k>/, rebuilt each time. `eval` runs a model over the prepared test tiles and scores it: a detection is
right when its centre is within the ball's box size of a labelled ball not yet matched. The scores
are appended to external/ball-tune/scores.csv.
"""

import argparse
import collections
import csv
import datetime
import glob
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import compare_ball  # noqa: E402
import detect_ball  # noqa: E402

# Params
SRC = os.path.join(ROOT, "external", "football-ball-detection.v4i.yolov11")
OUT_ROOT = os.path.join(ROOT, "external", "ball-tune")
RUNS = os.path.join(OUT_ROOT, "runs")
SPLITS = ("train", "valid", "test")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BALL_PX = (4.0, 14.0)       # ball width on the tactical camera: 0.22 m (Law 2) on 1920 px showing
                            # 105 m (the whole pitch) ... 30 m of play
TILE = detect_ball.TILE     # the tiles detect_ball.py runs on
NEG_SHARE = 0.1             # tiles with no ball per tile with one (Ultralytics: 0-10 % background)
TRAIN_SCALE = 0.25          # Ultralytics random zoom +-25 %: BALL_PX stays about 3-17 px
PAD = 114                   # Ultralytics' letterbox grey, where a tile overhangs a small image
SPLIT_CLIPS = {"valid": 2, "test": 2}  # whole source clips held out, so no frame has a twin in train
SEED = 0
TRAIN_BATCH = 4             # tiles per step: the GTX 1660 Ti trains in FP32 (no AMP), and 9 overflowed 6 GB
EPOCHS = 50                 # a small dataset: validation was level by epoch 2-5 of the first run ...
PATIENCE = 15               # ... so stop after this many epochs without a better score
MINE_CONF = 0.25           # a false ball this confident is worth showing again (Ultralytics' default conf)
MINE_COPIES = 1             # extra copies of each such tile
SIZE_BUCKETS = (6, 10, 16)  # px, the recall breakdown in `eval` (reporting only)
THRESHOLDS = (detect_ball.CONF_MIN,) + compare_ball.THRESHOLDS
BATCH = 16                  # tiles per predict call in `eval` (speed only)


def prepared(args):
    """external/ball-tune/<dataset>-fold<k>, or -export for the export's own split."""
    return os.path.join(OUT_ROOT, "{}-{}".format(os.path.basename(os.path.normpath(args.src)),
                                                 "export" if args.fold is None else "fold{}".format(args.fold)))


def read_names(src):
    """{class id: name} from the export's data.yaml."""
    import yaml
    with open(os.path.join(src, "data.yaml")) as handle:
        names = yaml.safe_load(handle)["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def ball_ids(names):
    """The export's classes that are the ball: every name with "ball" in it."""
    ids = {k for k, n in names.items() if "ball" in str(n).lower()}
    if not ids:
        raise SystemExit("no ball class among {}".format(names))
    return ids


def images(folder):
    """[(image path, label path)] of one split of a YOLO export."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "images", "*"))):
        if path.lower().endswith(IMAGE_EXT):
            stem = os.path.splitext(os.path.basename(path))[0]
            out.append((path, os.path.join(folder, "labels", stem + ".txt")))
    return out


def read_boxes(path, ids, W, H):
    """The ball boxes of one image as [(cx, cy, w, h)] in pixels. Polygon labels (segmentation
    exports) become their bounding box."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as handle:
        for line in handle:
            v = line.split()
            if len(v) < 5 or int(float(v[0])) not in ids:
                continue
            if len(v) == 5:
                cx, cy, w, h = (float(x) for x in v[1:])
            else:
                xs, ys = [float(x) for x in v[1::2]], [float(y) for y in v[2::2]]
                cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
            out.append((cx * W, cy * H, w * W, h * H))
    return out


def clip_key(path):
    """The source a Roboflow image came from, roughly: its name before ".rf.<hash>", without the
    original extension and a trailing frame number. Frames of one clip share it."""
    stem = os.path.basename(path).split(".rf.")[0]
    stem = re.sub(r"_(jpe?g|png|bmp|webp)$", "", stem, flags=re.I)
    return re.sub(r"[\W_]*\d+$", "", stem) or stem


def stats(args):
    """What the export holds: classes, image sizes, ball sizes in pixels, and shared sources
    across the splits."""
    import numpy as np
    from PIL import Image
    names = read_names(args.src)
    ids = ball_ids(names)
    print("classes {}; ball: {}".format(names, sorted(names[k] for k in ids)))
    a = dataset_aspect(args.src, ids)
    print("ball box h/w median {:.2f} (1 = as shot; widths below are at the restored aspect)".format(a))
    splits_of = collections.defaultdict(set)
    for split in SPLITS:
        items = images(os.path.join(args.src, split))
        if not items:
            print("\n{}: no images".format(split))
            continue
        sizes, widths, per = collections.Counter(), [], []
        for img, lab in items:
            W, H = Image.open(img).size
            sizes[(W, H)] += 1
            boxes = read_boxes(lab, ids, W, H)
            per.append(len(boxes))
            widths += [min(w * a, h) for _, _, w, h in boxes]
            splits_of[clip_key(img)].add(split)
        per, widths = np.array(per), np.array(widths)
        print("\n{}: {} images, {} balls; images with 0 / 1 / 2+ balls: {} / {} / {}".format(
            split, len(items), int(per.sum()), int((per == 0).sum()), int((per == 1).sum()), int((per >= 2).sum())))
        print("  image sizes: {}".format(", ".join("{}x{} ({})".format(w, h, n) for (w, h), n in sizes.most_common(5))))
        if len(widths):
            q = np.percentile(widths, [5, 25, 50, 75, 95])
            print("  ball width px (min of the box): p5 {:.1f}  p25 {:.1f}  median {:.1f}  p75 {:.1f}  p95 {:.1f}".format(*q))
            print("  already within BALL_PX {}: {:.0%}; smaller: {:.0%}".format(
                BALL_PX, ((widths >= BALL_PX[0]) & (widths <= BALL_PX[1])).mean(), (widths < BALL_PX[0]).mean()))
    shared = sorted(k for k, s in splits_of.items() if len(s) > 1)
    print("\nsources (by file name) in more than one split: {} of {}{}".format(
        len(shared), len(splits_of), "; e.g. " + ", ".join(shared[:5]) if shared else ""))


def crop(img, x0, y0):
    """TILE x TILE of img from (x0, y0), PAD where it overhangs."""
    import numpy as np
    H, W = img.shape[:2]
    tile = np.full((TILE, TILE, 3), PAD, np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + TILE), min(H, y0 + TILE)
    if xb > xa and yb > ya:
        tile[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return tile


def window(rng, size, centre, margin):
    """A tile origin along one axis: the whole image at a random place in the tile when it is
    smaller than one, else a random start that keeps `centre` at least `margin` from the edges."""
    if size <= TILE:
        return -int(rng.integers(0, TILE - size + 1))
    lo, hi = max(0, centre + margin - TILE), min(size - TILE, centre - margin)
    if lo > hi:
        return int(min(max(0, centre - TILE / 2), size - TILE))
    return int(rng.integers(int(lo), int(hi) + 1))


def tile_labels(boxes, x0, y0):
    """YOLO label lines for the balls whose centre is in the tile at (x0, y0), boxes clipped to it."""
    lines = []
    for cx, cy, w, h in boxes:
        if not (x0 <= cx < x0 + TILE and y0 <= cy < y0 + TILE):
            continue
        xa, ya = max(x0, cx - w / 2), max(y0, cy - h / 2)
        xb, yb = min(x0 + TILE, cx + w / 2), min(y0 + TILE, cy + h / 2)
        lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
            ((xa + xb) / 2 - x0) / TILE, ((ya + yb) / 2 - y0) / TILE, (xb - xa) / TILE, (yb - ya) / TILE))
    return lines


def dataset_aspect(src, ids):
    """Median h/w of the ball boxes over all splits. A ball is round: 1 for frames as shot, the
    frames' own aspect for an export stretched to a square (Roboflow "Stretch")."""
    import numpy as np
    from PIL import Image
    ratios = []
    for split in SPLITS:
        for img, lab in images(os.path.join(src, split)):
            W, H = Image.open(img).size
            ratios += [h / w for _, _, w, h in read_boxes(lab, ids, W, H) if w > 0]
    return float(np.median(ratios)) if ratios else 1.0


def split_items(src, fold):
    """{split: [(image, label)]}: the export's own split when fold is None, else by whole source
    clip (clip_key): SPLIT_CLIPS clips each for test and valid, a different set per fold, the
    rest train."""
    import numpy as np
    pooled = {split: images(os.path.join(src, split)) for split in SPLITS}
    if fold is None:
        return pooled
    clips = collections.defaultdict(list)
    for items in pooled.values():
        for item in items:
            clips[clip_key(item[0])].append(item)
    keys = [sorted(clips)[k] for k in np.random.default_rng(SEED).permutation(len(clips))]
    held = SPLIT_CLIPS["test"] + SPLIT_CLIPS["valid"]
    if held >= len(keys):
        raise SystemExit("{} clips: too few to hold out {}".format(len(keys), held))
    start = fold * held
    order = [keys[(start + k) % len(keys)] for k in range(len(keys))]
    of = {"test": order[:SPLIT_CLIPS["test"]], "valid": order[SPLIT_CLIPS["test"]:held], "train": order[held:]}
    for split in SPLITS:
        print("{}: {} clips: {}".format(split, len(of[split]), ", ".join(of[split])))
    return {split: sorted(item for k in of[split] for item in clips[k]) for split in SPLITS}


def prepare(args):
    import cv2
    import numpy as np
    import yaml
    from PIL import Image
    rng = np.random.default_rng(SEED)
    names = read_names(args.src)
    ids = ball_ids(names)
    a = dataset_aspect(args.src, ids)
    print("ball box h/w median {:.2f}: images widened by that".format(a))
    out = prepared(args)
    if os.path.isdir(out):
        shutil.rmtree(out)
    counts = {}
    for split, items in split_items(args.src, args.fold).items():
        if not items:
            continue
        # Scale per image, after the aspect is undone: its median ball to a draw from BALL_PX, never
        # up. Frames without a label are left out (the export labels some frames of each clip only).
        plan = []
        for img, lab in items:
            W, H = Image.open(img).size
            boxes = [(cx * a, cy, w * a, h) for cx, cy, w, h in read_boxes(lab, ids, W, H)]
            if boxes:
                d = float(np.median([min(w, h) for _, _, w, h in boxes]))
                plan.append((img, boxes, min(1.0, rng.uniform(*BALL_PX) / max(d, 1e-9))))
        os.makedirs(os.path.join(out, split, "images"))
        os.makedirs(os.path.join(out, split, "labels"))

        def write(name, tile, lines):
            cv2.imwrite(os.path.join(out, split, "images", name + ".jpg"), tile)
            with open(os.path.join(out, split, "labels", name + ".txt"), "w") as handle:
                handle.write("".join(line + "\n" for line in lines))

        pos, balls, frames = 0, 0, []
        for i, (img_path, boxes, s) in enumerate(plan):
            img = cv2.imread(img_path)
            if img is None:
                continue
            sx, sy = a * s, s
            if (sx, sy) != (1.0, 1.0):
                img = cv2.resize(img, (max(1, round(img.shape[1] * sx)), max(1, round(img.shape[0] * sy))),
                                 interpolation=cv2.INTER_AREA if max(sx, sy) <= 1 else cv2.INTER_LINEAR)
            boxes = [(cx * s, cy * s, w * s, h * s) for cx, cy, w, h in boxes]
            frames.append((i, img, boxes))
            H, W = img.shape[:2]
            for k, (cx, cy, w, h) in enumerate(boxes):
                x0, y0 = window(rng, W, cx, max(w, 1)), window(rng, H, cy, max(h, 1))
                lines = tile_labels(boxes, x0, y0)
                write("{:05d}_{}".format(i, k), crop(img, x0, y0), lines)
                pos += 1
                balls += len(lines)
            if len(frames) > 64:  # keep a pool for the background tiles, not every image in memory
                frames.pop(int(rng.integers(len(frames))))
        # Background tiles: random windows that touch no ball, from the images kept in the pool.
        neg, want, tries = 0, round(NEG_SHARE * pos), 0
        while neg < want and frames and tries < 20 * max(want, 1):
            tries += 1
            i, img, boxes = frames[int(rng.integers(len(frames)))]
            H, W = img.shape[:2]
            x0 = -int(rng.integers(0, TILE - W + 1)) if W <= TILE else int(rng.integers(0, W - TILE + 1))
            y0 = -int(rng.integers(0, TILE - H + 1)) if H <= TILE else int(rng.integers(0, H - TILE + 1))
            if any(x0 - w <= cx < x0 + TILE + w and y0 - h <= cy < y0 + TILE + h for cx, cy, w, h in boxes):
                continue
            write("{:05d}_bg{}".format(i, neg), crop(img, x0, y0), [])
            neg += 1
        counts[split] = {"images": len(items), "labelled": len(plan), "ball_tiles": pos, "balls": balls,
                         "background_tiles": neg}
        print("{}: {} images, {} labelled -> {} tiles with a ball ({} balls), {} without".format(
            split, len(items), len(plan), pos, balls, neg))
    with open(os.path.join(out, "data.yaml"), "w") as handle:
        yaml.safe_dump({"path": out, "train": "train/images", "val": "valid/images", "test": "test/images",
                        "names": {0: "ball"}}, handle)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump({"src": os.path.relpath(args.src, ROOT), "ball_classes": sorted(names[k] for k in ids),
                   "aspect": a, "fold": args.fold, "BALL_PX": BALL_PX, "TILE": TILE, "NEG_SHARE": NEG_SHARE,
                   "SPLIT_CLIPS": SPLIT_CLIPS, "SEED": SEED, "counts": counts}, handle, indent=1)
    print("-> {}".format(os.path.relpath(out, ROOT)))


def train(args):
    data_yaml = os.path.join(prepared(args), "data-hard-{}.yaml".format(args.hard) if args.hard else "data.yaml")
    if not os.path.isfile(data_yaml):
        raise SystemExit("run `prepare` first")
    if os.path.isdir(os.path.join(RUNS, args.name)):  # Ultralytics would quietly train into <name>2
        raise SystemExit("{} exists: pick another --name".format(os.path.relpath(os.path.join(RUNS, args.name), ROOT)))
    model = detect_ball.weights(args.model)
    model.train(data=data_yaml, imgsz=args.imgsz, epochs=args.epochs, patience=PATIENCE, batch=args.batch,
                scale=TRAIN_SCALE, seed=SEED, deterministic=True, project=RUNS, name=args.name)
    print("-> {}".format(os.path.relpath(os.path.join(RUNS, args.name, "weights", "best.pt"), ROOT)))


KAGGLE_SCRIPT = '''"""Generated by model/tune_ball.py kaggle; change it there, not here.
Trains the ball detector on the prepared tiles (Kaggle dataset {dataset}) as tune_ball.py train does."""
import os, subprocess, sys, time
t0 = time.time()


def log(*what):
    """Timestamped and flushed: Kaggle shows a script's log only after the run, so every step leaves a trace."""
    print("[{{:6.0f}} s]".format(time.time() - t0), *what, flush=True)


import socket
try:
    socket.getaddrinfo("pypi.org", 443)
except OSError:
    raise SystemExit("no internet in this session: Kaggle grants it (and the GPU) only to phone-verified "
                     "accounts, kaggle.com > Settings > Phone verification")
log("installing ultralytics=={ultralytics}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics=={ultralytics}"], check=True)
import torch
import yaml
from ultralytics import YOLO, settings
settings.update({{"sync": False}})
log("torch", torch.__version__, "| cuda", torch.version.cuda, "| GPUs:",
    [torch.cuda.get_device_name(k) for k in range(torch.cuda.device_count())])
if not torch.cuda.is_available():  # Ultralytics would quietly train on the CPU, for hours
    raise SystemExit("no GPU visible: stopping instead of training on the CPU")

# The dataset is mounted read-only under /kaggle/input; find the folder holding train/images.
root = next((d for d, subs, _ in os.walk("/kaggle/input") if os.path.isdir(os.path.join(d, "train", "images"))), None)
if root is None:
    raise SystemExit("no train/images under /kaggle/input: {{}}".format(os.listdir("/kaggle/input")))
cfg = {{"path": root, "train": {train}, "val": "valid/images", "test": "test/images", "names": {{0: "ball"}}}}
with open("/kaggle/working/data.yaml", "w") as handle:
    yaml.safe_dump(cfg, handle)
log("tiles from", root)
model = YOLO("{model}")
model.add_callback("on_fit_epoch_end", lambda trainer: log(  # also called once more for the final validation
    "final validation" if trainer.epoch + 1 > trainer.epochs else "epoch {{}} of {{}}".format(trainer.epoch + 1, trainer.epochs),
    "; metrics", trainer.metrics))
model.train(data="/kaggle/working/data.yaml", imgsz={imgsz}, epochs={epochs}, patience={patience}, batch={batch},
            scale={scale}, seed={seed}, deterministic=True, device=0, project="/kaggle/working/runs", name="{name}")
log("finished")
'''


def kaggle(args):
    """A Kaggle kernel that runs `train` on Kaggle's GPU: dataset-metadata.json in the prepared
    tiles' folder, and external/ball-tune/kaggle/<name>/ with the script and kernel-metadata.json.
    Kaggle's GPUs have working AMP, so the batch is AutoBatch's (-1), not TRAIN_BATCH (THRESHOLDS
    v4). Prints the commands to upload, run and fetch."""
    import ultralytics
    base = prepared(args)
    if not os.path.isfile(os.path.join(base, "data.yaml")):
        raise SystemExit("run `prepare` first")
    if args.hard and not os.path.isdir(os.path.join(base, "hard-" + args.hard)):
        raise SystemExit("no hard-{}: run `mine --name {}` first".format(args.hard, args.hard))
    if os.path.isdir(os.path.join(RUNS, args.name)):
        raise SystemExit("{} exists: pick another --name".format(os.path.relpath(os.path.join(RUNS, args.name), ROOT)))
    # ball-tiles-<first word of the export>-<fold>, e.g. ball-tiles-football-fold0
    name = os.path.basename(base).lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", "ball-tiles-{}-{}".format(re.split(r"[^a-z0-9]", name)[0],
                                                                 name.rsplit("-", 1)[-1]))[:50]
    dataset = "{}/{}".format(args.user, slug)
    with open(os.path.join(base, "dataset-metadata.json"), "w") as handle:
        json.dump({"title": slug, "id": dataset, "isPrivate": True, "licenses": [{"name": "other"}],
                   "description": "Tiles derived from football-ball-detection v4 by roboflow-jvuqo on Roboflow "
                                  "Universe, CC BY 4.0, rescaled and re-split by model/tune_ball.py prepare."},
                  handle, indent=1)
    kernel_slug = re.sub(r"[^a-z0-9-]+", "-", "ball-train-" + args.name.lower()).strip("-")[:50]
    folder = os.path.join(OUT_ROOT, "kaggle", args.name)
    os.makedirs(folder, exist_ok=True)
    train_dirs = ["train/images"] + (["hard-{}/images".format(args.hard)] if args.hard else [])
    with open(os.path.join(folder, "train.py"), "w") as handle:
        handle.write(KAGGLE_SCRIPT.format(dataset=dataset, ultralytics=ultralytics.__version__, train=train_dirs,
                                          model=os.path.basename(args.model), imgsz=args.imgsz, epochs=args.epochs,
                                          patience=PATIENCE, batch=-1, scale=TRAIN_SCALE, seed=SEED, name=args.name))
    with open(os.path.join(folder, "kernel-metadata.json"), "w") as handle:
        json.dump({"id": "{}/{}".format(args.user, kernel_slug), "title": kernel_slug, "code_file": "train.py",
                   "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                   "enable_internet": True, "dataset_sources": [dataset], "competition_sources": [],
                   "kernel_sources": []}, handle, indent=1)
    rel = lambda p: os.path.relpath(p, ROOT).replace("\\", "/")  # noqa: E731
    # Absolute paths for the CLI: Kaggle CLI 2.2.5 on Windows breaks on relative ones with forward slashes.
    cli = lambda p: '"{}"'.format(os.path.abspath(p))  # noqa: E731
    print("wrote {}/dataset-metadata.json and {}/".format(rel(base), rel(folder)))
    print("\n1. upload the tiles (first time; afterwards `kaggle datasets version -p ... -m <note> --dir-mode zip`):")
    print("   .venv/Scripts/kaggle datasets create -p {} --dir-mode zip".format(cli(base)))
    print("2. once Kaggle shows the dataset as ready, start the run:")
    print("   .venv/Scripts/kaggle kernels push -p {} --accelerator NvidiaTeslaT4".format(cli(folder)))
    print("3. follow it, then fetch the weights when it is complete:")
    print("   .venv/Scripts/kaggle kernels status {}/{}".format(args.user, kernel_slug))
    print("   .venv/Scripts/kaggle kernels output {}/{} -p {}".format(
        args.user, kernel_slug, cli(os.path.join(OUT_ROOT, "kaggle", args.name, "output"))))
    print("   then copy output/runs/{0}/ to {1}/{0}/ and `eval --model {1}/{0}/weights/best.pt`".format(args.name, rel(RUNS)))


def match(dets, gts):
    """Greedy by confidence: each detection takes the nearest labelled ball not yet taken whose
    box size, max(w, h), reaches it. [(conf, hit)] per detection, and the conf each ball was found
    at (0 if never)."""
    found = [0.0] * len(gts)
    out = []
    for cx, cy, c in sorted(dets, key=lambda d: -d[2]):
        best, best_d = None, None
        for j, (gx, gy, gw, gh) in enumerate(gts):
            d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
            if not found[j] and d <= max(gw, gh) and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            found[best] = c
        out.append((c, best is not None))
    return out, found


def evaluate(args):
    import numpy as np
    folder = os.path.join(prepared(args), args.split)
    items = images(folder)
    if not items:
        raise SystemExit("no prepared {} tiles: run `prepare` first".format(args.split))
    model = detect_ball.weights(args.model)
    ball = detect_ball.ball_class(model)
    people = detect_ball.weights(detect_ball.MODEL)  # COCO players, to tell boots and socks apart
    scored, found, sizes, bg_tiles, bg_conf, on_player = [], [], [], 0, [], []
    for k in range(0, len(items), BATCH):
        chunk = items[k:k + BATCH]
        paths = [img for img, _ in chunk]
        results = model.predict(paths, imgsz=args.imgsz, classes=[ball], conf=detect_ball.CONF_MIN, verbose=False)
        persons = people.predict(paths, imgsz=TILE, classes=[detect_ball.PERSON_CLASS], conf=detect_ball.PERSON_CONF,
                                 verbose=False)
        for (_, lab), result, person in zip(chunk, results, persons):
            gts = read_boxes(lab, {0}, TILE, TILE)
            dets = [(cx, cy, c) for (cx, cy, _, _), c in zip(result.boxes.xywh.tolist(), result.boxes.conf.tolist())]
            s, f = match(dets, gts)
            scored += s
            found += f
            sizes += [min(w, h) for _, _, w, h in gts]
            # match() goes by falling confidence, a stable sort: the same order here.
            for (cx, cy, c), (_, hit) in zip(sorted(dets, key=lambda d: -d[2]), s):
                if not hit and any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in person.boxes.xyxy.tolist()):
                    on_player.append(c)
            if not gts:
                bg_tiles += 1
                bg_conf += [c for _, _, c in dets]
    n = len(found)
    conf = np.array([c for c, _ in scored])
    hit = np.array([h for _, h in scored], bool)
    found, sizes = np.array(found), np.array(sizes)
    order = np.argsort(-conf)
    tp = np.cumsum(hit[order])
    precision = tp / np.arange(1, len(tp) + 1) if len(tp) else np.zeros(0)
    recall = tp / max(n, 1)
    # Average precision, all points (precision made monotone from the right).
    p = np.concatenate([[0], precision, [0]])
    r = np.concatenate([[0], recall, [recall[-1] if len(recall) else 0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    ap = float(np.sum((r[1:] - r[:-1]) * p[1:]))
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(np.argmax(f1)) if len(f1) else None
    best_conf = float(conf[order][best]) if best is not None else float("nan")

    print("{} on {} {} ({} tiles, {} balls, {} background tiles), imgsz {}".format(
        os.path.basename(args.model), os.path.basename(prepared(args)), args.split, len(items), n, bg_tiles,
        args.imgsz))
    print("  AP {:.3f}; best F1 {:.3f} at conf {:.2f}".format(ap, f1[best] if best is not None else 0, best_conf))
    bg_conf, on_player = np.array(bg_conf), np.array(on_player)
    for t in THRESHOLDS:
        kept = conf >= t
        tp_t = int(hit[kept].sum())
        print("  conf >= {:.2f}: precision {:.3f}  recall {:.3f}  false per background tile {:.3f}  "
              "false {} ({} on a player)".format(
                  t, tp_t / max(int(kept.sum()), 1), tp_t / max(n, 1), (bg_conf >= t).sum() / max(bg_tiles, 1),
                  int((~hit & kept).sum()), int((on_player >= t).sum())))
    edges = (0,) + SIZE_BUCKETS + (float("inf"),)
    print("  recall by ball width at conf {:.2f}:".format(best_conf))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sizes >= lo) & (sizes < hi)
        if sel.any():
            print("    {:>4}-{:<4} px: {:.3f} ({} balls)".format(lo, hi if hi != float("inf") else "",
                  float((found[sel] >= best_conf).mean()), int(sel.sum())))
    bg_best = float((bg_conf >= best_conf).sum() / max(bg_tiles, 1))

    false_best = int((~hit & (conf >= best_conf)).sum())
    print("  false at conf {:.2f}: {}, {} of them on a player".format(best_conf, false_best,
                                                                     int((on_player >= best_conf).sum())))

    path = os.path.join(OUT_ROOT, "scores.csv")
    header = ["date", "dataset", "split", "model", "imgsz", "tiles", "balls", "ap", "best_f1", "best_conf",
              "background_false_per_tile_at_best", "false_at_best", "false_on_player_at_best"]
    rows = []
    if os.path.isfile(path):
        with open(path, newline="") as handle:
            rows = list(csv.reader(handle))[1:]  # older rows may lack the later columns
    rows.append([datetime.date.today().isoformat(), os.path.basename(prepared(args)), args.split,
                 os.path.relpath(args.model, ROOT) if os.path.isfile(args.model) else args.model, args.imgsz,
                 len(items), n, round(ap, 4), round(float(f1[best]), 4) if best is not None else "",
                 round(best_conf, 3), round(bg_best, 4), false_best, int((on_player >= best_conf).sum())])
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(header)
        out.writerows(row + [""] * (len(header) - len(row)) for row in rows)
    print("-> {}".format(os.path.relpath(path, ROOT)))


def mine(args):
    """Copies of the training tiles where the trained model `name` finds a ball at MINE_CONF or more
    that matches no label (the boots and socks), MINE_COPIES each, labels unchanged, to
    hard-<name>/ beside the prepared splits, and data-hard-<name>.yaml to train on both."""
    import yaml
    base = prepared(args)
    model = detect_ball.weights(os.path.join(RUNS, args.name, "weights", "best.pt"))
    ball = detect_ball.ball_class(model)
    items = images(os.path.join(base, "train"))
    hard = os.path.join(base, "hard-" + args.name)
    if os.path.isdir(hard):
        shutil.rmtree(hard)
    os.makedirs(os.path.join(hard, "images"))
    os.makedirs(os.path.join(hard, "labels"))
    tiles, false = 0, 0
    for k in range(0, len(items), BATCH):
        chunk = items[k:k + BATCH]
        results = model.predict([img for img, _ in chunk], imgsz=args.imgsz, classes=[ball], conf=MINE_CONF,
                                verbose=False)
        for (img, lab), result in zip(chunk, results):
            dets = [(cx, cy, c) for (cx, cy, _, _), c in zip(result.boxes.xywh.tolist(), result.boxes.conf.tolist())]
            s, _ = match(dets, read_boxes(lab, {0}, TILE, TILE))
            n = sum(not h for _, h in s)
            if not n:
                continue
            tiles += 1
            false += n
            stem = os.path.splitext(os.path.basename(img))[0]
            for c in range(MINE_COPIES):
                shutil.copyfile(img, os.path.join(hard, "images", "{}_hard{}.jpg".format(stem, c)))
                shutil.copyfile(lab, os.path.join(hard, "labels", "{}_hard{}.txt".format(stem, c)))
    with open(os.path.join(base, "data-hard-{}.yaml".format(args.name)), "w") as handle:
        yaml.safe_dump({"path": base, "train": ["train/images", "hard-{}/images".format(args.name)],
                        "val": "valid/images", "test": "test/images", "names": {0: "ball"}}, handle)
    print("{} of {} training tiles with a false ball at conf >= {} ({} false balls), x{} -> {}".format(
        tiles, len(items), MINE_CONF, false, MINE_COPIES, os.path.relpath(hard, ROOT)))
    print("next: train --name {}-hard --hard {}".format(args.name, args.name))


def promote(args):
    best = os.path.join(RUNS, args.name, "weights", "best.pt")
    if not os.path.isfile(best):
        raise SystemExit("no {}".format(os.path.relpath(best, ROOT)))
    os.makedirs(detect_ball.WEIGHTS_DIR, exist_ok=True)
    dst = os.path.join(detect_ball.WEIGHTS_DIR, "ball-{}.pt".format(args.name))
    shutil.copyfile(best, dst)
    print("-> {}; use with detect_ball.py --ball-model {}".format(os.path.relpath(dst, ROOT), os.path.basename(dst)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="a YOLO export: train/ valid/ test/ and data.yaml")
    ap.add_argument("--fold", type=int, default=0, help="which clips are held out (split_items)")
    ap.add_argument("--export-split", dest="fold", action="store_const", const=None,
                    help="keep the export's own split instead (frames of one clip in every split)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    sub.add_parser("prepare")
    p = sub.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default=detect_ball.MODEL, help="the starting weights")
    p.add_argument("--imgsz", type=int, default=TILE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=TRAIN_BATCH, help="-1: Ultralytics AutoBatch (GPUs with AMP)")
    p.add_argument("--hard", help="also train on the tiles `mine --name <this>` copied")
    p = sub.add_parser("kaggle", help="the same training as a Kaggle kernel, on Kaggle's GPU")
    p.add_argument("--user", required=True, help="your Kaggle username")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default=detect_ball.MODEL, help="the starting weights (downloaded on Kaggle)")
    p.add_argument("--imgsz", type=int, default=TILE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--hard", help="also train on the tiles `mine --name <this>` copied")
    p = sub.add_parser("mine")
    p.add_argument("--name", required=True, help="the trained run whose false balls to mine")
    p.add_argument("--imgsz", type=int, default=TILE, help="as that run was trained")
    p = sub.add_parser("eval")
    p.add_argument("--model", required=True, help="a weights name (weights/, COCO downloads) or a .pt path")
    p.add_argument("--imgsz", type=int, default=TILE)
    p.add_argument("--split", default="test", choices=SPLITS)
    p = sub.add_parser("promote")
    p.add_argument("--name", required=True)
    args = ap.parse_args()
    {"stats": stats, "prepare": prepare, "train": train, "kaggle": kaggle, "mine": mine, "eval": evaluate,
     "promote": promote}[args.cmd](args)


if __name__ == "__main__":
    main()
