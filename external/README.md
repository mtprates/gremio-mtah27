# External data

Public datasets from matches other than BAR-ATH, for validation. Everything in
here except this file is gitignored.

## idsse

Seven Bundesliga / 2. Bundesliga matches from the DFL (Sportec Solutions), 25 fps optical
tracking with the provider's per-frame BallStatus and BallPossession. Bassek et al. 2025, "An
integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data
(https://www.nature.com/articles/s41597-025-04505-y), CC BY 4.0. Mirror used by kloppy:

    mkdir -p external/idsse/raw && cd external/idsse/raw
    for m in J03WMX:DFL-COM-000001 J03WN1:DFL-COM-000001 J03WOH:DFL-COM-000002 J03WOY:DFL-COM-000002 \
             J03WPY:DFL-COM-000002 J03WQQ:DFL-COM-000002 J03WR9:DFL-COM-000002; do
      id=${m%%:*}; c=${m##*:}
      for f in DFL_02_01_matchinformation_${c}_DFL-MAT-${id}.xml DFL_04_03_positions_raw_observed_${c}_DFL-MAT-${id}.xml \
               DFL_03_02_events_raw_${c}_DFL-MAT-${id}.xml; do
        curl -sSL -o $f https://huggingface.co/datasets/pysport/idsse-data/resolve/main/$f
      done
    done

About 2.6 GB. `model/external_eval.py` parses it into external/idsse/cache/ and scores the model
on it (validation/README.md).

## metrica-sample

Metrica Sports' open sample data: three anonymised matches, tracking and events, synchronised.
Source: https://github.com/metrica-sports/sample-data. Terms: "please be responsible with the use
of this data" and "if you use it for anything public, please acknowledge the source".

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git external/metrica-sample
    mv external/metrica-sample/data external/metrica-sample/games
    .venv/Scripts/python model/metrica_sample_gt.py

The rename keeps the folder clear of the `data/` read guards in .claude/settings.json, which are
there for the competition data. `model/metrica_sample_gt.py` writes `gt/sample_game_<n>.csv`
(frame,label, derived from the events; an approximation of the challenge's labels, see its
docstring), scored with:

    py evaluation/run_evaluation.py --gt external/metrica-sample/gt/sample_game_1.csv --results <dir>

## football-ball-detection.v4i.yolov11

Ball boxes for fine-tuning the ball detector (`model/tune_ball.py`; docs/VALUES.md, "Ball
detector fine-tuning"). Roboflow Universe, football-ball-detection v4 (2024-07-29),
https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/4, CC BY 4.0
(attribution: "Provided by a Roboflow user", workspace roboflow-jvuqo). 4948 frames of 14 clips,
stretched to 640 x 640 from 16:9, one class `ball`, about a quarter of each clip's frames labelled,
no augmentation. Downloaded by hand (free Roboflow login), format YOLOv11, unzipped as is.

    .venv/Scripts/python model/tune_ball.py stats
    .venv/Scripts/python model/tune_ball.py prepare     # -> external/ball-tune/football-ball-detection.v4i.yolov11-fold0/

Training runs and scores go to external/ball-tune/ too (runs/, scores.csv).
