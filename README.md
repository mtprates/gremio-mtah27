# gremio: who has the ball

Our submission to [More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027): for every
frame of the BAR-ATH match, home (H), away (A) or dead ball (D).

- **Technical document:** [docs/TECHNICAL.md](docs/TECHNICAL.md): the approach, the possession rules, results,
  how every value was set, limitations.
- **Every value and its basis:** [docs/VALUES.md](docs/VALUES.md).
- **Overview of the model:** [MODEL.md](MODEL.md).
- **External validation (10 public matches):** [validation/README.md](validation/README.md).

We find the ball and the players ourselves in both videos (YOLO, a tracker, a 3D ball with two-camera
fusion), and a three-state hidden Markov model weighs the touches, kicks, controls, carries, the players'
distances and the dead-ball signals. No value is trained, fitted or chosen by a score on the match; values
come from physics, the Laws of the Game, reasoning, or public matches (IDSSE, Metrica held out).

## Reproduce `gremio.csv`

The match data is licensed to each participant and is not in this repository. Lay it out as
[data/README.md](data/README.md) describes (in `./data`, or set `MTAH_DATA_DIR`), then:

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-detect.txt     # install torch for your GPU first
.venv/Scripts/python reproduce.py --team gremio                    # -> submission/gremio.csv, validated
```

`reproduce.py` runs every step whose output is missing: YOLO detection over both videos (about 13 h on a
GTX 1660 Ti), the broadcast's shots and cuts, the tracker and team labels, then the possession model
(minutes) and the format check. `python reproduce.py --list` shows which steps are done. Everything after
detection is deterministic; detection on a different GPU, driver or the CPU can change boxes in their last
digits and so a few labels (docs/TECHNICAL.md, "Hardware and determinism").

Python 3.13; versions pinned in the requirements files. Scoring a file against the published 5-minute
window: `python evaluation/run_evaluation.py` (Metrica's scorer).

## External validation

`external/README.md` lists the public datasets and how to download them. Then:

```
.venv/Scripts/python model/external_eval.py                                     # labels and references
TUNE_RULES='{"TOUCH_ECHO": true}' .venv/Scripts/python model/tune_external.py export submitted
.venv/Scripts/python model/tune_external.py export before r1
.venv/Scripts/python model/readiness.py                                         # validation/readiness_*.csv
.venv/Scripts/python docs/make_figures.py                                       # docs/figures
```

## Layout

`CHALLENGE.md`, `EVALUATION.md`, `DATA.md`, `RULES.md` and `SUBMISSION.md`, referred to throughout, are the
challenge's documents, in [its repository](https://github.com/metrica-sports/more-than-a-hack-2027).

| path | what |
|---|---|
| `reproduce.py` | data to submission file, one command |
| `model/` | the pipeline: detection, teams, ball track, events, the possession models, external validation |
| `docs/` | the technical document, the value list, figures |
| `validation/` | external validation results (committed aggregates) |
| `evaluation/`, `games.py`, `starter/load_smart_tagging.py` | Metrica's scorer, validator and loaders (their licence) |

## Licence

Our code and documents: MIT ([LICENSE](LICENSE)). Metrica's files (`evaluation/`, `games.py`,
`starter/load_smart_tagging.py`, `data/README.md`, `requirements.txt`): their MIT licence
([LICENSE-METRICA](LICENSE-METRICA)). Neither covers the match data.
