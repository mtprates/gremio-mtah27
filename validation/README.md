# External validation

The possession model on 10 public matches that are not BAR-ATH: does it get possession right, and how
does it compare with simple references?

| dataset | games | tracking | labels |
|---|---|---|---|
| IDSSE / DFL (Bassek et al. 2025, CC BY 4.0) | 7 Bundesliga / 2. Bundesliga | optical, 25 fps, every player + ball | the provider's per-frame `BallStatus` (0 -> D) and `BallPossession` (1 -> H, 2 -> A) |
| Metrica sample | 3 anonymised | optical, 25 fps | derived from events by `model/metrica_sample_gt.py` |

Neither labeller is the challenge's. These numbers check the model on other matches; they are **not** a
BAR-ATH score estimate. Download steps: `external/README.md`.

## How the matches are used

- **IDSSE is the tune set:** where a value needed data, it was measured or chosen here.
- **Metrica is held out:** reported, never used to choose.
- **Clean and noisy copies:** every match is scored as provided and with noise that mimics our own
  detector's input (the ball unseen in runs whose start depends on the nearest player's distance, a
  heavy-tailed run length, position jitter, players missing) (`model/tune_external.py`).
- **Adapted per match** (`model/external_eval.py`): halves, kick-offs and pitch size point at that game;
  there is no broadcast camera, so every broadcast rule is off; player speeds come from positions.

## The submitted model

Model C with the submission's rules, against references. `readiness_sets.csv` and `readiness_games.csv`
(written by `model/readiness.py`); figures 3-4 in `docs/TECHNICAL.md`.

| set | model | spellF1 | macroF1 | possessions vs truth | found, true length < 2 s / 2-4 / 4-8 / > 8 s |
|---|---|---|---|---|---|
| IDSSE clean | **submitted** | **0.486** | 0.836 | ×0.59 | 8% / 20% / 50% / 83% |
| | R1: every touch hands over | 0.577 | 0.762 | ×0.94 | 33% / 49% / 62% / 83% |
| | ground truth 1 s late | 0.557 | 0.900 | ×1.00 | 0% / 43% / 100% / 100% |
| IDSSE noisy | **submitted** | **0.417** | 0.815 | ×0.48 | 5% / 11% / 34% / 76% |
| | R1 | 0.432 | 0.626 | ×0.73 | 18% / 29% / 39% / 66% |
| Metrica clean | **submitted** | **0.543** | 0.833 | ×0.57 | 11% / 22% / 46% / 78% |
| | R1 | 0.608 | 0.732 | ×0.90 | 25% / 49% / 67% / 84% |
| | ground truth 1 s late | 0.608 | 0.920 | ×1.00 | 0% / 49% / 100% / 100% |
| Metrica noisy | **submitted** | **0.435** | 0.790 | ×0.48 | 6% / 13% / 32% / 64% |
| | R1 | 0.423 | 0.524 | ×0.72 | 16% / 26% / 36% / 58% |

What it says:

- **Long possessions are found as well as by R1** (8 s and longer: 78-83% on clean tracking), with a far
  better macroF1 and live / dead boundary. On noisy input the model is level with R1 on spellF1, and R1
  falls apart on our own detector input.
- **Short possessions are the gap, partly by design.** One uncontrolled touch does not change possession
  (CHALLENGE.md), while the labellers' possession flags flip on single touches. Even a perfect labeller
  running 1 s late finds none under 2 s.
- **Two choices cost here on purpose.** The dead-ball acceleration signal counts only when present: on
  these datasets a dead ball is parked by the provider and visibly still, so its absence is real evidence of
  live play here but not on our input, where the ball is unseen on 44% of frames. And the rules that rest
  on the broadcast cannot fire here: these datasets have no broadcast.

## Files

- `scores.csv`: `model/external_eval.py`'s reference variants, one row per game.
- `readiness_sets.csv`, `readiness_games.csv`: the submitted model against references.
- `tuning.csv`: every external candidate scored (`model/tune_external.py`).
- `gt/`, `predictions/`, `reports/`: per-frame labels and full metric JSONs (gitignored, rebuilt by the
  scripts).
