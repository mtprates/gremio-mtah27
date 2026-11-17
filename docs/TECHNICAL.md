# Who has the ball: possession from our own detections and a team HMM

**Team:** gremio · **Members:** Matias Prates, Eduardo Amorim

**Repository:** https://github.com/mtprates/gremio-mtah27

**Submission file:** `gremio.csv`, from `python reproduce.py --team gremio`

## Summary

We label every frame of BAR-ATH as home (H), away (A) or dead ball (D) in two stages:

1. **Perception.** We find the ball and the players ourselves, in both videos.
2. **Inference.** A three-state hidden Markov model (HMM) weighs everything the ball and the players
   say about possession.

Two facts shaped the design. **Ball errors become fake possession changes**: a jump between two
ball-like objects reads as a kick, and a kick near an opponent as a turnover, so most of our work
went into a ball track that cannot do what a ball cannot do. And **the ranking metric counts
possessions, not frames**: spellF1 punishes splitting one possession far more than a few late
frames, so the decoder needs more than one touch to hand possession over.

Nothing was hand-labelled or trained on BAR-ATH, and no value was chosen by a score on it
(section 6).

![Pipeline](figures/pipeline.png)

*Figure 1. The pipeline. Everything after the inputs is ours; the provider's tracking only aligns
the two videos, and no audio is used.*

## 1. Approach, and why

**Our own detections.** The provided ball tracking was our first input, but its errors (a ball
jumping between objects, players swapping) look the same as the possession events we need to
detect, and we had no ground truth to correct them against. So we detect and track the ball and the
players ourselves, and use the provider's tracking only to align the two videos.

**An HMM rather than rules.** Our first model was a state machine: every touch hands possession to
the toucher's team (R1). It is the best decoder on clean tracking, but on our detector input false
touches split possessions apart and it falls below the published floor (0.284 spellF1 on the public
window, against 0.326 for `baseline_smooth50`). The HMM weighs evidence in nats instead of yes or
no, charges a price for every change of state, and finds the labels that explain the whole match at
once, so one stray touch cannot flip possession unless other evidence agrees.

![HMM](figures/hmm_states.png)

*Figure 2. Model C's three states and the prices of moving between them, from the mean lengths of
possessions, live play and stoppages on public matches.*

**Design choices.**

- **One touch is not enough to change possession.** Possession changes on a kick, a control, a carry
  or a second touch. Two reasons: our tracking makes false touches (a detected ball's jitter looks
  exactly like a touch, and better tracking is the real cure), and CHALLENGE.md says a touch without
  control, a deflection or an aerial duel won without control do not change possession.
- **As independent of the provided data as we could make it.** Our own tracking (above). The
  homographies are still the provider's: our own pitch calibration is built (within ~0.4 m on a check
  clip) but not wired in. The two cameras are kept separable: each has its own detections and ball
  trace before fusion, every broadcast rule can be switched off, and the external validation runs on
  a single camera stream that way. A broadcast-only version would feed that stream where the
  tactical camera is used now; one dependency remains, the alignment of the two videos, which uses
  the provider's tracking.
- **No audio.** The model uses images only, so it would work on a broadcast with commentary over the
  referee or no sound at all. A whistle detector is built but not submitted, for that reason and
  because its thresholds were set on this match's whistles.
- **No `X` labels on graded frames.** macroF1 counts an `X` as a miss, and spellF1 replaces it with
  the previous label. The decoder already keeps the previous label where the evidence is weak, which
  is what spellF1 would do with an `X`, so abstaining would add nothing to spellF1 and cost macroF1.
  `X` appears only on the 40,462 frames that are never graded.

**Tried and dropped:**

| tried | outcome |
|---|---|
| A rules state machine (R1, V1) | Fragments on noisy input; replaced by the HMM. |
| Possession won at touches by pitch control (model T); a player-level HMM (model P) | T too sticky, P called 5% more of the match dead; kept as comparisons. |
| A ball detector fine-tuned on public images (Roboflow, CC BY 4.0) | Better on its own test set (AP 0.92 vs 0.64), worse on the tactical camera; COCO weights kept. |
| Our own pitch calibration (PnLCalib) | Agrees within ~0.4 m; a full run takes ~24 h, so not wired in. |
| Referee whistles from the audio (rule 57) | Built, not submitted (see above). |
| Other ball-track and touch variants (rules 28, 32, 37, 38, 40, 49-52, 54, 61) | Behind switches, off. |

## 2. Assets used

| asset | how we use it |
|---|---|
| Tactical and broadcast video | YOLO detection of balls and people; shirt colours from the person boxes. |
| Provided homographies | Image to pitch, per frame, for both cameras. |
| Provided tracking | Only to align the broadcast to the tactical timeline (by matching player layouts), and as a diagnostic yardstick. Never a model input. |
| Smart Tagging | The kick-off frames only (the FAQ allows it). |
| Public 5-minute ground truth | A pipeline check only: the frozen model was scored on it once. |
| COCO-pretrained `yolo11m` (Ultralytics) | Ball and person detector, no fine-tuning. |
| IDSSE / DFL open tracking, 7 matches (Bassek et al. 2025, CC BY 4.0) | Where a value needed data: measured or chosen here. |
| Metrica sample games, 3 matches | Held out: reported, never used to choose. |

There are no hand labels anywhere in the project.

## 3. The model

**Two cameras on one timeline.** `align.py` finds the broadcast-to-tactical offset per half from the
layout of players and detects replays as jumps back in time; `shots.py` and `cameras.py` find cuts
and close-ups. Only the broadcast's live wide shots are used.

**Players and teams.** YOLO person boxes are projected to the pitch and tracked; shirt colours
(torso pixels, grass removed) are clustered by a seeded Gaussian mixture into home, away, two keepers
and officials, unsupervised, on the match itself.

**The ball.** (1) *Picks per camera*: a Viterbi search over each camera's candidates, where confident
candidates and ones the other camera supports are cheap, and standing still, the penalty spot, the
advertising boards, impossible speeds, appearing from nowhere and restarting where the Laws do not
allow are expensive. (2) *Height*: each camera sees where its line of sight meets the grass, which
for a ball in the air is well past it (Figure 3); triangulation and arcs fitted with gravity and drag
give the height, and each camera's point is corrected to it. (3) *Fusion*: a two-camera Kalman
smoother, forward and backward, that bridges gaps but never faster than 35 m/s (about the hardest
shot), and keeps a ball caught by a keeper in his hands until released.

![Line of sight](figures/line_of_sight.png)

*Figure 3. Two cameras see a ball in the air at two different ground points; their disagreement
gives its height.*

**Events.** A *touch* is a change of the ball's velocity of 3.5 m/s or more with a player within 4 m,
not above head height. A *control* is the same team within 1.3 m of the ball for 0.4 s. A *carry* is
a player who keeps his pace through a gap in which the ball is unseen.

### The decoder: model C

On every frame each rule below adds evidence, in nats, for one state against another; changing
state costs a price; Viterbi picks the labels that explain all the evidence at once, and a last pass
cleans up short spells. Every rule that changes a label:

| | rule | what it does | weight | why |
|---|---|---|---|---|
| **Who has the ball** | touch (1) | credits a touch to the team of the player within 2 m | 3 nats | less than a switch (5.2): a touch without control is not a change, and jitter makes false touches |
| | far touch (1, 26) | a touch with the nearest player 2-4 m away, on a ball in the air, or with the ball unseen on one side | 1.5 nats | less sure who touched it; an aerial duel is not control |
| | control (1.a), carry (2) | the ball held near one team's player | 4 nats | control is what changes possession |
| | kick (55) | a touch that sends a ground ball off at 8 m/s or more, no slower than it came, away from a player within 2 m | switch price + 1 | a struck pass, shot or clearance is control; a deflection is not |
| | detection precision (53, 53b) | every event is multiplied by how sure the ball detections around it are (precision measured on public images) | × 0 to 1 | events on weak detections are doubtful |
| | measured touch first (58) | where the ball is briefly unseen, a stronger change measured beside the gap replaces the one fitted inside it | | the fitted one is the same kick, blurred |
| | clear nearest team (60) | a far touch counts only if the other team is 1 m or more farther from the ball | 1 m | nearer, our position errors cannot tell who touched it |
| | distances confirm only (59) | the players' distances to the ball may support the latest event's team, never the other | capped, once per 0.5 s | a loose ball stays with the last team |
| **Dead or live** | out of bounds (27) | the ball over a line, dead from the crossing until the restart | 3-6 nats per 0.5 s | Law 9 |
| | broadcast cut-away | the broadcast loses its homography (a close-up) | +2.0 / −1.3 | directors cut away at stoppages, rarely in play (reasoned) |
| | replays neutral (62) | a replay gives no evidence either way | 0 | it shows earlier play |
| | players still, stillness | most players under 1 m/s; the mean under 0.9 m/s with the ball still | measured on IDSSE | players stop at a stoppage |
| | ball acceleration near zero | the ball smooth or still for 1 s | +1.75 present, 0 absent | its absence is no signal on our input, where the ball is unseen on 44% of frames |
| | still ball in a cut-away (64) | the broadcast has cut away *and* the ball is still whenever seen | 3 nats | a still ball alone is no evidence (a player can hold a live ball) |
| | restart or kick is live (56) | a kick from rest inside the pitch, or any kick | 9 nats against dead | only a restart sends a stationary ball off a foot |
| **After decoding** | short spells (63) | a live spell under 5 s between stoppages becomes dead unless the ball goes out at its end; a stoppage under 5 s inside one team's possession, seen only by the broadcast, becomes that team's | 5 s | a touch inside a stoppage reads as a restart; the broadcast alone does not overrule the tactical camera |

Changing state costs log(mean length × 25 fps), with the mean possession (7.2 s), live stretch
(31.2 s) and stoppage (25.3 s) on IDSSE's labels: 5.2 nats between H and A, 6.7 to dead, 7.1 back to
live.

## 4. The possession rules we implemented

CHALLENGE.md defines a change of possession as an opponent gaining **controlled** possession.
Situation by situation (rule numbers from the table above):

- **A single touch** by the other team: not enough by itself; it needs a second touch, a control, a
  carry or a kick (1).
- **A kick** by the other team: changes possession on its own; an interception struck straight on is
  control (55).
- **A deflection:** either no touch at all, or a touch that fails the kick's "no slower than it came"
  test; not enough by itself.
- **A header or aerial duel:** a weak touch; possession follows whoever controls the ball afterwards
  (26).
- **A loose ball or a tackle that only knocks it loose:** stays with the team that last played it
  (59); a far touch between two equally near opponents counts for nobody (60).
- **A ball out of bounds:** dead from the crossing until the restart (27).
- **A stoppage:** dead when the players stop or the broadcast cuts away, surely so when the ball lies
  still during a cut-away (64); a cut-away under 5 s inside a possession is not a stoppage (63).
- **A restart:** live from the restart kick, and the team taking it has the ball (56); a "restart"
  followed within 5 s by a stoppage was the ball being set up (63).
- **A goalkeeper's catch:** his team's, held until release.

## 5. Results

**Public window** (17 possessions), scored once on the frozen model as a pipeline check:

| | spellF1 (95% interval) | macroF1 | possessions |
|---|---|---|---|
| **submitted model** | **0.634** (0.43-0.80) | **0.894** | 12 |
| published floor, `baseline_smooth50` | 0.326 (0.21-0.45) | 0.621 | 26 |
| for scale: the ground truth itself 0.5 / 1 / 2 s late | 0.906 / 0.776 / 0.588 | 0.966 / 0.931 / 0.871 | 17 |

One possession moves spellF1 here by about six points (EVALUATION.md), so this is a check, not a
ranking. (It was scored just before two small final changes, which alter one 2.7-second stretch of
the match, and not again.)

**External matches**, scored clean and with noise that mimics our detector's input
(`validation/README.md`):

| set | spellF1 | for scale: the ground truth 1 s / 2 s late | possessions predicted vs true |
|---|---|---|---|
| IDSSE, clean | 0.486 | 0.557 / 0.356 | ×0.59 |
| IDSSE, noisy | 0.417 | | ×0.48 |
| Metrica, clean | 0.543 | 0.608 / 0.426 | ×0.57 |
| Metrica, noisy | 0.435 | | ×0.48 |

![By length](figures/match_by_length.png)

*Figure 4. Share of true possessions found, by their length (clean external tracking).*

- **The datasets agree:** the model scores like a perfect labeller about 1.2 s late on clean external
  tracking and about 1.5 s late on the public window, a sign it is not fitted to BAR-ATH.
- **Its strengths** are long possessions (about 80% of those over 8 s are found), the live / dead
  boundary, and robustness: on noisy input it is level with R1, which collapses on our own detector
  input.
- **Its gap** is short possessions: it predicts about half as many possessions as there are, almost
  all the missing ones short (section 7).

## 6. How every value was set

Every value had a dated rationale, written before the code; `docs/VALUES.md` lists each with its
basis. They fall into these groups:

- **Physics, the Laws of the Game, published figures:** the maximum ball pace, drag, pitch markings,
  the restart tests.
- **Measured or chosen on public matches** (IDSSE to choose, Metrica held out): the mean possession,
  live and dead lengths, the dead-ball signals' reliability, the kick and live-evidence weights.
- **Reasoning:** the broadcast cut-away weight, the player relink window, the keeper's hold, the 5 s
  short-spell window, the still-ball-in-a-cut-away rule.
- **Our own noise floors**, measured once on our detections without labels or scores: the touch
  threshold (3.5 m/s), "acceleration near zero" (3 m/s²), the still-player speed (0.9 m/s).

BAR-ATH was watched to find bugs in our logic and understand its data's errors: spot checks in the
viewer showed where a rule was needed (a kick lost to a gap in the ball track, a replay read as
live play), and the rule's values came from outside the match. Which rules to submit was decided
with those checks; one of them (rule 64) came from a moment inside the public window.

## 7. Limitations, and what we would do next

- **Short possessions, partly by design.** We follow CHALLENGE.md: an uncontrolled touch does not
  change possession. The ground truth, derived mechanically from the provider's tracking, does
  contain very short possessions, some of them single touches (2 of the 17 public ones last under
  2 s), and we find only 5-11% of possessions under 2 s on the external matches. Of those of 2-4 s
  we find 11-23%, where R1 finds 26-49% on clean data: that part is a real limitation. Next: a
  duration-aware decoder (an HSMM).
- **Dead balls against keeper saves and stalling.** A stoppage, a keeper holding the ball after a
  save and a player standing on the ball look alike to the model: the ball still or unseen,
  someone near it, play paused. A catch while the ball is unseen gives no event, and play close to a
  dead ball (a quick restart, a ball knocked about after a whistle) is error-prone for the same
  reason. Next: player movement patterns as evidence, and finer player labels (identities and
  roles, keepers in particular).
- **The live / dead boundary.** Our stoppage signals arrive a few seconds after the whistle, and
  while the ball is kicked about during a stoppage only the broadcast shows it, which we weigh below
  clear tactical evidence. The whistle would catch both; it is not submitted (section 1).
- **False touches and tracking.** Jumps between ball candidates and team-label flips are our main
  noise; the tactical camera's ball is about 5 px, so detector recall at players' feet is the ceiling
  on everything. The external matches' provider never loses the ball (ours is unseen on 44% of
  frames), so signals that depend on the ball at a stoppage transfer badly. Next: a ball detector
  fine-tuned on wide-angle footage, and tuned tracking filters.
- **Dependence on provided data:** the homographies and the alignment of the two videos.

## 8. Reproduction

```
git clone https://github.com/mtprates/gremio-mtah27 && cd gremio-mtah27
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-detect.txt   # torch for your GPU first
# lay out the match data as DATA.md describes (./data, or set MTAH_DATA_DIR)
.venv/Scripts/python reproduce.py --team gremio      # -> submission/gremio.csv, validated
```

`reproduce.py` runs every step whose output is missing: YOLO detection over both videos (about
13 h on a GTX 1660 Ti), the broadcast's shots and cuts, the tracker and team labels, then the
possession model (minutes) and the format check. Python 3.13, versions pinned.

**Hardware and determinism.** The detections were produced on a GPU (GTX 1660 Ti, CUDA 12.6, torch
2.14.1, ultralytics 8.4.183). GPU inference can differ in the last digits across GPU models, drivers
or CPU-only runs, which can move a ball pick and so a few labels. Everything after detection is
deterministic: from the same detections a fresh clone of the repository rebuilt `gremio.csv` byte for
byte. `model/readiness.py` and `model/tune_external.py` rebuild the external results from the public
datasets (`external/README.md`).
