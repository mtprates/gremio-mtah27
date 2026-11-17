# Who has the ball: possession from our own detections and a team HMM

**Team:** gremio · **Members:** Matias Prates, Eduardo Amorim

**Repository:** https://github.com/mtprates/gremio-mtah27

**Submission file:** `gremio.csv`, from `python reproduce.py --team gremio`

## Summary

We label every frame of BAR-ATH as home (H), away (A) or dead ball (D) in two stages:

1. **Perception.** We find the ball and the players ourselves, in both videos.
2. **Inference.** A three-state hidden Markov model weighs everything the ball and players say
   about possession.

The provided tracking is used only to put the two videos on one timeline.

Two things shaped the design:

- **Ball errors become fake possession changes.** A jump between two ball-like objects reads
  as a kick, and a kick near an opponent reads as a turnover. Most of our work went into a ball
  track that cannot do what a ball cannot do.
- **The ranking metric counts possessions, not frames.** spellF1 punishes fragmenting one
  possession into several far more than it punishes a few late frames. So the decoder needs
  more than one touch before it hands possession over.

Nothing was hand-labelled on BAR-ATH, nothing was trained on it, and no value was chosen by
searching scores on it. The code, and the one command that rebuilds the submission, are at
https://github.com/mtprates/gremio-mtah27. Section 6 says exactly how every value was set, including the ones
that rest on observations of this match.

![Pipeline](figures/pipeline.png)

*The pipeline. Everything after the inputs is ours; the provider's tracking only aligns the two videos, and no audio is used.*

## 1. Approach, and why this one

**Why our own detections rather than the provided tracking.** The provided ball (ATD) was our
first input. We found that its errors look the same as the possession events we need to detect.
We also wanted a pipeline that would carry over to a match with only a broadcast camera. So we
detect balls and people ourselves, and treat the provider's ball and players as a point of
comparison, never as the truth.

**Why an HMM rather than rules.** Our first model was a state machine (rules R1 and V1 below).
On clean external tracking, R1 is the best decoder: every touch hands possession to the toucher's
team (Figure 3). On our noisy detector input it falls below the published floor: 0.284 spellF1 on
the public window, against 0.326 for `baseline_smooth50`. False touches split possessions apart.

The HMM (Figure 2) avoids this:

- Each frame gives evidence in nats, never a yes or no.
- Switching state has a price set from how long possessions and stoppages last.
- Viterbi finds the labels that explain the whole match at once.

As a result, one stray touch cannot flip possession unless other evidence agrees.

![HMM](figures/hmm_states.png)

*Figure 2. Model C: three states and the prices of moving between them, from the mean lengths of possessions, live play and stoppages on IDSSE's labels.*

### Design choices, and why

**One touch is not enough to change possession.** An opposing touch is worth 3 nats against a
5.2-nat switch: possession changes on a kick, a control, a carry or a second touch. Two reasons:

- *Our tracking makes false touches.* A touch is a sudden change in the ball's velocity, and the
  jitter of a detected ball, or a jump from one ball-like object to another, looks exactly like
  one. Weighing a single touch less is the decoder's defence; the better cure is better
  tracking. We tried fine-tuning the ball detector on public images (it won there and lost on the
  tactical camera, section 1's table), and the tracking filters' weights are reasoned values, not
  tuned ones; both are where we would look next.
- *The challenge's definition.* CHALLENGE.md says a simple touch without control, a deflection
  that does not intercept a pass and an aerial duel won without control do not change
  possession. The cost is short possessions (section 7): the ground truth, derived mechanically
  from the provider's tracking, does contain some single-touch possessions.

**As independent of the provided data as we could make it.**

- *Our own tracking.* We detect and track the ball and the players ourselves. The provided
  tracking's errors (a ball jumping between objects, players swapping) look the same as the
  possession events we need to detect, and we had no ground truth to correct them against; so it
  is a point of comparison, never an input to the model.
- *Homography: not yet our own.* We use the provided per-frame homographies. Our own pitch
  calibration (PnLCalib) is built and agrees within ~0.4 m on a check clip, but is not wired in;
  doing so would make the pipeline independent of the provider apart from the alignment below.
- *Two cameras, kept separable.* Each camera has its own detections, ball picks and trace before
  the two are fused, and every broadcast-based rule (cut-aways, replays, rules 62-64) can be
  switched off. The external validation runs exactly that way, on a single camera stream with no
  broadcast at all. A broadcast-only version is therefore a matter of feeding that stream where
  the tactical camera is used now, which we have not tried. One dependency remains: the
  broadcast-to-tactical alignment matches player layouts using the provided tracking.
- *No audio.* The submitted model uses images only, so it would work on a broadcast with no
  sound or with commentary over the referee. A referee-whistle detector is built (rule 57), but
  it is not submitted, for that reason and because its thresholds were set on this match's
  whistles.

**Every graded frame gets H, A or D; no `X`.** The file writes `X` only on the 40,462 frames that
are never graded (before kick-off, half-time, after the final whistle). EVALUATION.md scores an
`X` two ways: macroF1 counts it as a miss, and spellF1 replaces it with the previous label before
matching possessions. So an abstention is free only inside a possession and costly at a change of
possession, where it carries the old team across the hand-over. Our decoder already does the
useful part: where the evidence is weak, the switch price keeps the previous label, which is
what spellF1 would do with an `X`. Abstaining on top would add nothing to spellF1 and cost
macroF1, and we have no reliable way to tell, in a weak stretch, whether a change is happening.

**Alternatives tried and dropped** (each value's basis in `docs/VALUES.md`):

| tried | outcome |
|---|---|
| Rules state machine on the provider's tracking (R1, V1) | Replaced by the HMM. R1 fragments on noisy input. |
| Model T: possession changes only at touches, won by pitch control (Spearman) | Too sticky: two opposing touches were needed to switch. Kept as a viewer comparison. |
| Model P: player-level HMM (a pass to a teammate is cheaper than a turnover) | Called 5% more of the match dead than C. Not submitted. |
| Ball detector fine-tuned on a public Roboflow set (CC BY 4.0) | Better on its own test set (AP 0.92 vs 0.64), worse on the tactical camera: static balls off the pitch, a domain gap. COCO weights kept. |
| Our own pitch calibration (PnLCalib) | Within ~0.4 m of the provided homography on a check clip. A full run takes ~24 h, so it is not wired in. |
| Referee whistles from the audio | Built. Rule 57 (a stoppage starts at the whistle just before it) is not submitted: five of the detector's values were set against this match's whistles, and no open dataset with audio exists to set them elsewhere. |
| Rules 28, 32, 37, 38, 40, 49-52, 54 (ball-track and touch variants) | Behind switches in the code, off. Rule 52 (pricing picks by precision in the tracker) backfired. |
| Rule 61: a kick from rest is no restart if the ball is back at rest soon after | Cost on the external matches; off. Rule 63 handles the same false restarts by their length instead. |

## 2. Assets used

| asset | how we use it |
|---|---|
| Tactical and broadcast video | YOLO detection of balls and people in both. Shirt colours from the person boxes. |
| Provided homographies | Image to pitch, per frame, for both cameras. |
| Provided tracking (ATD) | Only to align the broadcast to the tactical timeline (`align.py`, by matching player layouts), and as a diagnostic yardstick. Never as model input for C. |
| Smart Tagging | The kick-off frames only (the FAQ allows it). |
| Public 5-minute ground truth | A pipeline check, plus three measured dead-ball signal weights (section 6). |
| COCO-pretrained `yolo11m` (Ultralytics) | Ball and person detector, no fine-tuning. |
| IDSSE / DFL open tracking, 7 matches (Bassek et al. 2025, CC BY 4.0) | External validation and the tuning set. |
| Metrica Sports sample games, 3 matches | Held-out external validation. Labels derived from their events by `model/metrica_sample_gt.py`. |

No hand labels exist anywhere in the project. The external labels are the providers' own.

## 3. The model

### 3.1 Two cameras on one timeline

`align.py` finds the broadcast-to-tactical frame offset per half from the layout of players, and
detects replays as jumps back in time. `shots.py` and `cameras.py` find cuts and close-ups from
image changes and from the height of people's boxes under the homography. Only the broadcast's
live wide shots are used. A broadcast shot whose homography contradicts the tactical camera is
dropped (rule 41).

### 3.2 Players and teams

- YOLO person boxes are projected to the pitch and tracked (`teams.py`).
- Shirt colours (torso pixels, with the grass removed) are clustered by a seeded Gaussian
  mixture into home, away, two keepers and officials. This runs unsupervised on the match at
  inference time: no labels, nothing trained.
- Short detection gaps are relinked (`detector_model.py`).

### 3.3 The ball

**1. Picks per camera** (`ball_track.py`). A Viterbi search runs over each camera's candidates,
and the cheapest path wins.

Cheaper:
- confident candidates;
- candidates the other camera supports (rule 29).

Dearer:
- standing still, the penalty spot, the advertising boards;
- moving faster than a ball can.

Rule 45 adds what a ball cannot do:
- reappear farther away than it could have travelled since it was last seen;
- turn up at a player's feet with no pass toward him;
- restart anywhere other than where the Laws put it;
- exist for a single isolated frame.

**2. Height** (`ball3d.py`, Figure 1). Each camera reports where its line of sight meets the
grass, which for a ball in the air is past the ball.

- Where both cameras see it, triangulation gives the height.
- Flights are fitted as whole arcs, with gravity and quadratic drag (rule 34, physical constants).
  Rule 47 also fits flights that only one camera sees.
- Each camera's point is then corrected to its height.

**3. Fusion** (`kalman.py`). A two-camera Kalman smoother runs forward and backward. It weights
picks by confidence and restarts only on evidence: both cameras agree, a kick, or a consistent
run. It bridges gaps, but never at a pace above 35 m/s, about the hardest shot (rules 39, 48).
A ball caught by a keeper stays with him until released (rules 36, 48).

![Line of sight](figures/line_of_sight.png)

*Figure 1. A camera sees where its line of sight to the ball meets the grass: for a ball in the air, a point well past it. Two cameras, or a fitted flight, give the height.*

### 3.4 Events

From the fused ball and our detected players (`rules.py`):

- **Touch:** a change of the ball's velocity of 3.5 m/s or more with a player within 4 m, not above
  head height; touches closer than 6 frames are one.
- **Control:** the same team within 1.3 m of the ball for 0.4 s.
- **Carry:** a player who keeps his pace through a gap in which the ball is unseen.
- **Bounce:** the landing of a fitted arc with no horizontal change. It is not a touch.

### 3.5 The decoder: model C

Three hidden states: H, A and D (Figure 2). On every frame each rule below adds evidence, in nats,
for one state against another; changing state costs a price; Viterbi picks the labels that explain
all the evidence at once, and a last pass cleans up short spells. Every rule that changes a label:

| | rule | what it does | weight | why |
|---|---|---|---|---|
| **Who has the ball** | touch (1) | credits a touch to the team of the player within 2 m | 3 nats | less than a switch (5.2): an uncontrolled touch does not change possession (CHALLENGE.md), and tracking jitter makes false touches |
| | far touch (1, 26) | a touch with the nearest player 2-4 m away, on a ball in the air, or with the ball unseen on one side of it | 1.5 nats | less sure who touched it; an aerial duel won is not control |
| | control (1.a), carry (2) | the ball held near one team's player | 4 nats | control is what changes possession |
| | kick (55) | a touch that sends a ground ball off at 8 m/s or more, no slower than it came, away from a player within 2 m, seen on both sides | switch price + 1 | a pass, shot or clearance struck by a player is an act of control; a deflection fails "no slower than it came" |
| | detection precision (53, 53b) | every event above is multiplied by how sure the ball detections around it are: the weaker side's precision, measured on public images, over the best band's | × 0 to 1 | an event resting on weak detections is doubtful whatever its kind |
| | measured touch first (58) | where the ball is unseen for a few frames, a stronger change measured beside the gap replaces the one the smoother fits inside it | | the fitted one is the measured kick seen through the smoother's lag |
| | far touch needs a clear nearest team (60) | a far touch counts only if the other team's nearest player is 1 m or more farther from the ball | 1 m | nearer than that, our position errors cannot tell who touched it |
| | distances confirm only (59) | the players' distances to the ball may support the team of the latest event, never the other | capped per frame, counted once per 0.5 s | a loose ball stays with the team that last played it |
| **Dead or live** | out of bounds (27) | the ball beyond a line by 0.5 m for 3 frames, dead from the crossing until the restart | 3 nats per 0.5 s; 6 once out for 1 s; 1 for a short run with nothing after it | Law 9; a lofted ball's ground point leaves the field only briefly |
| | broadcast cut-away | the broadcast loses its homography (a close-up) | +2.0 present, −1.3 absent | a director cuts away for most of a stoppage and little of live play (reasoned; no open dataset has a broadcast) |
| | replays neutral (62) | on replay frames the broadcast gives no evidence either way | 0 | a replay shows earlier play |
| | players still, stillness | most players under 1 m/s; mean player speed under 0.9 m/s with the ball still | precision and recall measured on IDSSE | players stop at a stoppage |
| | ball acceleration near zero | the ball smooth or still for 1 s | +1.75 present, 0 absent | presence from IDSSE; its absence is not a valid signal on our input, where the ball is unseen on 44% of frames |
| | still ball in a cut-away (64) | the broadcast has cut away *and* the ball is still (median speed of the frames it is seen on under 1 m/s over 3 s, seen on a third of them or more) | 3 nats | a still ball alone is no evidence (a player can hold a live ball); with a cut-away it is a stoppage |
| | restart or kick is live (56) | a kick from rest inside the pitch, or any kick | 9 nats against dead, over 1 s | nothing but a restart sends a stationary ball off a player's foot |
| | before kick-off | the second half before its kick-off | forced dead | the Smart Tagging kick-off |
| **After decoding** | short spells (63) | a live spell under 5 s between two stoppages becomes dead, unless the ball goes out within 1 s of its end; then a stoppage under 5 s between two spells of the same team, with the cut-away as its only evidence, becomes that team's | 5 s | a touch inside a stoppage reads as a restart; the broadcast alone does not overrule a tactical camera that shows play |
| **In the ball track** (3.3) | keeper hold (36, 48) | a ball caught by a keeper stays in his hands until released | | no phantom touches during a hold |

Changing state costs log(mean length × 25 fps), with the mean lengths of possessions (7.2 s),
live play (31.2 s) and stoppages (25.3 s) on IDSSE's labels: H ↔ A 5.2 nats, live → dead 6.7,
dead → live 7.1. Rule 63's 5 s window is a round value set by reasoning (4, 5 and 6 s differ by under
0.002 on IDSSE; its targets are artefacts of our own input that the external data does not have).
Every value and its basis is in `docs/VALUES.md`. The submitted file writes `X` only on the 40,462
ungraded frames (before kick-off, half-time, after the final whistle); every graded frame gets H, A
or D.

## 4. The possession rules we implemented

CHALLENGE.md defines a change of possession as an opponent gaining **controlled** possession.
Where we drew the line, situation by situation (rule numbers from the table in 3.5):

| situation | how the model decides it |
|---|---|
| **A single touch** by the other team | Not enough by itself (touch, 1): it needs a second touch, a control, a carry or a kick. CHALLENGE.md: "a simple touch without control" does not change possession. |
| **A kick** by the other team | Changes possession on its own (55): an interception struck straight on is control, even with no dwell. |
| **A deflection** | Either no touch at all (the ball keeps its course), or a touch that fails the kick's "no slower than it came" test: not enough by itself. |
| **A header or aerial duel** | A far touch (26): possession follows whoever controls the ball afterwards. |
| **A loose or contested ball** | Stays with the team that last played it (59), until the other side touches, kicks, controls or carries it; a far touch between two equally near opponents counts for nobody (60). |
| **A tackle** | One that wins the ball shows up as the tackler's team's control or carry; one that only knocks it loose stays with the previous team, as a loose ball. |
| **A ball out of bounds** | Dead from the crossing until the restart (27). |
| **A stoppage** (foul, injury) | Dead when the players are still or the broadcast cuts away, and surely so when the ball lies still during a cut-away (64). A cut-away under 5 s inside one team's possession, with nothing on the tactical camera to support it, is not a stoppage (63); replays say nothing (62). |
| **A restart** | Live from the restart kick (56); the team taking it gets the touch, so the restarting team has the ball (CHALLENGE.md: "a clear possessor"). A "restart" followed by a stoppage within 5 s, with no ball going out, was the ball being set up (63). |
| **A goalkeeper's catch** | His team's possession, held until release (36, 48). |
| **Before kick-off / half-time** | Forced dead. |

## 5. Results

**Public window** (17 possessions). The frozen model was scored on it once, as a pipeline check:

| | spellF1 (95% bootstrap interval) | macroF1 | possessions |
|---|---|---|---|
| **submitted model** | **0.634** (0.43-0.80) | **0.894** | 12 |
| published floor, `baseline_smooth50` | 0.326 (0.21-0.45) | 0.621 | 26 |
| for scale: the ground truth itself, 0.5 / 1 / 2 s late | 0.906 / 0.776 / 0.588 | 0.966 / 0.931 / 0.871 | 17 |

The submission scores like a perfect labeller running about 1.5 s late. EVALUATION.md is explicit
that one possession here moves spellF1 by about six points, so this is a check, not a ranking. (It
was scored on the frozen model just before its last two small changes, rule 63's window and
rule 64, which together change one 2.7-second stretch of the match; it was not scored again. The
window was looked at once more before the freeze, on the re-derived values, and nothing was
decided by it.)

**External matches.** Values that needed data were chosen on the 7 IDSSE matches and reported on
the 3 Metrica games, never chosen on them; every match is scored clean and with noise that mimics
our own detector's input (`validation/README.md`). The submitted model:

| set | spellF1 | for scale: the ground truth 1 s / 2 s late | possessions predicted vs true |
|---|---|---|---|
| IDSSE, clean | 0.486 | 0.557 / 0.356 | ×0.59 |
| IDSSE, noisy | 0.417 | | ×0.48 |
| Metrica, clean | 0.543 | 0.608 / 0.426 | ×0.57 |
| Metrica, noisy | 0.435 | | ×0.48 |

![External](figures/external_scores.png)

*Figure 3. The submitted model on the 10 external matches (clean tracking), against R1 (every touch
hands over), our first HMM, and the ground truth delayed by 1 s for scale.*

![By length](figures/match_by_length.png)

*Figure 4. Share of true possessions found, by their length. Long possessions are found about as
often as by R1; short ones much less, partly by design (section 7).*

What the results say:

- **The two datasets agree.** On clean tracking the model scores like a perfect labeller about
  1.2 s late, on the public window about 1.5 s late; we take this as a sign the model is not
  fitted to BAR-ATH.
- **Long possessions and the live / dead boundary are its strength:** about 80% of possessions
  longer than 8 s are found, and macroF1 is well above the touch-by-touch rule R1's.
- **It is robust to noise.** On the noisy copies it is level with R1, the best decoder on clean
  data, which falls below the published floor on our own detector input.
- **Short possessions are its gap.** It predicts about half as many possessions as there are,
  almost all of the missing ones short, partly by design (section 7).
- **One choice costs here on purpose.** On the external matches a dead ball is parked by the
  provider and visibly still, so "the ball is not still" is real evidence of live play there; on
  our input the ball is unseen on 44% of frames and moving at most stoppages. We count only the
  presence of a still ball, which fits our input and costs on theirs.

## 6. How every value was set

Every constant had a dated rationale entry in our development log, written before the code: its
value, its reason, the alternatives, and how to revisit it. `docs/VALUES.md` lists every value with
its basis. Values fall into four groups:

| group | examples | basis |
|---|---|---|
| **Physics, the Laws of the Game, published figures** | 35 m/s maximum ball pace, drag coefficient, penalty-area size, the restart tests | external |
| **Measured or chosen on external matches** | the mean possession, live and dead lengths and the dead share (IDSSE's labels); the dead-ball signals' precision and recall (IDSSE); the HMM weights left at their values by the IDSSE search; `KICK_MARGIN`, `LIVE_KICK_LLR` (rules 55-56); `ref_delay` 0 and rule 27's out-of-bounds values confirmed there; rules 58-60 tested there | external (IDSSE, Metrica held out) |
| **Reasoned** | the broadcast cut-away signal (no open dataset has a broadcast); the player relink window and view margin; the keeper's hold; rule 63's 5 s window and rule 64 (checked on IDSSE where they can fire) | reasoning, a physical limit or the Laws |
| **Adopted after spot checks on BAR-ATH** | rules 53 / 53b (Matias checked the spells they change in the viewer); the choice of rules 58-60 (a kick lost to a ball-gap artefact and a loose ball); rules 62-64 (replays, short spells, a still ball in a cut-away); counting only the presence of "ball acceleration near zero" (stoppages missed on our input). Their values are external or reasoned; which rules to submit was decided with these checks. One of them (rule 64) came from a moment inside the public window | external or reasoned values, BAR-ATH check |
| **Reasoned heuristics, motivated by a failure seen on BAR-ATH** | most ball-track switches (rules 29-48): e.g. a junk pick 30 m away that the tracker took. The values are physical or reused round numbers ("initial guess, untested"), never swept to a score. | heuristic, BAR-ATH case |
| **Own measurement** | the noise floors of our own detections, set once without labels or scores: the touch threshold (3.5 m/s velocity change), "acceleration near zero" (3 m/s²) and the still-player speed (0.9 m/s) | BAR-ATH, disclosed |

No value in the submitted model is trained, fitted, or chosen by a score on BAR-ATH. The full list,
with each value's basis, is in `docs/VALUES.md`.

## 7. Limitations, and what we would do next

- **Short possessions, partly by design.** CHALLENGE.md says a simple touch without control, a
  deflection that does not intercept a pass, and an aerial duel won without control do not change
  possession. We follow it: one uncontrolled touch is worth less than a switch, and possession
  changes on a kick, a control, a carry or a second touch. The ground truth, derived mechanically
  from the provider's tracking, does contain very short possessions, some of them single touches:
  2 of the 17 public possessions last under 2 s (0.8 s and 1.3 s), and 6 under 4 s; on the external
  matches 25-37% of true possessions last under 2 s. We find 5-11% of those under 2 s on the
  external matches (Figure 4), so part of this gap is a deliberate reading of the rules rather than
  a tracking error. The rest is a real limitation: of possessions of 2-4 s we find 11-23%, where
  the touch-by-touch rule R1 finds 26-49% (on clean data, where every touch is real). Next: a
  duration-aware decoder (HSMM), or a switch price that falls where the ball is seen cleanly.
- **False touches.** Jumps between ball candidates and team-label flips are BAR-ATH's main
  noise, and our external noise model does not include them yet. Adding them comes before any
  new search.
- **Live/dead boundary.** Our stoppage signals (still players) arrive a few seconds after the
  whistle. Rule 57 moves a stoppage back to a broadcast whistle just before it, but it is not
  submitted: the model uses no audio, and the whistle detector's thresholds were set on this
  match's whistles.
- **Short spells.** Rule 63 removes live spells under 5 s between stoppages and fills short
  broadcast-only stoppages. Real play that short (a corner cleared at once before the next
  stoppage) can be lost, and a false restart longer than 5 s is not caught. A test of where the
  ball comes to rest (the goal area, the corner arc) would separate a ball set up for a restart
  from one in play; a stillness-only version (rule 61) cost on the external matches.
- **A ball played around during a stoppage.** Kicked away, passed to the taker or juggled, it
  looks like live play to the tactical camera; only the broadcast and the referee's whistle show
  the stoppage, and we weigh the broadcast below clear tactical evidence. Rule 64 catches the
  part of such a stoppage where the ball lies still during a cut-away, not
  the part where it is kicked about. The whistle detector would catch these, but its thresholds
  were set on this match's whistles, so it is not submitted.
- **Tracking differs from the external data.** The external matches' provider parks the ball at
  stoppages and never loses it; our detector's ball is unseen on 44% of graded frames. Signals
  that depend on the ball at a stoppage transfer badly (the acceleration signal above), and our
  noise model for the external copies (ball dropouts, jitter) does not reproduce this.
- **Dead balls against keeper saves and stalling.** A stoppage, a keeper holding the ball after
  a save, and a player standing on the ball to slow the game look alike to our model: the ball
  is still or unseen, someone is near it, and play has paused. The difference is in how the
  players move (walking back into shape, lining up a wall, the keeper waiting for team-mates to
  push up) and in who exactly has the ball. A catch while the ball is unseen gives no event, so
  the attacking team keeps possession until the next one; a low-confidence pick near
  a keeper can end a stoppage early. Play close to a dead ball (a quick restart, a ball knocked
  about after a whistle) is error-prone for the same reason. Next: player movement patterns as
  evidence, and higher-resolution player labels (individual identities and roles, keepers in
  particular, rather than team and keeper groups), and extending the keeper hold (rules 36, 48)
  to a ball that disappears near a keeper.
- **Dependence on provided data.** The homographies and the video-to-video alignment are still
  the provider's. Our own calibration is built but not wired in.
- **The tactical camera's ball is small** (about 5 px). Detector recall at players' feet is the
  ceiling on everything above. Fine-tuning on a wide-angle dataset (SoccerTrack v2) is the lead.

## 8. Reproduction

```
git clone https://github.com/mtprates/gremio-mtah27 && cd gremio-mtah27
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-detect.txt   # torch for your GPU first
# lay out the match data as DATA.md describes (./data, or set MTAH_DATA_DIR)
.venv/Scripts/python reproduce.py --team gremio      # -> submission/gremio.csv, validated
```

`reproduce.py` runs every step whose output is missing:

| step | what it does | time (GTX 1660 Ti) |
|---|---|---|
| detection | YOLO over both videos | ~13 h |
| shot features, cuts | the broadcast's shots | ~10 min |
| tracker and teams | per video chunk | not stated |
| model | ball track, events and HMM | ~15 min first time |
| submit | copy and validate the file | seconds |

**Hardware and determinism.** The detections were produced on a GPU (NVIDIA GTX 1660 Ti, CUDA
12.6, torch 2.14.1+cu126, ultralytics 8.4.183). GPU inference is not bit-for-bit reproducible
across GPU models, drivers or CPU-only runs: box coordinates and confidences can differ in their
last digits, which near a decision threshold can change which ball candidate the tracker picks,
and so a few labels. Everything after detection is deterministic (the kit clustering is seeded),
and given the same detections the file is reproduced byte for byte: checked on a fresh clone of
the submitted commit, run against the same detections, which rebuilt `gremio.csv` identically
(224 possessions, validated). Small differences in a rerun from the videos are therefore
expected, and come from detection, not from the model.

`model/readiness.py` and `model/tune_external.py` rebuild the external results from the public
datasets (download steps in `external/README.md`).

Python 3.13. The tested versions are pinned in the requirements files. `docs/make_figures.py`
redraws these figures from committed files only.
