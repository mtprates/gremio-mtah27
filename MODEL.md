# How the model works

A short overview. `docs/TECHNICAL.md` is the full write-up, `docs/VALUES.md` lists every value with its
basis, and `README.md` covers setup.

**The task:** for every frame of the match, say who has the ball: home (H), away (A) or dead ball (D).
**The approach:** find the ball and the players ourselves in both videos, then let a probabilistic
model weigh everything the ball and players say about possession.

```
videos ──> detections ──> players + teams ─────────────────────┐
              │                                                 v
              └──> ball picks per camera ──> 3D ball ──> events ──> possession model ──> H / A / D
```

## 1. Two cameras, one timeline

The match comes as two recordings.
- **Tactical:** a wide view of the whole pitch.
- **Broadcast:** the TV feed, with cuts, close-ups and replays.

Each comes with a homography per frame, mapping the image to the pitch. The broadcast is not
frame-aligned to the tactical, so `align.py` matches the two by the layout of players on the pitch,
using the provider's tracking. Replays fall out because they go back in time. `shots.py` and
`cameras.py` find the broadcast's cuts and close-ups, so only its live wide shots are used.

## 2. Detection and players

- **Detection** (`detect_all.py`, `detect_ball.py`): YOLO finds ball candidates and person boxes in both
  videos. Each box's shirt colour is saved.
- **Teams** (`teams.py`): boxes are tracked on the pitch, and shirt colours are grouped into kits by a
  Gaussian mixture. That gives home, away, the two keepers and the officials.
- **Players** (`detector_model.py`): the tracks become player columns, relinked across short gaps.

## 3. The ball

This is where most of the work has gone, because ball errors become fake touches.

1. **Picks per camera** (`ball_track.py`). Each frame has several candidates or none. A Viterbi search
   picks the cheapest path through the whole window:
   - confident candidates are cheap;
   - standing still, a penalty spot, the boards beyond the lines, and moving faster than a ball can,
     are expensive;
   - a candidate the other camera supports is cheaper (rule 29).

   Rule 45 adds what a ball can't do:
   - appear far from where it was last seen sooner than it could have got there;
   - turn up at a player's feet with no pass seen toward him (a ball out of sight beside a player is
     covered, and moves with him);
   - restart anywhere other than where the Laws put it after it goes out;
   - be seen in one isolated frame.
2. **3D** (`ball3d.py`). Each camera reports where its line of sight to the ball meets the grass. For a
   ball in the air that point lies well past the ball, away from the camera.
   - **Two cameras:** triangulating them gives the height.
   - **Flights:** fitted as whole arcs with gravity and drag. Rule 47 also fits flights that only one
     camera sees.
   - **Correction:** each camera's point is moved to where its line of sight is at that height.
   - **Continuity:** the height may not jump (rule 46).
3. **Fusion** (`kalman.py`). A two-camera Kalman smoother runs forward and backward:
   - it rejects picks too far from where the ball should be;
   - it restarts only on evidence (both cameras agreeing, a kick, a consistent run);
   - it bridges gaps, but never at a pace no ball has (rules 39, 48);
   - a ball caught by a keeper stays in his hands until released (rules 36, 48).

## 4. Events

From the ball's motion and the players around it (`rules.py`):
- **Touches:** a change of the ball's velocity with a player within reach, and not above head height.
  Where the ball is unseen for a few frames, the smoother places at most one touch in the gap, and a
  stronger change measured just beside it wins (rule 58).
- **Kicks (rule 55):** a touch that sends a ground ball off at 8 m/s or more, no slower than it came,
  away from a player within 2 m.
- **Controls:** the ball staying near one player.
- **Carries:** a player moving with an unseen ball.
- **Bounces:** a landing with no horizontal change, so nobody touched it.

Dead-ball signals:
- the ball out of bounds;
- the broadcast cutting away (losing its homography); replays say nothing either way (rule 62);
- players standing still, and overall stillness;
- the ball's acceleration near zero (its presence only: our ball is missing on many frames);
- a still ball while the broadcast has cut away (rule 64).

No audio is used. A referee-whistle detector exists (rule 57) but is not part of the model.

## 5. Possession

Three models share the same ball and players; **C is the one submitted**, T and P are comparisons.

- **C, team HMM** (`hmm.py`). Three hidden states, H, A and D. Each rule gives evidence in nats
  instead of a yes or no:
  - a bonus at each touch, kick, control or carry, scaled by how sure the ball detections around it
    are (rules 53, 53b); a kick alone is enough to change possession;
  - players' distances to the ball, which may confirm the team of the latest event but never flip it
    (rule 59): a loose ball stays with the team that last played it;
  - each dead-ball signal by how reliable it is, and live evidence at a restart or a kick (rule 56).

  Switching state has a price, set from how long possessions, live play and stoppages last on IDSSE's
  labels (7.2 s, 31.2 s, 25.3 s). Viterbi picks the labels that explain everything at once, so one stray
  touch does not flip possession without more evidence. A last pass removes live spells under 5 s
  between stoppages and short stoppages seen only by the broadcast (rule 63).
- **T, touches and pitch control** (`touch_model.py`). Possession changes only at touches. Who wins a
  touch is decided by pitch control at the touch spot and by who is nearest. A comparison only.
- **P, player HMM** (`player_hmm.py`). Like C, but it tracks which player has the ball, so a pass to a
  teammate is cheaper than a turnover. A comparison only.

The output is one label per frame, written as a submission CSV (`SUBMISSION.md`). Every graded frame
gets H, A or D; `X` only on frames that are never graded.

## 6. How we judge it

- **No tuning on this match.** No hand labels, nothing trained on BAR-ATH, and no value chosen by a
  score on it. Values come from physics, the Laws of the Game, published figures, reasoning, or the
  external matches (IDSSE to choose, Metrica held out); the only BAR-ATH measurements left are the
  noise floors of our own detections, taken without labels or scores and disclosed. Every value's
  reason is in `docs/VALUES.md`, written before the code, and listed in `docs/VALUES.md`.
- **BAR-ATH is watched to find bugs, not to fit.** Spot checks in the viewer showed where a rule was
  needed (a kick lost to a ball gap, a replay read as live play); the rule's values come from outside
  the match. Those decisions are disclosed in `docs/TECHNICAL.md`.
- **Tracking is judged by internal consistency:** jumps no ball could make, picks both passes reject,
  restarts, and the cases flagged in the viewer. The provider's ball is a comparison, not ground truth.
- **The public window** (17 possessions) is reported, not optimised: the frozen model was scored on it
  once.

New rules go in behind switches, off by default, until they have been reviewed. The submitted set is
`MODEL_FLAGS` in `reproduce.py`; `docs/TECHNICAL.md` is the full write-up.
