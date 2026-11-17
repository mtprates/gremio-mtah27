# Every value in the submitted model

This lists every constant that can change a label of the submitted model (model C with rules
53, 53b, 55, 56, 58, 59, 60, 62, 63, 64; `reproduce.py`'s `MODEL_FLAGS`), with its value and where it comes
from. File paths, cache settings, display values and the switches of rules that are not
submitted are left out (listed at the end).

**How values were set.** The model began as heuristics built from physical limits and the Laws
of the Game. We watched BAR-ATH to find bugs in our logic and to understand the data's errors
(a spare ball by the boards, a broadcast cut-away), and wrote a general rule for each failure
we found. No value was trained or fitted on BAR-ATH, and none was chosen by its score against
any BAR-ATH labels. Where a value needed data, it came from public datasets: the IDSSE / DFL
matches (search) with the Metrica sample games held out, and public ball images (Roboflow, CC
BY 4.0).

**Basis codes**

| code | meaning |
|---|---|
| Physics | a physical limit, geometry or a statistical quantile |
| Laws | the Laws of the Game (pitch markings, restarts, squad size) |
| Published | a paper, a data specification or a library default |
| External | measured or chosen on public data: IDSSE (search), Metrica (held out), Roboflow images |
| Reasoned | a round number or a reused constant, set by reasoning. *Spot check* means a failure seen while watching BAR-ATH showed the rule was needed; its value was not compared against any score |
| Own measurement | a noise level or camera geometry of our own detections on BAR-ATH, measured once without labels or scores |

## 1. Detection (`detect_ball.py`)

| constant | value | basis |
|---|---|---|
| `MODEL` | COCO-pretrained `yolo11m`, no fine-tuning | Published (Ultralytics) |
| `PERSON_CONF` | 0.25 | Published (Ultralytics default) |
| `CONF_MIN` | 0.05 (ball candidates kept; the tracker decides) | Reasoned |
| `TILE`, `TILE_X`, `TILE_Y` | 640 px tiles, 4 × 2 over the tactical frame | Physics (the ball is ~5 px at native resolution) |
| `BROADCAST_IMGSZ` | 1280 px | Reasoned |
| `PERSON_OVERLAP`, `MERGE_PX` | 0.6, 8 px | Reasoned |
| `TORSO`, `TORSO_WIDTH` | 20-50 % of box height, 50 % of width | Physics (body proportions) |
| `GRASS_HUE`, `GRASS_SAT`, `SHIRT_MIN_PX`, `HUE_BINS` | hue 35-85, sat ≥ 40, 6 px, 6 bins | Reasoned |

## 2. Broadcast shots and the two timelines (`shots.py`, `cameras.py`, `align.py`)

| constant | value | basis |
|---|---|---|
| `CUT_SPIKE`, `CUT_NEIGHBOURS` | 0.2 (histogram distance over its local median), ±6 frames | Reasoned, spot check |
| `HEIGHT_RATIO_MAX` | 2.0 × a 1.8 m person's height under the homography | Physics, spot check |
| `ERR_MAX` | 2 px | Own measurement |
| `MAIN_ANGLE`, `DIRECTION_WINDOW`, `OTHER_ANGLE` | 8°, 25 frames, 20° | Own measurement (camera geometry) |
| `CENTRE_RADIUS`, `F_AGREE` | 5 m, 10 % | Reasoned |
| `HALF_OFFSETS`, `PAIR_RADIUS` | one offset per half, 1.5 m | Own measurement |
| `SEED_STRIDE_*`, `MIN_PLAYERS`, `GOOD_COST`, `UNIQUE_RATIO`, `SAME_OFFSET`, `MAX_SEED_GAP`, `FINE_TUNE`, `LIVE_COST` | 3 s / 10 frames, 5, 1.6 m, 1.4, 4 frames, 60 s, ±4 frames, 2 m | Reasoned |

## 3. Players and teams (`teams.py`, `detector_model.py`)

| constant | value | basis |
|---|---|---|
| `GATE_CHI2` | 13.8 (χ², 2 dof, p = 0.001) | Physics |
| `PLAYER_HEIGHT_M` | 1.8 m | Physics |
| `GROUPS` | 4-6 kit clusters (two kits, two keepers, officials) | Laws |
| `TEAM_MAX` | 11 | Laws |
| `BOX_DEPTH`, `BOX_HALF_WIDTH` | 16.5 m, 20.16 m | Laws |
| `DUPLICATE` | 0.3 | Physics |
| `COLOUR_STD`, `COLOUR_GATE` | 12, 35 Lab units (the two outfield kits are ~40 apart) | Own measurement |
| `OFF_PITCH`, `MAX_GAP`, `VEL0_PLAYER`, `COLOUR_EMA` | 1 m, 10 frames, 4 m/s, 0.1 | Reasoned |
| `MERGE_MARGIN`, `MERGE_ASPECT`, `MERGE_MAX` | 0.7 m, 0.85, 2 s | Reasoned |
| `L_WEIGHT`, `VOTE_WINDOW` | 0.5, ±0.5 s | Reasoned, spot check |
| `SPREAD_FLOOR`, `MATCH_M`, `SECOND_MIN`, `OFFICIAL_MATCH`, `SPLIT` | 16, 1.5 m, 0.2, 0.3, 0.7 | Reasoned |
| `SIDELINE_M`, `OFFICIAL_TRACK`, `OFFICIAL_LEAVE` | 2.5 m, 0.5, 0.15 | Reasoned, spot check |
| `PAIR_M`, `OTHER_WEIGHT` | 1.5 m, 1.0 | Reasoned |
| `KEEPER_SHARE`, `KEEPER_GOAL_M`, `KEEPER_MIN`, `KEEPER_SHARE_DEEP` | 12 %, 25 m, 25 frames, 60 % | Reasoned |
| `KEEPER_DEPTH` (= `rules.GK_RANGE`) | 30 m from the goal line | Reasoned, spot check |
| `COLUMN_GAP` | 5 frames | Reasoned |
| `RELINK_M`, `RELINK_SPEED` | 2 m + 8 m/s × gap | Physics (sprint pace) |
| `FEET_STD` family | 0.15 m at 60 px, within 0.08-0.8 m | Physics |
| `RELINK_MAX` | 1 s | Physics (an occlusion in the wide camera lasts under ~1 s; beyond it a straight-line bridge is unsafe) |
| `VIEW_MARGIN` | 2 m | Physics (a body width, the feet's projection error and a few frames at sprint pace from the view's edge) |

## 4. Ball picks per camera (`ball_track.py`; rules 18, 29-30, 33, 41, 45, 48)

| constant | value | basis |
|---|---|---|
| `V_GAP` | 35 m/s across gaps | Physics (about the hardest shot) |
| `V_MAX`, `V_SLACK` | 70 m/s frame to frame, 1 m | Physics (a rising ball's ground point outruns the ball) |
| `AIR_MAX` | 25 m | Physics |
| `V_AIR`, `V_LANDED` | 38 m/s, 25 m/s | Physics, Own measurement |
| `SPOT_PEN`, `SPOT_M` | 2 nats within 0.6 m of a penalty or centre spot | Laws, Reasoned |
| `THROW_M`, `EXIT_M` | 5 m, 3 m (rule 45: restarts where the Laws put them) | Laws, Reasoned |
| `V_PLAYER` | 8 m/s (rule 45: how far a ball can be carried unseen) | Physics (sprint pace) |
| `PRECISION` | detection precision by confidence band | External (COCO `yolo11m` on 1,392 public Roboflow tiles) |
| `CONF_TRACK` | 0.1 | Reasoned (consistent with the external precision table) |
| `DETECTION_NOISE`, `TRACK_OUT` | 0.25 m, 0.5 m | Own measurement |
| `REWARD`, `MISS`, `START` | 2, 0.5, 12 nats | Reasoned, spot check |
| `STATIC_PEN`, `STATIC_M`, `STATIC_HITS`, `STATIC_FRAMES`, `STATIC_STEP`, `STATIC_NEAR` | 3 nats, 0.5 m, 3, 2 s, 5, 10 | Reasoned, spot check |
| `LONELY_M`, `LONELY_PEN` | 15 m, 1 nat | Reasoned, spot check |
| `OUT_PEN`, `HINT_MAX`, `HINT_Z`, `SKIP`, `SPEED_COST`, `V_REF`, `GAP_FRAMES` | 4 nats/m, 4 s, 3.5 m, 3 s, 0.3 nats, 30 m/s, 5 | Reasoned |
| `REF_CONF`, `ANCHOR_FRAMES`, `AGREE_M`, `AGREE_STATIC`, `REF_NEAR`, `ANCHOR_PEN`, `ANCHOR_SCALE`, `ANCHOR_STRONG`, `START_REF` (rules 29-30) | 0.2, 2 s, 3 m, 1.5 m, 3 frames, 6 nats, 5 m, 0.5, 3 nats | Reasoned, spot check |
| `OUT_SOFT`, `OUT_CAP`, `OUT_MAX` (rule 33) | 0.5 nats/m, 3 nats, 10 m | Reasoned |
| `SIDE_MISS`, `SIDE_MIN`, `SIDE_JOIN`, `SIDE_AGREE`, `SIDE_END_GAP` (rule 41) | 3 m, 8, 6 frames, 1 m, 12 frames | Reasoned, spot check |
| `MEMORY`, `ZONE_SLACK`, `ISO_GAP` (rule 45); reused `HC_CONF`, `FEET_M` | 8 s, 2 m, 2 frames; 0.5, 1.5 m | Reasoned, spot check |
| `OFFICIAL_FEET`, `OFFICIAL_KEEP` | 0.3 box heights, 0.7 | Reasoned, spot check |
| `KEEPER_HOLD_RADIUS` (= `rules.CARRY_RADIUS`), `KEEPER_HOLD_MAX` | 1.5 m, 8 s | Reasoned |
| `KEEPER_HOLD_MIN` | 1 s | Reasoned, Laws (a parry changes direction within a frame or two; a catch holds the ball; Law 12 allows up to 6 s) |

## 5. The ball's height (`ball3d.py`, `airborne.py`; rules 17, 34-35, 42, 46-47)

| constant | value | basis |
|---|---|---|
| `G` | 9.81 m/s² | Physics |
| `DRAG_C` | 0.0133 per m (quadratic drag) | Physics (published drag of a football) |
| `DZ_MAX` | 1.0 m per frame (25 m/s vertically) | Physics |
| `REACH`, `CONTROL_REACH` | 2.3 m, 1.2 m | Physics (body reach) |
| `PERFECT_RMS` | 0.02 m | Physics |
| `FIT_RMS_BASE`, `FIT_RMS_PER_S` (rule 35) | 1.0 m + 0.45 m/s × flight time | Reasoned, spot check (drag on long balls) |
| `BRIDGE`, `BRIDGE_UNKNOWN`, `VALLEY_RISE`, `VALLEY_FRAMES` | 6 frames, 2 s, 0.4 m, 10 frames | Reasoned, spot check |
| `FLIGHT_JOIN`, `FILL_MAX` (rule 42), `SOLO_BASE`, `SOLO_COVER` (rule 47) | 3 frames, 25 frames, 5 frames, 0.5 | Reasoned |
| `GROUND_GAP`, `Z_SMOOTH`, `AIR_SIGMA`, `AIR_MIN`, `MIN_FLIGHT`, `MIN_AIR`, `GROW_MAX`, `TURN_UP`, `EDGE_Z`, `FIT_ITER`, `TRIM_Z`, `PEAK_MIN`, `Z_FLOOR`, `SINGLE_GAIN`, `SINGLE_PEAK` | 1 m, 5 frames, 2.5σ, 0.5 m, 6 and 4 frames, 1 s, 0.25 m, 0.2 m, 12, −0.2 m, 0.8 m, −0.5 m, 0.7, 1 m | Reasoned |
| `airborne`: `SPEED_MIN`, `MIN_FRAMES`, `MAX_FRAMES`, `MAX_GAP`, `RMS_MAX`, `JUMP_MAX` | 8 m/s, 0.6 s, 6 s, 6 frames, 0.35 m, 0.6 m | Reasoned |

## 6. Two-camera fusion (`kalman.py`; rules 13, 31, 39, 48)

| constant | value | basis |
|---|---|---|
| `MAX_PACE`, `MAX_PACE_ANY` (rule 39) | 35 m/s, 50 m/s | Physics (hardest shot ≈ 36 m/s) |
| `GATE`, `CONFLICT_Z` | 18.4 (χ², 2 dof, p = 1e-4) | Physics |
| `TOUCH_Z` | 3σ (χ², 2 dof, p ≈ 0.01) | Physics |
| `ACCEL_STD`, `VEL0_STD` | 3 m/s² (rolling friction, bounces, spin), 15 m/s | Physics |
| `MEAS_STD` (detector sources) | 0.5 / 1.0 m | Own measurement |
| `BIAS0_STD`, `BIAS_TAU` | 0.5 m, 30 s | Reasoned, spot check |
| `PACE_CONF`, `RESTART_MIN_CONF` (rule 39) | 0.5, 0.3 | Reasoned |
| `CONF_REF`, `VAR_MAX`, `RESYNC_CONSISTENT`, `RESYNC_SPREAD`, `RESYNC_BREAK`, `RESYNC_MARGIN` (rule 31) | 0.5, 16, 8 picks, 1.0 m, 5 frames, 0.1 | Reasoned, spot check |
| `RESET_GAP`, `POS_STD_MAX`, `RESYNC`, `RESYNC_AGREE`, `RESYNC_LONG`, `LONE_FRAMES`, `WEAK_AGREE`, `PLAYER_ACCEL`, `VEL_FRAMES` | 2 s, 1 m, 3 frames, 1.5 m, 1 s, 12, 2 m, 4 m/s², 3 | Reasoned |

## 7. Events (`rules.py`)

| constant | value | basis |
|---|---|---|
| `SPIKE` | 1.6 m in one frame (40 m/s) | Physics |
| `DV_THRESHOLD` | 3.5 m/s velocity change for a touch | Own measurement (~3σ of our velocity noise) |
| `TOUCH_RADIUS`, `TOUCH_CLOSE` | 4 m (far touch), 2 m (touch) | Reasoned (our position error) |
| `DV_WINDOW`, `SMOOTH`, `INTERP_MAX_GAP`, `TOUCH_NMS`, `TOUCH_SLACK` | ±4 frames, ±2 frames, 6 frames, 6 frames, 3 frames | Reasoned |
| `CONTROL_RADIUS`, `CONTROL_FRAMES` | 1.3 m, 0.4 s | Reasoned |
| `CARRY_SEEN`, `CARRY_RADIUS`, `CARRY_SPEED`, `CARRY_SLOW`, `CARRY_MAX` | 3 frames, 1.5 m, 0.5-1.6 × pace, 1.5 m/s, 5 s | Reasoned |
| `RESTART_IN_SPEED`, `RESTART_OUT_SPEED`, `MIN_DEAD`, `THROW_IN_LOOKBACK`, `RESTART_NEAR`, `RESUME_FRAMES`, `RESUME_SPEED`, `RESTART_MAX` | 2 m/s, 3 m/s, 1.5 s, 3 s, 3 s, 12 frames, 3 m/s, 60 s | Reasoned (Laws: a restart is a kick of a stationary ball) |
| `OOB_MARGIN`, `OOB_FRAMES` | 0.5 m beyond the line, 3 frames | Reasoned |
| `ACCEL_WINDOW`, `ACCEL_NEAR_ZERO`, `ACCEL_MIN` | ±6 frames, 3 m/s², 1 s | Reasoned; `ACCEL_NEAR_ZERO` Own measurement (our noise at that window) |
| `STILL_SPEED` | 0.9 m/s mean player speed | Own measurement |
| `STILL_BALL`, `STILL_MIN`, `QUIET_SPEED`, `QUIET_SHARE`, `QUIET_MIN_PLAYERS`, `QUIET_WINDOW` | 1 m/s, 3 s, 1 m/s, 60 %, 10, 1 s | Reasoned |
| `LOCAL_RADIUS`, `LOCAL_SPEED`, `LOCAL_MIN`, `LOCAL_MAX_AGE`, `LOCAL_BACKDATE` | 4 m, 1 m/s, 2 s, 10 s, 3 s | Reasoned, spot check |
| `EXTEND_MAX`, `EXTEND_SLACK` | 10 s, 0.5 s | Reasoned |
| `BROADCAST_DEAD_MIN` | 0 (no minimum length) | Reasoned |
| `ref_delay` (C) | 0: dead from the crossing | Laws (Law 9: out when wholly over the line); External (IDSSE `BallStatus`: 1 s scores lower) |

## 8. The decoder (`hmm.py`)

| constant | value | basis |
|---|---|---|
| `CORR_FRAMES` | 0.5 s | Reasoned; External (left unchanged by the IDSSE search) |
| `CONTROL_STD`, `PROX_CLIP` | 1.0 m, 3 nats | Reasoned; External (left unchanged by the search) |
| `BALL_STD` | 0.35 m | Own measurement |
| `EVENT_LLR` | touch 3, far touch 1.5, control 4, carry 4 nats | Reasoned (one touch < a switch, two > a switch); External (left unchanged by the search) |
| `KICK_SPEED` | 8 m/s | Reasoned (above a dribbling push, within a short pass) |
| `KICK_MARGIN` | 1 nat | External (IDSSE, Metrica held out) |
| `LIVE_KICK_LLR`, `LIVE_SPREAD` | 9 nats, 1 s | External (IDSSE); Reasoned |
| `FAR_MARGIN` | 1 m | Reasoned (our position error); External check |
| rule 59, 58, 53, 53b | no new numbers | External (reasoning; IDSSE / Metrica scores; the Roboflow precision table) |
| `POSSESSION_S` | 7.2 s → switch price ≈ 5.2 nats | External (IDSSE mean possession) |
| `LIVE_S`, `DEAD_S` | 31.2 s, 25.3 s | External (IDSSE mean live and dead stretches) |
| `BASE_DEAD` | 0.448 | External (IDSSE dead share; the base rate the precision / recall pairs below were measured at) |
| `SIGNALS`: broadcast without homography | 2.0 nats present, −1.3 absent | Reasoned (a director cuts away for ~3/4 of a stoppage, ~1/10 of live play; no external broadcast exists) |
| `SIGNALS`: players mostly still, stillness (precision, recall) | (0.966, 0.384), (0.992, 0.214) | External (measured on IDSSE's labels, clean copies; chosen over the noisy copies' values on the tune set) |
| `SIGNALS`: ball acceleration near zero | 1.75 nats present, 0 absent | External for the present weight (IDSSE precision 0.823, recall 0.947); Reasoned for the absent weight: not a valid signal on our input, where the ball is missing on many frames |
| `SIGNALS`: out of bounds, local stillness | 3.0, 1.5 nats | Reasoned |
| `OOB_LONG`, `OOB_SUPPORT`, "out of bounds, long" / "alone" (rule 27) | 1 s, 5 s, 6 / 1 nats | Reasoned (a lofted ball's ground point leaves the field for well under 1 s; a ball out stays out until the restart); External (IDSSE: alternatives within 0.001; "alone" at 0 is worse) |
| rule 62 (replays neutral) | no number | Reasoned (a replay shows earlier play) |
| rule 64: ball still in a cut-away | 3.0 nats present (the out-of-bounds weight), 0 absent; `STILL_SEEN` 1/3 of the 3 s window seen; reused `STILL_BALL`, `STILL_MIN` | Reasoned (a still ball alone is no evidence; with a cut-away it is a stoppage; the median needs 1 s of observations) |
| `SHORT_S` (rule 63) | 5 s | Reasoned (a round value; checked on IDSSE, 4-6 s differ by under 0.002; the rule targets artefacts of our own input) |

## Not listed

- **Rules in the code but not submitted** (off in `MODEL_FLAGS` or in model C's configuration):
  3, 3.a, 3.b, 6, 9, 10 (provider-tracking player rules), 23, 25, 28, 32, 37, 38, 40, 44, 49-52,
  54, 57 (referee whistles), 61 (restart confirmation), models T and P.
- **Rule 8** (`BALL_AGREE`, `BALL_RECENT`, `BALL_BIAS_DECAY`) and `dropouts.py` / `identity.py`
  serve the provider-tracking models only.
- File paths, cache keys, chunk sizes, label names and viewer display values.
