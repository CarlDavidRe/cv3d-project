# Next-best-view policy variants

The reconstruction dashboard compares nine policies at five total view budgets
(`1`, `2`, `3`, `5`, and `10`), giving 45 policy/view variants. Every rollout
starts with canonical camera anchor 0, and the budget includes that initial
view. A policy only chooses the later cameras; all variants use the same
camera calibration, reconstruction pipeline, and 2DGS/3DGS fitting settings.
The pipeline receives each policy's own ordered RGB history, so reconstruction
differences should be attributed to the selected views rather than to a
different fitting method.

## Phase 2: single-view and geometric policies

These policies either use geometry alone or combine predictions made from each
acquired image independently. They were not trained on complete view histories.

- **`phase2_random`** is the non-learned stochastic baseline. It assigns a
  deterministic seeded random ranking to the remaining cameras at every step.
  The seed includes the object and step, so reruns are reproducible and object
  processing order does not affect the result.

- **`phase2_farthest`** is the non-learned coverage-by-pose baseline. It selects
  the candidate whose viewing direction has the greatest angular distance from
  its nearest acquired camera. It encourages camera-space diversity but never
  examines RGB content or predicted object visibility.

- **`phase2_pun`** uses the released PUN/UPNet checkpoint. Each acquired RGB
  image independently predicts a source-relative 48-camera PSNR map. The maps
  are rotated into the canonical camera frame and combined multiplicatively.
  PUN's `small` rule first suppresses candidates that any acquired view predicts
  to have relatively high PSNR; the policy then visits the surviving camera
  with the smallest raw PSNR product. It uses no ground-truth geometry at test
  time.

- **`phase2_vggt`** replaces PUN's image encoder and head with the
  validation-selected Phase 1 frozen-VGGT feature plus a learned lightweight
  PSNR head. Like PUN, it processes every acquired image separately, aligns the
  source-relative maps, and uses the same filtering and aggregation rule. This
  isolates the effect of the learned visual representation while retaining the
  single-image PSNR formulation.

- **`phase2_oracle`** is a privileged upper-bound policy. At each step it uses
  the true incremental `VisA` surface gain for every candidate and greedily
  chooses the largest gain. It is useful as a ceiling for view selection, but
  it is not deployable and must not be compared as though it used only RGB.

## Phase 3: history-aware VGGT policies

All four Phase 3 policies are trained on object-disjoint histories to predict
the 48 candidates' incremental `VisA` surface gains directly. Higher output is
better. Unlike the Phase 2 learned policies, they make one prediction from the
whole acquired history and do not predict or aggregate PSNR maps.

- **`phase3_vggt_independent_history`** extracts one frozen, max-pooled VGGT
  vector per image in separate single-image forwards. Each vector is paired
  with its known camera direction; the pairs are averaged across the history
  and passed to a lightweight 48-output gain head. Views interact only in this
  final mean, making this the independent-feature control.

- **`phase3_vggt_joint_history`** sends the complete acquired image sequence
  through frozen VGGT jointly, allowing cross-view interaction inside the
  backbone. Its per-view vectors and camera directions are then averaged and
  scored by the same capacity-matched head used by the independent control.
  Comparing these two variants tests joint versus independent VGGT context.

- **`phase3_vggt_joint_pose_deepsets`** also runs VGGT jointly, but applies a
  shared nonlinear encoder to each `(view feature, camera direction)` pair
  before averaging. This preserves appearance/pose associations that direct
  averaging reduces to separate means. It is an expressive ablation, not a
  capacity-matched replacement for the preceding joint control.

- **`phase3_vggt_joint_token_attention`** retains a coarse 2-by-2 spatial token
  grid for every acquired view instead of max-pooling each view to one vector.
  Tokens receive the acquired-camera pose, while all 48 candidate directions
  act as queries in cross-attention and receive individual gain scores. It
  tests whether spatial pooling is the bottleneck, at greater cache and model
  cost than the capacity-matched controls.

The `phase2_` and `phase3_` prefixes in artifact directories identify the
experiment family; the policy names stored in metrics omit those prefixes.
The two expressive Phase 3 results should be reported as ablations, while the
independent-versus-joint pair is the controlled, capacity-matched comparison.

See [`training_runs_and_losses.md`](training_runs_and_losses.md) for the
training datasets, retained runs, optimizer settings, loss definitions, and
Gaussian-fitting objectives behind these variants.
