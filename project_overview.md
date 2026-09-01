# Project Overview — Direct Next-Best-View Prediction from Frozen VGGT Geometry Features

## 1. Purpose of this document

This file is the implementation reference for the project. It reorganizes the original proposal around the instructor's recommended three-phase execution plan so that each phase ends with a complete, usable checkpoint.

The core research goal remains:

> Determine how accurately frozen VGGT representations can rank unobserved camera poses by incremental surface coverage, and whether joint multi-view processing improves over independent per-view aggregation.

The project should be implemented so that:

1. **Phase 1 is already a complete single-image study.**
2. **Phase 2 is a complete end-to-end NBV project and the main fallback deliverable.**
3. **Phase 3 is the highest-risk / highest-novelty extension.**
4. Optional reconstruction-quality and real-world evaluations are added only after the core pipeline is stable.

---

## 2. Research problem

### Domain

Active 3D object reconstruction for robotics and embodied vision.

Given a limited observation budget, an agent repeatedly chooses the next camera pose that should expose the largest amount of previously unseen object geometry.

### Motivation

Classical next-best-view (NBV) methods often maintain explicit geometric state such as:

- occupancy grids,
- point clouds,
- uncertainty fields,
- voxelized representations.

The proposed approach asks whether **frozen VGGT latent features themselves already contain enough geometry-aware information to directly predict the value of candidate camera views**, without first constructing an explicit 3D uncertainty representation.

### Main research question

**How accurately can frozen VGGT representations rank unobserved candidate camera poses by incremental surface coverage, and does joint multi-view processing outperform capacity-matched independent-view aggregation?**

### Main hypotheses

- **H1 — Feature decodability:** NBV-relevant information is directly decodable from frozen VGGT features.
- **H2 — Geometry-aware advantage:** VGGT features outperform generic frozen visual features such as ImageNet-ViT and DINOv2 for the single-image prediction task.
- **H3 — End-to-end NBV usefulness:** A VGGT-based view-by-view predictor can be competitive with PUN and simple NBV baselines in closed-loop evaluation.
- **H4 — Joint multi-view advantage:** Joint VGGT processing of the observation history improves candidate-view ranking over independent-view processing followed by aggregation.

H4 is the main novel/high-risk hypothesis and belongs in Phase 3.

---

## 3. Project scope and priority

### Required core

The project is successful if **Phase 2 is complete and well evaluated**, even if Phase 3 is unsuccessful.

### Highest-priority research outputs

1. Controlled single-image backbone/feature comparison.
2. Reproduction or integration of the PUN baseline.
3. Complete closed-loop NBV evaluation pipeline.
4. VGGT-based independent per-view aggregation.
5. Joint multi-view VGGT comparison if time permits.

### Stretch outputs

- Reconstruction-quality evaluation using Chamfer distance or another geometric metric.
- Reconstruction evaluation after fixed view counts such as 5, 10, 15, ...
- Optional 3DGS-based reconstruction evaluator if computationally feasible.
- Real-world transfer evaluation following a PUN-like protocol, e.g. MipNeRF360.

---

# 4. Three-phase execution plan

## Phase 1 — Single-Image Feature Probe

### Goal

Answer the smallest clean research question first:

> Do frozen VGGT features contain more useful NBV information than generic frozen image representations on the original NUM single-image task?

This phase must include the full loop:

**method → training → evaluation → baseline comparison → visualization/demo**

No multi-view dataset extension should be required here.

### Data

Use the original PUN Neural Uncertainty Map (NUM) ShapeNet data with its 48 spherical anchor views and existing image-target pairs.

Prefer the original PUN/NUM train/validation/test split if available. If it cannot be recovered, define a fixed **object-disjoint** split and never change it between experiments.

### Models

Use the **same lightweight predictor head** whenever possible so that the backbone comparison is controlled.

Backbones:

- Frozen ImageNet-ViT
- Frozen DINOv2
- Frozen VGGT

VGGT feature variants to test, subject to what is cleanly exposed by the implementation:

- pooled patch tokens,
- camera token/features,
- register token/features,
- selected intermediate/final layers,
- a small fixed combination of the above.

The initial controlled sweep should use these fixed-size alternatives:

| Backbone | Feature variant | Purpose |
|---|---|---|
| None | training-set mean 48-anchor map | Measure performance available from dataset and anchor priors without looking at the image |
| Raw RGB | flattened 16×16 adaptive-average RGB grid + MLP | Test what the shared probe can learn directly from coarse pixels without a pretrained backbone |
| ImageNet-ViT | mean-pooled patch tokens | Generic spatial-feature baseline |
| ImageNet-ViT | classification token | Test the backbone's learned global summary |
| DINOv2 | mean-pooled patch tokens | Self-supervised spatial-feature baseline |
| DINOv2 | classification token | Test DINOv2's learned global summary |
| VGGT | mean-pooled patch tokens | Direct comparison with the generic patch baselines |
| VGGT | max-pooled patch tokens | Preserve strong localized geometric evidence that mean pooling may dilute |
| VGGT | camera token | Test VGGT's global camera-aware representation; VGGT has no classification token |
| VGGT | mean-pooled register tokens | Test the aggregator's learned global workspace separately |
| VGGT | camera token + mean-pooled patch tokens | One small global-and-spatial fusion variant |

Use the final cached VGGT layer for this first sweep. Treat intermediate-layer
selection and larger token combinations as follow-up ablations only if the
initial results justify them. This keeps the sweep small enough to interpret
while covering each feature family exposed by the current implementation.

#### Control baseline A — training-set mean map

This baseline measures how much of the task can be solved from dataset-wide
anchor preferences without observing the input image. For training target
`y_i,j` and validity mask `m_i,j`, compute one value per anchor using only the
training split:

```text
mean_map[j] = sum_i(m_i,j * y_i,j) / sum_i(m_i,j),  i in training split
```

Masked targets do not contribute. If an anchor has no valid training targets,
store zero as a finite placeholder and keep that anchor masked during
evaluation. The same resulting 48-value vector is returned for every
validation and test image. Validation and test targets must never contribute
to the mean.

This is an analytical, zero-trainable-parameter baseline: it does not use an
MLP, an optimizer, or image features. Save the computed map with the other
variant artifacts and evaluate it through the same candidate masks, Huber
loss, ranking loss, normalized regret, Spearman, and NDCG@5 implementation.

#### Control baseline B — raw-RGB MLP

This baseline tests whether the prediction head can learn the task directly
from coarse image appearance without a pretrained feature backbone:

```text
RGB image [3, H, W] in [0, 1]
    ↓ adaptive average pooling over the complete image
RGB grid [3, 16, 16]
    ↓ channel-first flattening
raw feature vector [768]
    ↓ LayerNorm → Linear(768, 128) → GELU → Dropout → Linear(128, 48)
48-anchor prediction map
```

Use `torch.nn.functional.adaptive_avg_pool2d(image, (16, 16))`. It averages
spatial regions into a fixed grid while retaining the complete image; do not
crop the image or apply ImageNet normalization. Keep RGB values in `[0, 1]`.
Do not flatten the full-resolution image, which would give this baseline a
much larger trainable head and make the capacity comparison misleading.

The 768-value input means this baseline uses exactly the same 768→128→48 MLP
and trainable parameter count as the ImageNet-ViT-B/16 and DINOv2-B/14 probes.
Train it with the same loss, optimizer, early stopping, seeds, split, and
candidate masks as the frozen-feature probes. Cache the flattened RGB vectors
with preprocessing and output-size metadata, and reuse compatible caches on
subsequent runs.

Run the sweep cache-first. Before loading a frozen backbone, reuse every
compatible per-variant cache whose model, preprocessing, layer, pooling,
dataset split, and target metadata match the request. Extract only missing or
explicitly invalidated variants, and extract all missing variants for the same
backbone from a shared forward pass. Cache rebuilding should remain disabled
by default; enable it only when intentionally invalidating prior features.

### Baseline

Include **PUN** in this phase.

Use the official implementation/evaluation behavior where possible. If any part is reimplemented, document the exact aggregation, preprocessing, target definition, and evaluation differences.

#### Official pretrained PUN/UPNet integration

Use the official released PSNR checkpoint rather than retraining PUN inside
this project:

```text
release: vit_small_patch16_224_PSNR_250425172703
model: timm vit_small_patch16_224 → classifier removed → Linear(384, 48)
checkpoint: best_vit_regressor.pth
official Drive file ID: 1vpVFy2LQMjN0jTJ_o1chZ6B4nJNfNVKP
SHA-256: 91b2065f7652aac0c84386d4af10cd0c1ae723c049c91e45ccc78722f70907ae
```

Pin the official PUN source revision and checkpoint checksum in the experiment
config. Download a missing checkpoint atomically into the shared model cache,
reject checksum mismatches, instantiate the official UPNet architecture with
`pretrained=False`, and then load the complete released state dict. Apply the
deterministic preprocessing resolved from the timm backbone configuration,
matching the official inference script.

Run PUN inference after every local probe entry so it is the final row in the
Phase 1 sweep. Do not create an optimizer, select an epoch, or use the local
train split for PUN. Evaluate the released predictions on the exact same
validation/test records, source-view mask, target direction, and common metric
implementation as all other entries. Also retain the official full-map
unmasked MSE as `official_unmasked_mse_loss`.

The released checkpoint does not include a machine-readable manifest of its
original training samples. Therefore, report that object/category overlap with
the project's test split cannot be ruled out; do not present the pretrained
PUN comparison as strictly training-data-controlled.

### Predictor

Conceptual interface:

```text
RGB image
    ↓
frozen backbone
    ↓
selected frozen feature tensor(s)
    ↓
pooling / normalization
    ↓
lightweight trainable head
    ↓
48-anchor prediction map
```

The head should be deliberately small because the experiment is about feature quality rather than large task-specific capacity.

### Loss

Use the proposal loss:

```text
L = L_huber + lambda_rank * L_rank
```

Where:

- `L_huber` regresses the target utility/uncertainty values.
- `L_rank` encourages correct ordering of candidate views.

Make `lambda_rank` configurable.

For early debugging, first verify that pure Huber regression trains correctly before enabling the ranking term.

### Phase 1 evaluation

Required:

- normalized utility regret, if compatible with the original NUM target representation,
- Spearman rank correlation,
- NDCG@5,
- PUN comparison,
- trainable parameter count,
- inference time,
- peak memory where practical.

Also report the original PUN/NUM metric if its official evaluation uses a metric not listed above. Keep official-baseline comparability separate from the project's common metric suite.

### Phase 1 training diagnostics

For every learned probe variant, record an epoch-zero baseline and one history
row after every completed epoch. Each row contains:

- the batch-time optimization training loss,
- evaluation-mode training and validation total loss,
- evaluation-mode training and validation Huber and pairwise-ranking losses,
- validation normalized regret, Spearman correlation, and NDCG@5,
- epoch and learning rate.

The evaluation-mode training pass uses the final weights from that epoch and
disables dropout, so its loss is directly comparable with validation loss. The
batch-time optimization loss is retained as a separate diagnostic because its
weights change throughout the epoch.

Write per-variant SVGs for the loss components and validation ranking metrics,
plus one validation-loss overlay across all learned variants. Mark the
validation-selected best epoch and last completed epoch, identifying the latter
as an early-stop point when patience ends training. Analytical baselines such
as the training-set mean map do not have training curves.

Do not compute or plot a test curve. Restore the checkpoint selected only by
validation loss, then evaluate the test split once for final reporting. Test
results must not influence early stopping or variant tuning. If Step 8 uses
test examples, choose their sample IDs without inspecting per-sample test
results.

### Phase 1 visualization/demo

Create a notebook or lightweight script that shows:

- input RGB image,
- ground-truth 48-anchor target map,
- predicted 48-anchor map,
- top-ranked predicted candidate,
- top-ranked ground-truth candidate,
- optional error/rank visualization.

### Phase 1 definition of done

- [x] Dataset loader is deterministic and tested.
- [x] All three frozen backbones can produce features through one common interface.
- [x] Feature caching works.
- [x] The same lightweight head can train on every backbone.
- [x] Evaluation produces one common metrics JSON/table.
- [ ] PUN baseline results are available in the same comparison table.
- [x] Official pretrained PUN is configured as the final inference-only sweep entry.
- [x] At least one qualitative visualization is reproducible from a saved checkpoint.
- [x] Experiment config, seed, checkpoint, and metrics are saved together.
- [x] A single command can reproduce the main Phase 1 comparison.

### Current Phase 1 repository status

Repository audit as of 2026-09-01:

| Area | Status | Current evidence / remaining work |
|---|---|---|
| Configuration and provenance | Implemented and tested | YAML loading/overrides, safe artifact paths, deterministic seeding, run directories, resolved config, environment metadata, and logs are implemented. |
| Canonical anchors | Implemented and tested | The checked-in 48-anchor CSV, ordering, directions, angular distances, camera poses, and candidate masking have unit coverage. |
| NUM data and split handling | Implemented, tested, and exercised at full scale | The loader validates RGB/target records and the checked-in object-disjoint PUN-compatible split. The completed sweep contains 41,760 training, 5,232 validation, and 14,400 test samples. |
| Target-only visualization | Implemented | `inspect_num_sample.py` produces a source-image/target SVG and self-contained interactive 3D anchor view. |
| Metrics and losses | Implemented and tested | Masked Huber, pairwise ranking, normalized regret, Spearman, NDCG@5, and coverage AUC are available. |
| Feature extractors | Implemented, tested, and exercised in the full sweep | Raw RGB, ImageNet-ViT-B/16, DINOv2, and single-image VGGT share one interface. The completed artifacts verify real-checkpoint extraction and training for every configured feature variant. |
| Feature/model caching | Implemented and tested | Model downloads use the shared model-cache root; feature vectors use metadata-fingerprinted, atomically written caches. Compatible caches are reused by default. |
| Probe training and diagnostics | Implemented and tested | The shared MLP, masked objectives, early stopping, best-state restoration, comparable post-epoch train/validation diagnostics, validation ranking histories, and dependency-free SVG curves are implemented. An ImageNet-ViT tiny-set overfit artifact succeeds. |
| Phase 1 controls | Implemented and tested | `train_mean_map` and `raw_rgb_16x16_mlp` are configured and emit the same evaluation/result schema as learned probes. |
| One-command sweep | Implemented; pre-PUN seed-0 run completed | The checked-in run contains the mean-map baseline and ten locally trained variants (11 rows). The current config appends official pretrained PUN as row 12 on rerun. Additional seeds remain desirable for final reporting. |
| Runtime/memory profiling | Not implemented | Trainable parameter counts are reported, but inference timing and peak-memory measurement still need a documented common protocol. |
| PUN comparison | Implemented and smoke-tested; full row pending | `inspect_pun.py` performs a one-image real-checkpoint smoke alongside the feature-model checks. The runner checksum-verifies and loads the official released PSNR UPNet checkpoint, applies official timm preprocessing, evaluates it last with both official unmasked MSE and common masked metrics, and emits the normal artifact schema. The complete 14,400-sample test row has not yet been generated. |
| Prediction demo | Implemented and tested | `visualize_phase1.py <experiment>` discovers every complete saved variant by default and writes one self-contained prediction-versus-target SVG per variant. Repeatable `--variant` filters select a subset. The checked-in raw-RGB validation example includes shared-scale target/prediction maps, absolute error, top candidates, regret, Spearman, NDCG@5, and MAE. |
| Phase 2 visibility infrastructure | Implemented and synthetic-tested; real-object run pending meshes | Deterministic OBJ surface sampling, canonical 48-anchor CPU depth visibility, atomic metadata-validated caches, split/subset precompute CLI, and debug SVG output are implemented. The local NUM release does not include its separately distributed ShapeNetCore.v2 meshes, so the required real-object precompute remains pending. No policy, closed-loop simulator, history dataset, or joint VGGT implementation is present. |

The automated suite currently contains 104 passing tests. This number records
the audit state rather than replacing the requirement for real-data,
real-checkpoint, and full-sweep validation.

### Phase 1 expected report result

A controlled statement such as:

> Frozen VGGT features are / are not more predictive of future-view utility than generic frozen image features under the same predictor capacity.

Even if the later phases fail, this can already be a meaningful result.

---

## Phase 2 — Independent Per-View VGGT + Complete Closed-Loop NBV Pipeline

### Goal

Build a complete NBV system using frozen VGGT **without joint history processing**.

This is the main project fallback and should be treated as a final-quality implementation.

The key question is:

> When observations are processed independently and combined across time, how well does a VGGT-based predictor perform in the full PUN-style closed-loop setting?

### Recommended independent-history design

To match the instructor's suggestion and PUN as closely as possible, the safest primary Phase 2 baseline is:

```text
I_1 ─→ frozen VGGT ─→ head ─→ per-view 48-anchor map ─┐
I_2 ─→ frozen VGGT ─→ head ─→ per-view 48-anchor map ─┼─→ history aggregation ─→ final map
...                                                    │
I_t ─→ frozen VGGT ─→ head ─→ per-view 48-anchor map ─┘
```

Use the official PUN post-prediction combination rule when known.

Possible controlled aggregation variants:

- mean,
- max,
- learned permutation-invariant aggregation.

If time is limited, prioritize the aggregation rule that most directly matches PUN, then add mean/max as small ablations.

A feature-level independent aggregation variant may also be useful:

```text
independent per-image VGGT features
    ↓
permutation-invariant feature aggregation
    ↓
shared predictor head
    ↓
48-anchor utility map
```

Treat this as an ablation unless it is required for the final capacity-matched Phase 3 comparison.

### Closed-loop evaluator

The evaluator must simulate sequential acquisition.

At time `t`:

1. Gather the observation history `I_1:t`.
2. Predict utility for all 48 anchor views.
3. Mask already acquired/invalid candidates.
4. Select `argmax` under the evaluated policy.
5. Acquire that view in the simulator.
6. Update visible surface state.
7. Compute coverage and per-step metrics.
8. Repeat until the view budget is exhausted.

Every policy must run through the **same evaluator**.

### Surface visibility representation

The proposal defines utility using ground-truth mesh surface coverage:

```text
u_t,j = Coverage(P_t ∪ z(v_j), W) - Coverage(P_t, W)
```

Implementation recommendation:

Precompute, per object and anchor view, a visibility mask over a fixed set of uniformly sampled mesh-surface points.

Then:

```text
seen_mask_t = OR of visibility masks for acquired views
candidate_gain(j) = count(visibility[j] AND NOT seen_mask_t) / N_surface
coverage_t = count(seen_mask_t) / N_surface
```

Benefits:

- exact consistency between labels and evaluation,
- much faster history-label generation,
- reproducible oracle utilities,
- easier unit testing.

Visibility must still be based on the proposal's depth-consistency rule.

Keep these configurable:

- number of sampled surface points,
- rendering/depth resolution,
- depth consistency tolerance,
- back-face/culling policy if applicable.

Do not silently change them between experiments.

### Required Phase 2 policies/baselines

From the proposal:

- Random policy
- Farthest-view heuristic
- PUN
- VGGT independent per-view aggregation
- Explicit geometry pipeline using VGGT-predicted depth
- Ground-truth-depth geometry upper bound
- Oracle one-step policy

If the explicit-geometry baselines threaten completion, implement them after the core Random / Farthest / PUN / VGGT / Oracle comparison is stable, but before declaring the final experiment suite frozen.

### One-step metrics

#### Normalized utility regret

For selected candidate `j_hat` and oracle candidate `j*`:

```text
R_t =
    (u_t,j* - u_t,j_hat)
    /
    (u_t,j* - min_j u_t,j + epsilon)
```

Lower is better.

#### Ranking metrics

- Spearman correlation
- NDCG@5

Important:

- Mask already observed candidate views before computing policy selection.
- Define consistently whether ranking metrics include or exclude invalid/already-seen anchors.
- Prefer evaluating over the valid candidate set.

### Closed-loop metrics

Required:

- surface coverage vs. number of acquired views,
- area under the coverage-vs-view curve,
- final coverage at the maximum view budget,
- median inference time,
- peak memory,
- trainable parameter count.

Recommended reporting points:

- early budget,
- medium budget,
- final budget.

The exact reported view counts should match the experiment horizon and be fixed before final evaluation.

### Phase 2 demo

Extend the visualization so a full rollout shows:

- acquired RGB views,
- current predicted spherical utility map,
- selected NBV,
- accumulated visible surface,
- coverage curve,
- optional policy comparison on the same object.

The demo should load saved checkpoints and results. It should not contain hidden training/evaluation logic that exists only inside the notebook.

### Phase 2 definition of done

- [ ] Closed-loop simulator works on a single object.
- [ ] Surface visibility/coverage cache is reproducible.
- [ ] Oracle candidate utility exactly matches the same coverage state used by evaluation.
- [ ] All policies share one evaluator.
- [ ] Already-seen views are masked consistently.
- [ ] PUN and VGGT can both run in the same history protocol.
- [ ] Coverage curves can be produced for the entire test split.
- [ ] One-step ranking metrics and closed-loop metrics are saved together.
- [ ] Runtime, memory, and parameter counts are reported.
- [ ] Demo can replay at least one saved rollout.
- [ ] Main results can be regenerated from config files without editing source code.

### Phase 2 expected report result

Phase 2 should be sufficient for a complete final project:

> A frozen-VGGT NBV predictor is evaluated in the same closed-loop framework as PUN and classical/simple policy baselines, establishing whether VGGT representations are useful in a practical sequential acquisition pipeline.

---

## Phase 3 — Joint Multi-View VGGT Prediction

### Goal

Test the central novel hypothesis:

> Does joint geometry-aware VGGT processing of the complete observation history improve NBV prediction over independent per-view processing and aggregation?

This is the most uncertain phase and must not block completion of Phase 2.

### Dataset extension

Create history examples:

```text
I_1:t = {I_1, ..., I_t},  t <= T
```

with one geometric utility target for every candidate view.

Target:

```text
u_t,j = Coverage(P_t ∪ z(v_j), W) - Coverage(P_t, W)
```

### History sampling

The proposal does not prescribe a history-sampling distribution.

Implementation recommendation:

1. Start with random unique-view histories so the pipeline is easy to generate and debug.
2. Sample multiple history lengths up to `T`.
3. Ensure training histories do not contain duplicate anchors unless deliberately testing revisits.
4. Later, if useful, add histories generated by realistic policies to reduce train/evaluation distribution mismatch.

Store the sampling strategy in the dataset metadata.

### Random object rotations

The proposal calls for object-centered utility anchors and random object rotations to reduce canonical-orientation shortcuts.

Implementation rule:

Rotation augmentation must transform **all relevant quantities consistently**:

- rendered/image observation,
- camera/object pose relationship,
- anchor directions,
- visibility/utility labels.

Do not add rotation augmentation until the non-rotated history pipeline is verified end-to-end.

### Joint model

Conceptual flow:

```text
I_1:t
   ↓
joint frozen VGGT forward pass
   ↓
multi-frame geometry-aware patch / camera / register tokens
   ↓
history-aware pooling
   ↓
lightweight predictor
   ↓
48-anchor utility map
```

Backbone remains frozen.

Only the pooling/aggregation and prediction head are trained unless a deliberate ablation says otherwise.

### Independent comparison model

The comparison must receive exactly the same observation histories.

Primary comparison:

```text
same I_1:t
   ↓
independent frozen VGGT processing per image
   ↓
permutation-invariant aggregation
   ↓
capacity-matched prediction head
   ↓
48-anchor utility map
```

### Capacity matching

To make the research conclusion defensible:

- use the same frozen backbone,
- keep output dimensionality identical,
- keep trainable head capacity as close as practical,
- report trainable parameter counts,
- use the same training split,
- use the same history samples,
- use the same optimizer and training budget unless there is a documented reason not to,
- evaluate with exactly the same metrics and closed-loop protocol.

### Phase 3 experiments

Minimum:

1. Independent VGGT aggregation vs. joint VGGT.
2. One-step ranking comparison.
3. Closed-loop coverage comparison.
4. Runtime/memory comparison.

Useful ablations if time permits:

- history length,
- VGGT layer used,
- token type,
- aggregation type,
- ranking-loss weight,
- effect of random rotations.

### Phase 3 definition of done

- [ ] History dataset generation is deterministic.
- [ ] Utility labels are verified against brute-force coverage on small examples.
- [ ] Joint VGGT forward pass supports variable history length.
- [ ] Independent baseline uses the exact same histories.
- [ ] Capacity/training differences are documented.
- [ ] Joint and independent models run through the Phase 2 evaluator unchanged.
- [ ] Main result includes both one-step ranking and closed-loop coverage.
- [ ] Failures/negative results are saved and reportable rather than discarded.

### Phase 3 fallback rule

If joint training is unstable, too slow, or incomplete near the project deadline:

1. Freeze the Phase 2 implementation.
2. Use Phase 2 as the main complete project.
3. Report Phase 3 as an exploratory extension.
4. Include what was attempted, what failed, and any partial quantitative findings.

Do not weaken the Phase 2 result to rescue Phase 3.

---

# 5. Optional extensions

## Extension A — Reconstruction-quality evaluation

The instructor noted that surface coverage is only a proxy for the actual goal: reconstruction quality.

If the core project is complete, evaluate whether predicted NBV sequences produce better reconstructions.

### Preferred first version

Use VGGT-produced 3D points/depth from the selected views and compare reconstructed geometry to the ground-truth mesh.

Possible metric:

- Chamfer distance

Evaluate after fixed numbers of acquired views, e.g.:

```text
5 views
10 views
15 views
...
```

Use the same selected-view sequence generated by each policy.

### Optional expensive version

Run a 3D Gaussian Splatting reconstruction/optimization per policy trajectory and compare reconstruction quality.

Only attempt this if compute and implementation time are clearly available.

## Extension B — Real-world transfer

If time permits, follow the PUN-style real-world evaluation protocol on a dataset such as MipNeRF360.

Questions:

- Does the ShapeNet-trained VGGT predictor transfer at all?
- Is VGGT more robust than PUN under real-image domain shift?
- Does joint history processing help or hurt transfer?

Keep this strictly optional. Do not delay the main ShapeNet evaluation for it.

---

# 6. Current and planned repository structure

## Current Phase 1 tree

The repository currently contains the Phase 1 implementation shown below.
Generated directories such as `.venv/`, `build/`, `*.egg-info/`, `__pycache__/`,
downloaded model weights, and feature-cache payloads are intentionally omitted.

```text
project_root/
│
├── .gitignore
├── README.md
├── project_overview.md
├── colab.ipynb
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   └── experiments/
│       ├── phase1.yaml
│       ├── phase1_probe_tiny.yaml
│       └── phase1_sweep.yaml
│
├── data/
│   ├── NUM/                         # local dataset; not tracked
│   ├── cache/                       # models/features; not tracked
│   └── splits/
│       └── num_v1.json
│
├── src/nbv/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_utils.py
│   ├── reproducibility.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── num_dataset.py
│   │   └── num_splits.py
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── phase1.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cache.py
│   │   ├── dinov2.py
│   │   ├── imagenet_vit.py
│   │   ├── model_cache.py
│   │   ├── preprocessing.py
│   │   ├── raw_rgb.py
│   │   ├── selection.py
│   │   └── vggt.py
│   │
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── anchors.py
│   │   └── anchors_v1.csv
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── huber.py
│   │   └── ranking.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── heads.py
│   │   └── pun.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── phase1.py
│   │   ├── probe.py
│   │   └── pun.py
│   └── visualization/
│       ├── __init__.py
│       ├── phase1_prediction.py
│       └── training_curves.py
│
├── scripts/
│   ├── init_experiment.py
│   ├── inspect_features.py
│   ├── inspect_num_sample.py
│   ├── inspect_pun.py
│   ├── prepare_num_split.py
│   ├── run_phase1.py
│   ├── train_probe.py
│   └── visualize_phase1.py
│
├── tests/
│   ├── test_anchors.py
│   ├── test_config.py
│   ├── test_feature_cache.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_num_dataset.py
│   ├── test_num_splits.py
│   ├── test_num_visualization.py
│   ├── test_phase1_experiment.py
│   ├── test_phase1_prediction_visualization.py
│   ├── test_probe.py
│   ├── test_pun.py
│   └── test_reproducibility.py
│
└── outputs/
    ├── README.md
    └── phase1/backbone_sweep/       # selected seed-0 result artifacts
```

The implemented Phase 1 separation is:

**data → cached inputs/features → predictor or fixed baseline → evaluator → results → visualization**

The repository now also includes the first Phase 2 infrastructure step:
deterministic mesh-surface sampling and per-anchor visibility caches. There is
still no `policies/`, coverage/oracle utility abstraction, history dataset,
closed-loop evaluator, or joint multi-view model; those later Phase 2/3
additions should be introduced only in their corresponding development steps.

## Planned full-project tree

The original planned structure for future Phase 2/3 work remains:

```text
project_root/
│
├── README.md
├── project_overview.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── eval/
│   └── experiments/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   │   ├── features/
│   │   └── visibility/
│   └── splits/
│
├── src/
│   ├── data/
│   │   ├── num_dataset.py
│   │   ├── history_dataset.py
│   │   ├── transforms.py
│   │   └── splits.py
│   │
│   ├── features/
│   │   ├── base.py
│   │   ├── imagenet_vit.py
│   │   ├── dinov2.py
│   │   ├── vggt.py
│   │   └── cache.py
│   │
│   ├── models/
│   │   ├── heads.py
│   │   ├── single_view.py
│   │   ├── independent_multiview.py
│   │   └── joint_multiview.py
│   │
│   ├── losses/
│   │   ├── huber.py
│   │   └── ranking.py
│   │
│   ├── geometry/
│   │   ├── anchors.py
│   │   ├── mesh_sampling.py
│   │   ├── visibility.py
│   │   └── coverage.py
│   │
│   ├── policies/
│   │   ├── base.py
│   │   ├── random_policy.py
│   │   ├── farthest_policy.py
│   │   ├── pun_policy.py
│   │   ├── learned_policy.py
│   │   ├── geometry_policy.py
│   │   └── oracle_policy.py
│   │
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── one_step.py
│   │   ├── closed_loop.py
│   │   ├── profiling.py
│   │   └── result_schema.py
│   │
│   ├── visualization/
│   │   ├── spherical_map.py
│   │   ├── rollout.py
│   │   └── coverage_plot.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── checkpoint.py
│
├── scripts/
│   ├── prepare_num.py
│   ├── cache_features.py
│   ├── precompute_visibility.py
│   ├── build_history_dataset.py
│   ├── train.py
│   ├── evaluate_one_step.py
│   ├── evaluate_closed_loop.py
│   ├── profile_model.py
│   └── make_demo_results.py
│
├── notebooks/
│   ├── phase1_single_view_demo.ipynb
│   └── closed_loop_demo.ipynb
│
├── tests/
│   ├── test_anchors.py
│   ├── test_visibility.py
│   ├── test_coverage.py
│   ├── test_metrics.py
│   ├── test_candidate_masking.py
│   └── test_history_dataset.py
│
└── outputs/
    ├── checkpoints/
    ├── metrics/
    ├── rollouts/
    ├── figures/
    └── tables/
```

The exact future file names can change. The important full-project separation
remains:

**data → frozen features → predictor → policy → evaluator → visualization**

---

# 7. Core software interfaces

Keep the implementation modular enough that every baseline can be plugged into the same evaluator.

### Feature extractor

```python
class FeatureExtractor:
    def extract(self, images, **kwargs):
        # Return frozen features plus masks/metadata required by the predictor.
        ...
```

Implementations:

- `RawRGBExtractor`
- `ImageNetViTExtractor`
- `DINOv2Extractor`
- `VGGTExtractor`

`RawRGBExtractor` is deliberately backbone-free but follows the same cached
fixed-vector contract. `FixedMapHead` adapts the training-set mean map to the
common evaluator; `LightweightProbeHead` is used for every trainable variant.

### Predictor

```python
class UtilityPredictor:
    def predict(self, observations, observation_anchors, **kwargs):
        # Return [B, 48] predicted utility/ranking scores.
        ...
```

### Policy

```python
class NBVPolicy:
    def select_next(self, state):
        # Return one valid unobserved anchor index.
        ...
```

### Evaluator state

Recommended state fields:

```text
object_id
acquired_anchor_ids
observation_images / image references
seen_surface_mask
current_coverage
step_index
candidate_mask
```

The learned model should not receive ground-truth `seen_surface_mask`; that belongs only to the simulator/evaluator unless explicitly running a geometry upper bound.

---

# 8. Data schemas

## Phase 1 sample

Recommended logical schema:

```yaml
object_id: str
image_path: str
source_anchor_id: int
target_map: float[48]
split: train|val|test
metadata: ...
```

## Phase 3 history sample

```yaml
object_id: str
history_image_paths:
  - ...
history_anchor_ids:
  - ...
history_length: int
target_utility: float[48]
valid_candidate_mask: bool[48]
rotation_metadata: ...
split: train|val|test
```

### Dataset invariants

- Anchor ordering must be global and immutable.
- The same anchor index must always refer to the same object-centered direction.
- Train/validation/test separation should be at the object level.
- Already-observed anchors must have a consistent target/mask policy.
- Utilities must be generated from the exact same visibility definition used in closed-loop evaluation.

---

# 9. 48-anchor coordinate system

Treat the anchor map as a first-class project object.

Create one canonical file containing:

```text
anchor_id
azimuth
elevation
unit direction
optional camera transform
```

All of these must use it:

- dataset loading,
- spherical-map visualization,
- utility targets,
- candidate masking,
- farthest-view heuristic,
- PUN alignment,
- VGGT outputs,
- closed-loop evaluator,
- oracle policy.

Add tests for anchor ordering because a silent permutation bug can invalidate every result while producing plausible-looking maps.

---

# 10. Feature caching strategy

Frozen backbone features can make training much cheaper.

### Cache in Phase 1

Cache per-image frozen features for:

- flattened 16×16 raw-RGB vectors,
- ImageNet-ViT,
- DINOv2,
- independent single-image VGGT.

Cache metadata must include:

- backbone/model version,
- input preprocessing,
- image resolution,
- feature layer,
- token type,
- pooling choice if pooling happens before caching.

The current implementation stores one fixed-size tensor per configured
variant, fingerprints the full extraction metadata, writes cache files
atomically, and validates exact metadata on load. Missing variants that share
a backbone are selected from the same forward pass. Backbone downloads use the
separate `data/cache/models/` root. The training-set mean baseline does not
need an image-feature cache; it is derived only from valid cached training
targets.

### Phase 2

Reuse per-image VGGT caches for independent per-view aggregation.

### Phase 3

Do **not** assume independent VGGT features can replace the joint forward pass.

Joint multi-view VGGT features may depend on the complete input set, so the joint model should run the frozen backbone on the history as a group.

---

# 11. Training conventions

Use configuration files rather than hard-coded experiment values.

Every run should save:

```text
resolved config
random seed
git commit hash if available
checkpoint
training log
validation metrics
test metrics
runtime information
```

Recommended experiment identity:

```text
phase / model / feature_variant / seed
```

Example:

```text
phase1/vggt_patch_final/seed_0
phase2/vggt_postpred_mean/seed_0
phase3/vggt_joint/seed_0
```

Run multiple seeds for the final reported comparisons if compute allows.

---

# 12. Loss implementation details

### Huber regression

Input:

```text
predicted scores: [B, 48]
target utility:   [B, 48]
candidate mask:   [B, 48]
```

Only valid candidates should contribute where appropriate.

### Pairwise ranking

The ranking loss should encourage:

```text
u_i > u_j  =>  score_i > score_j
```

Avoid materializing all `48 x 48` pairs if it becomes inefficient; sampling informative pairs is acceptable if documented.

Potential pair selection:

- random valid pairs,
- top-vs-bottom pairs,
- pairs separated by a minimum target margin.

Keep the strategy fixed for controlled experiments.

---

# 13. Metric implementation notes

All metrics should be unit-tested on tiny manually constructed examples.

### Normalized regret tests

Include cases where:

- predicted best equals oracle best,
- predicted best equals worst candidate,
- all utilities are equal,
- some candidates are masked.

### Spearman

Define behavior for ties explicitly.

### NDCG@5

Compute only over valid candidate anchors.

### Coverage AUC

Define the x-axis consistently:

```text
number of acquired views
```

Do not mix "initial views included" vs. "new views selected" between policies.

---

# 14. Closed-loop evaluation protocol

A single deterministic rollout function should support every policy.

Pseudo-code:

```python
state = initialize_object(initial_view_or_views)

for t in range(max_steps):
    valid = get_unobserved_candidates(state)

    scores = policy.score(state)
    scores[~valid] = -inf

    next_anchor = argmax(scores)

    oracle_utilities = compute_ground_truth_utilities(state)
    record_one_step_metrics(scores, oracle_utilities, valid)

    state = acquire_view_and_update_visibility(state, next_anchor)

    record_coverage(state)
```

For the Random policy, use a seeded generator.

For the Oracle policy, select directly from ground-truth candidate gains.

---

# 15. Baseline fairness rules

Before final experiments, use this checklist:

- Same test objects.
- Same initial view(s).
- Same view budget.
- Same 48 anchors.
- Same invalid-view masking.
- Same ground-truth coverage evaluator.
- Same stopping rule.
- Same metric implementation.
- Same history available to each method unless the baseline definition explicitly differs.
- Runtime measured with a documented warm-up/timing procedure.
- Frozen/trained parameters reported clearly.

---

# 16. Experiment matrix

## Phase 1 — configured sweep

| Entry | Image representation | Prediction rule | Current status |
|---|---|---|---|
| `train_mean_map` | none | fixed per-anchor training-target mean | implemented |
| `raw_rgb_16x16_mlp` | flattened 16×16 adaptive-average RGB | shared lightweight head | implemented |
| `imagenet_vit_pooled_patch` | ImageNet-ViT mean-pooled patches | shared lightweight head | implemented |
| `imagenet_vit_cls_token` | ImageNet-ViT classification token | shared lightweight head | implemented |
| `dinov2_pooled_patch` | DINOv2 mean-pooled patches | shared lightweight head | implemented |
| `dinov2_cls_token` | DINOv2 classification token | shared lightweight head | implemented |
| `vggt_pooled_patch` | VGGT mean-pooled patches | shared lightweight head | implemented |
| `vggt_max_pooled_patch` | VGGT max-pooled patches | shared lightweight head | implemented |
| `vggt_camera_token` | VGGT camera token | shared lightweight head | implemented |
| `vggt_pooled_register` | VGGT mean-pooled registers | shared lightweight head | implemented |
| `vggt_camera_patch` | VGGT camera + mean-pooled patches | shared lightweight head | implemented |
| `pun_upnet` | official released PUN ViT-S/16 checkpoint | inference only; official preprocessing + common evaluator | implemented; full sweep pending |

“Implemented” here means the entry is configured and covered by the Phase 1
runner/tests; it does not mean the complete dataset sweep has already been run.

## Phase 2 — minimum policy table

| Policy | Learned? | Uses explicit geometry? | History processing |
|---|---:|---:|---|
| Random | no | no | none |
| Farthest view | no | camera geometry only | acquired poses |
| PUN | yes | official PUN behavior | per-view + post-prediction aggregation |
| VGGT independent | yes | no | per-view + aggregation |
| VGGT-depth geometry | partly | yes | explicit geometry |
| GT-depth geometry | no / upper bound | yes | explicit geometry |
| Oracle | no | ground-truth utility | direct oracle |

## Phase 3 — central comparison

| Model | VGGT use | Multi-view interaction inside backbone? | Aggregation |
|---|---|---:|---|
| Independent VGGT | frozen | no | permutation-invariant |
| Joint VGGT | frozen | yes | joint token pooling |

---

# 17. Recommended development order for VSCode + Codex

Ask Codex to implement small, testable units in this order.

### Step 1 — repository and config skeleton

Create modules, config loading, seed utility, logging, and result directories.

**Do not start with VGGT integration.**

### Step 2 — anchor representation

Implement canonical 48-anchor loading and candidate masking.

Tests:

- unique IDs,
- normalized directions,
- stable ordering,
- distance calculation for farthest-view baseline.

### Step 3 — NUM dataset loader

Load one sample and visualize its target map.

Verify:

- image preprocessing,
- target shape `[48]`,
- object split,
- source-view anchor mapping.

### Step 4 — metric library

Implement and test:

- normalized regret,
- Spearman,
- NDCG@5,
- coverage AUC.

### Step 5 — generic frozen feature interface

Implement one simple backbone first.

Then add:

- ImageNet-ViT,
- DINOv2,
- VGGT.

### Step 6 — lightweight probe head

Train on a tiny subset until it overfits.

Only after overfitting succeeds, run full Phase 1 training.

### Step 7 — Phase 1 experiment runner

One command should sweep the selected backbone/feature configurations and
write the comparison table, best checkpoints, machine-readable training
histories, per-variant train/validation diagnostic curves, validation ranking
curves, and the cross-variant validation-loss overlay. Test metrics are a
single final-checkpoint evaluation, never an epoch-by-epoch curve.

### Step 8 — Phase 1 demo

Load an experiment through the visualization-only CLI and render one prediction
versus ground-truth plot for every complete saved variant:

```bash
python scripts/visualize_phase1.py \
  outputs/phase1/backbone_sweep/seed_0 \
  --split val \
  --index 0
```

The experiment directory is the positional argument. Omit `--variant` to
discover all complete directories under `variants/`; repeat `--variant NAME` to
select a subset. The command writes only self-contained SVGs below the
experiment's `figures/predictions/` directory (or `--output-dir`). The explicit
single-file `--output` path requires exactly one selected variant. It must not
train, alter metrics/checkpoints, or write feature caches. Default all variants
to the same deterministic validation sample. Reuse compatible frozen-feature
caches when present; require explicit `--extract-missing-features` permission
before running a missing pretrained backbone in memory.

### Step 9 — geometry visibility cache

Sample mesh points, compute anchor-view visibility, and validate on one object.

### Step 10 — closed-loop simulator

Start with Oracle and Random only.

If Oracle does not monotonically improve coverage as expected, stop and debug the evaluator before adding learned policies.

### Step 11 — farthest-view baseline

This tests pose/anchor geometry independently of learning.

### Step 12 — integrate PUN

Verify the PUN output anchor ordering and history aggregation.

### Step 13 — independent VGGT policy

Use cached per-view features/predictions and reproduce the Phase 2 closed-loop evaluation.

### Step 14 — complete Phase 2 baselines

Add explicit geometry and GT-depth upper-bound pipelines.

Freeze the Phase 2 results once stable.

### Step 15 — history dataset generator

Generate `I_1:t` examples and utility targets from the visibility cache.

### Step 16 — independent history model

Build the exact comparison model required for Phase 3.

### Step 17 — joint VGGT model

Only now implement joint-history VGGT processing.

### Step 18 — final experiment suite

Run the fixed configs, aggregate seeds, create tables/figures, and archive checkpoints.

### Step 19 — optional extensions

Only after the primary results are complete:

- Chamfer/reconstruction evaluation,
- 3DGS evaluator,
- MipNeRF360 transfer.

---

# 18. Codex task style

When asking Codex to implement components, prefer bounded tasks with a definition of done.

Good example:

```text
Implement src/eval/metrics.py with normalized_regret, spearman_rank,
ndcg_at_k, and coverage_auc. All functions must accept a valid-candidate
mask where relevant. Add unit tests covering ties, masked candidates,
oracle selection, worst selection, and all-equal utilities. Do not modify
other modules unless required for imports.
```

Avoid requests like:

```text
Implement the whole project.
```

For each major component, ask Codex to:

1. inspect the relevant existing files first,
2. make the smallest coherent change,
3. add/update tests,
4. run those tests,
5. report assumptions or unresolved interface issues.

---

# 19. Debugging checkpoints

### Before full Phase 1 training

Confirm:

- one batch loads,
- one frozen feature tensor has expected shape,
- one forward pass gives `[B, 48]`,
- loss is finite,
- a tiny subset can be overfit.

### Before Phase 2 evaluation

Confirm:

- visibility masks look geometrically correct,
- coverage never decreases,
- acquiring an already-seen view is impossible,
- oracle gain equals explicit coverage difference,
- anchor IDs match image/camera directions.

### Before Phase 3 training

Confirm:

- history labels equal brute-force utility on small cases,
- variable-length history padding/masking is correct,
- joint and independent models receive the same observations,
- random rotation augmentation preserves anchor/label consistency.

---

# 20. Highest-risk implementation areas

### Risk 1 — Anchor-coordinate mismatch

Symptom:

- training loss decreases,
- predictions look smooth,
- ranking metrics remain unexpectedly poor.

Mitigation:

- canonical anchor file,
- visual orientation test,
- permutation/unit tests.

### Risk 2 — PUN target semantics differ from direct surface-gain semantics

The original NUM target may encode uncertainty differently from the Phase 2/3 incremental coverage utility.

Mitigation:

- keep Phase 1's official target semantics intact,
- clearly separate Phase 1 target handling from new history-utility labels,
- do not silently reinterpret PUN labels.

### Risk 3 — Data leakage across ShapeNet objects

Mitigation:

- object-level splits,
- split files checked into the project/configuration,
- assertions preventing object overlap.

### Risk 4 — Joint VGGT compute/memory

Mitigation:

- freeze backbone,
- use short histories first,
- mixed precision where safe,
- gradient only through the trainable head/pooling,
- measure memory before full runs.

### Risk 5 — History distribution mismatch

Random training histories may differ from trajectories produced by trained policies.

Mitigation:

- start random for correctness,
- measure performance by history length,
- optionally add policy-generated histories later.

### Risk 6 — Evaluation code divergence between policies

Mitigation:

- one evaluator,
- policy objects only choose/score candidates,
- geometry and metric state remain evaluator-owned.

---

# 21. Reproducibility checklist

For every final table/figure:

- [ ] Exact config saved.
- [ ] Seed saved.
- [ ] Dataset split version saved.
- [ ] Backbone version saved.
- [x] PUN version/commit and checkpoint SHA-256 documented.
- [ ] Feature/token layer documented.
- [ ] Input resolution documented.
- [ ] Loss weights documented.
- [ ] Candidate masking behavior documented.
- [ ] View budget documented.
- [ ] Number of surface samples documented.
- [ ] Visibility rule/tolerance documented.
- [ ] Runtime hardware documented.
- [ ] Number of evaluation objects documented.

---

# 22. Result files

Use machine-readable output so tables can be regenerated automatically. The
implemented Phase 1 runner uses this structure:

```text
outputs/<phase>/<experiment>/seed_<n>/
├── config.yaml
├── metadata.json
├── run.log
├── metrics/
│   ├── comparison.csv
│   ├── comparison.json
│   └── comparison.md
├── figures/
│   ├── training/
│   │   ├── <variant>_losses.svg
│   │   ├── <variant>_validation_metrics.svg
│   │   └── validation_loss_comparison.svg
│   └── predictions/
│       └── <variant>_<split>_<sample_id>.svg
└── variants/<variant>/
    ├── best.pt
    ├── summary.json
    ├── test_per_sample.csv
    └── training_history.json
```

The tiny-overfit command instead stores `checkpoints/best.pt` and
`metrics/tiny_overfit.json` in the same run root. Later closed-loop work may
add `rollouts/` and per-step files without changing the Phase 1 contract.

For learned variants, `training_history.json` starts at epoch zero and then has
one row per completed epoch. Its stable fields are:

```json
{
  "epoch": 1,
  "learning_rate": 0.001,
  "optimization_train_loss": null,
  "train_loss": null,
  "train_huber_loss": null,
  "train_ranking_loss": null,
  "validation_loss": null,
  "validation_huber_loss": null,
  "validation_ranking_loss": null,
  "validation_normalized_regret_mean": null,
  "validation_spearman_mean": null,
  "validation_ndcg_at_5_mean": null
}
```

`train_loss` is measured in evaluation mode after the epoch; it is not the
same quantity as `optimization_train_loss`. The analytical mean-map baseline
stores an empty history. The pretrained PUN baseline also stores an empty
history because it performs official-checkpoint inference only; its `best.pt`
is a lightweight descriptor containing the release, cache path, preprocessing,
and checksum rather than a duplicate of the 83 MB state dict. No test metric
appears in a training history: the test split is evaluated only after restoring
the validation-selected checkpoint or loading the pinned official release.

The implemented per-variant `summary.json` shape is:

```json
{
  "variant": "vggt_pooled_patch",
  "backbone": "vggt",
  "feature": "pooled_patch",
  "feature_components": ["pooled_patch"],
  "input_dim": 1024,
  "trainable_parameters": 139440,
  "best_epoch": null,
  "epochs_completed": null,
  "best_validation_loss": null,
  "target_name": "PSNR",
  "target_direction": "lower",
  "train_samples": null,
  "validation": {
    "num_samples": null,
    "loss": null,
    "huber_loss": null,
    "ranking_loss": null,
    "normalized_regret_mean": null,
    "spearman_mean": null,
    "ndcg_at_5_mean": null
  },
  "test": {
    "num_samples": null,
    "loss": null,
    "huber_loss": null,
    "ranking_loss": null,
    "normalized_regret_mean": null,
    "spearman_mean": null,
    "ndcg_at_5_mean": null
  }
}
```

Runtime, memory, coverage, and per-step fields belong to the future Phase 2
result schema and are not currently emitted by the Phase 1 runner.

For future closed-loop phases, retain the planned machine-readable additions:

```text
outputs/
  metrics/
    <experiment_id>/
      config.yaml
      summary.json
      per_object.csv
      per_step.csv
  rollouts/
    <experiment_id>/
      <object_id>.npz
  checkpoints/
    <experiment_id>/
      best.pt
```

The future common summary/result schema should retain these fields in addition
to the Phase 1 fields above:

```json
{
  "normalized_regret_mean": null,
  "spearman_mean": null,
  "ndcg_at_5_mean": null,
  "coverage_auc_mean": null,
  "final_coverage_mean": null,
  "median_inference_ms": null,
  "peak_memory_mb": null,
  "trainable_parameters": null
}
```

---

# 23. Figures/tables to plan for the final report

### Phase 1

- Table: ViT vs. DINOv2 vs. VGGT vs. PUN.
- Qualitative spherical target/prediction visualization.
- Optional ablation plot for VGGT token/layer choice.

### Phase 2

- Main coverage-vs-view curve for all policies.
- Table with regret, Spearman, NDCG@5, coverage AUC, runtime, memory, parameters.
- Qualitative rollout showing chosen NBVs and accumulated surface.

### Phase 3

- Joint vs. independent coverage curve.
- Joint vs. independent ranking metrics.
- Runtime/memory comparison.
- History-length ablation if available.

### Optional reconstruction extension

- Chamfer distance vs. number of acquired views.
- Qualitative predicted reconstruction vs. ground-truth mesh.

---

# 24. Decision gates

### Gate A — after Phase 1

Proceed only when the single-image comparison is reproducible and PUN is represented fairly.

If VGGT does not outperform generic features:

- do not hide the result,
- continue to Phase 2,
- test whether sequential aggregation still provides value.

### Gate B — after Phase 2

Treat the project as **complete** when the full independent-VGGT closed-loop evaluation is stable.

At this point:

- freeze a reportable result set,
- tag/save configs and checkpoints,
- only then spend the remaining time on joint multi-view processing.

### Gate C — during Phase 3

If joint processing threatens the final deliverable:

- stop adding complexity,
- preserve Phase 2 as the main result,
- report Phase 3 honestly as partial/negative/exploratory.

---

# 25. Final project narrative

The final report should tell a progressive story rather than presenting Phase 3 as the only success condition.

### Story 1 — Representation probe

Can frozen VGGT features predict NBV-relevant quantities from a single image better than generic visual representations?

### Story 2 — Practical sequential policy

Do those frozen features remain useful when placed into a complete closed-loop NBV pipeline and compared with PUN, heuristics, and geometry-based policies?

### Story 3 — Novel multi-view test

Does VGGT's joint multi-view reasoning provide additional value beyond independent per-view processing and aggregation?

This structure ensures that every completed phase produces a self-contained research conclusion.

---

# 26. Immediate implementation backlog

Start here.

### Milestone 1 — infrastructure

- [x] Create the Phase 1 repository/module structure.
- [x] Add config loading and typed command-line overrides.
- [x] Add deterministic seed utility.
- [x] Add the Phase 1 experiment/artifact schema.
- [x] Add canonical 48-anchor representation.

### Milestone 2 — Phase 1 data + metrics

- [x] Implement original NUM sample loading and validation.
- [x] Reproduce/verify target-map orientation and source-relative anchor zero.
- [x] Implement regret, Spearman, NDCG@5, and coverage AUC.
- [x] Add metric unit tests.
- [x] Validate the loader over the complete local NUM dataset.

### Milestone 3 — feature probes

- [x] Add ImageNet-ViT extractor.
- [x] Add DINOv2 extractor.
- [x] Add single-image VGGT extractor.
- [x] Add raw-RGB and training-mean controls.
- [x] Add model-download and feature-vector caching.
- [x] Add shared lightweight predictor head.
- [x] Verify tiny-set overfitting with ImageNet-ViT.
- [x] Record real-checkpoint DINOv2 and VGGT runs.

### Milestone 4 — Phase 1 result

- [x] Implement the cache-first one-command sweep runner.
- [x] Implement common validation/test metrics and comparison-table writers.
- [x] Run backbone comparison.
- [x] Integrate official pretrained PUN baseline.
- [x] Produce the populated seed-0 comparison table.
- [x] Build saved-checkpoint prediction-versus-target visualization.
- [ ] Freeze Phase 1 checkpoint.

### Milestone 5 — closed-loop geometry

- [ ] Mesh surface sampling.
- [ ] Per-anchor visibility cache.
- [ ] Coverage state.
- [ ] Oracle utility.
- [ ] Closed-loop simulator.
- [ ] Random + oracle sanity tests.

### Milestone 6 — complete Phase 2

- [ ] Farthest-view policy.
- [ ] PUN history policy.
- [ ] Independent VGGT aggregation policy.
- [ ] Full metric suite.
- [ ] Coverage curves.
- [ ] Runtime/memory profiling.
- [ ] Closed-loop demo.
- [ ] Freeze Phase 2 checkpoint.

### Milestone 7 — Phase 3 dataset

- [ ] History sampling.
- [ ] Incremental utility labels.
- [ ] Variable-length batching.
- [ ] Rotation augmentation after non-rotated pipeline passes tests.

### Milestone 8 — joint experiment

- [ ] Capacity-matched independent model.
- [ ] Joint frozen-VGGT model.
- [ ] One-step evaluation.
- [ ] Closed-loop evaluation.
- [ ] History-length/token ablations if useful.
- [ ] Freeze Phase 3 result or document failure mode.

### Milestone 9 — stretch only

- [ ] Chamfer/reconstruction metric.
- [ ] Fixed-budget reconstruction comparison.
- [ ] 3DGS evaluator if feasible.
- [ ] MipNeRF360 transfer if feasible.

---

# 27. Non-goals until the core project is finished

Avoid spending early time on:

- custom UI/web applications,
- large hyperparameter searches,
- fine-tuning VGGT,
- replacing the 48-anchor action space,
- sophisticated learned policy optimization,
- 3DGS reconstruction,
- real-world transfer,
- extra datasets,
- major architecture changes to VGGT.

The project is strongest when the comparison is controlled and the evaluation is complete.

---

# 28. One-sentence implementation priority

**First prove the feature probe, then build the complete independent-view closed-loop pipeline, and only then attempt joint multi-view VGGT processing.**
