# Project Overview — Direct Next-Best-View Prediction from Frozen VGGT Geometry Features

## 1. Purpose of this document

This file is the implementation reference for the project. It integrates the revised PUN-compatible Phase 2 and direct surface-gain Phase 3 plan while retaining the completed Phase 1 study. Each phase ends with a complete, usable checkpoint.

The revision separates **single-image NUM supervision**, **aggregated policy scores**, and **ground-truth geometric evaluation**. Phase 2 reuses the Phase 1 image-target pairs; a new supervised history dataset is introduced only in Phase 3. Per the project decision, retain the current rasterized mesh-face visibility implementation and its explicit `Vis`/`VisA` metrics. The attachment's sampled-surface-point proposal does not apply.

The core research goal remains:

> Determine whether frozen VGGT predictions of the original single-image NUM target transfer to effective sequential surface coverage, then test whether joint multi-view processing improves direct history-dependent surface-gain prediction over a capacity-matched independent model.

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
- **H3 — End-to-end NBV usefulness:** Independently predicting the original NUM/PUN target with VGGT and combining maps using PUN-style aggregation produces a competitive closed-loop coverage policy. Better proxy prediction may or may not improve geometric coverage.
- **H4 — Joint multi-view advantage:** When both models train on identical history-dependent surface-gain targets, joint VGGT processing improves prediction and ranking over capacity-matched independent VGGT feature aggregation.

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

- `L_huber` regresses the original NUM/PUN target values, preserving their dataset semantics; these are not direct surface-gain labels.
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
- [x] PUN baseline results are available in the same comparison table.
- [x] Official pretrained PUN is configured as the final inference-only sweep entry.
- [x] At least one qualitative visualization is reproducible from a saved checkpoint.
- [x] Experiment config, seed, checkpoint, and metrics are saved together.
- [x] A single command can reproduce the main Phase 1 comparison.

### Current Phase 1 repository status — complete

Phase 1 was marked complete on 2026-09-07. The frozen primary result is the
complete seed-1 sweep under `outputs/phase1/backbone_sweep/seed_1`, with all 12
configured entries evaluated through the common validation/test pipeline.

Repository audit as of 2026-09-07:

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
| One-command sweep | Implemented; complete seed-1 run frozen | The primary result contains the mean-map baseline, ten locally trained variants, and official pretrained PUN as the final row (12 rows total). Seed 0 remains available as an earlier supporting run. |
| Runtime/memory profiling | Deferred to Phase 2 | Trainable parameter counts are reported. A common inference-time and peak-memory protocol remains useful for the closed-loop system but does not block the frozen Phase 1 feature-probe result. |
| PUN comparison | Complete | The official released PSNR UPNet checkpoint was checksum-verified, evaluated last with official timm preprocessing, and recorded with both official unmasked MSE and common masked metrics. The full row covers all 5,232 validation and 14,400 test samples. |
| Prediction demo | Implemented and tested | `visualize_phase1.py <experiment>` discovers every complete saved variant by default and writes one self-contained prediction-versus-target SVG per variant. Repeatable `--variant` filters select a subset. The checked-in raw-RGB validation example includes shared-scale target/prediction maps, absolute error, top candidates, regret, Spearman, NDCG@5, and MAE. |
| Phase 2 visibility and simulator | Implemented and subset-validated; full-split work pending | Canonical face rasterization, `Vis`/`VisA` metrics, schema-2 caches, and the Random/Oracle simulator are implemented. Prepared PLY meshes use bounding-box centering supported by six independent validation objects. Three local caches and their replayable rollouts use this convention. Full-split precomputation, learned policy adapters, the history dataset, and joint VGGT remain pending. |

The Phase 1 audit recorded 104 passing tests. This historical count does not
validate the planned Phase 2/3 changes or replace real-data, real-checkpoint,
and full-sweep validation.

### Phase 1 expected report result

A controlled statement such as:

> Frozen VGGT features are / are not more predictive of the original single-image NUM target than generic frozen image features under the same predictor capacity. Whether this translates into increased surface coverage is tested in Phase 2.

Even if the later phases fail, this can already be a meaningful result.

---

## Phase 2 — PUN-Compatible Independent Per-View VGGT + Complete Closed-Loop NBV Pipeline

### Goal

Build the complete closed-loop next-best-view system using frozen VGGT **without joint history processing**, while keeping the learned prediction problem as close as possible to the original PUN/NUM formulation.

The main Phase 2 question is:

> If frozen VGGT features are trained to predict the same single-image 48-anchor target used by PUN/NUM, does independently applying that predictor to every acquired observation and aggregating the resulting maps produce an effective closed-loop NBV policy?

Phase 2 is the main project fallback and should be treated as a final-quality deliverable.

The important distinction is:

* **training target:** original PUN/NUM single-image target,
* **policy score:** aggregated PUN/NUM-style prediction,
* **final evaluation:** true incremental surface coverage.

Phase 2 does **not** require a new supervised history dataset.

---

### Phase 2 data and training target

Reuse the original PUN Neural Uncertainty Map (NUM) ShapeNet image-target pairs from Phase 1.

For every single image:

```text
RGB image
    ↓
original NUM target
    ↓
48-anchor target map
```

The 48 target values retain exactly the semantics of the original NUM/PUN dataset.

Do not reinterpret these values as direct surface-coverage gains.

The same object-disjoint train/validation/test split and canonical 48-anchor ordering used in Phase 1 should be retained.

Reuse the validation-selected Phase 1 VGGT feature variant and saved head as
the default Phase 2 predictor. Pin its experiment path, feature metadata,
checkpoint checksum, target name, and target direction in the Phase 2 config.
Change the variant only when validation results justify it; do not choose it
using Phase 2 test coverage.

---

### VGGT per-view predictor

Each observed image is processed independently:

```text
RGB image
    ↓
frozen VGGT
    ↓
selected frozen feature representation
    ↓
lightweight predictor
    ↓
48-value NUM/PUN-style prediction map
```

The predictor is trained exactly as a single-image model.

VGGT never receives multiple acquired images in the same forward pass during Phase 2.

Because processing is independent, frozen VGGT features and optionally complete per-image prediction maps can be cached.

---

### Official PUN baseline

Use the official released PUN/UPNet checkpoint as the primary published-method baseline.

Do **not** retrain official PUN to predict direct surface gain.

PUN should preserve:

* its released checkpoint,
* its original preprocessing,
* its original per-image output semantics,
* its official history-combination behavior as closely as possible.

The purpose of PUN in Phase 2 is to answer:

> How does the published PUN policy perform under the project's common closed-loop surface-coverage evaluator?

Any mismatch between the original PUN/NUM target semantics and direct incremental surface coverage must be reported explicitly.

A separately retrained PUN-like architecture using direct surface-gain supervision may be added later as an ablation, but it must not be presented as the official PUN baseline.

---

### Independent history processing

At time `t`, each acquired image produces its own 48-anchor prediction:

```text
I_1 → frozen VGGT → head → map_1 ┐
I_2 → frozen VGGT → head → map_2 ├→ history aggregation → final 48 scores
...                              │
I_t → frozen VGGT → head → map_t ┘
```

The primary aggregation rule should match the official PUN post-prediction combination rule whenever that rule can be reproduced reliably.

Possible secondary ablations:

* mean aggregation,
* max aggregation,
* other simple deterministic PUN-compatible aggregation rules.

The primary Phase 2 experiment should avoid adding a large learned history
module because that would change the research question and make the comparison
with PUN less direct. Feature-level history aggregation trained on surface gain
belongs to the Phase 3 independent control.

Before combining maps, reproduce the official conversion from each source-view
map frame into the common rollout anchor frame. The Phase 1 source-relative
anchor-zero convention is not a global already-acquired mask. Pin any rotation,
resampling, normalization, and combination behavior after checking the official
implementation; do not assume all frame conversions are exact permutations.

Preserve raw target predictions separately from final policy scores. The
adapter returns higher-is-better scores while respecting the original target
direction and the official order of operations. The current PSNR target is
lower-is-more-uncertain, so an unconditional argmax of raw PSNR predictions
would reverse its intended ranking. Mean/max ablations must specify whether
they aggregate raw maps or oriented scores.

---

### Interpretation of the Phase 2 scores

The final 48 values after history aggregation are **policy scores**.

They should not automatically be described as predicted surface gain.

Instead:

```text
aggregated PUN/NUM-style scores
            ↓
rank currently valid candidate views
            ↓
select next view
```

The policy is successful if those rankings cause the system to acquire views that reveal useful new geometry.

This separates:

```text
what the model is trained to predict
            ≠
what the evaluator ultimately measures
```

---

### Surface-coverage evaluator

Although Phase 2 models are trained on PUN/NUM targets, all policies are evaluated using the same ground-truth geometric criterion.

For observation history `P_t` and candidate view `v_j`:

```text
u_t,j =
    Coverage(P_t ∪ z(v_j), W)
    -
    Coverage(P_t, W)
```

where:

* `P_t` is the surface already visible from acquired views,
* `z(v_j)` is the surface visible from candidate anchor `j`,
* `W` is the complete ground-truth mesh surface, represented by its faces under the configured `Vis` or `VisA` coverage definition.

These ground-truth gains are evaluator quantities.

They are **not Phase 2 VGGT training labels**.

---

### Surface visibility cache

Retain the existing per-object, per-anchor rasterized mesh-face visibility
cache. A face is visible when it survives occlusion in the triangle-ID z-buffer
for at least one pixel. Keep both existing coverage definitions explicit:

```text
Vis  = number of visible faces / total number of faces
VisA = total area of visible faces / total mesh area
```

Select `vis` or `vis_a` in config before evaluation; the current configuration
uses `vis_a`. The selected definition must be identical for all policies,
oracle gains, and subsequent Phase 3 training labels.

```text
seen_face_mask_t = OR of face_visibility masks for acquired views

weight[f] = 1 / N_faces                         # Vis
         or face_area[f] / total_mesh_area       # VisA

candidate_gain(j) = sum_f(
    weight[f] * (face_visibility[j, f] AND NOT seen_face_mask_t[f])
)

coverage_t = sum_f(weight[f] * seen_face_mask_t[f])
```

The same cache supports closed-loop coverage, oracle selection, one-step
regret, ranking against true geometric gain, and later Phase 3 history labels.
The same visibility definition must be reused everywhere.

#### Existing implementation and remaining validation

Reuse `src/nbv/geometry/visibility.py`, `src/nbv/geometry/coverage.py`,
`src/nbv/data/visibility_cache.py`, `scripts/precompute_visibility.py`, and
`configs/experiments/phase2_visibility.yaml`. Preserve the current face-mask
representation, schema-2 metadata validation, and `Vis`/`VisA` helpers.

Keep mesh checksum/scale, anchor ordering, camera geometry, triangle-ID render
resolution, clipping, culling, and visibility definition in cache provenance.
Do not change the geometry definition between experiments or introduce a
surface-point sampler. The remaining Step 9 work is real-object validation
and complete split precomputation using this existing implementation.

Coverage is normalized by the full face count or full mesh area, including
faces unreachable from the candidate set; maximum achievable coverage can be
below 1. Report the chosen face-based definition explicitly when describing
true geometric gain.

---

### Closed-loop Phase 2 evaluator

For each test object:

1. Initialize the rollout from the fixed starting view or views.
2. Gather all currently acquired RGB observations.
3. Independently run the VGGT/PUN predictor on each observation.
4. Aggregate the per-view prediction maps.
5. Mask already acquired and invalid candidate anchors.
6. Select the highest-scoring valid candidate.
7. Compute the true surface gain of that selection.
8. Acquire the selected view.
9. Update the visible-surface mask.
10. Record coverage and per-step metrics.
11. Repeat until the view budget is exhausted.

Every policy must use the same evaluator.

---

### Required Phase 2 policies

Minimum core comparison:

* Random
* Farthest-view heuristic
* Official PUN
* Independent per-view VGGT + PUN-style aggregation
* Oracle one-step policy

Add after the core comparison is stable:

* VGGT-predicted-depth geometry baseline
* ground-truth-depth geometry upper bound
* mean/max history-aggregation ablations

---

### Phase 2 metrics

#### Closed-loop metrics

These are the primary Phase 2 outcome metrics:

* surface coverage versus number of acquired views,
* coverage AUC,
* final coverage at the maximum view budget,
* median inference time,
* peak memory,
* trainable parameter count.

These metrics make comparison possible even though different policies may internally assign different meanings to their 48 scores.

#### One-step policy-quality metrics

For each rollout state, compute the true geometric gain of every remaining candidate.

Then compare the policy's ranking with those gains using:

* normalized utility regret of the selected action,
* Spearman rank correlation,
* NDCG@5.

These metrics measure how well the policy scores align with actual geometric usefulness.

They do **not** imply that PUN or Phase 2 VGGT was trained directly on these surface-gain targets.

#### Original target metrics

For the learned single-image VGGT predictor, retain the original Phase 1 NUM-target metrics separately.

This gives two complementary results:

```text
single-image prediction quality
    → how well VGGT predicts the original NUM target

closed-loop geometric quality
    → whether those predictions lead to useful NBV decisions
```

---

### Phase 2 demo

A saved rollout should visualize:

* acquired RGB views,
* individual per-view prediction maps if useful,
* aggregated current 48-anchor score map,
* selected NBV,
* ground-truth candidate surface gains for diagnostic comparison,
* accumulated visible surface,
* coverage curve,
* optional side-by-side PUN/VGGT rollout.

The demo must use saved checkpoints and the same evaluator as the quantitative experiment.

---

### Phase 2 definition of done

* [ ] Surface visibility cache is reproducible.
* [ ] Oracle surface gain matches explicit coverage differences.
* [ ] Closed-loop simulator is deterministic.
* [ ] Official PUN runs without retraining.
* [ ] PUN's original history aggregation is reproduced or any deviation is documented.
* [ ] VGGT predicts the original NUM/PUN target from each image independently.
* [ ] Per-image VGGT features/predictions can be cached.
* [ ] VGGT and PUN receive every image in the supplied history under the same observation protocol; paired fixed-history checks use identical histories. Each closed-loop policy then follows its own selected trajectory from the same initial views.
* [ ] Already-seen candidates are masked consistently.
* [ ] Random, Farthest, PUN, VGGT, and Oracle use one evaluator.
* [ ] True surface gain is used only as evaluation/oracle information and is never exposed to learned Phase 2 policies.
* [ ] Coverage curves are generated for the complete test split.
* [ ] Per-step geometric ranking metrics and closed-loop metrics are stored together.
* [ ] Runtime, memory, and parameter counts are reported.
* [ ] At least one saved rollout can be replayed.
* [ ] Main results can be reproduced from fixed configs.

---

### Phase 2 expected result

Phase 2 should answer:

> Does replacing the visual representation used for single-image PUN/NUM prediction with frozen VGGT features lead to an effective sequential NBV policy when the independently predicted maps are combined across observations?

A useful result could therefore be:

> Frozen VGGT predicts the original NUM target more accurately than generic visual representations and, when used with PUN-style history aggregation, produces improved / comparable / worse true surface coverage in closed-loop acquisition.

Importantly, this phase tests whether improvement on the proxy prediction task translates into improvement on the real geometric objective.

A complete Phase 2 remains sufficient for the final project.

---

## Phase 3 — Direct History-Dependent Surface-Gain Prediction

### Goal

Phase 3 changes the learning problem.

Instead of independently predicting the original single-image PUN/NUM target and combining those predictions, train models directly to answer:

> Given everything observed so far, how much previously unseen surface would each remaining candidate view reveal?

This phase tests the central novel hypothesis:

> Does joint geometry-aware VGGT processing of the complete observation history improve direct surface-gain prediction over capacity-matched independent VGGT processing of exactly the same history?

Phase 3 is the highest-risk extension and must not block completion of Phase 2.

---

### Phase 3 history dataset

Create a new supervised dataset of observation histories:

```text
I_1:t = {I_1, ..., I_t}
```

For every history, generate the 48-value target:

```text
u_t,j =
    Coverage(P_t ∪ z(v_j), W)
    -
    Coverage(P_t, W)
```

Now, unlike Phase 2, these values are the actual **training targets**.

Each value explicitly means:

> How much additional surface would candidate `j` reveal given the current observation history?

The same face-visibility cache and configured `vis` or `vis_a` target used for
Phase 2 evaluation generate these labels. Name the field `target_surface_gain`
and keep it distinct from the original NUM `target_map`. No Phase 2 training
run consumes these generated labels.

A logical sample contains:

```yaml
object_id: str
history_image_paths:
  - ...
history_anchor_ids:
  - ...
history_length: int
target_surface_gain: float[48]
valid_candidate_mask: bool[48]
rotation_metadata: ...
split: train|val|test
```

The dataset should remain object-disjoint.

---

### History sampling

Start with random unique-view histories.

Sample multiple history lengths up to the chosen maximum training horizon.

Do not include duplicate anchors unless revisits are being studied deliberately.

Record the sampling strategy in dataset metadata.

If time permits, later add policy-generated histories to test or reduce
train/evaluation distribution mismatch. Keep the fixed object split and record
the policy/checkpoint provenance without using test objects for training.

### Random object rotations

Retain rotation augmentation as an optional Phase 3 ablation after the
non-rotated pipeline is verified. Transform the rendered RGB observations,
camera/object poses, anchor directions, and visibility labels consistently;
rotating only labels or cached vectors is invalid. Record the transformation
in sample/cache metadata and give both models identical augmented histories.

---

### Phase 3 independent control model

The first Phase 3 model processes exactly the same history as the joint model but keeps VGGT independent across images:

```text
I_1 → frozen VGGT → f_1 ┐
I_2 → frozen VGGT → f_2 ├→ permutation-invariant aggregation
...                      │
I_t → frozen VGGT → f_t ┘
                               ↓
                       lightweight predictor
                               ↓
                    48 direct surface gains
```

Unlike Phase 2, this model is **trained on history-dependent surface-gain targets**.

Therefore it is not simply the Phase 2 policy reused unchanged.

It is the controlled independent-processing baseline required for the Phase 3 scientific comparison.

Frozen per-image VGGT features may still be cached for this model.

---

### Phase 3 joint model

The joint model receives the complete observation history in one VGGT forward pass:

```text
{I_1, I_2, ..., I_t}
          ↓
 joint frozen VGGT
          ↓
multi-view geometry-aware tokens
          ↓
 history-aware pooling
          ↓
 lightweight predictor
          ↓
48 direct surface gains
```

Information from different observations can interact inside VGGT before the prediction head.

The VGGT backbone remains frozen.

---

### Controlled Phase 3 comparison

The central comparison is:

```text
Independent:
same history
    ↓
VGGT separately per image
    ↓
aggregate
    ↓
predict surface gain


Joint:
same history
    ↓
one joint VGGT forward
    ↓
pool
    ↓
predict surface gain
```

Everything except the location of multi-view interaction should be matched as closely as possible.

Use:

* identical history examples,
* identical surface-gain targets,
* identical object splits,
* identical valid-candidate masks,
* the same frozen VGGT checkpoint,
* comparable trainable head capacity,
* the same optimizer,
* the same training budget,
* the same loss,
* the same evaluator,
* the same rollout starting conditions,
* the same view budget.

This makes the central Phase 3 conclusion interpretable:

> A systematic difference supports an effect of allowing observations to interact inside VGGT rather than only after independent feature extraction, subject to the documented capacity, pooling, and compute differences.

---

### Relationship between PUN and Phase 3

Official PUN remains an external closed-loop baseline.

Do not retrain the official PUN checkpoint as part of the main Phase 3 experiment.

The main controlled scientific comparison is:

```text
independent VGGT trained on surface gain
                vs.
joint VGGT trained on surface gain
```

PUN can still appear on the same coverage curves because all policies are evaluated using the same ground-truth coverage metric.

An optional additional ablation may retrain a PUN-like architecture on the Phase 3 history/surface-gain dataset.

If included, label it explicitly as something such as:

```text
PUN-architecture surface-gain control
```

and keep it separate from the official pretrained PUN result.

---

### Phase 3 evaluation

Because the models now directly predict the evaluator's geometric utility, one-step metrics have a particularly direct interpretation:

* Huber/regression error on surface gain,
* normalized utility regret,
* Spearman correlation,
* NDCG@5.

Closed-loop metrics remain:

* coverage versus acquired views,
* coverage AUC,
* final coverage,
* runtime,
* peak memory,
* trainable parameters.

The Phase 2 evaluator itself should not need to change.

Only the policy's method for generating the 48 candidate scores changes.

---

### Phase 3 definition of done

* [ ] History dataset generation is deterministic.
* [ ] Surface-gain labels match brute-force coverage calculations.
* [ ] Training and evaluation use the same visibility definition.
* [ ] Multiple history lengths are supported.
* [ ] Independent and joint models receive exactly the same histories.
* [ ] Independent VGGT processing never allows cross-view backbone interaction.
* [ ] Joint VGGT genuinely processes multiple views together.
* [ ] Trainable capacities are documented and approximately matched.
* [ ] Both models use the same direct surface-gain supervision.
* [ ] Both run through the Phase 2 evaluator unchanged.
* [ ] One-step and closed-loop results are reported.
* [ ] Runtime and memory differences are reported.
* [ ] Negative or inconclusive joint-processing results remain reportable.

### Phase 3 experiments and fallback rule

Run the independent-versus-joint one-step, closed-loop, and runtime/memory
comparison first. Add history-length, feature-layer/token, aggregation,
ranking-loss-weight, or rotation ablations only if time permits. Select model
settings on validation data and retain all final test outcomes.

If joint training is unstable, too slow, or incomplete near the deadline,
keep the frozen Phase 2 result as the complete project and report Phase 3 as an
exploratory extension with attempted configurations, failure modes, and any
partial quantitative results. Do not delay or weaken Phase 2 to rescue it.

---

## Target semantics across the three phases

The project should explicitly distinguish the three cases:

| Phase       | Model input                       | Training target                                | History treatment                  | Final geometric evaluation                      |
| ----------- | --------------------------------- | ---------------------------------------------- | ---------------------------------- | ----------------------------------------------- |
| **Phase 1** | one image                         | original PUN/NUM 48-value target               | none                               | primarily original target metrics               |
| **Phase 2** | each observed image independently | original PUN/NUM 48-value target               | aggregate per-view prediction maps | true surface gain and closed-loop coverage      |
| **Phase 3** | complete observation history      | direct history-dependent 48-value surface gain | independent vs. joint VGGT         | same true surface gain and closed-loop coverage |

This creates a progressive research story.

### Phase 1 — representation

> Can VGGT predict the original single-image NBV-related target better than generic frozen representations?

### Phase 2 — practical sequential transfer

> If VGGT predicts that proxy target well, does the improvement translate into better actual surface coverage when the predictions are used sequentially?

### Phase 3 — direct geometric reasoning

> If we train directly on history-dependent surface gain, does VGGT benefit from processing all observations jointly rather than independently?

The three phases therefore change one major idea at a time:

```text
Phase 1:
single image → proxy target

Phase 2:
multiple independently processed images
→ proxy predictions
→ aggregation
→ evaluate actual geometry

Phase 3:
multi-view history
→ direct geometric target
→ independent vs. joint processing
```

This preserves PUN as a meaningful published baseline, keeps Phase 2 comparatively low-risk, and gives Phase 3 a clean capacity-matched experiment for the project's main multi-view hypothesis.

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

## Current implementation tree

The repository contains the completed Phase 1 implementation and initial
Phase 2 face-visibility infrastructure shown below.
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
│       ├── phase1_sweep.yaml
│       └── phase2_visibility.yaml
│
├── data/
│   ├── NUM/                         # local dataset; not tracked
│   ├── ShapeNetCore.v2/              # prepared NUM mesh subset; not tracked
│   ├── cache/                       # models/features/visibility; not tracked
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
│   │   ├── num_splits.py
│   │   └── visibility_cache.py
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
│   │   ├── anchors_v1.csv
│   │   ├── mesh.py
│   │   ├── visibility.py
│   │   └── coverage.py
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
│       ├── training_curves.py
│       └── visibility.py
│
├── scripts/
│   ├── init_experiment.py
│   ├── inspect_features.py
│   ├── inspect_num_sample.py
│   ├── inspect_pun.py
│   ├── prepare_num_split.py
│   ├── precompute_visibility.py
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
│   ├── test_reproducibility.py
│   └── test_visibility.py
│
└── outputs/
    ├── README.md
    └── phase1/backbone_sweep/       # frozen seed-1 results; seed 0 supporting
```

The implemented Phase 1 separation is:

**data → cached inputs/features → predictor or fixed baseline → evaluator → results → visualization**

The Phase 2 geometry infrastructure already provides deterministic PUN-style
per-anchor mesh-face visibility caches, coverage, and candidate marginal gains
for `Vis` and `VisA`. There is still no `policies/`, closed-loop evaluator,
supervised history dataset, or joint multi-view model. Extend the existing
`src/nbv` package without reorganizing completed Phase 1 modules.

## Planned additions for Phases 2 and 3

These are proposed module/config names, not implemented commands:

```text
project_root/
├── configs/experiments/
│   ├── phase2_closed_loop.yaml           # five core policies + evaluator
│   ├── phase2_aggregation_ablation.yaml  # optional mean/max rules
│   ├── phase3_histories.yaml             # direct surface-gain dataset
│   └── phase3_controlled.yaml            # matched independent/joint runs
├── data/
│   ├── cache/predictions/                # optional per-image NUM maps
│   └── processed/histories/              # Phase 3 only
├── src/nbv/
│   ├── data/
│   │   ├── observation_store.py         # acquired RGB/anchor lookup
│   │   ├── prediction_cache.py
│   │   └── history_dataset.py           # Phase 3 labels and batching
│   ├── models/
│   │   ├── independent_multiview.py     # Phase 3 feature aggregation + head
│   │   └── joint_multiview.py           # Phase 3 joint history + head
│   ├── features/vggt_joint.py           # separate from single-image VGGT
│   ├── policies/
│   │   ├── base.py
│   │   ├── aggregation.py              # map alignment + PUN rule
│   │   ├── random_policy.py
│   │   ├── farthest_policy.py
│   │   ├── pun_policy.py               # released checkpoint, no retraining
│   │   ├── vggt_policy.py              # Phase 2 independent NUM predictions
│   │   ├── history_policy.py           # Phase 3 direct gain predictions
│   │   ├── oracle_policy.py            # privileged evaluator adapter
│   │   └── geometry_policy.py          # optional depth baselines
│   ├── eval/
│   │   ├── one_step.py
│   │   ├── closed_loop.py
│   │   ├── profiling.py
│   │   └── result_schema.py
│   └── visualization/
│       ├── rollout.py
│       └── coverage_plot.py
├── scripts/
│   ├── evaluate_closed_loop.py
│   ├── make_demo_results.py
│   ├── build_history_dataset.py         # Phase 3 only
│   ├── train_history.py                # Phase 3 only
│   └── evaluate_one_step.py
└── tests/
    ├── test_closed_loop.py
    ├── test_policies.py
    ├── test_history_dataset.py
    └── test_history_models.py
```

Keep the existing geometry implementation and `precompute_visibility.py` as
the common source of face coverage and gains. Exact future file names may
change; the separation remains:

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

Use separate semantic contracts even though each output has 48 entries:

```python
class SingleImageNUMPredictor:
    def predict(self, images):
        # Independent images -> [B, 48] original NUM/PUN target values.
        ...

class HistorySurfaceGainPredictor:
    def predict(self, history_images, history_anchor_ids, history_mask):
        # Phase 3 only: complete histories -> [B, 48] direct surface gains.
        ...
```

Phase 2 calls the single-image predictor for each acquired image, aligns and
aggregates the maps, and converts the aggregate to higher-is-better policy
scores. VGGT's current one-frame sequence contract is retained; acquired views
never interact inside the Phase 2 backbone. Independent batch examples are
not a multi-view sequence.

Phase 3's independent model aggregates independently extracted **features**
before a history-trained head. Its joint counterpart extracts features from
the complete history together. Both use the second predictor contract and
identical direct surface-gain supervision.

### Policy

```python
class NBVPolicy:
    def score(self, observation_state):
        # Return [48] higher-is-better scores in the common rollout frame.
        # No mesh, visibility cache, seen-face mask, or true gains are supplied.
        ...
```

The evaluator applies one candidate mask and deterministic argmax rule. Keep
raw NUM maps available for diagnostics without presenting aggregated Phase 2
scores as calibrated surface gains. Oracle is a separately identified
privileged adapter that returns evaluator-computed gains.

### Observation and evaluator state

Policy-visible observation state:

```text
object_id (lookup/provenance only, not a learned feature)
acquired_anchor_ids
acquired RGB images / image references
known camera/anchor geometry
step_index
valid_candidate_mask
```

Evaluator-private additions:

```text
mesh / visibility cache
seen_face_mask
current_coverage
candidate_surface_gains
```

Pass only acquired images or their cached features/maps to learned policies;
precomputing every anchor must not make unacquired RGB observations available
to them. Neither ground-truth gains nor coverage state is a Phase 2 training
input. Optional GT-depth controls receive only the privileged information
specified by their baseline definition and are labeled accordingly.

---

# 8. Data schemas

## Phase 1 and Phase 2 training sample

Phase 2 reuses the original Phase 1 records and fixed object split:

```yaml
object_id: str
image_path: str
source_anchor_id: int
target_map: float[48]             # original NUM/PUN values
split: train|val|test
metadata: ...                    # target name/direction and map frame
```

## Phase 2 rollout observation

This is simulator state, not a new supervised training dataset:

```yaml
object_id: str
acquired_image_paths: [str, ...]
acquired_anchor_ids: [int, ...]
valid_candidate_mask: bool[48]
map_frame_metadata: ...
```

## Phase 3 history sample

```yaml
object_id: str
history_image_paths:
  - ...
history_anchor_ids:
  - ...
history_length: int
target_surface_gain: float[48]   # direct history-dependent Vis/VisA gain
valid_candidate_mask: bool[48]
rotation_metadata: ...
split: train|val|test
visibility_cache_id: str
coverage_target: vis|vis_a
sampling_metadata: ...
```

### Dataset invariants

- Preserve the global canonical 48-anchor ordering and Phase 1 object splits.
- Distinguish source-relative NUM map coordinates from the common rollout frame and document their conversion.
- Preserve original NUM targets and direction in Phases 1/2; never replace them with geometric gain labels.
- Generate Phase 3 `target_surface_gain` only from the same face-visibility definition/configuration used in evaluation.
- Acquired anchors have zero direct marginal gain and are masked; other invalid anchors are also excluded from loss, selection, and ranking.
- Histories contain unique anchors by default; variable-length batches carry an observation-padding mask separate from the candidate mask.
- Both Phase 3 models receive identical split records, histories, labels, masks, and rotation metadata.

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

Reuse compatible Phase 1 per-image VGGT feature caches and the saved NUM head.
Optionally cache complete raw 48-value maps for both VGGT and official PUN.
Fingerprint prediction caches with the image/sample ID, model/head checkpoint
checksum, preprocessing, feature metadata, target name/direction, and map-frame
convention. Cache raw per-image outputs so aggregation ablations do not require
backbone inference again.

At a rollout step, retrieve only the acquired views. Compute each new image
once and reuse the previous maps. Cache-backed rollout latency and live model
inference latency must be reported separately; precomputation is not free
inference.

### Phase 3

Do **not** assume independent VGGT features can replace the joint forward pass.

The Phase 3 independent control can reuse frozen per-image VGGT features, but
must train its history aggregation/head on the new direct gain labels. Its
head checkpoint is distinct from the Phase 2 single-image NUM head.

Joint VGGT features depend on the complete input history. Run the frozen
backbone on that history as a group; a joint cache, if introduced, must key the
entire ordered history, anchor/rotation metadata, checkpoint, preprocessing,
and feature selection. Changing or adding a view invalidates that history's
features. Fix the history ordering convention for paired comparisons.

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
phase2/vggt_num_pun_aggregation/seed_0
phase3/vggt_joint/seed_0
```

Run multiple seeds for the final reported comparisons if compute allows.

Phase-specific rules:

- Phase 2 reuses the validation-selected Phase 1 NUM predictor by default. Any additional predictor training uses only original single-image NUM labels and the Phase 1 loss/split protocol.
- Official PUN remains inference-only in every main experiment; retain the released checkpoint and original preprocessing.
- Phase 3 trains new independent and joint history heads with identical `target_surface_gain`, valid masks, optimizer, loss, validation selection, and training budget. Document approximately matched trainable capacity, batch/effective batch size, seeds, and any unavoidable compute differences.
- Never use test coverage or test history metrics to select a feature variant, aggregation rule, loss weight, epoch, or checkpoint.

---

# 12. Loss implementation details

### Huber regression

Input:

```text
raw predictions: [B, 48]
target values:   [B, 48]
candidate mask:  [B, 48]
```

Only valid candidates contribute. In Phases 1/2, regress the original NUM
values in their original units and orient scores/targets separately for the
ranking term. In Phase 3, regress nonnegative direct `Vis`/`VisA` surface gains;
higher gain is always better. Keep the configured Huber and ranking weights
identical between the Phase 3 control models.

### Pairwise ranking

The ranking loss should encourage:

```text
oriented_target_i > oriented_target_j  =>  score_i > score_j
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

Retain average ranks for ties and the current undefined result when fewer than two candidates remain or either vector is constant. Store undefined values as JSON `null`, and report valid metric counts when aggregating.

### NDCG@5

Compute only over valid candidate anchors, with `k = min(5, valid_count)`. Retain the existing tie-aware implementation and all-zero-relevance result of zero. Phase 2/3 geometric NDCG uses nonnegative true gains as relevance; original NUM-target NDCG remains a separate metric.

### Coverage AUC

Define the x-axis consistently:

```text
number of acquired views
```

Count initial views in every policy's x-axis and record the initial coverage
point. The existing `coverage_auc` computes the unnormalized trapezoidal area;
retain that definition and use the same view-count interval for every policy.
If reporting a normalized variant, give it a separate name and denominator.

For Phase 2/3 regret, use true geometric gains over the valid candidate set:

```text
R_t = (max_valid(u_t) - u_t[selected])
      / (max_valid(u_t) - min_valid(u_t) + epsilon)
```

Use zero regret for all-equal valid gains, consistent with the current metric
implementation. These geometric metrics do not imply Phase 2 gain supervision.
Store original NUM-target metrics under a distinct namespace.

---

# 14. Closed-loop evaluation protocol

One deterministic rollout function supports every policy. Its view budget is
the **total** number of acquired views, including initial views:

```python
state = evaluator.initialize(object_id, initial_anchor_ids)
record_coverage(state)  # include initial observations in the curve

while len(state.acquired_anchor_ids) < max_acquired_views:
    valid = evaluator.valid_candidates(state)
    if not valid.any():
        record_stop_reason("no_valid_candidates")
        break

    true_gains = evaluator.candidate_gains(state)  # evaluator-private
    if policy.is_oracle:
        scores = true_gains.copy()               # explicit privileged branch
    else:
        observations = evaluator.observation_state(state)
        scores = policy.score(observations)      # only acquired observations

    require_finite_valid_scores(scores, valid)
    next_anchor = canonical_masked_argmax(scores, valid)
    record_one_step_metrics(scores, true_gains, valid, next_anchor)
    record_rollout_step(state, scores, true_gains, valid, next_anchor)
    state = evaluator.acquire_and_update(state, next_anchor)
    record_coverage(state)
```

`candidate_gains` and the state update use the existing face-cache helpers with
the configured `vis` or `vis_a` target. The selected true gain must equal the
post-acquisition coverage minus pre-acquisition coverage. Coverage is
nondecreasing, but zero-gain steps are valid; do not require strict improvement
or introduce policy-dependent early stopping when all remaining gains are zero.

Random uses a recorded seed and random candidate scores. Farthest scores each
valid candidate by its minimum angular distance to the acquired views and
selects the largest value. Ties use canonical anchor order. Fail clearly on
invalid/nonfinite policy scores rather than silently substituting a policy.

Each policy chooses its own subsequent history from the same initial views.
For paired fixed-history diagnostics, explicitly feed PUN/VGGT or the Phase 3
control pair identical saved histories. Do not force identical trajectories
when measuring closed-loop policy performance.

# 15. Baseline fairness rules

Before final experiments, verify:

- Same test objects, canonical anchors, initial view(s), total view budget, candidate masks, stopping rule, and coverage interval.
- Same face-visibility cache, `Vis`/`VisA` target, geometric metrics, and evaluator for every policy.
- Same acquired-observation access protocol; no unacquired RGB, ground-truth visibility, or candidate gains in learned policy inputs. Oracle and optional GT-depth controls are explicitly privileged.
- Official PUN keeps its released checkpoint, preprocessing, raw output semantics, and reproduced history rule. Pin and report all deviations; carry forward the unknown original training-overlap limitation from Phase 1.
- Phase 2 VGGT retains original NUM supervision and the same reproducible PUN-style combination rule wherever possible. Its aggregate is a policy score, not a surface-gain regression output.
- Phase 3 independent/joint controls match histories, labels, masks, frozen checkpoint, optimizer, loss, validation selection, training budget, starting conditions, and approximately matched head capacity.
- Runtime uses documented hardware, warm-up, device synchronization, precision, and history lengths. Separate live inference from cached policy scoring, include feature extraction/head/aggregation in live latency, and report precompute cost separately.
- Report peak memory and trainable/frozen parameter counts, including a separate indication that official PUN is not trained locally.

An optional retrained architecture is labeled **PUN-architecture surface-gain
control**, with its own training provenance. It never replaces the official
pretrained PUN result.

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
| `pun_upnet` | official released PUN ViT-S/16 checkpoint | inference only; official preprocessing + common evaluator | complete in the frozen seed-1 sweep |

All 12 configured entries were evaluated in the frozen seed-1 Phase 1 sweep.
Preserve those artifacts and use validation evidence to select the Phase 2
VGGT predictor.

## Phase 2 — minimum policy table

| Policy | Training / checkpoint | Policy-score meaning | History treatment |
|---|---|---|---|
| Random | none | seeded random ranking | candidate mask |
| Farthest view | none | minimum angular distance from acquired views | known camera poses |
| Official PUN | released UPNet, no retraining | official aggregated NUM/PUN score | official per-view map combination |
| Independent VGGT | Phase 1 NUM-trained head + frozen VGGT | aggregated NUM/PUN score | independent images + PUN-style map combination |
| Oracle one-step | none; privileged evaluator | true incremental `Vis`/`VisA` gain | evaluator's acquired-face union |

Every row uses the same ground-truth face-coverage evaluator. A complete,
validated five-policy comparison is the core Phase 2 deliverable.

Add after the core comparison is stable:

| Optional comparison | Purpose |
|---|---|
| VGGT-predicted-depth geometry | compare with an explicit geometry policy |
| Ground-truth-depth geometry upper bound | privileged depth control |
| Mean/max per-view map aggregation | test simple deterministic combination rules |

These additions do not block completion or freezing of the core Phase 2 result.
The oracle is greedy for the current step; it is not a proof of globally
optimal coverage at every multi-step budget.

## Phase 3 — central comparison

| Model | Training target | Multi-view interaction inside frozen VGGT? | History aggregation |
|---|---|---|---|
| Independent VGGT control | direct history-dependent `Vis`/`VisA` surface gains | no | permutation-invariant feature aggregation + new head |
| Joint VGGT | identical direct surface-gain labels | yes | history-aware joint-token pooling + matched head |

Both models train on the same history dataset and run through the unchanged
Phase 2 evaluator. Official PUN and the Phase 2 VGGT policy remain external
coverage references with their original NUM semantics. Any optional
PUN-architecture surface-gain control is a separate, clearly labeled row.

---

# 17. Recommended development order for VSCode + Codex

Steps 1–8 are complete and remain the record of Phase 1 implementation.
Continue at Step 9. Steps 9–14 complete the main Phase 2 deliverable without a
new supervised history dataset; Steps 15–18 are the optional Phase 3 extension.
Future script/config names below describe planned deliverables, not commands
that already exist.

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

### Step 9 — validate and precompute existing face visibility

**Starting point:** the face rasterizer, mesh loader, `Vis`/`VisA` coverage and
candidate-gain helpers, schema-2 cache, CLI, and synthetic tests already exist.
Maintain this implementation; no surface-point sampling or cache migration is
required.

1. Validate one real NUM object using `scripts/precompute_visibility.py` and `configs/experiments/phase2_visibility.yaml`, then a small fixed subset. Compare debug visibility views with the RGB/camera orientation.
2. Pin mesh scale, camera settings, render resolution, culling, anchor ordering, and the selected coverage target (currently `vis_a`). Verify cache metadata rejects incompatible settings.
3. Confirm candidate gains equal explicit coverage differences, acquired anchors have zero gain, and unions are independent of acquisition order.
4. Precompute every object in the fixed Phase 2 test split; record completeness and any failures. Do not silently evaluate only successful objects. Prepare validation objects when needed for policy selection/debugging; training-split geometry is not required for NUM training.

**Exit artifact:** reproducible face caches, a precompute manifest, and saved
real-object sanity/debug results. Extend targeted geometry tests only where
remaining correctness checks are not already covered.

### Step 10 — deterministic closed-loop simulator with Random and Oracle

**Implementation status:** implemented with `scripts/evaluate_closed_loop.py`
and `configs/experiments/phase2_closed_loop.yaml`. The shared evaluator uses
acquired-only RGB snapshots, explicit Oracle privilege, deterministic Random
scores, common masks/ties, total-view budgets, zero-gain continuation, and
candidate-exhaustion stopping. NPZ rollouts and per-step/per-object metrics
support replay through the same evaluator; missing caches require an explicit
partial-run option and are listed in the completeness manifest.

**Step 9 camera validation:** PUN's Blender renderer uses a pixel focal length of
`(525/512) * width`, giving a 51.98948897809546° FOV; its tracking convention
uses world Z to determine roll. The implementation pins the pole
rolls to the released RGB and matches an independent PUN/Blender reference at
all 46 non-pole anchors. Mesh scale remains 2.0 and radius remains 2.73.
See `docs/num_camera_alignment.md` and the real-object validation report in
`outputs/phase2/visibility_alignment`. Schema-2 face caches use
`data/cache/visibility` with an explicit camera-convention identifier.
The retained `random_oracle` run covers three objects; full-split
precomputation is tracked separately. Prepared meshes use bounding-box
centering before scale 2.0, supported by six independent validation objects
(30 views; mean silhouette IoU 51.3% to 86.8%). Cache metadata pins centering
and the source-to-world transform. Exact original-OBJ equivalence and
full-dataset RGB registration are not claimed.

Add the observation store, policy-visible/private state separation, common
result schema, and `src/nbv/eval/closed_loop.py`. Wrap the existing face-cache
coverage/gain helpers; do not duplicate geometry logic inside policies.

Implement Random and one-step Oracle first. Pin total view budget, initial
anchors, masks, seeded randomness, canonical tie-breaking, and candidate
exhaustion behavior. Store initial coverage and per-step gains/coverage.

**Exit check:** a saved single-object rollout replays identically; no acquired
view is selected twice; coverage never decreases; each selected oracle gain
matches the explicit coverage difference; zero-gain and exhausted-candidate
states terminate under the documented protocol. Verify learned-policy state
cannot access geometry or unacquired observations.

### Step 11 — farthest-view baseline

Implement max-min angular distance from the acquired camera directions using
the canonical anchor utilities. Run it through the same mask, tie-breaking,
rollout, and metric path as Random and Oracle.

**Exit artifact:** a deterministic three-policy subset comparison establishing
that camera geometry and evaluator behavior work without learned predictions.

### Step 12 — official PUN closed-loop adapter and aggregation

Reuse the existing official UPNet loader/checkpoint from Phase 1. Do not train
or replace it. Inspect the pinned official history implementation and record
its source revision, map-frame conversion, target direction, normalization,
combination order, and final candidate selection behavior.

Implement the reproducible post-prediction rule in a shared aggregation module
and add the PUN policy adapter. Check one-image behavior, aligned multi-image
maps, and acquisition masks against official outputs where reproducible.
If a behavior cannot be reproduced, document the exact deviation in config,
results, and report; label deterministic fallbacks as approximations rather
than claiming an official aggregation reproduction.

**Exit artifact:** official-checkpoint PUN rollouts and aggregation provenance,
including anchor alignment and raw-output versus policy-score diagnostics.

### Step 13 — independent per-view VGGT NUM policy

Pin the validation-selected Phase 1 VGGT variant and saved head. Reuse the
single-image extractor and compatible per-image features; optionally add raw
prediction-map caches with checkpoint/target/frame metadata. No history-label
generation or history-trained module is introduced in this step.

Use the shared PUN-style post-prediction combination rule. At each step, process
only the newly acquired RGB and reuse previous maps; every VGGT sequence still
contains one image. Validate cached versus live predictions and fixed-history
PUN/VGGT access to exactly the same acquired observations.

**Exit artifact:** VGGT runs through the same simulator as the other four core
policies and saves per-view raw maps, aggregate policy scores, selected actions,
and evaluator-only true gains. Any retraining retains original NUM targets.

### Step 14 — complete and freeze the core Phase 2 result

Add geometric regret, Spearman, NDCG@5, coverage AUC/final coverage, complete
per-object/per-step exports, live/cached runtime profiling, peak memory, and
parameter counts. Keep original NUM-target results in a separate namespace.

Run Random, Farthest, official PUN, independent VGGT, and Oracle on the full
fixed test split with shared starts/budgets and recorded seeds. Produce coverage
curves, comparison tables, and a saved-checkpoint rollout replay showing RGB,
aggregate scores, true gain diagnostics, and accumulated visible faces.

**Exit artifact:** fixed Phase 2 configs, complete results, checkpoints or
checksum-pinned references, profiling records, a replayable demo, and all Phase 2
completion checks satisfied. This is a final-quality project deliverable.

After the core is stable, optionally add VGGT-depth geometry, GT-depth geometry,
and mean/max aggregation ablations. These do not gate freezing the core result
or justify changing its target semantics.

### Step 15 — Phase 3 direct surface-gain history dataset

Only after freezing Phase 2, create `history_dataset.py` and
`scripts/build_history_dataset.py`. Extend face-cache precomputation to the
fixed training/validation splits as needed, preserving the evaluation geometry
settings. Sample seeded unique-anchor histories at multiple lengths and store
`target_surface_gain`, valid-candidate masks, cache IDs, split, and sampling
metadata. Include observation-padding masks in variable-length batches.

**Exit check:** histories reproduce from config, splits remain object-disjoint,
and labels match brute-force coverage differences. Training/evaluation share
one visibility definition. Verify non-rotated data first; add consistent random
rotations or policy-generated histories only as later documented ablations.

### Step 16 — Phase 3 independent history control

Build a new history-trained model from independent frozen VGGT features,
permutation-invariant feature aggregation, and a lightweight 48-gain head.
Reuse feature caches, but do not reuse Phase 2's NUM-trained policy unchanged.
Train with the direct history labels and valid masks from Step 15.

**Exit check:** a tiny history subset can be overfit; multiple lengths and
padding work; history permutation leaves the independent aggregate unchanged;
no backbone cross-view interaction occurs. Save validation-selected weights,
one-step regression/ranking metrics, and a closed-loop smoke result.

### Step 17 — joint frozen VGGT control

Add a joint-history extractor separately from the existing single-image path.
Feed complete histories through one frozen VGGT forward and train history-aware
pooling plus a lightweight direct-gain head. Confirm multiple acquired views
actually interact inside VGGT and padding is handled without contaminating
real-view features.

**Exit check:** the independent/joint pair uses identical histories, labels,
valid masks, checkpoint, optimizer, loss, training budget, and approximately
matched head capacity. Document parameter counts and memory at short histories
before longer runs; do not substitute independent caches for joint features.

### Step 18 — controlled Phase 3 experiment and reporting

Run both models on identical held-out histories for Huber/regression error,
geometric regret, Spearman, and NDCG@5. Run their own closed-loop trajectories
through the unchanged Phase 2 evaluator with the same starts and budgets.
Report coverage, AUC, final coverage, runtime, peak memory, and capacity.

**Exit artifact:** reproducible configs, validation-selected checkpoints,
paired one-step and closed-loop tables/figures, and a documented conclusion,
including negative or inconclusive joint-processing results. Keep official
PUN and Phase 2 VGGT as external coverage references. Aggregate configured
seeds and add history-length/token ablations only if feasible.

### Step 19 — optional extensions

Only after the core result is complete, consider reconstruction/Chamfer
metrics, a 3DGS evaluator, MipNeRF360 transfer, or a separately labeled
PUN-architecture surface-gain control. Preserve the frozen Phase 2 deliverable
if Phase 3 or any extension exceeds the available time/compute.

---

# 18. Codex task style

When asking Codex to implement components, prefer bounded tasks with a definition of done.

Good example:

```text
Implement src/nbv/eval/metrics.py with normalized_regret, spearman_rank,
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

- existing face masks look geometrically correct on real objects,
- every policy uses the same configured `Vis`/`VisA` cache,
- coverage never decreases and repeat acquisition is impossible,
- oracle gain equals explicit coverage difference, including zero-gain cases,
- source-relative maps align to the common rollout anchor frame,
- PUN's aggregation and score direction are verified or deviations documented,
- VGGT uses only one-frame sequences and original NUM supervision,
- cached/live outputs agree and only acquired observations reach policies,
- true gains/visible-face state remain evaluator-private,
- original NUM metrics and geometric metrics are stored separately.

### Before Phase 3 training

Confirm:

- direct surface-gain labels equal brute-force `Vis`/`VisA` differences,
- the Phase 2 face-visibility definition/config is unchanged,
- variable-length history padding/masking is correct,
- joint and independent models receive identical histories, labels, and masks,
- independent backbone calls never mix views; joint calls do mix views,
- head capacities, optimizer, loss, and training budget are matched/documented,
- optional random rotations preserve image/anchor/label consistency.

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

The original NUM/PUN target is a proxy, while incremental face coverage is the
common evaluator quantity. Better proxy-target regression does not guarantee
better sequential coverage. Lower-is-more-uncertain targets also require
explicit score orientation before candidate selection.

Mitigation:

- preserve original NUM targets in both Phases 1 and 2,
- call Phase 2 aggregates policy scores rather than predicted surface gains,
- generate direct history-dependent gain training labels only in Phase 3,
- retain official PUN as an unretrained published-method baseline,
- report the target/evaluator mismatch and any official aggregation deviations,
- evaluate NUM prediction quality and geometric policy quality separately.

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
- [ ] Triangle-ID render resolution documented.
- [ ] Visibility rasterization and culling policy documented.
- [ ] `Vis`/`VisA` coverage target and face-cache provenance documented.
- [ ] Training-target semantics and policy-score direction documented.
- [ ] PUN map alignment, aggregation rule, and deviations documented.
- [ ] Initial views, tie-breaking, and stopping behavior documented.
- [ ] Phase 3 history sampling, cache IDs, and padding/rotation conventions saved.
- [ ] Phase 3 matched histories, targets, capacity, optimizer, loss, and budget recorded.
- [ ] Cached versus live inference timing and precomputation cost distinguished.
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

For Phases 2/3, extend the existing semantic run-directory convention:

```text
outputs/<phase>/<experiment>/seed_<n>/
├── config.yaml
├── metadata.json
├── run.log
├── checkpoints/                    # trained heads or pinned model references
├── metrics/
│   ├── summary.json
│   ├── comparison.csv
│   ├── per_object.csv
│   ├── per_step.csv
│   ├── profiling.json
│   └── num_target_metrics.json     # Phase 2: original-target results/reference
├── rollouts/<policy>/<object_id>.npz
└── figures/
    ├── coverage.svg
    └── rollouts/
```

The planned common summary separates training and evaluation semantics:

```json
{
  "phase": "phase2",
  "policy": "vggt_num_pun_aggregation",
  "training_target_semantics": "original_num",
  "num_target_name": "PSNR",
  "num_target_direction": "lower",
  "policy_score_semantics": "aggregated_num_policy_score",
  "aggregation_rule": null,
  "aggregation_deviations": [],
  "coverage_target": "vis_a",
  "visibility_definition": "pun_unoccluded_rasterized_mesh_faces_v1",
  "visibility_cache_manifest": null,
  "checkpoint_sha256": null,
  "geometry_metrics": {
    "normalized_regret_mean": null,
    "spearman_mean": null,
    "spearman_valid_count": null,
    "ndcg_at_5_mean": null,
    "coverage_auc_mean": null,
    "final_coverage_mean": null
  },
  "profiling": {
    "median_live_inference_ms": null,
    "median_cached_policy_ms": null,
    "precompute_seconds": null,
    "peak_memory_mb": null,
    "trainable_parameters": null,
    "frozen_parameters": null
  }
}
```

These future fields are a planned schema, not currently emitted Phase 1 data.
For Phase 3, set training/policy semantics to direct history-dependent surface
gain, record the history dataset ID and matched-control settings, and add
masked surface-gain Huber/regression error. Official PUN retains original NUM
semantics in either phase. Use `none`/not-applicable metadata for heuristics and
Oracle, rather than implying they were trained on NUM labels.

Each per-step record should identify object, policy, seed, history anchors,
acquired-view count, valid mask, selected anchor, selected/oracle true gain,
coverage before/after, regret, Spearman, NDCG@5, and timing. Saved rollouts also
retain score arrays, evaluator-only candidate gain arrays, optional raw
per-image maps, image references, and enough cache/config provenance to replay
the visible-face union and coverage curve. Store diagnostics together without
passing them back into learned policy inputs.

Aggregate per-object coverage curves over the complete fixed test split and
report the object count. State whether ranking summaries are per-step or
per-object means and how undefined metrics are excluded. Saved arrays and
checkpoints must support the demo through the same evaluator, without hidden
notebook-only training or geometry logic.

---

# 23. Figures/tables to plan for the final report

### Phase 1

- Table: ViT vs. DINOv2 vs. VGGT vs. PUN.
- Qualitative spherical target/prediction visualization.
- Optional ablation plot for VGGT token/layer choice.

### Phase 2

- Main coverage-vs-view curve for all policies.
- Table with geometric regret, Spearman, NDCG@5, coverage AUC, final coverage, live/cached runtime, memory, and parameters.
- Separate original NUM-target table or Phase 1 reference to test whether proxy-task improvements transfer to coverage.
- Qualitative rollout showing acquired RGB, aggregated policy scores, chosen NBV, evaluator-only candidate gains, accumulated visible faces, and coverage.
- Explicit PUN target/evaluator mismatch, history-aggregation deviations, and training-overlap limitation.

### Phase 3

- Joint vs. independent coverage curve.
- Joint vs. independent direct surface-gain regression and ranking metrics on identical held-out histories.
- Runtime/memory comparison.
- History-length ablation if available.

### Optional reconstruction extension

- Chamfer distance vs. number of acquired views.
- Qualitative predicted reconstruction vs. ground-truth mesh.

---

# 24. Decision gates

### Gate A — after Phase 1

The complete seed-1 Phase 1 result satisfies this gate. Keep its artifacts
frozen, pin the validation-selected VGGT variant, and proceed with original NUM
supervision and official pretrained PUN. Complete the missing closed-loop
aggregation/evaluator work in Steps 9–14.

If VGGT does not outperform generic features:

- do not hide the result,
- continue to Phase 2,
- test whether sequential aggregation still provides value.

### Gate B — after Phase 2

Treat the core project as **complete** when Random, Farthest, official PUN,
independent NUM-trained VGGT, and Oracle have reproducible full-test-split
coverage results, geometric ranking metrics, profiling, and a replayable demo.
PUN's aggregation is reproduced or deviations are documented. No sampled-point
visibility replacement, history-supervised training, or depth-based ablation
is required for this gate.

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

Can frozen VGGT features predict the original single-image NUM/PUN 48-value target better than generic visual representations?

### Story 2 — Practical sequential policy

Does improved prediction of that proxy target translate into better true face coverage when independently predicted maps are combined across observations, compared with official PUN, Random, Farthest, and Oracle?

### Story 3 — Novel multi-view test

When both models train on direct history-dependent surface gains from the same face-visibility cache, does joint VGGT processing improve over capacity-matched independent feature aggregation?

This structure ensures that every completed phase produces a self-contained research conclusion.

---

# 26. Immediate implementation backlog

Milestones 1–4 are complete. Resume at Milestone 5 / Step 9 using the existing
face-visibility implementation. Milestone 6 is the required final-quality
project checkpoint; Milestones 7–8 change the learning problem and follow only
after Phase 2 is frozen.

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
- [x] Produce the populated seed-1 comparison table, including official PUN.
- [x] Build saved-checkpoint prediction-versus-target visualization.
- [x] Freeze Phase 1 checkpoint.

### Milestone 5 — closed-loop geometry (Steps 9–11)

- [x] Mesh loading, canonical face rasterization, and schema-2 face caches.
- [x] `Vis`/`VisA` coverage and candidate marginal-gain helpers with synthetic tests.
- [x] Split/subset precompute CLI and debug visibility visualization.
- [x] Check camera calibration and reference poses; retain explicit NUM pole-roll convention.
- [x] Apply bounding-box centering and validate on six independent validation objects; record transforms in caches.
- [ ] Complete fixed-test-split cache precomputation and completeness manifest.
- [x] Verify explicit coverage differences and pin one `Vis`/`VisA` definition (`vis_a` in the evaluator).
- [x] Add acquired-RGB observation store and evaluator-private geometry access.
- [x] Implement deterministic shared closed-loop simulator and result schema.
- [x] Add Random/Oracle replay, masks, zero-gain, and exhaustion checks.
- [ ] Add the max-min angular-distance Farthest policy.

### Milestone 6 — complete Phase 2 (Steps 12–14)

- [ ] Reuse official PUN checkpoint/preprocessing without retraining.
- [ ] Reproduce PUN map alignment, direction, and aggregation; document deviations.
- [ ] Pin the validation-selected Phase 1 VGGT NUM head and feature metadata.
- [ ] Implement independent per-image VGGT + shared PUN-style map aggregation.
- [ ] Reuse feature caches; support optional raw prediction-map caching.
- [ ] Verify live/cache equivalence and identical supplied-history access.
- [ ] Verify no true gains, visible-face state, or unacquired RGB reaches learned policies.
- [ ] Run Random, Farthest, PUN, VGGT, and Oracle on the complete test split.
- [ ] Save per-step geometric metrics and coverage metrics together; retain NUM metrics separately.
- [ ] Report coverage curves/AUC/final coverage, live/cached runtime, memory, and parameter counts.
- [ ] Save and replay a full rollout using checkpoints and the common evaluator.
- [ ] Freeze reproducible Phase 2 configs and final-quality results.
- [ ] Optional after the core is stable: VGGT-depth/GT-depth geometry and mean/max aggregation ablations.

### Milestone 7 — Phase 3 history dataset (Step 15)

- [ ] Precompute training/validation face caches with the frozen evaluation definition.
- [ ] Generate deterministic, unique-anchor histories at multiple lengths.
- [ ] Generate `target_surface_gain` and masks from the existing face-cache helpers.
- [ ] Verify labels against brute-force coverage differences and preserve object splits.
- [ ] Add variable-length batching and separate observation/candidate masks.
- [ ] Save sampling strategy, cache IDs, coverage target, and rotation metadata.
- [ ] Optional: consistent rotations or policy-generated histories after the base pipeline passes.

### Milestone 8 — controlled direct-gain experiment (Steps 16–18)

- [ ] Train a new independent VGGT feature-aggregation control on history surface gains.
- [ ] Train joint frozen VGGT on the identical histories, targets, and masks.
- [ ] Verify absent/present cross-view backbone interaction in independent/joint paths.
- [ ] Approximately match head capacity and use the same optimizer, loss, and training budget.
- [ ] Compare held-out surface-gain regression, regret, Spearman, and NDCG@5.
- [ ] Run both models through the unchanged Phase 2 evaluator and compare coverage.
- [ ] Report runtime, peak memory, parameter counts, and remaining control differences.
- [ ] Add history-length/token ablations only if useful and feasible.
- [ ] Freeze Phase 3 results or document negative/inconclusive outcomes and failure modes.

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
- replacing the current face-visibility implementation with surface-point sampling,
- creating history-supervised datasets or large learned history modules for Phase 2,
- retraining official PUN on direct surface gain,
- replacing the 48-anchor action space,
- sophisticated learned policy optimization,
- 3DGS reconstruction,
- real-world transfer,
- extra datasets,
- major architecture changes to VGGT.

The project is strongest when the comparison is controlled and the evaluation is complete.

---

# 28. One-sentence implementation priority

**Retain the completed Phase 1 NUM probe, finish Phase 2 with independent NUM predictions and PUN-style aggregation under the existing face-coverage evaluator, then train capacity-matched independent and joint Phase 3 models on direct history-dependent surface gains.**
