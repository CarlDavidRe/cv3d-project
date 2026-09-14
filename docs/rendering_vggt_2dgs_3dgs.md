# How the VGGT, 2DGS, and 3DGS renderings were created

This document describes the rendering and visualization pipeline used in this
repository. It separates three outputs that are easy to conflate:

- **VGGT reconstruction visualization** shows a filtered VGGT point map aligned
  with a ground-truth ShapeNet point sample. It is a point-cloud viewer, not a
  photorealistic novel-view renderer.
- **2D Gaussian Splatting (2DGS)** produces RGB turntable renders and a surface
  reconstructed from rendered median depth. That surface is evaluated
  geometrically.
- **3D Gaussian Splatting (3DGS)** produces RGB turntable and canonical-view
  renders. In this project it is deliberately qualitative and has no geometry
  score.

The implementation is mainly in:

- `src/nbv/eval/reconstruction.py` — frozen VGGT inference, filtering, alignment,
  and point-cloud metrics;
- `scripts/visualize_reconstruction.py` — VGGT/ground-truth visualization;
- `src/nbv/eval/gaussian_splatting.py` — Gaussian model, optimization,
  rasterization, depth fusion, and HTML galleries;
- `scripts/evaluate_gaussian_splatting.py` — one object, policy, and view count;
- `scripts/evaluate_gaussian_splatting_variants.py` — batch orchestration across
  Phase 2/3 policies and view budgets.

## End-to-end data flow

```text
policy rollout's ordered anchor history
                  |
                  v
          acquired NUM RGB images
                  |
                  v
 frozen VGGT full-model forward (cached)
  - world point map
  - confidence map
  - predicted camera poses
                  |
     white-background + confidence filtering
                  |
                  v
 filtered VGGT points -- Sim(3) from predicted to known NUM cameras
                  |
          aligned VGGT point cloud
           /                    \
          /                      \
 point-cloud visualization       initialize 2DGS and 3DGS Gaussian means
 and shared VGGT metrics                    |
                                  project acquired RGB onto points
                                             |
                                  optimize against acquired images
                                    /                     \
                                   v                       v
                          2DGS RGB + depth            3DGS RGB only
                          surface fusion              qualitative galleries
                          geometry metrics
```

The policy determines **which views are in the history**. All policies then use
the same evaluator-owned VGGT reconstruction and Gaussian-fitting pipeline. PUN,
random, farthest, and oracle therefore do not introduce their own rendering
backends.

## 1. Creating the VGGT reconstruction

The reconstruction evaluation reads the RGB images for an ordered prefix of a
policy rollout and runs the frozen `facebook/VGGT-1B` full model. Input images
are converted to RGB, resized to 518 x 518 by default, scaled to `[0, 1]`, and
submitted together as one sequence.

VGGT returns a world-space point map, a confidence value for every predicted
point, and encoded camera poses. The pose encoding is converted to OpenCV
world-to-camera extrinsics and then inverted to camera-to-world matrices.

The raw point map is reduced as follows:

1. Pixels close to white are classified as background and removed. The default
   foreground test is `min(R, G, B) < 245/255`.
2. Non-finite points and confidence values are removed.
3. Points below the within-history 50th confidence percentile are removed by
   default.
4. The remaining points are deterministically capped at 10,000. The seed is
   derived from the ordered input-image checksums.

### Coordinate alignment

VGGT predicts geometry in its own coordinate frame, while NUM has known
canonical Blender/OpenGL cameras. VGGT's predicted OpenCV camera poses are
converted to the NUM convention, and a similarity transform (rotation,
translation, and one uniform scale) is fitted between predicted and known
camera poses. The transform is then applied to the filtered VGGT points.

For histories with at least two views, camera centres and orientations determine
the transform. A fallback handles inconsistent orientation-derived scales. A
one-view result needs the ground-truth bounding-box diameter to resolve scale and
must therefore be described as a scale-normalized diagnostic rather than a
metric-scale reconstruction. More detail, including known antipodal-camera
failure cases, is in [reconstruction_evaluation.md](reconstruction_evaluation.md).

### VGGT visualization artifacts

`scripts/visualize_reconstruction.py` loads the cached VGGT output and recomputes
the camera-based alignment; it does not run VGGT again. It writes:

| File | Meaning |
| --- | --- |
| `prediction_aligned.ply` | aligned VGGT points, colored orange |
| `ground_truth.ply` | area-sampled ShapeNet surface, colored blue |
| `comparison.ply` | both point clouds in one file |
| `comparison.png` | fixed XY, XZ, and YZ projections |
| `comparison_interactive.html` | dependency-free orbitable point-cloud viewer |
| `metadata.json` | inputs, history, alignment, transform, and display metadata |

The interactive viewer embeds a deterministic subset of each point cloud. It
supports rotation, zoom, fixed-axis views, auto-rotation, cloud toggles, and
overlay/side-by-side layouts. It is rendered with the browser's 2D canvas; it
does not rasterize a learned radiance field.

Example:

```bash
python3 scripts/visualize_reconstruction.py \
  outputs/phase2/phase2_closed_loop_reconstruction/seed_0/metrics/reconstruction_per_object.csv \
  --object-id 02691156/1628b65a9f3cd7c05e9e2656aff7dd5b \
  --policy vggt \
  --views 10
```

## 2. Shared initialization for 2DGS and 3DGS

For a selected object, policy, and acquired-view count,
`scripts/evaluate_gaussian_splatting.py` reads the corresponding row of
`reconstruction_per_object.csv`. The row supplies the ordered anchor history and
paths to the cached VGGT prediction and metric result. The script then:

1. loads the filtered VGGT points and predicted cameras;
2. finds the matching ground-truth sample and visibility metadata;
3. constructs the known NUM poses for the acquired anchors;
4. repeats the same camera-based VGGT-to-NUM alignment;
5. uses the aligned VGGT points as the initial Gaussian means;
6. builds camera intrinsics from NUM's horizontal field of view and requested
   square render resolution; and
7. converts NUM Blender/OpenGL camera poses to gsplat's OpenCV convention by
   flipping the camera Y and Z axes before inversion.

Initial view-independent RGB colors are estimated by projecting every aligned
VGGT point into every acquired image. Foreground pixel colors at valid
projections are averaged; points not observed in any acquired view start at
mid-gray.

Each VGGT point becomes one learnable Gaussian with:

- a 3D mean;
- three log-scales;
- a normalized quaternion rotation;
- an opacity logit initialized to `2.0`; and
- a view-independent RGB logit.

The initial scale is based on the point-cloud bounding-box diagonal divided by
the square root of the point count. For 2DGS, the local third scale is reduced
to 5% of that radius, creating a thin oriented disk. For 3DGS, all three initial
scales are equal, creating a volumetric ellipsoid.

This implementation does not use spherical-harmonic, view-dependent color and
does not perform Gaussian densification or pruning. The number of Gaussians is
therefore the number of aligned VGGT initialization points.

## 3. Optimizing the Gaussian models

Both backends are trained independently from the same initialization, acquired
RGB images, foreground masks, and known cameras. Training requires CUDA and
`gsplat>=1.5,<2`:

```bash
python3 -m pip install -e '.[gaussian-splatting]'
```

Images are bilinearly resized to the configured resolution (256 x 256 by
default). Training cycles deterministically through the acquired cameras:
iteration `i` uses camera `i mod number_of_views`. The default budget is 1,500
iterations **per acquired view**, so a five-view model receives 7,500 updates.
Every view budget starts a fresh model; the five-view model is not continued
from the three-view checkpoint.

Adam uses a lower learning rate for Gaussian means (`2e-4` by default) and
`1e-2` for scale, rotation, opacity, and color. Both backends minimize:

```text
RGB L1 + 0.1 * foreground-alpha L1
```

2DGS additionally minimizes:

```text
0.01 * depth-distortion loss + 0.05 * normal-consistency loss
```

The target background and rasterizer background are white. Loss samples are
stored at iteration 1, every 100 iterations, and the final iteration. The
optimized parameter state and history are saved to `checkpoint.pt`.

### Rasterizer details

- 2DGS uses `gsplat.rasterization_2dgs`. Camera-specific expanded color tensors
  work around missing color broadcasting in affected gsplat releases.
- 3DGS uses `gsplat.rasterization` with antialiasing, unpacked rasterization,
  and a per-camera white background. Unpacked mode avoids the packed-background
  shape mismatch present in affected gsplat releases.
- Renders are produced in batches of eight cameras by default.

## 4. Creating the 2DGS outputs

After training, 2DGS is rendered from two camera sets:

- a smooth 60-frame orbit at 20 degrees elevation for `turntable.html`; and
- all 48 canonical NUM cameras for geometric surface extraction.

Canonical rendering explicitly requests `RGB+ED` and median depth. This is
important: in affected gsplat versions, requesting RGB alone makes the reported
median channel contain the final RGB feature (blue), not depth.

For every canonical camera, pixels are retained when alpha is at least 0.5 and
depth is finite and positive. Pixel centres and intrinsics back-project each
depth to camera coordinates, after which the known camera-to-world pose maps it
to NUM world coordinates. Points from all 48 views are concatenated, fused by a
voxel grid whose cell width is `bounding-box diagonal / 512`, and
deterministically reduced to at most 10,000 surface points.

The fused surface is compared directly with the ground-truth ShapeNet surface
sample. There is no ground-truth-assisted ICP. Reported accuracy, completeness,
Chamfer-L1, precision, recall, and F-scores are normalized or thresholded by the
ground-truth bounding-box diameter.

The 2DGS directory contains:

| File | Meaning |
| --- | --- |
| `checkpoint.pt` | trained Gaussian parameters and loss history |
| `turntable.html` | self-contained RGB orbit gallery with play/pause and slider |
| `surface.ply` | alpha-filtered, depth-fused 2DGS surface |
| `ground_truth.ply` | ground-truth surface sample |
| `comparison.ply` | fused surface and ground truth together |
| `comparison.png` | static orthographic point-cloud comparison |
| `comparison_interactive.html` | orbitable 2DGS-surface/ground-truth viewer |
| `metrics.csv` | one row of geometric metrics |
| `summary.json` | settings, provenance, alignment, losses, paths, and metrics |

The original CUDA extraction bug and the CPU recovery path for existing
checkpoints are documented in
[2dgs_cpu_fixes.md](2dgs_cpu_fixes.md). New runs use the corrected `RGB+ED`
path.

## 5. Creating the 3DGS outputs

After training, 3DGS is rendered from:

- the same smooth 60-frame orbit used by 2DGS; and
- all 48 canonical NUM cameras.

`turntable.html` embeds every orbit frame as a base64 PNG and provides autoplay
and a frame slider. `ground_truth_comparison.html` pairs each canonical render
with the corresponding NUM RGB observation. It provides synchronized
side-by-side and opacity-overlay modes and labels the anchor as an acquired
training view or a held-out view.

The 3DGS directory contains only:

| File | Meaning |
| --- | --- |
| `checkpoint.pt` | trained Gaussian parameters and loss history |
| `turntable.html` | self-contained novel-view orbit gallery |
| `ground_truth_comparison.html` | 48 rendered/reference canonical view pairs |
| `summary.json` | settings, provenance, alignment, losses, and gallery paths |

3DGS does not write geometry metrics because its Gaussian centres are model
parameters, not samples of the reconstructed object surface. The contract is
therefore appearance-only and qualitative.

## 6. Running the rendering pipeline

### One policy and view count

The input CSV must already contain the cached reconstruction history for the
requested object, policy, and view count:

```bash
python3 scripts/evaluate_gaussian_splatting.py \
  outputs/phase2/phase2_closed_loop_reconstruction/seed_0/metrics/reconstruction_per_object.csv \
  --object-id 02691156/1628b65a9f3cd7c05e9e2656aff7dd5b \
  --policy vggt \
  --views 5 \
  --backend both \
  --iterations 1500 \
  --resolution 256 \
  --render-frames 60
```

If `--output-dir` is omitted, output is placed under a `gaussian_splatting`
directory beside the metrics directory. Use `--backend 2dgs` or `--backend
3dgs` to create only one representation.

### Every Phase 2/3 variant for one object

```bash
python3 scripts/evaluate_gaussian_splatting_variants.py \
  --object-id 02691156/1628b65a9f3cd7c05e9e2656aff7dd5b \
  --views 1 2 3 5 10 \
  --backend both \
  --iterations 1500
```

The batch script discovers these policies in the available reconstruction CSVs:

- Phase 2: `random`, `farthest`, `pun`, `vggt`, and `oracle`;
- Phase 3: `vggt_independent_history`, `vggt_joint_history`,
  `vggt_joint_pose_deepsets`, and `vggt_joint_token_attention`.

It validates that every requested history exists, starts the single-run script
for every variant/view pair, and writes:

```text
outputs/gaussian_splatting_per_view_budget/
  <category>/<object>/
    index.html
    manifest.json
    <N>views/<phase>_<policy>/
      2dgs/...
      3dgs/...
```

`index.html` is the main entry point: it links every available 2DGS comparison,
2DGS turntable, 3DGS reference comparison, and 3DGS turntable. A rerun skips a
backend only when its summary exists and records the requested per-view training
budget; `--force` retrains it. Use `--dry-run` to validate inputs and print all
commands without starting CUDA work.

To view the HTML through a local server:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then browse to the desired output path below `http://127.0.0.1:8000/`.

## Interpretation and limitations

- The shared VGGT point map is both a reconstruction result and the
  initialization for Gaussian fitting. Poor VGGT camera alignment can therefore
  affect the point-cloud comparison, 2DGS, and 3DGS together.
- 2DGS geometry scores measure the depth-fused rendered surface, not the raw
  Gaussian means and not the original VGGT point map.
- 3DGS results assess rendered appearance only. Do not compare its means with a
  mesh and call the result a surface reconstruction metric.
- Ground truth is used to evaluate VGGT and 2DGS. It is not used to optimize the
  Gaussian models or to align the standard 2DGS result with ICP.
- The separate CPU placement-repair workflow changes the evaluation procedure
  by fitting a silhouette-based similarity transform. Keep repaired scores
  separate from the original camera-aligned scores; see
  [gaussian_alignment_cpu_repair.md](gaussian_alignment_cpu_repair.md).
- The HTML galleries are self-contained: rendered PNG frames are embedded in
  the document, so they can be copied or opened without a JavaScript package or
  image directory.
