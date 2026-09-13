# CPU placement repair for saved 2DGS and 3DGS checkpoints

Run from the repository using its existing Python environment:

```bash
python scripts/repair_gaussian_alignment_cpu.py \
  --object-id 02691156/1628b65a9f3cd7c05e9e2656aff7dd5b \
  --variant phase2_pun --views 3
```

The default is now `--backend both`. Use `--backend 2dgs` for geometry repair
only, or `--backend 3dgs` to transform saved 3DGS checkpoints and regenerate
RGB comparisons on CPU. Each backend fits its own trained Gaussian centers;
neither depends on the other backend having been repaired.

Omit the three filters to process all available histories. `--views 3 5 10`,
`--limit`, `--threads`, and `--dry-run` can restrict the work or inspect inputs.
Repeating a command skips matching completed repairs; `--force` recomputes them.
One-view histories are explicitly skipped because their depth and scale are
underconstrained. Two-view repairs are possible but can also be ambiguous.
Failed histories are reported individually and the command returns nonzero if
any fail. A full batch may take hours: fitting runs on CPU and surface extraction
renders all 48 canonical cameras for every history.

The script uses the repository's existing NumPy, PyTorch, and Pillow dependencies.
It does not import gsplat, run VGGT, retrain Gaussian parameters, or write caches.
Both backends need the original `checkpoint.pt`, `summary.json`, the existing
visibility cache, and acquired NUM RGB images for fitting. 2DGS additionally
needs `ground_truth.ply` for evaluation. 3DGS needs all 48 canonical RGB images
for its comparison gallery; unacquired images are loaded only as gallery
references and never enter placement fitting or candidate selection.
Existing VGGT prediction caches are optional and used only for camera diagnostics.
Absolute `/content/...` paths in original summaries are resolved using local
roots and artifact filenames. Override roots if the files are elsewhere:

```bash
python scripts/repair_gaussian_alignment_cpu.py \
  --input-root outputs/gaussian_splatting_variant_comparison \
  --output-root outputs/gaussian_splatting_alignment_repair \
  --data-root data/NUM \
  --visibility-cache-root data/cache/visibility \
  --prediction-cache-root data/cache/reconstruction/predictions
```

## Cause established from the saved airplane artifacts

For object `02691156/1628b65a9f3cd7c05e9e2656aff7dd5b`, three-view PUN and Oracle
geometry is already displaced before 2DGS training. Average Gaussian-center
movement during training is approximately 0.0094 world units for PUN and zero
for Oracle. The ground truth is centered near the origin, while PUN's initial
bounding-box center is approximately `(0.739, 1.373, -0.396)` and Oracle's is
`(-2.017, 0.469, 0.493)`.

The saved predicted camera rotations disagree with the known NUM rotations.
Across camera pairs, median relative-rotation disagreement is approximately
34.1 degrees for three-view PUN, 90.6 degrees for three-view Oracle, and
34.4 degrees for ten-view PUN. Relative camera rotations are invariant to a
global similarity transform, so no choice of one rotation/translation/scale
can reconcile these predictions with the known cameras exactly.

The existing alignment averages orientation-derived rotations and fits scale
and translation from camera centers. It accepts a finite positive scale even
when residuals are large. Fitting centers alone reduces some residuals but did
not fix the displaced geometry in the checked histories. With two cameras,
center fitting also leaves rotation around their baseline unconstrained.

The inversion and local-axis conversion match VGGT's documented
[OpenCV world-to-camera extrinsics](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/pose_enc.py).
These checks establish inconsistent predicted poses and inadequate alignment
validation in these artifacts, not a universal coordinate-axis correction.
They do not establish why VGGT made those predictions; object symmetry, image
overlap, and disagreement between its point and camera heads remain possible
contributors. A small camera-center residual alone does not validate a fit.

## What the script repairs

The script fits a single positive scale, proper rotation, and translation to
the foreground silhouettes of the **acquired** RGB images, projected through
the known NUM cameras. It tries deterministic initial rotations and minimizes
symmetric distances between projected Gaussian centers and foreground pixels.
The camera-orbit origin is an initialization prior; translation remains free.
A separate, denser sample selects the fit and checks it improves this image
placement proxy by at least 1%; otherwise the original placement is retained.
This proxy is not rendered silhouette IoU and does not guarantee better geometry.

Ground-truth geometry is **not an input to fitting or candidate selection**.
It is used only after placement is fixed, to calculate geometric evaluation
metrics and draw comparisons. The script transforms Gaussian centers, rotates
their quaternion frames, and scales all covariance axes consistently. Colors
and opacity stay as trained. It then regenerates depth and surface geometry
using the existing CPU renderer; transforming an old fused point cloud alone
would retain the old visibility and depth-extraction artifacts.

The script repairs placement, not missing or deformed geometry. RGB silhouettes
can admit multiple poses, particularly for symmetric objects or sparse views.
Training performed at an incorrect location cannot generally be undone by one
similarity transform. This is a practical post-processing method, not an exact
recovery of the model that correct initialization and retraining would produce.

Verified end-to-end on the three-view airplane histories above (32 starts,
200 steps, seed 0). Before values are from the existing corrected CPU depth
recovery, not the original affected RGB-depth extraction:

| Policy | Recall @ 2%, before → after | F-score @ 2%, before → after |
| --- | --- | --- |
| PUN | 0% → 71.30% | 0% → 31.47% |
| Oracle | 0% → 53.91% | 0% → 28.15% |

The outputs visibly retain thickness and shape errors. These two histories
demonstrate placement recovery, not dataset-wide policy improvements. Tests
cover covariance transformation, camera-diagnostic invariance, synthetic
image-only placement recovery, and protected output paths. The existing CPU
renderer and Gaussian tests also pass (25 tests total). A repeated real PUN run
skipped the completed repair, and its source checkpoint, summary, RGB, ground
truth, visibility cache, and prediction cache hashes were unchanged.

## Outputs and interpretation

Results go to `outputs/gaussian_splatting_alignment_repair` using the existing
`category/object/Nviews/variant/backend` hierarchy. The 2DGS outputs are:

- transformed `checkpoint.pt`;
- freshly extracted `surface.ply` and `metrics.csv`;
- `comparison.png` and `comparison_interactive.html`;
- `summary.json`, including the source hashes, camera diagnostics, fitted
  transform, before/after image-placement proxy, and protocol limitations;
- root-level `recovered_metrics.csv` aggregating completed outputs.

3DGS writes a transformed `checkpoint.pt`, a fresh 48-anchor
`ground_truth_comparison.html`, a canonical-anchor `turntable.html`, `preview.png`,
and `summary.json`. It remains qualitative: no 3DGS geometry metrics are added
to `recovered_metrics.csv`. Its CPU renderer projects full Gaussian covariances
and uses front-to-back alpha compositing with the antialias opacity compensation
used by the original gsplat wrapper. It supports this project's constant-RGB
checkpoints, not arbitrary spherical-harmonic checkpoints. The implementation
follows [gsplat's projection](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu)
and [RGB compositing](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu);
CPU/CUDA parity has not been verified. A better silhouette fit does not guarantee
better RGB colors, which remain those learned at the original placement.

The three-view PUN airplane was also run through the complete 3DGS path:
48 comparison frames were generated, a repeated invocation skipped the completed
repair, and the original checkpoint hash stayed unchanged. Its repaired RGB is
still faint and incomplete; moving the Gaussians does not recover colors or
opacity learned poorly during the original training. The renderer's analytic
projection/compositing tests and the repair integration tests pass (29 tests
across the repair, CPU extraction, and Gaussian modules).

The original checkpoints, cached predictions, and original evaluations are
preserved. Interrupted runs do not publish a completion marker. Old RGB
turntables are not relabeled as repaired renders.

Treat these results as a **separate silhouette-refined evaluation protocol**.
Apply it consistently across policies for comparisons; do not combine repaired
and original scores in one policy curve. The existing CPU renderer still has
`cuda_parity_verified: false`. The dashboard loads this directory by default
through `GAUSSIAN_SPLATTING_REPAIR_ROOT` in `dashboard/app.py`. Its 2DGS curves
include only completed repairs. The gallery shows repaired 3DGS RGB, unrepaired
2DGS from `gaussian_splatting_cpu_recovery`, and repaired 2DGS side by side.
Missing artifacts show an availability message. The unrepaired 2DGS column
uses correct depth extraction at the original placement, not the older
blue-channel-as-depth artifacts. Original files are preserved.
