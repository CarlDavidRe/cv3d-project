# Shared VGGT reconstruction evaluation

Phase 2 evaluates PUN and the validation-selected VGGT prediction-head policy
with two complementary geometric outcomes. Rasterized `VisA` remains the
backend-independent coverage metric. Reconstruction quality is measured by
passing each policy's selected RGB history through the same frozen full VGGT
model and comparing its point map with the paired ShapeNet mesh. PUN is a view
selector here; it does not supply a separate reconstruction backend.

The default checkpoints are 1, 2, 3, 5, and 10 acquired views. VGGT world
points are filtered by the NUM foreground mask and within-history confidence
percentile, then deterministically capped at 10,000 points. The ground-truth
mesh uses the same bounding-box centering and scale as visibility evaluation
and is sampled uniformly by triangle area at 10,000 points.

VGGT's OpenCV camera poses are converted to the NUM OpenGL/Blender axes. With
at least two views, a similarity transform is estimated from predicted camera
centres and orientations to the known NUM cameras and applied to the point
cloud. The one-view diagnostic uses camera orientation/centre plus the
ground-truth bounding-box diameter for scale; it must be reported as a
scale-normalized diagnostic, not metric-scale reconstruction.

Reported distances are divided by the ground-truth point cloud's bounding-box
diameter:

- accuracy: mean predicted-to-ground-truth nearest-neighbour distance;
- completeness: mean ground-truth-to-predicted nearest-neighbour distance;
- Chamfer-L1: the mean of normalized accuracy and completeness;
- precision, recall, and F-score at 1% and 2% of ground-truth diameter.

## Cache contract

`data/cache/reconstruction` contains three independently invalidated layers:

- deterministic area-sampled ground truth, keyed by mesh checksum and mesh
  normalization;
- filtered VGGT point maps and predicted cameras, keyed by ordered anchors,
  image checksums, model/input settings, filtering, and output point count;
- alignment and metric results, keyed by the point-map and ground-truth
  identities plus metric thresholds.

The point-map key contains no policy name. Identical ordered histories are
therefore reconstructed once and reused across Phase 2 policies, reruns, and
Phase 3. A metric-only change retains the expensive VGGT prediction. Cache
files are atomically replaced and never used when their embedded identity
differs.

Phase 2 writes `reconstruction_per_object.csv`,
`reconstruction_curves.csv`, reconstruction fields in `comparison.csv` and
`summary.json`, and `figures/closed_loop/reconstruction_curves.svg`. Phase 3
uses the same evaluator and exports the same reconstruction tables and fields
for its independent and joint policies. This supports the scoped claim that
one policy selects better views under a common frozen VGGT reconstruction
backend; visibility remains the complementary backend-independent check.
