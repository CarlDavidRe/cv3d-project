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
cloud. If an inconsistent predicted pose would make the orientation-derived
scale non-positive, a recorded fallback minimally corrects the rotation from
the widest corresponding camera baseline and obtains a positive scale from the
camera-centre spread. The one-view diagnostic uses camera orientation/centre
plus the ground-truth bounding-box diameter for scale; it must be reported as
a scale-normalized diagnostic, not metric-scale reconstruction.

Reported distances are divided by the ground-truth point cloud's bounding-box
diameter:

- accuracy: mean predicted-to-ground-truth nearest-neighbour distance;
- completeness: mean ground-truth-to-predicted nearest-neighbour distance;
- Chamfer-L1: the mean of normalized accuracy and completeness;
- precision, recall, and F-score at 1% and 2% of ground-truth diameter.

## Observed farthest-policy two-view outliers

The 300-object Phase 2 reconstruction run in
`outputs/phase2/phase2_closed_loop_reconstruction/seed_0` has a misleading
spike in the farthest-policy mean Chamfer-L1 at two views. The mean rises from
2.811 at one view to 30.608 at two views, although the two-view median is only
0.163. Three two-view objects have Chamfer-L1 values of approximately 1,515,
1,610, and 5,899; excluding those three reduces the two-view mean to 0.536.
The three-view mean is 0.599.

This is an alignment failure rather than broad reconstruction degradation.
With initial anchor 0, the farthest policy selects anchor 46 next, so every
two-view history is `[0, 46]`. These are the north and south poles and are
exactly 180 degrees apart. Their limited visual overlap can make VGGT estimate
a nearly collapsed camera baseline. Here, "collapsed baseline" means that
VGGT places its two predicted camera centres almost at the same 3D position,
even though the known NUM cameras are on opposite sides of the object. It does
not mean that the actual input cameras or canonical anchors are colocated.

The alignment estimates a similarity transform consisting of rotation,
translation, and one global scale. Informally, the scale must satisfy

```text
alignment scale ~= known camera separation / predicted camera separation
```

Therefore, when the predicted separation approaches zero, the estimated scale
becomes extremely large. The implementation detects only an exactly or nearly
zero squared camera spread below `1e-12`; a small but nonzero spread can still
pass that check and yield a finite but implausible scale. The positive-scale
fallback also divides the known camera spread by the predicted spread, so it
cannot repair a collapsed prediction by itself. Applying the resulting Sim(3)
transform enlarges VGGT's entire predicted point cloud, not just its camera
centres. Most reconstructed points then lie thousands of object diameters from
the ground-truth surface, inflating both the predicted-to-ground-truth accuracy
distance and the ground-truth-to-predicted completeness distance in Chamfer-L1.

The per-object table makes this failure visible in `alignment_scale` and
`camera_center_rmse_normalized`. The three dominant two-view outliers have
alignment scales of approximately 3,053, 3,263, and 11,841, compared with a
median two-view scale of 2.77. Two use the positive-scale fallback. A small
camera-centre RMSE does not necessarily validate the reconstruction: with only
two camera correspondences, a huge scale can force a tiny predicted baseline
to match the known endpoints while scaling the point cloud incorrectly. Adding
a non-antipodal third view supplies a better-constrained camera configuration
and stabilizes the result in this run.

Consequently, the farthest-policy two-view arithmetic mean and its aggregate
Chamfer AUC must not be interpreted as typical reconstruction quality. Report
the median and alignment-failure rate alongside the mean, and treat collapsed
predicted baselines or implausible alignment scales as failed alignments before
using the AUC for policy comparison. Chamfer is also not expected to be
monotonic because VGGT reconstructs each complete view prefix independently.

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

The common evaluation reports F-scores at 1%, 2%, and 10% of the ground-truth
bounding-box diameter. The 10% threshold is the less strict diagnostic for
coarse reconstruction overlap; it complements rather than replaces the stricter
geometric-fidelity thresholds.

## Recovering saved 2DGS evaluation on CPU

The original 2DGS surface extraction requested RGB-only rendering and interpreted
its `render_median` output as depth. In the affected gsplat implementation, the
median output reads the last feature channel, so this back-projected blue
intensities. The corrected CUDA path requests `RGB+ED`. Training already used
`RGB+ED`; checkpoints do not need retraining for this fix.

For machines without CUDA, recover geometry from the existing artifacts:

```bash
.venv/bin/python scripts/reevaluate_gaussian_splatting_cpu.py --dry-run
.venv/bin/python scripts/reevaluate_gaussian_splatting_cpu.py
```

The default input is `outputs/gaussian_splatting_variant_comparison`; corrected
results go to `outputs/gaussian_splatting_cpu_recovery` with the same
category/object/views/variant/2dgs hierarchy. The original artifacts are retained.
No gsplat, extra dependencies, source RGB images, VGGT model, or reconstruction
prediction caches are required. Each input needs `checkpoint.pt`, `summary.json`,
`ground_truth.ply`, and the corresponding local `data/cache/visibility` entry.
Old `/content/...` paths in summaries are not used to locate inputs.

To recover a single history first:

```bash
.venv/bin/python scripts/reevaluate_gaussian_splatting_cpu.py \
  --object-id 02691156/165c4491d10067b3bd46d022fd7d80aa \
  --variant phase2_vggt --views 5
```

`--limit`, `--views`, and `--variant` restrict the batch. `--input-root`,
`--output-root`, and `--visibility-cache-root` override locations. Repeating a
command skips completed recoveries whose input hashes and renderer version match;
`--force` recomputes them. Incomplete outputs are recomputed. Each failure is
reported and the batch continues, returning a nonzero exit status if any failed.

The command regenerates surface and comparison PLYs, static and interactive
comparisons, per-history metrics, and summaries. `recovered_metrics.csv` at the
output root combines recovered per-history metrics, including variant labels;
policy averages should be calculated from these corrected rows. Original RGB
turntables and trained checkpoints remain in the source tree. Saved summaries
record that source and the CPU renderer identity. The dashboard still reads its
`GAUSSIAN_SPLATTING_ROOT` constant in `dashboard/app.py`; that path must point at
the recovery directory to display corrected metrics. Recovery does not change
the dashboard or combine corrected rows with the original zero-valued files.

CPU rendering follows gsplat v1.5.3's projection, screen-space filter, median
**Gaussian-center camera-z** depth, tile coverage, and alpha compositing rules.
It is a separate NumPy inference implementation, not a CPU build of gsplat.
Analytic tests cover projection and compositing, but CPU/CUDA parity is not
verified: floating-point differences and equal-depth ordering can differ.
The summary explicitly records `cuda_parity_verified: false`. Compare a sample
against the corrected CUDA path before mixing CPU and CUDA numbers in a report.
CPU recovery can take several minutes per checkpoint; a full batch may take hours.
