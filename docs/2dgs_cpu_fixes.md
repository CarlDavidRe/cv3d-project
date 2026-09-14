# Gaussian CPU fixes: depth extraction and placement

Both scripts read the original trained artifacts from
`outputs/gaussian_splatting_variant_comparison`. They preserve those artifacts
and existing caches; neither requires CUDA, VGGT inference, or Gaussian retraining.

| Script | Problem addressed | What changes |
| --- | --- | --- |
| `scripts/reevaluate_gaussian_splatting_cpu.py` | The affected original extraction interpreted the blue image channel as depth. | Extracts correct depth and surface geometry from the saved checkpoint, keeping Gaussian placement unchanged. |
| `scripts/repair_gaussian_alignment_cpu.py` | Inconsistent VGGT camera predictions can leave the initial geometry misplaced. | Fits one rotation, translation, and scale using acquired RGB silhouettes and known cameras. For 2DGS it performs corrected depth extraction and evaluation; for 3DGS it regenerates RGB comparisons. |

**For 2DGS, both recalculate all geometric metrics:** accuracy, completeness, Chamfer,
precision, recall, and F-score at the configured thresholds. Both regenerate
surface point clouds and static/interactive ground-truth comparisons.

## Which script to run

Run from the repository with its Python environment:

```bash
# Correct evaluation of the original trained checkpoint
python scripts/reevaluate_gaussian_splatting_cpu.py

# Repair both backends: 2DGS geometry evaluation and 3DGS RGB comparisons
python scripts/repair_gaussian_alignment_cpu.py

# Or repair only 3DGS (use --backend 2dgs for geometry only)
python scripts/repair_gaussian_alignment_cpu.py --backend 3dgs
```

**There is no required order.** The repair script calls the corrected CPU
extraction/evaluation routine internally, so reevaluation is not a prerequisite.
For a before/after comparison, run both against the original artifacts in either
order. Do not use the reevaluation output directory as the repair input.

Both commands support `--object-id`, `--variant`, `--views`, `--limit`, and
`--dry-run` to restrict or inspect a batch. Completed matching outputs are reused.

## Inputs and outputs

Both need the original `checkpoint.pt`, `summary.json`, `ground_truth.ply`, and
existing visibility cache. The repair additionally needs the acquired NUM RGB
images. Existing VGGT prediction caches are optional for its camera diagnostics.
For 3DGS, `ground_truth.ply` is not required; all 48 RGB references are needed
to build the comparison gallery, but only acquired views participate in fitting.
3DGS remains qualitative and does not receive geometry metrics.

Default output directories are separate:

```text
outputs/gaussian_splatting_cpu_recovery/      # Original placement, corrected extraction
outputs/gaussian_splatting_alignment_repair/  # Refined placement, corrected extraction
```

Placement repair cannot restore missing or distorted geometry and skips
one-view histories because depth/scale are ambiguous. Ground truth is used
only for evaluation, not fitting. Repaired results use a separate
silhouette-refined evaluation procedure: apply it consistently across policies
and keep its scores separate from original-placement scores. CPU/CUDA renderer
parity has not been verified.

New Gaussian training runs interpret `--iterations` as iterations per acquired
view, so every view receives the same optimization budget. Placement candidates
must improve mean rendered silhouette IoU by at least 0.01 before they are
accepted; configure this guard with `--min-iou-improvement`. Existing checkpoints
must be retrained and existing repairs rerun to benefit from these changes.

See [placement repair details and dashboard setup](gaussian_alignment_cpu_repair.md)
and [CPU surface recovery details](reconstruction_evaluation.md#recovering-saved-2dgs-evaluation-on-cpu).
