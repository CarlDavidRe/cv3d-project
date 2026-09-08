# Phase 2 implementation checks

These are NUM geometry and simulator checks. A frozen Phase 2 result requires
the learned policies and complete-test-split evaluation.
Meshes use bounding-box centering, supported by six independent validation
objects. Successful cache/replay checks establish internal consistency;
full-dataset RGB registration and exact source-OBJ equivalence are not claimed.

- `visibility_alignment/`: RGB/mesh comparisons using the NUM camera,
  including both pole views, native-resolution masks, boundary measurements,
  and diagnostic translations. The `validation/` subdirectory compares
  centered and scaled source coordinates on six independent validation objects. See
  `docs/num_camera_alignment.md` in the repository for the source analysis.
- `random_oracle/seed_0/`: Random and Oracle on three explicitly
  selected objects. Complete for that subset, with
  `geometry_validation_status: num_camera_bbox_centered` and
  `complete_fixed_split: false`. All six trajectories replay through the
  shared evaluator.
- `visibility_debug/`: projected face-centroid diagnostics from precomputation.

Reproduce the available-cache check in a new run directory:

```bash
python3 scripts/evaluate_closed_loop.py --skip-missing-caches \
  --set experiment.name=random_oracle_evaluation
```

As more caches finish, a new availability snapshot may contain more objects.
Use the saved manifest's evaluated IDs as explicit `object_ids` to repeat that
exact cohort. Decisions/scores/coverage reproduce; wall-clock timings do not.
