# Independent VGGT closed-loop adapter

Phase 2 uses the `vggt_max_pooled_patch` entry from the completed seed-1
Phase 1 sweep. It was selected without Phase 2 test coverage: among VGGT
variants it has the best validation normalized regret (0.1998), Spearman
(0.5363), and NDCG@5 (0.8162). The resolved config pins the Phase 1 config,
summary, head checkpoint, and test feature cache by SHA-256.

Each cache row was extracted from one RGB image passed to VGGT as a one-frame
sequence. At rollout time the policy looks up only acquired sample IDs. It
runs the saved Phase 1 head on a newly encountered feature once, retains the
resulting 48-value source-relative PSNR map in memory, and reuses earlier
maps. A cache miss fails by default. `allow_live_extraction: true` explicitly
enables lazy VGGT loading; each missing RGB is still processed independently.

The raw predictions preserve the lower-is-more-uncertain PSNR semantics and
the source-relative anchor-zero map frame. The adapter then calls the same
Step 12 functions as official PUN: 30-degree exponential alignment, per-map
`small` suppression, raw-map product, and negation so the common evaluator's
argmax implements PSNR-product minimization. These aggregate values are
policy scores, not predicted surface gains.

Acquired-only raw maps and whether each came from the feature cache or live
VGGT are saved under `prediction_maps/vggt/`. Rollouts separately store the
aggregate scores, selected anchors, and evaluator-only geometric gains. The
policy never receives NUM labels, unacquired RGB, face visibility, coverage,
or true gains.

The controlled tests compare cached and live features through the same head,
verify incremental inference, and reject implicit live extraction. The
retained real-data five-policy smoke artifact is
`outputs/phase2/five_policy_smoke/seed_0`; it runs Random, Farthest, official
PUN, independent VGGT, and Oracle through the common evaluator. It is a
one-object implementation check, not a final policy result.
