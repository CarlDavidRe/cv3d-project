# Phase 3 joint frozen-VGGT control

Step 17 adds a joint-history path without changing the Phase 1 single-image
extractor or the Step 16 independent control. `VGGTJointExtractor` sends each
complete acquired history to the frozen VGGT aggregator as one tensor with
shape `[B, H, 3, 518, 518]`. The selected final-layer max-pooled patch vector
for each view is therefore conditioned on the other views before history
pooling.

VGGT has no observation-padding-mask input. A mixed-length minibatch is grouped
by its real history length, and each group is forwarded without padded images.
The resulting vectors are restored to the padded batch layout with exact zeros
at padded positions. Suffix-padding validation prevents a malformed batch from
silently dropping real views.

## Matched control

`configs/experiments/phase3_joint.yaml` points to
`phase3_independent.yaml`. Before loading VGGT, the runner verifies that both
controls use the same:

- Step 15 history manifest, direct `target_surface_gain`, candidate masks, and
  `vis_a` definition;
- VGGT model ID, input size, final layer, and max-pooled-patch feature selection;
- 2,048-value feature width and optional fixed canonical camera directions;
- 128-unit lightweight head, dropout, AdamW settings, Huber/ranking loss,
  epoch budget, patience, and seed;
- effective batch size of 64. The joint run uses one history per microbatch and
  64-step gradient accumulation to bound backbone memory.

The Step 16 feature-cache metadata does not contain a VGGT weight-file digest.
The runner therefore verifies the recorded `facebook/VGGT-1B` model ID,
preprocessing size, layer, component, cache checksum, and feature width, and
records that limitation rather than claiming weight-level identity.

The joint and independent trainable heads each contain **272,950 parameters**.
The VGGT aggregator is frozen, excluded from optimizer traversal, and counted
separately in each completed run's summary.

## Memory preflight

Before optimization, the joint runner forwards one real history of lengths 1
and 2. On CUDA it records peak allocated and reserved bytes, including the
resident frozen backbone and prediction head; on CPU those accelerator-memory
fields are explicitly `null`. Results are written to
`metrics/memory_preflight.json`. Adjusting the configured preflight lengths is
allowed for hardware bring-up, but longer experiments should start only after
the short-history measurements succeed.

Run the control with:

```bash
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint.yaml
```

The command trains on the train histories, selects the checkpoint by validation
loss, and saves validation diagnostics. It intentionally does not run held-out
test comparison or closed-loop evaluation; those belong to Step 18. Joint
features are always computed live from the complete history. The independent
feature cache is opened only to validate backbone and control provenance and is
never substituted for a joint forward.
