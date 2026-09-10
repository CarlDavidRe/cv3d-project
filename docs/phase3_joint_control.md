# Phase 3 joint frozen-VGGT control

Step 17 adds a joint-history path without changing the Phase 1 single-image
extractor or the Step 16 independent control. `VGGTJointExtractor` sends each
complete acquired history to the frozen VGGT aggregator as one tensor with
shape `[B, H, 3, 518, 518]`. The selected final-layer max-pooled patch vector
for each view is therefore conditioned on the other views before history
pooling.

VGGT has no observation-padding-mask input. Feature precomputation batches only
histories of the same real length and forwards them without padded images. The
resulting joint-conditioned vectors are restored to dataset order, cached in
atomic resumable disk shards, and retained in system RAM for head training.
Suffix-padding validation prevents a malformed batch from silently dropping
real views.

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
- effective batch size of 64. After one frozen VGGT pass per complete history,
  the joint run trains the small head directly with batches of 64.

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

The command first computes each train and validation history's joint features
once, trains on the in-memory vectors, selects the checkpoint by validation
loss, and saves validation diagnostics. The checked-in A100 configuration uses
four equal-length histories per frozen-backbone call, four image-loading
workers, pinned transfers, and automatic CUDA OOM backoff to smaller batches.
The retained vectors require roughly 0.6--1.2 GB for the current dataset,
depending on autocast dtype. It intentionally does not run held-out test
comparison or closed-loop evaluation; those belong to Step 18. Every joint
feature is still computed from its complete history. The independent feature
cache is opened only to validate backbone and control provenance and is never
substituted for a joint forward.

The joint runner also checkpoints the trainable head, optimizer, validation
selection, early-stopping state, partial epoch, shuffle order, and RNG state.
The complete Drive sync, restore, and `--resume` commands are in the Phase 3
section of the repository README.
