# Phase 3 joint-processing follow-up variants

The retained capacity-matched Phase 3 experiment found that joint frozen-VGGT
processing reduced Huber error but did not improve candidate ranking or
closed-loop coverage. These two follow-up variants test whether that result is
caused by the downstream representation bottleneck rather than by an absence
of useful cross-view information in VGGT.

They use the same object-disjoint history dataset, direct `VisA` surface-gain
targets, frozen VGGT checkpoint, final backbone layer, optimizer, loss,
validation selection, seed, and effective training batch size as the retained
controls. They deliberately add trainable structure after VGGT and are
therefore expressive ablations, not capacity-matched replacements for the
original H4 result.

## Variant 1: pose-conditioned DeepSets

The original model concatenates each pooled view feature with its camera
direction and immediately averages across views. Algebraically this exposes
the head only to the mean feature and mean direction, losing the association
between a particular appearance and the pose from which it was observed.

The `pose_deepsets` model instead applies a shared nonlinear element encoder to
each pair before pooling:

```text
joint VGGT history -> per-view max-pooled patch vector z_i
                  -> element MLP phi([z_i, direction_i])
                  -> padding-aware mean over phi outputs
                  -> lightweight 48-gain head
```

This is the lowest-cost test of the feature/pose-association hypothesis. Joint
VGGT vectors retain the existing `[H, 2048]` cache contract.

Run it with:

```bash
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint_pose_deepsets.yaml
```

## Variant 2: token-preserving candidate attention

The original `max_pooled_patch` representation collapses all spatial VGGT
patches to one vector per view. The `token_candidate_attention` model retains a
small spatial grid, adds the corresponding acquired-view pose to every token,
and scores candidates using their canonical camera directions as queries:

```text
joint VGGT history -> final-layer spatial patch grid
                  -> adaptive 2x2 grid (four tokens per view)
                  -> token projection + acquired-pose embedding
candidate direction -> query projection
queries x observed tokens -> multi-head cross-attention
                         -> shared scalar score head -> 48 gains
```

The 2x2 reduction is intentional: it preserves coarse spatial layout while
bounding the full-data feature cache to four times the pooled baseline. Change
`model.token_grid_size` only as a recorded memory/accuracy ablation.

Run it with:

```bash
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint_token_attention.yaml
```

Both runs support `--resume` and use architecture-specific cache identities,
so baseline, DeepSets, and token shards cannot be mixed.

## Held-out and closed-loop evaluation

After training, substitute one variant checkpoint into the established Step 18
pipeline. The command computes and records its SHA-256 digest and writes to a
separate experiment directory:

```bash
python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml \
  --joint-checkpoint outputs/phase3/joint_history_pose_deepsets/seed_0/checkpoints/best.pt \
  --experiment-name controlled_pose_deepsets

python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml \
  --joint-checkpoint outputs/phase3/joint_history_token_attention/seed_0/checkpoints/best.pt \
  --experiment-name controlled_token_attention
```

The resulting summaries label these as `expressive_joint_variant`, report the
independent and joint parameter counts separately, and preserve the same fixed
test histories and closed-loop evaluator. Length-one vector equivalence is
still checked for DeepSets. It is recorded as not shape-comparable for the
token model because four spatial tokens intentionally replace one max-pooled
vector; model ID, preprocessing, layer, and data provenance remain checked.

Interpret the variants in order:

1. Improvement from DeepSets supports the feature/pose-association explanation.
2. Additional improvement from token attention supports the spatial-bottleneck
   explanation.
3. No improvement from either strengthens the case that frozen VGGT's
   contextual representation is not aligned with direct `VisA` gain under the
   current data and loss.

Report fixed-history metrics by history length, closed-loop coverage AUC and
final coverage, policy latency, peak memory, and trainable parameters. Do not
replace the retained capacity-matched H4 conclusion with an expressive-variant
result.
