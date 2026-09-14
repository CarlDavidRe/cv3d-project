# Training runs and losses

This guide summarizes the training behind the policies in
[`policy_variants.md`](policy_variants.md) and the per-object Gaussian models
shown in the reconstruction dashboard. PUN training is intentionally out of
scope: the project imports its released checkpoint and does not train it.

## What is and is not trained

| Dashboard variant | Training used by this project |
|---|---|
| `phase2_random` | None; deterministic random baseline |
| `phase2_farthest` | None; camera-geometry baseline |
| `phase2_pun` | No local training; frozen released checkpoint |
| `phase2_vggt` | Phase 1 lightweight PSNR head; VGGT is frozen |
| `phase2_oracle` | None; true visibility gain is supplied at evaluation time |
| Four `phase3_*` variants | Phase 3 history head; VGGT is frozen |
| Every policy/view/backend cell | A fresh 2DGS or 3DGS model is fitted to that selected RGB history |

The policy-training loss and Gaussian-fitting loss are separate. Policy heads
learn which camera to select. Gaussian splats are optimized afterward to show
what can be reconstructed from the selected cameras; their loss never updates
or selects a policy.

## Shared policy-head objective

The locally trained Phase 1 and Phase 3 heads use the same two-part objective,
averaged only over valid candidate cameras:

```text
total loss = masked Huber loss + 0.1 * masked pairwise ranking loss
```

For prediction error `e = prediction - target`, the Huber term is

```text
0.5 * e^2                         when |e| <= delta
delta * (|e| - 0.5 * delta)       otherwise.
```

The ranking term considers every unordered pair of valid candidates whose
target values are not tied. It is the mean logistic loss

```text
softplus(-sign(target_i - target_j) * (prediction_i - prediction_j)).
```

Thus Huber fits the target values, while the ranking term explicitly rewards
the correct candidate ordering. The configured minimum target margin is zero,
so only exact target ties are omitted. Acquired or otherwise invalid cameras
are masked out of both terms. Validation `total_loss` selects `best.pt`, and
training stops after the configured patience without improvement. Regret,
Spearman correlation, and NDCG@5 are reported metrics, not additional loss
terms.

## Phase 1 run for `phase2_vggt`

The retained policy checkpoint comes from the `vggt_max_pooled_patch` entry in
the Phase 1 backbone sweep:

- run: `outputs/phase1/backbone_sweep/seed_1`;
- supervision: one RGB view mapped to its 48-value NUM PSNR map;
- data: 41,760 training, 5,232 validation, and 14,400 test samples from the
  object-disjoint NUM split;
- representation: final-layer, 2,048-dimensional max-pooled patch features
  from `facebook/VGGT-1B` at 518-pixel input size;
- trainable model: a 128-hidden-unit lightweight head with 272,560 parameters;
  VGGT remains frozen and cached;
- optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 256;
- schedule: at most 50 epochs, early-stopping patience 8, no learning-rate
  scheduler;
- loss: the shared objective above with Huber `delta = 1.0`, ranking weight
  `0.1`, and ranking margin `0.0`.

The retained seed-1 run completed all 50 epochs and selected epoch 50 by
validation total loss. The target is raw PSNR, so the head learns the raw PSNR
ordering even though lower PSNR is interpreted as greater uncertainty when the
policy chooses a view. The Phase 1 sweep selected this VGGT representation by
validation ranking behavior across regret, Spearman, and NDCG@5; the selected
checkpoint itself is still the minimum-total-loss epoch for that run.

Configuration and retained summary:
[`phase1_sweep.yaml`](../configs/experiments/phase1_sweep.yaml) and
[`summary.json`](../outputs/phase1/backbone_sweep/seed_1/variants/vggt_max_pooled_patch/summary.json).

## Phase 3 history runs

All four Phase 3 runs use the same object-disjoint direct-gain history task:

- 34,800 training, 4,360 validation, and 12,000 test histories;
- history lengths 1, 2, 4, 6, and 8, balanced within each split;
- target: the incremental `VisA` surface coverage obtained by adding each
  candidate camera to the acquired history;
- output: 48 direct gain predictions, where higher is better;
- frozen backbone: `facebook/VGGT-1B`, final layer, 2,048 feature channels;
- optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`;
- schedule: at most 80 epochs, patience 10, no learning-rate scheduler;
- loss: the shared objective with Huber `delta = 0.01`, ranking weight `0.1`,
  and ranking margin `0.0`;
- deterministic training seed 0 and effective batch size 64.

Only the head and variant-specific downstream layers are optimized. Frozen
VGGT features are precomputed and stored in resumable shards for joint runs;
the complete history is still processed live by VGGT during closed-loop policy
evaluation.

### `phase3_vggt_independent_history`

Each image is encoded independently, then its feature and camera direction are
masked-mean pooled. The trainable head has 272,950 parameters. Its physical and
effective batch size is 64. The retained run selected epoch 41 and stopped at
epoch 51 after ten non-improving epochs.

Config: [`phase3_independent.yaml`](../configs/experiments/phase3_independent.yaml).

### `phase3_vggt_joint_history`

VGGT encodes each complete history jointly before the same masked mean and
capacity-matched head. It also has 272,950 trainable parameters and uses batch
size 64. The retained run selected epoch 40 and stopped at epoch 50. This is
the controlled comparison with the independent run: target, loss, optimizer,
effective batch size, head capacity, and stopping rules match.

Config: [`phase3_joint.yaml`](../configs/experiments/phase3_joint.yaml).

### `phase3_vggt_joint_pose_deepsets`

The shared nonlinear element encoder added before pooling raises the trainable
count to 289,718 parameters. It uses batch size 64. The retained run selected
epoch 28 and stopped at epoch 38. Its optimizer, loss, effective batch size,
and patience match the controls, but its added capacity makes it an expressive
ablation.

Config:
[`phase3_joint_pose_deepsets.yaml`](../configs/experiments/phase3_joint_pose_deepsets.yaml).

### `phase3_vggt_joint_token_attention`

The spatial-token attention model has 350,209 trainable parameters. Because it
retains four tokens per view, it uses microbatches of 32 with two-step gradient
accumulation, preserving the effective batch size of 64. The retained run
selected epoch 48 and stopped at epoch 58. It uses the same objective and
optimizer settings as the other Phase 3 runs but is also an expressive
ablation.

Config:
[`phase3_joint_token_attention.yaml`](../configs/experiments/phase3_joint_token_attention.yaml).

The retained run metadata and validation losses are recorded under
`outputs/phase3/<run>/seed_0/metrics/summary.json`. Controlled test and
closed-loop evaluation is a separate checkpoint-consuming step; it performs no
additional policy training.

## Per-object 2DGS and 3DGS fitting

Every policy and view budget receives a fresh Gaussian model initialized from
the camera-aligned VGGT point map. Models are not continued from a smaller view
budget. With the default `1,500` iterations per acquired view, the budgets 1,
2, 3, 5, and 10 run for 1,500, 3,000, 4,500, 7,500, and 15,000 optimizer steps,
respectively. Training cycles evenly through the acquired images, giving each
camera approximately 1,500 updates.

Both backends use Adam. Gaussian positions use learning rate `2e-4`; scale,
rotation, opacity, and view-independent color parameters use `1e-2`. At each
step they optimize one 256-by-256 acquired image. Their shared image loss is

```text
RGB L1 + 0.1 * foreground-opacity L1.
```

2DGS additionally uses the rasterizer's depth-distortion term and a normal
consistency term over pixels with alpha greater than `0.05`:

```text
2DGS loss = RGB L1
           + 0.1  * foreground-opacity L1
           + 0.01 * mean depth distortion
           + 0.05 * mean (1 - |cos(rendered normal, surface normal)|).
```

3DGS has no distortion or normal term in this implementation:

```text
3DGS loss = RGB L1 + 0.1 * foreground-opacity L1.
```

There is no SSIM term. Chamfer-L1, accuracy, completeness, precision, recall,
and F-score are post-training geometric evaluation metrics for 2DGS, not
optimization losses. 3DGS is used for qualitative appearance rendering and
does not produce the reported surface metrics.

Implementation:
[`gaussian_splatting.py`](../src/nbv/eval/gaussian_splatting.py).

