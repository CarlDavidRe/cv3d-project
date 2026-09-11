# Frozen-feature next-best-view study

This repository studies next-best-view (NBV) selection for 3D object coverage.
It asks whether frozen visual and geometry-aware representations—especially
VGGT—can predict useful camera views without maintaining an explicit learned 3D
uncertainty model.

The implementation follows the research plan in
[`project_overview.md`](project_overview.md):

- **Phase 1 — complete:** predict PUN/NUM single-image uncertainty maps with
  frozen ImageNet ViT, DINOv2, and VGGT features.
- **Phase 2 — complete:** Random, Farthest, PUN, VGGT, and Oracle were evaluated
  on the complete 300-object test split in a deterministic closed loop using
  mesh-face coverage.
- **Phase 3 — complete:** the complete 1,279-object,
  51,160-history direct-gain dataset is generated, and the cache-backed
  independent and capacity-matched joint frozen-VGGT controls have retained
  validation-selected checkpoints. The controlled experiment evaluated both
  models on all 12,000 held-out histories and all 300 test objects through the
  unchanged evaluator. Joint processing did not improve a clear majority of
  the prespecified outcomes.

## Quick start

Python 3.10 or newer is required.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Initialize a reproducible run and verify the installation:

```bash
python3 scripts/init_experiment.py \
  --config configs/experiments/phase1.yaml
```

Run artifacts are written to
`outputs/<phase>/<experiment>/seed_<n>/`. Reusable model, feature, and geometry
caches are stored under `data/cache/`.

## Data setup

Two datasets are required:

1. The [PUN NUM dataset](https://github.com/ZhangLab-DeepNeuroCogLab/PUN) for
   RGB observations and single-image targets.
2. The gated
   [ShapeNetCore v2 dataset](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
   for Phase 2 and Phase 3 mesh visibility.

Place them in the following layout:

```text
data/
├── NUM/<category_id>/<object_id>/
│   ├── images/viewpoint_<anchor>_offset_phi_0.png
│   └── uncertainties/viewpoint_<anchor>_offset_phi_0.json
└── ShapeNetCore.v2/<category_id>/<object_id>/models/
    └── model_normalized.ply
```

The fixed object-disjoint split is checked in at
[`data/splits/num_v1.json`](data/splits/num_v1.json). To verify that the local
NUM dataset matches it, run:

```bash
python3 scripts/prepare_num_split.py \
  --data-root data/NUM \
  --output data/splits/num_v1.json
```

### Download the required ShapeNet categories

After ShapeNet grants access, authenticate with the Hugging Face CLI:

```bash
python3 -m pip install --upgrade huggingface_hub
hf auth login
```

The NUM split uses 13 ShapeNet synsets. Download only those archives:

```bash
mkdir -p data/downloads/shapenetcore
hf download ShapeNet/ShapeNetCore \
  02691156.zip 02828884.zip 02933112.zip 02958343.zip \
  03001627.zip 03211117.zip 03636649.zip 03691459.zip \
  04090263.zip 04256520.zip 04379243.zip 04401088.zip \
  04530566.zip \
  --repo-type dataset \
  --local-dir data/downloads/shapenetcore

mkdir -p data/ShapeNetCore.v2
for archive in data/downloads/shapenetcore/*.zip; do
  unzip -q -n "$archive" -d data/ShapeNetCore.v2
done
```

The default geometry configuration expects `model_normalized.ply`. If the
download contains OBJ meshes, set
`phase2.visibility.mesh_relative_path=models/model_normalized.obj` when
precomputing visibility.

## Phase 1: single-image feature probes

Phase 1 predicts a 48-anchor NUM uncertainty map from one RGB image. All frozen
backbones use the same lightweight prediction head and object-disjoint split.
The comparison also includes a training-set mean map, a coarse-RGB MLP, and the
official pretrained PUN/UPNet checkpoint.

Run a small overfit check first:

```bash
python3 scripts/train_probe.py \
  --config configs/experiments/phase1_probe_tiny.yaml
```

Then run the configured comparison:

```bash
python3 scripts/run_phase1.py \
  --config configs/experiments/phase1_sweep.yaml
```

The first run downloads model weights and extracts features. Compatible
features in `data/cache/features/` and complete experiment entries are reused
automatically. VGGT is a 1B-parameter model and should be run on a CUDA GPU.

Useful smoke tests:

```bash
python3 scripts/inspect_features.py \
  --backbone imagenet_vit --data-root data/NUM --device cpu

python3 scripts/inspect_features.py \
  --backbone dinov2 --data-root data/NUM --device auto

python3 scripts/inspect_pun.py \
  --data-root data/NUM --device auto
```

To use VGGT, install its official package in the active environment:

```bash
git clone https://github.com/facebookresearch/vggt.git \
  data/cache/sources/vggt
python3 -m pip install -e data/cache/sources/vggt
python3 scripts/inspect_features.py \
  --backbone vggt --data-root data/NUM --device cuda
```

Visualize predictions from a completed experiment:

```bash
python3 scripts/visualize_phase1.py \
  outputs/phase1/backbone_sweep/seed_0
```

Phase 1 writes checkpoints, per-sample metrics, comparison tables, training
curves, and prediction SVGs under the experiment directory.

## Phase 2: closed-loop NBV evaluation

Phase 2 uses rasterized ShapeNet faces as ground-truth geometry. Each cache
stores the visible face set for all 48 camera anchors and supports two coverage
targets:

- `vis`: fraction of mesh faces observed;
- `vis_a`: fraction of mesh surface area observed (the default).

The calibrated NUM camera model and mesh normalization are defined in
[`configs/experiments/phase2_visibility.yaml`](configs/experiments/phase2_visibility.yaml).
See [`docs/num_camera_alignment.md`](docs/num_camera_alignment.md) for the
validation protocol and limitations.

### 1. Precompute visibility

Test one object before launching a full split:

```bash
python3 scripts/precompute_visibility.py \
  --object 02691156/10155655850468db78d106ce0a280f87 \
  --debug-anchor 0
```

Then build the required caches:

```bash
python3 scripts/precompute_visibility.py --split test
```

Compatible caches are skipped. Metadata or mesh-checksum mismatches fail
instead of silently rebuilding; use `--overwrite` only when replacement is
intentional.

### 2. Run the policies

For a quick subset evaluation:

```bash
python3 scripts/evaluate_closed_loop.py \
  --limit 10 \
  --skip-missing-caches \
  --set experiment.name=phase2_subset
```

To reproduce the reportable full test-split evaluation:

```bash
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml
```

The full run evaluates Random, angular-distance Farthest, official-checkpoint
PUN, independently processed VGGT, and one-step Oracle. The default budget is
10 total views, including initial anchor 0. Do not use `--limit`, `--object`,
or `--skip-missing-caches` for the final result.

The retained full run is under
[`outputs/phase2/phase2_closed_loop/seed_0/`](outputs/phase2/phase2_closed_loop/seed_0/).
It contains 1,500 replay-verified rollouts: five policies for each of the 300
fixed test objects. Its
[`phase2_completion.json`](outputs/phase2/phase2_closed_loop/seed_0/metrics/phase2_completion.json)
records `status: complete` with every required completion check passing.

Each run records configuration and cache provenance, per-step and per-object
metrics, replayable policy rollouts, resource profiling, and SVG summaries.
New schema-2 rollout and summary exports include both absolute `VisA` and a
companion reachable-normalized value. The latter divides each object's coverage
by the union visible from all available, non-invalid anchors before averaging
across objects; it is zero for a degenerate object with no reachable surface.
Retained schema-1 rollouts remain readable and expose this new metric as
unavailable.
`metrics/phase2_completion.json` is the machine-readable completeness gate.

Replay a saved rollout with:

```bash
python3 scripts/evaluate_closed_loop.py \
  --replay outputs/phase2/RUN/seed_0/rollouts/POLICY/CATEGORY/OBJECT.npz
```

Render it with:

```bash
python3 scripts/visualize_phase2.py \
  outputs/phase2/RUN/seed_0/rollouts/POLICY/CATEGORY/OBJECT.npz
```

The learned-policy adaptations and leakage boundaries are documented in the
[`PUN adapter note`](docs/pun_closed_loop_adapter.md) and
[`VGGT adapter note`](docs/vggt_closed_loop_adapter.md).

## Phase 3: direct surface-gain histories

Phase 3 reuses the same visibility caches and `vis_a` evaluator target to
create supervised, variable-length observation histories. Precompute all three
splits, then build the dataset:

```bash
python3 scripts/precompute_visibility.py --split train
python3 scripts/precompute_visibility.py --split val
python3 scripts/precompute_visibility.py --split test
python3 scripts/build_history_dataset.py
```

The result is written to
`data/processed/histories/random_unique_v1/`. Existing datasets are preserved
unless `--overwrite` is passed. For a small validation build:

```bash
python3 scripts/build_history_dataset.py \
  --split train \
  --limit 2 \
  --set phase3.histories.dataset_name=random_unique_smoke
```

Load the generated histories with:

```python
from torch.utils.data import DataLoader
from nbv.data import HistoryDataset, collate_history_samples

dataset = HistoryDataset(
    "data/processed/histories/random_unique_v1",
    split="train",
    load_images=True,
)
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_history_samples,
)
batch = next(iter(loader))
```

Each sample contains an ordered, duplicate-free anchor history, RGB paths,
48 direct surface-gain targets, a valid-candidate mask, split identity, cache
fingerprint, and deterministic sampling metadata.

Train the Step 16 independent-history control after all three history shards
have been generated:

```bash
python3 scripts/train_history.py \
  --config configs/experiments/phase3_independent.yaml
```

The default control reads the existing per-image
`vggt_max_pooled_patch` caches. It never feeds more than one observation per
VGGT sequence: real views are independently extracted (or loaded), combined
with a padding-aware permutation-invariant mean, and mapped to 48 direct
surface-gain predictions. The run saves the validation-selected checkpoint,
epoch diagnostics, validation/test one-step metrics, and a replay-verified
validation-object closed-loop smoke rollout. It also emits the same automatic
reporting artifacts as the Phase 1 sweep: per-model loss and validation-metric
SVGs, a validation-loss comparison SVG, and comparison tables in CSV, JSON,
and Markdown formats. Progress is logged after the initial baseline evaluation
and every completed epoch. The Phase 2 NUM head is not used.
The retained full-data Step 16 validation artifact is under
`outputs/phase3/independent_history_validation/seed_0`; it uses a deliberately
bounded one-epoch budget to validate the workflow on CPU. It is not the final
full-budget Phase 3 research result.

Train the Step 17 joint frozen-VGGT control on the identical history dataset:

```bash
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint.yaml
```

Each complete history is sent through frozen VGGT once. Equal-length histories
are batched together so padded views never enter the backbone, and the joint
features are retained in system RAM while the small head trains. The config
matches the Step 16 supervision, optimizer, loss, training budget, effective
batch size, and 272,950-parameter head, and performs a length-1/2
accelerator-memory preflight before precomputation. See
[`docs/phase3_joint_control.md`](docs/phase3_joint_control.md). The retained
real A100 run selected epoch 40 and stopped at epoch 50; its validation Huber,
regret, Spearman, and NDCG@5 are 0.008017, 0.162318, 0.727445, and 0.856920.

Two expressive follow-up variants test the pooling bottlenecks exposed by the
retained result:

```bash
# Preserve each joint feature's association with its acquired camera pose.
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint_pose_deepsets.yaml

# Retain a 2x2 spatial token grid and score candidate-direction queries by
# cross-attention over all acquired-view tokens.
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint_token_attention.yaml
```

Both variants reuse the Step 17 frozen-feature precompute, atomic shard, and
head-training resume path. They intentionally have more trainable downstream
capacity than the original control and are reported as expressive follow-up
ablations, not replacements for the capacity-matched H4 comparison. Their
architectures, rationale, evaluation commands, and interpretation rules are in
[`docs/phase3_joint_variants.md`](docs/phase3_joint_variants.md). The code and
configs are checked in; full real-VGGT GPU results for these follow-ups are not
yet retained in this repository.

Run the Step 18 paired test-history and closed-loop comparison on the same
VGGT-capable environment:

```bash
python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml
```

To evaluate either trained follow-up on the identical fixed test histories and
closed-loop cohort, supply its checkpoint and a distinct output name:

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

This override computes and records the checkpoint digest, switches Step 18 to
`expressive_joint_variant` validation, assigns an architecture-specific policy
name, and reports both trainable parameter counts separately.

If the runtime disconnects, restore the partial run and pass `--resume`. The
runner checksum-verifies both validation-selected checkpoints and the
independent test cache, restores verified joint-test shards, replays saved
rollouts, and writes `metrics/phase3_completion.json` only after reporting.
See [`docs/phase3_controlled_experiment.md`](docs/phase3_controlled_experiment.md).

The retained complete run is under
[`outputs/phase3/controlled_history_comparison/seed_0/`](outputs/phase3/controlled_history_comparison/seed_0/).
Its completion gate passes every check for both models on 12,000 fixed test
histories and 300 ten-view closed-loop rollouts per policy:

| Evaluation | Metric | Independent history | Joint history |
|---|---|---:|---:|
| One-step | Huber loss ↓ | 0.008110 | **0.007683** |
| One-step | Normalized regret ↓ | **0.175766** | 0.198544 |
| One-step | Spearman ↑ | **0.706572** | 0.661221 |
| One-step | NDCG@5 ↑ | **0.848879** | 0.826619 |
| Closed loop | Coverage AUC ↑ | **6.783379** | 6.753729 |
| Closed loop | Final `VisA` coverage ↑ | **0.864034** | 0.862319 |
| Closed loop | Median policy time ↓ | **1.563 ms** | 242.346 ms |
| Closed loop | Peak CUDA allocation ↓ | **3.649 GB** | 4.497 GB |

Thus joint processing improves regression error alone; the independent control
has better ranking and closed-loop outcomes. The recorded descriptive
conclusion is `does_not_support_h4`; it is not a statistical-significance
claim. Closed-loop timing also reflects the intended deployment paths:
independent scoring uses cached per-image features, whereas joint scoring runs
live VGGT on the complete history.

### Plot all closed-loop policies together

Generate one coverage figure containing the five Phase 2 policies and both
Phase 3 history policies:

```bash
python3 scripts/plot_all_policy_coverage.py
```

The SVG is written to
[`outputs/all_policy_comparison/coverage_curves.svg`](outputs/all_policy_comparison/coverage_curves.svg).
Solid curves identify the Phase 2 policies (Random, Farthest, official PUN,
independent VGGT, and Oracle); dashed curves identify the Phase 3 independent-
and joint-history policies.

The script always requires the completed Phase 2 `coverage.csv`. It includes
Phase 3 curves only when
`outputs/phase3/controlled_history_comparison/seed_0/metrics/phase3_completion.json`
records `status: complete`. Before that point, the SVG labels both Phase 3
entries as pending instead of mixing partial or smoke-run results into the
300-object comparison. The retained run passes this gate, so the checked-in SVG
contains all seven curves. Different run locations or an output filename can
be supplied explicitly:

```bash
python3 scripts/plot_all_policy_coverage.py \
  --phase2-run outputs/phase2/phase2_closed_loop/seed_0 \
  --phase3-run outputs/phase3/controlled_history_comparison/seed_0 \
  --output outputs/all_policy_comparison/coverage_curves.svg
```

The plotter rejects mismatched coverage targets and object-cohort sizes.

### Back up and resume the Step 18 controlled run

The Step 17 `JOINT_RUN` backup does not include Step 18. The controlled run has
its own output directory containing resumable joint test-feature shards,
completed per-object rollouts, metrics, figures, logs, and resolved config.
Before starting Step 18 in Colab, start a five-minute background sync to the
mounted Drive:

```bash
cd /content/cv3d-project
CONTROLLED_RUN=outputs/phase3/controlled_history_comparison/seed_0
CONTROLLED_DRIVE=/content/drive/MyDrive/cv3d-project/outputs/phase3/controlled_history_comparison/seed_0

mkdir -p "$CONTROLLED_RUN" "$CONTROLLED_DRIVE"
(
  while true; do
    rsync -av --exclude='*.tmp' "$CONTROLLED_RUN/" "$CONTROLLED_DRIVE/"
    sleep 300
  done
) > /tmp/cv3d-phase3-controlled-sync.log 2>&1 &
echo $! > /tmp/cv3d-phase3-controlled-sync.pid
```

Before intentionally shutting down the runtime, interrupt evaluation with
`Ctrl-C`, force a final sync, and stop the background process:

```bash
cd /content/cv3d-project
CONTROLLED_RUN=outputs/phase3/controlled_history_comparison/seed_0
CONTROLLED_DRIVE=/content/drive/MyDrive/cv3d-project/outputs/phase3/controlled_history_comparison/seed_0

mkdir -p "$CONTROLLED_DRIVE"
rsync -av --exclude='*.tmp' "$CONTROLLED_RUN/" "$CONTROLLED_DRIVE/"

if [ -f /tmp/cv3d-phase3-controlled-sync.pid ]; then
  kill "$(cat /tmp/cv3d-phase3-controlled-sync.pid)" 2>/dev/null || true
fi
```

In a fresh runtime, mount Drive and prepare the repository, then restore the
controlled run. Restart the background sync with the first command block above
before resuming evaluation:

```bash
cd /content/cv3d-project
CONTROLLED_RUN=outputs/phase3/controlled_history_comparison/seed_0
CONTROLLED_DRIVE=/content/drive/MyDrive/cv3d-project/outputs/phase3/controlled_history_comparison/seed_0

mkdir -p "$CONTROLLED_RUN"
rsync -av --exclude='*.tmp' "$CONTROLLED_DRIVE/" "$CONTROLLED_RUN/"
find "$CONTROLLED_RUN" -type f \
  \( -name 'shard_*.pt' -o -name '*.npz' -o -name 'phase3_completion.json' \) \
  -print

python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml \
  --resume
```

`--resume` validates the saved config and each test-feature shard, restores
completed shards, and computes only missing test histories. It also validates
and replays existing rollout files, so closed-loop evaluation resumes at
object granularity. Use matching paths when overriding `experiment.name` or
the seed.

### Resume and sync the joint Phase 3 training run

Only the joint control writes a resumable intermediate state. With the default
configuration it atomically updates
`outputs/phase3/joint_history_control/seed_0/checkpoints/training_state.pt`
after every 64 full head-training batches, after the optimization part of every
epoch, and after each completed epoch. Frozen joint features are also written
atomically in resumable 64-precompute-batch shards under
`joint_feature_cache/{train,val}/`. The training state contains the current
head, optimizer, best-validation head, early-stopping counters, completed
history, partial-epoch loss counters, deterministic shuffle state, and random
number generator state. `best.pt` remains the final inference checkpoint and
is not used for training resume.

For a new Colab run, start a five-minute background sync to the mounted Drive
before starting training. These commands use the default experiment name; use
matching paths if `experiment.name` is overridden:

```bash
cd /content/cv3d-project
JOINT_RUN=outputs/phase3/joint_history_control/seed_0
JOINT_DRIVE=/content/drive/MyDrive/cv3d-project/outputs/phase3/joint_history_control/seed_0

mkdir -p "$JOINT_RUN" "$JOINT_DRIVE"
(
  while true; do
    rsync -av --exclude='*.tmp' "$JOINT_RUN/" "$JOINT_DRIVE/"
    sleep 300
  done
) > /tmp/cv3d-phase3-joint-sync.log 2>&1 &
echo $! > /tmp/cv3d-phase3-joint-sync.pid
```

Start a new training run without `--resume`:

```bash
cd /content/cv3d-project
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint.yaml
```

Before intentionally shutting down the runtime, interrupt training with
`Ctrl-C`, force one final sync, and stop the background sync process:

```bash
cd /content/cv3d-project
JOINT_RUN=outputs/phase3/joint_history_control/seed_0
JOINT_DRIVE=/content/drive/MyDrive/cv3d-project/outputs/phase3/joint_history_control/seed_0

mkdir -p "$JOINT_DRIVE"
rsync -av --exclude='*.tmp' "$JOINT_RUN/" "$JOINT_DRIVE/"

if [ -f /tmp/cv3d-phase3-joint-sync.pid ]; then
  kill "$(cat /tmp/cv3d-phase3-joint-sync.pid)" 2>/dev/null || true
fi
```

In a fresh runtime, first mount Drive and prepare the repository as described
below. Then restore the joint run and list its resumable training state and/or
feature shards. A runtime interrupted during feature precompute may have shards
before `training_state.pt` exists:

```bash
cd /content/cv3d-project
JOINT_RUN=outputs/phase3/joint_history_control/seed_0
JOINT_DRIVE=/content/drive/MyDrive/cv3d-project/outputs/phase3/joint_history_control/seed_0

mkdir -p "$JOINT_RUN"
rsync -av --exclude='*.tmp' "$JOINT_DRIVE/" "$JOINT_RUN/"
find "$JOINT_RUN" -type f \
  \( -name 'training_state.pt' -o -name 'shard_*.pt' \) \
  -print
```

Restart the background sync using the first command block, then resume with
the same config and the same `--set` overrides used for the original run:

```bash
cd /content/cv3d-project
python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint.yaml \
  --resume
```

Resume validates the full experiment identity and training configuration
before loading state. Running without `--resume` refuses to overwrite an
existing resumable joint state; use a new `experiment.name` for a genuinely
new run. An unexpected runtime loss can repeat only the joint-feature
precompute or head-training batches since the most recent atomic
shard/checkpoint.

## Configuration and reproducibility

Primary experiment configurations live in
[`configs/experiments/`](configs/experiments/):

| Configuration | Purpose |
|---|---|
| `phase1.yaml` | installation/infrastructure check |
| `phase1_probe_tiny.yaml` | probe-head overfit check |
| `phase1_sweep.yaml` | complete frozen-feature comparison |
| `phase2_visibility.yaml` | camera model and visibility cache |
| `phase2_closed_loop.yaml` | closed-loop policy evaluation |
| `phase3_histories.yaml` | surface-gain history dataset |
| `phase3_independent.yaml` | independent history direct-gain control |
| `phase3_joint.yaml` | joint frozen-VGGT history direct-gain control |
| `phase3_joint_pose_deepsets.yaml` | pose-conditioned joint follow-up |
| `phase3_joint_token_attention.yaml` | spatial-token candidate-attention follow-up |
| `phase3_controlled.yaml` | paired Step 18 test and closed-loop comparison |

CLI settings can be overridden repeatably with `--set KEY=VALUE`. Use a new
`experiment.name` for a distinct run; completed output directories are not
silently overwritten.

The canonical 48-anchor definition is
[`src/nbv/geometry/anchors_v1.csv`](src/nbv/geometry/anchors_v1.csv). Runs save
their resolved configuration, random seed, environment metadata, Git commit,
and relevant cache/checkpoint fingerprints.

## Repository map

```text
configs/       experiment definitions
data/splits/   fixed NUM split manifest
docs/          camera-validation and policy-adapter notes
scripts/       command-line entry points
src/nbv/       datasets, features, geometry, policies, training, evaluation
tests/         unit and integration tests
outputs/       generated experiment artifacts
```

## Google Colab and Drive

Colab storage under `/content` is temporary. Keep the working copy and
extracted datasets there for speed, but store dataset archives and generated
artifacts in Google Drive.

### 1. Archive the datasets locally

Run these commands from the repository root on the machine that contains the
datasets:

```bash
tar -czf data/NUM.tar.gz -C data NUM

tar --exclude='ShapeNetCore.v2/archive.zip' \
  -czf data/ShapeNetCore.v2-num-subset.tar.gz \
  -C data ShapeNetCore.v2

ls -lh data/NUM.tar.gz data/ShapeNetCore.v2-num-subset.tar.gz
tar -tzf data/NUM.tar.gz | head
tar -tzf data/ShapeNetCore.v2-num-subset.tar.gz | head
```

The ShapeNet command excludes a source archive named `archive.zip` if one is
present, avoiding a redundant archive inside the new tarball. Upload both
`.tar.gz` files to `MyDrive/cv3d-datasets/` using the Google Drive web
interface. Large archives should not be uploaded through VS Code's Colab file
upload action.

If a local visibility cache already exists, archive it separately:

```bash
tar -czf data/visibility-cache.tar.gz -C data/cache visibility
tar -tzf data/visibility-cache.tar.gz | head
```

Upload that file to the same `MyDrive/cv3d-datasets/` directory.

### 2. Mount Drive and extract in Colab

Mount Drive from a Colab notebook cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

In the Colab terminal, authenticate with GitHub and clone the repository:

```bash
gh auth login
git clone https://github.com/CarlDavidRe/cv3d-project.git /content/cv3d-project
```

Then extract the datasets:

```bash
mkdir -p /content/cv3d-project/data

tar -xzf /content/drive/MyDrive/cv3d-datasets/NUM.tar.gz \
  -C /content/cv3d-project/data

tar -xzf \
  /content/drive/MyDrive/cv3d-datasets/ShapeNetCore.v2-num-subset.tar.gz \
  -C /content/cv3d-project/data

ls /content/cv3d-project/data/NUM
find /content/cv3d-project/data/ShapeNetCore.v2 \
  -type f -path '*/models/model_normalized.ply' | head
```

Restore the optional archived visibility cache with:

```bash
mkdir -p /content/cv3d-project/data/cache

if [ -f /content/drive/MyDrive/cv3d-datasets/visibility-cache.tar.gz ]; then
  tar -xzf /content/drive/MyDrive/cv3d-datasets/visibility-cache.tar.gz \
    -C /content/cv3d-project/data/cache
fi
```

### 3. Restore previous work from Drive

Run this before resuming an experiment in a new Colab runtime:

```bash
cd /content/cv3d-project

if [ -d /content/drive/MyDrive/cv3d-project/outputs ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/outputs/ outputs/
fi

if [ -d /content/drive/MyDrive/cv3d-project/data/cache/features ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/data/cache/features/ \
    data/cache/features/
fi
```

The trailing slashes copy directory contents and prevent an extra nested
directory from being created.

### 4. Sync results back to Drive

Run this after experiments and before disconnecting or recycling the runtime:

```bash
cd /content/cv3d-project

mkdir -p /content/drive/MyDrive/cv3d-project/outputs
rsync -av outputs/ /content/drive/MyDrive/cv3d-project/outputs/

mkdir -p /content/drive/MyDrive/cv3d-project/data/cache/features
rsync -av data/cache/features/ \
  /content/drive/MyDrive/cv3d-project/data/cache/features/
```

To sync only one phase's outputs, set `PHASE` to `phase1`, `phase2`, or
`phase3`:

```bash
cd /content/cv3d-project
PHASE=phase1

mkdir -p "/content/drive/MyDrive/cv3d-project/outputs/$PHASE"
rsync -av "outputs/$PHASE/" \
  "/content/drive/MyDrive/cv3d-project/outputs/$PHASE/"
```

If Phase 3 generated processed histories, persist those as well:

```bash
mkdir -p /content/drive/MyDrive/cv3d-project/data/processed/histories
rsync -av data/processed/histories/ \
  /content/drive/MyDrive/cv3d-project/data/processed/histories/
```

Re-run the restore commands at the start of the next Colab session. Model
downloads under `data/cache/models/` can usually be recreated and are omitted
from the default sync to save Drive space.
