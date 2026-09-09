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
- **Phase 3 — independent control validated:** the complete 1,279-object,
  51,160-history direct-gain dataset is generated, and the cache-backed
  independent VGGT history workflow has passed a full-data training/evaluation
  smoke run. Joint-history processing remains Step 17.

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
mkdir -p outputs data/cache/features data/cache/visibility

if [ -d /content/drive/MyDrive/cv3d-project/outputs ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/outputs/ outputs/
fi

if [ -d /content/drive/MyDrive/cv3d-project/data/cache/features ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/data/cache/features/ \
    data/cache/features/
fi

if [ -d /content/drive/MyDrive/cv3d-project/data/cache/visibility ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/data/cache/visibility/ \
    data/cache/visibility/
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

mkdir -p /content/drive/MyDrive/cv3d-project/data/cache/visibility
rsync -av data/cache/visibility/ \
  /content/drive/MyDrive/cv3d-project/data/cache/visibility/
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
