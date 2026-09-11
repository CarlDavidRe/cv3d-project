# Google Colab terminal workflow

Select a GPU runtime and mount Google Drive with Colab's left-sidebar folder
icon. Drive authorization is the only UI prerequisite. Every command below is
ordinary Bash intended to be pasted directly into the **Colab terminal**; do
not add `%%bash` or `set -euo pipefail`.

Colab's `/content` storage is temporary. Work there for speed and copy durable
data and results to `/content/drive/MyDrive`.

## 1. Check the runtime and clone the repository

```bash
nvidia-smi

if [ -d /content/drive/MyDrive ]; then
  echo "Google Drive is mounted."
else
  echo "Mount Google Drive from the Colab sidebar before continuing."
fi
```

Authenticate interactively, then clone the repository:

```bash
gh auth login
```

```bash
REPO=/content/cv3d-project

if [ ! -d "$REPO/.git" ]; then
  gh repo clone CarlDavidRe/cv3d-project "$REPO"
fi

cd "$REPO"
```

## 2. Download and archive the datasets

### Download ShapeNetCore v2

The ShapeNet data is gated. Request access to the
[ShapeNetCore v2 dataset](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
and accept its terms. After access is granted, install the Hugging Face CLI and
authenticate:

```bash
python3 -m pip install --upgrade huggingface_hub
hf auth login
```

The NUM split uses 13 ShapeNet synsets. Download only those category archives:

```bash
cd /content/cv3d-project

mkdir -p data/downloads/shapenetcore
hf download ShapeNet/ShapeNetCore \
  02691156.zip 02828884.zip 02933112.zip 02958343.zip \
  03001627.zip 03211117.zip 03636649.zip 03691459.zip \
  04090263.zip 04256520.zip 04379243.zip 04401088.zip \
  04530566.zip \
  --repo-type dataset \
  --local-dir data/downloads/shapenetcore

mkdir -p data/ShapeNetCore.v2
for ARCHIVE in data/downloads/shapenetcore/*.zip; do
  unzip -q -n "$ARCHIVE" -d data/ShapeNetCore.v2
done
```

The default geometry configuration expects `model_normalized.ply`. If the
download contains OBJ meshes, set
`phase2.visibility.mesh_relative_path=models/model_normalized.obj` in
`configs/experiments/phase2_visibility.yaml` before precomputing visibility,
and set the matching
`phase2.evaluation.reconstruction.mesh_relative_path` value in
`configs/experiments/phase2_closed_loop.yaml`.

### Create the dataset archives

Run this one-time step when the extracted datasets are available at
`/content/cv3d-project/data/NUM` and
`/content/cv3d-project/data/ShapeNetCore.v2`. It creates the archives locally,
verifies them, and copies them to `MyDrive/cv3d-datasets`.
Skip this step when those archives are already in Drive.

```bash
cd /content/cv3d-project

DATASET_DRIVE=/content/drive/MyDrive/cv3d-datasets
mkdir -p "$DATASET_DRIVE"

tar -czf /content/NUM.tar.gz -C data NUM

tar --exclude='ShapeNetCore.v2/archive.zip' \
  -czf /content/ShapeNetCore.v2-num-subset.tar.gz \
  -C data ShapeNetCore.v2

ls -lh /content/NUM.tar.gz /content/ShapeNetCore.v2-num-subset.tar.gz
tar -tzf /content/NUM.tar.gz | head
tar -tzf /content/ShapeNetCore.v2-num-subset.tar.gz | head

rsync -av /content/NUM.tar.gz "$DATASET_DRIVE/"
rsync -av /content/ShapeNetCore.v2-num-subset.tar.gz "$DATASET_DRIVE/"
```

If visibility or reconstruction caches already exist, archive and save them too:

```bash
cd /content/cv3d-project

DATASET_DRIVE=/content/drive/MyDrive/cv3d-datasets
tar -czf /content/visibility-cache.tar.gz -C data/cache visibility
tar -tzf /content/visibility-cache.tar.gz | head
rsync -av /content/visibility-cache.tar.gz "$DATASET_DRIVE/"

if [ -d data/cache/reconstruction ]; then
  tar -czf /content/reconstruction-cache.tar.gz -C data/cache reconstruction
  tar -tzf /content/reconstruction-cache.tar.gz | head
  rsync -av /content/reconstruction-cache.tar.gz "$DATASET_DRIVE/"
fi
```

The archives must contain these top-level directories:

```text
NUM/<category_id>/<object_id>/...
ShapeNetCore.v2/<category_id>/<object_id>/models/model_normalized.ply
visibility/...  # optional
reconstruction/...  # optional cached VGGT point clouds, GT samples, and metrics
```

## 3. Restore data and previous work

Run this at the start of each fresh Colab runtime:

```bash
REPO=/content/cv3d-project
DRIVE=/content/drive/MyDrive

mkdir -p "$REPO/data"

tar -xzf "$DRIVE/cv3d-datasets/NUM.tar.gz" \
  -C "$REPO/data"

tar -xzf "$DRIVE/cv3d-datasets/ShapeNetCore.v2-num-subset.tar.gz" \
  -C "$REPO/data"

if [ -f "$DRIVE/cv3d-datasets/visibility-cache.tar.gz" ]; then
  mkdir -p "$REPO/data/cache"
  tar -xzf "$DRIVE/cv3d-datasets/visibility-cache.tar.gz" \
    -C "$REPO/data/cache"
fi

if [ -f "$DRIVE/cv3d-datasets/reconstruction-cache.tar.gz" ]; then
  mkdir -p "$REPO/data/cache"
  tar -xzf "$DRIVE/cv3d-datasets/reconstruction-cache.tar.gz" \
    -C "$REPO/data/cache"
fi

for PATH_TO_RESTORE in \
  data/cache/features \
  data/cache/visibility \
  data/cache/reconstruction
do
  if [ -d "$DRIVE/cv3d-project/$PATH_TO_RESTORE" ]; then
    mkdir -p "$REPO/$PATH_TO_RESTORE"
    rsync -av "$DRIVE/cv3d-project/$PATH_TO_RESTORE/" \
      "$REPO/$PATH_TO_RESTORE/"
  fi
done
```

The versioned history dataset is included in the repository. Do not rebuild it
when resuming Phase 3: rebuilding can change its identity and cause the matching
feature cache and training checkpoint to be rejected.

Restore Phase 3 outputs from Drive, excluding rollouts:

```bash
REPO=/content/cv3d-project
DRIVE=/content/drive/MyDrive

if [ -d "$DRIVE/cv3d-project/outputs/phase3" ]; then
  mkdir -p "$REPO/outputs/phase3"
  rsync -av --exclude='rollouts/' "$DRIVE/cv3d-project/outputs/phase3/" \
    "$REPO/outputs/phase3/"
fi
```

## 4. Install the project and VGGT

```bash
cd /content/cv3d-project

python3 -m pip install -q -r requirements.txt
python3 -m pip install -q -e .

mkdir -p data/cache/sources
if [ ! -d data/cache/sources/vggt/.git ]; then
  git clone https://github.com/facebookresearch/vggt.git \
    data/cache/sources/vggt
fi
python3 -m pip install -q -e data/cache/sources/vggt

python3 -c 'import torch, nbv; print("nbv import: OK"); print("device:", "cuda" if torch.cuda.is_available() else "cpu")'
```

## 5. Run experiments

Run only the blocks you need.

### Phase 1

```bash
cd /content/cv3d-project

python3 scripts/run_phase1.py \
  --config configs/experiments/phase1_sweep.yaml
```

### Phase 2

Phase 2 evaluates every rollout with rasterized `VisA`. It additionally compares
the PUN and validation-selected VGGT-head policies using one shared frozen VGGT
point-map reconstructor at 1, 2, 3, 5, and 10 acquired views. Predicted clouds
are aligned through the predicted and known NUM cameras, then scored against
area-sampled points from the paired ShapeNet mesh with normalized Chamfer-L1,
accuracy, completeness, and F-score. No NeRF training is required.

The first run downloads/loads the full VGGT model and creates reusable caches
under `data/cache/reconstruction`. A cached ordered history is reused across
policies, reruns, and downstream Phase 3 evaluation. Changing only Chamfer or
F-score settings reuses the expensive point-cloud cache. Exact definitions and
alignment limitations are in `docs/reconstruction_evaluation.md`.

```bash
cd /content/cv3d-project

python3 scripts/precompute_visibility.py --split test
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml
```

### Phase 3

The checked-in history dataset is ready for the configured Phase 3 runs. To
intentionally generate a new history dataset for a new run, create visibility
caches for every split and then build it. Skip this block when resuming:

```bash
cd /content/cv3d-project

for SPLIT in train val test; do
  python3 scripts/precompute_visibility.py --split "$SPLIT"
done

python3 scripts/build_history_dataset.py
```

Train and compare the independent and joint controls:

```bash
cd /content/cv3d-project

python3 scripts/train_history.py \
  --config configs/experiments/phase3_independent.yaml

python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint.yaml

python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml
```

This downstream comparison reports the same visibility curves and applies the
same cached VGGT reconstruction evaluator to both history policies. Run the
configured Phase 2 evaluation first so its summary is available as the
external baseline reference.

Train the pose-conditioned DeepSets and spatial-token attention follow-ups:

```bash
cd /content/cv3d-project

python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint_pose_deepsets.yaml

python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint_token_attention.yaml
```

Evaluate both follow-ups on the same controlled test histories and closed-loop
cohort. Each evaluation uses a distinct experiment name and output directory:

```bash
cd /content/cv3d-project

python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml \
  --joint-checkpoint outputs/phase3/joint_history_pose_deepsets/seed_0/checkpoints/best.pt \
  --experiment-name controlled_pose_deepsets

python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml \
  --joint-checkpoint outputs/phase3/joint_history_token_attention/seed_0/checkpoints/best.pt \
  --experiment-name controlled_token_attention
```

Generate the combined Phase 2/3 coverage plot:

```bash
cd /content/cv3d-project
python3 scripts/plot_all_policy_coverage.py
```

### Quick smoke test

```bash
cd /content/cv3d-project

python3 scripts/train_probe.py \
  --config configs/experiments/phase1_probe_tiny.yaml

python3 scripts/evaluate_closed_loop.py \
  --limit 10 \
  --skip-missing-caches \
  --set phase2.evaluation.reconstruction.enabled=false \
  --set experiment.name=phase2_subset
```

## 6. Save experiment results to Drive

Run this after any experiment. It saves every result directory that currently
exists and skips phases that have not been run.

```bash
cd /content/cv3d-project

DRIVE_OUTPUTS=/content/drive/MyDrive/cv3d-project/outputs

for RESULT_DIR in phase1 phase2 phase3 all_policy_comparison; do
  if [ -d "outputs/$RESULT_DIR" ]; then
    mkdir -p "$DRIVE_OUTPUTS/$RESULT_DIR"
    rsync -av "outputs/$RESULT_DIR/" "$DRIVE_OUTPUTS/$RESULT_DIR/"
  fi
done
```

Save reusable caches as well:

```bash
cd /content/cv3d-project

BACKUP=/content/drive/MyDrive/cv3d-project

for PATH_TO_SYNC in \
  data/cache/features \
  data/cache/visibility \
  data/cache/reconstruction
do
  if [ -d "$PATH_TO_SYNC" ]; then
    mkdir -p "$BACKUP/$PATH_TO_SYNC"
    rsync -av "$PATH_TO_SYNC/" "$BACKUP/$PATH_TO_SYNC/"
  fi
done
```

## 7. Back up long runs automatically

Start this once per runtime. It syncs results and reusable data to Drive every
five minutes:

```bash
REPO=/content/cv3d-project
BACKUP=/content/drive/MyDrive/cv3d-project
mkdir -p "$BACKUP"

if [ -f /tmp/cv3d-sync.pid ] && \
   kill -0 "$(cat /tmp/cv3d-sync.pid)" 2>/dev/null
then
  echo "Backup is already running."
else
  (
    while true; do
      for PATH_TO_SYNC in \
        outputs \
        data/cache/features \
        data/cache/visibility \
        data/cache/reconstruction
      do
        if [ -d "$REPO/$PATH_TO_SYNC" ]; then
          mkdir -p "$BACKUP/$PATH_TO_SYNC"
          rsync -av --exclude='*.tmp' "$REPO/$PATH_TO_SYNC/" \
            "$BACKUP/$PATH_TO_SYNC/"
        fi
      done
      sleep 300
    done
  ) >/tmp/cv3d-sync.log 2>&1 &

  echo $! >/tmp/cv3d-sync.pid
  echo "Backup started. Log: /tmp/cv3d-sync.log"
fi
```

Before disconnecting, interrupt the active experiment and force a final sync:

```bash
REPO=/content/cv3d-project
BACKUP=/content/drive/MyDrive/cv3d-project

for PATH_TO_SYNC in \
  outputs \
  data/cache/features \
  data/cache/visibility \
  data/cache/reconstruction
do
  if [ -d "$REPO/$PATH_TO_SYNC" ]; then
    mkdir -p "$BACKUP/$PATH_TO_SYNC"
    rsync -av --exclude='*.tmp' "$REPO/$PATH_TO_SYNC/" \
      "$BACKUP/$PATH_TO_SYNC/"
  fi
done

if [ -f /tmp/cv3d-sync.pid ]; then
  kill "$(cat /tmp/cv3d-sync.pid)" 2>/dev/null || true
fi
```

## 8. Resume after a runtime reset

Repeat steps 1, 3, and 4, then restart the automatic backup. Resume joint
training with:

```bash
cd /content/cv3d-project

python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint.yaml \
  --resume
```

Or resume the controlled evaluation with:

```bash
cd /content/cv3d-project

python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml \
  --resume
```
