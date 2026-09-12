# Google Colab terminal workflow

## Local dashboard

Launch the interactive experiment dashboard from the repository root:

```bash
make dashboard
```

The command creates a local `.venv` when needed and installs the dashboard
dependencies on first launch.

### Run the dashboard without cloning the full repository

The dashboard does not require model checkpoints or ML dependencies. A shallow
sparse clone can download only the dashboard, its dataset metadata, and the
Phase 1 summary files it displays:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  git@github.com:CarlDavidRe/cv3d-project.git cv3d-dashboard

cd cv3d-dashboard

git sparse-checkout set --no-cone \
  '/dashboard/' \
  '/requirements-dashboard.txt' \
  '/data/splits/num_v1.json' \
  '/src/nbv/geometry/anchors_v1.csv' \
  '/outputs/phase1/backbone_sweep/**/summary.json' \
  '/outputs/phase2/phase2_closed_loop_reconstruction/seed_0/metrics/**' \
  '/outputs/phase2/phase2_closed_loop_reconstruction/seed_0/rollouts/**' \
  '/outputs/phase3/controlled_history_comparison_reconstruction/seed_0/metrics/**' \
  '/outputs/phase3/controlled_history_comparison_reconstruction/seed_0/rollouts/**'

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

This keeps the directory layout expected by the app while avoiding the full
repository history and large experiment artifacts. The dataset explorer still
shows every category, split assignment, object ID, and camera anchor. Clone or
place the NUM dataset at `data/NUM` to also show the selected RGB observation,
and ShapeNet meshes at `data/ShapeNetCore.v2` to place the selected object inside
the anchor sphere. The closed-loop page reads the saved Phase 2 and Phase 3
metric tables and rollout replays directly; it does not run model inference.

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

Run this one-time step in a terminal on your local computer after placing the
extracted datasets in `data/NUM` and `data/ShapeNetCore.v2` in your local
repository checkout. The archives are written to the repository's existing
`data` directory and are ignored by Git. These commands create `.tar.gz`
archives; they do not extract or unzip the datasets.

```bash
cd /home/reese/computer_vision/cv3d-project

ARCHIVE_DIR=data

tar -czf "$ARCHIVE_DIR/NUM.tar.gz" -C data NUM

tar --exclude='ShapeNetCore.v2/archive.zip' \
  -czf "$ARCHIVE_DIR/ShapeNetCore.v2-num-subset.tar.gz" \
  -C data ShapeNetCore.v2

ls -lh "$ARCHIVE_DIR/NUM.tar.gz" \
  "$ARCHIVE_DIR/ShapeNetCore.v2-num-subset.tar.gz"
# List archive contents for verification without extracting them.
tar -tzf "$ARCHIVE_DIR/NUM.tar.gz" | head
tar -tzf "$ARCHIVE_DIR/ShapeNetCore.v2-num-subset.tar.gz" | head
```

Open Google Drive in your browser and manually upload `NUM.tar.gz` and
`ShapeNetCore.v2-num-subset.tar.gz` from the local
`/home/reese/computer_vision/cv3d-project/data` directory to the Drive folder
`/cv3d-datasets`. Skip this step if the archives are already there.

If visibility or reconstruction caches already exist locally, archive them in
the same directory and upload the resulting files too:

```bash
ARCHIVE_DIR=data

tar -czf "$ARCHIVE_DIR/visibility-cache.tar.gz" \
  -C data/cache visibility
tar -tzf "$ARCHIVE_DIR/visibility-cache.tar.gz" | head

if [ -d data/cache/reconstruction ]; then
  tar -czf "$ARCHIVE_DIR/reconstruction-cache.tar.gz" \
    -C data/cache reconstruction
  tar -tzf "$ARCHIVE_DIR/reconstruction-cache.tar.gz" | head
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
  data/cache/features
do
  if [ -d "$DRIVE/cv3d-project/$PATH_TO_RESTORE" ]; then
    mkdir -p "$REPO/$PATH_TO_RESTORE"
    rsync -av "$DRIVE/cv3d-project/$PATH_TO_RESTORE/" \
      "$REPO/$PATH_TO_RESTORE/"
  fi
done
```

The reconstruction cache is restored only from
`reconstruction-cache.tar.gz`; it is not synchronized separately from the
project backup directory.

The versioned history dataset is included in the repository. Do not rebuild it
when resuming Phase 3: rebuilding can change its identity and cause the matching
feature cache and training checkpoint to be rejected.

Restore Phase 3 outputs from Drive, excluding rollouts:

```bash
REPO=/content/cv3d-project
DRIVE=/content/drive/MyDrive

if [ -d "$DRIVE/cv3d-project/outputs/phase3" ]; then
  mkdir -p "$REPO/outputs/phase3"

  RSYNC_ARGS=(-av --exclude='rollouts/')
  while IFS= read -r -d '' CACHE_DIR; do
    CACHE_PATH=${CACHE_DIR#"$REPO/outputs/phase3/"}
    RSYNC_ARGS+=(--exclude="/$CACHE_PATH/")
  done < <(find "$REPO/outputs/phase3" -type d -name joint_feature_cache -print0)

  rsync "${RSYNC_ARGS[@]}" "$DRIVE/cv3d-project/outputs/phase3/" \
    "$REPO/outputs/phase3/"
fi
```

Existing joint feature-cache directories are left untouched. If a run does not
have a local `joint_feature_cache`, its cache is restored from Drive normally.

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

# Evaluate all five Phase 2 variants together (recommended).
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml \
  --set 'phase2.evaluation.reconstruction.policies=[random,farthest,pun,vggt,oracle]'
```

The combined command evaluates `random`, `farthest`, `pun`, `vggt`, and
`oracle` on exactly the same cohort. To evaluate only one variant, use the
corresponding command below. Each command has a unique experiment name
so it cannot overwrite the combined comparison. The combined run applies the
same frozen VGGT reconstruction evaluator to every policy. Interpret `oracle`
as a privileged upper bound because its view selection uses ground-truth
visibility.

To visually compare any cached reconstruction with its ground-truth surface
sample, select its object and policy from `reconstruction_per_object.csv`:

```bash
python3 scripts/visualize_reconstruction.py \
  outputs/phase2/phase2_closed_loop_reconstruction/seed_0/metrics/reconstruction_per_object.csv \
  --object-id 02691156/1628b65a9f3cd7c05e9e2656aff7dd5b \
  --policy vggt \
  --views 10
```

The command uses only the reconstruction and visibility caches; it does not
run VGGT again. It writes a three-projection `comparison.png`, a self-contained
interactive `comparison_interactive.html`, colored `comparison.ply`, separate
aligned-prediction and ground-truth PLY files, and the recomputed alignment
transform. Omit `--views` to use the largest cached view count. Open the HTML
directly in a browser for Phase-1-style rotation, zoom, auto-rotation, cloud
toggles, overlay or side-by-side layouts, and a toggleable ground-truth box
with X/Y/Z world dimensions, or open the PLY in MeshLab or CloudCompare
(ground truth is blue; prediction is orange).

To serve the interactive HTML from localhost, run this from the repository
root and keep the terminal open:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open the generated viewer at:

```text
http://127.0.0.1:8000/outputs/phase2/phase2_closed_loop_reconstruction/seed_0/metrics/reconstruction_visualizations/02691156_1628b65a9f3cd7c05e9e2656aff7dd5b_vggt_10views/comparison_interactive.html
```

Replace the final visualization directory or filename to inspect another run;
use `comparison_oracle_icp_interactive.html` for the oracle-ICP diagnostic.
Stop the server with Ctrl+C.

Add `--oracle-icp` to also export a ground-truth-assisted Sim(3) ICP overlay.
This is useful for separating residual global alignment error from structural
reconstruction error, but its metrics are an oracle diagnostic and must not be
reported as official evaluation results.
It also writes `comparison_oracle_icp_interactive.html` for that diagnostic.

#### Gaussian-splatting reconstruction

Gaussian splatting is an optional CUDA workflow and is not part of the base
installation. Install the official `gsplat` rasterizer in the project
environment:

```bash
python3 -m pip install -e '.[gaussian-splatting]'
```

To evaluate every object in the test split, train every Phase 2 and Phase 3
variant independently at each incremental view count, geometrically evaluate
2D Gaussian Splatting, and render 3D Gaussian Splatting for qualitative
visualization:

```bash
mapfile -t test_objects < <(
  python3 - <<'PY'
import json
import sys
from pathlib import Path

with open("data/splits/num_v1.json", encoding="utf-8") as handle:
    manifest = json.load(handle)

test_objects = manifest["splits"]["test"]
views = (1, 2, 3, 5, 10)
variants = (
    "phase2_random", "phase2_farthest", "phase2_pun", "phase2_vggt",
    "phase2_oracle", "phase3_vggt_independent_history",
    "phase3_vggt_joint_history", "phase3_vggt_joint_pose_deepsets",
    "phase3_vggt_joint_token_attention",
)
backends = ("2dgs", "3dgs")
output_root = Path("outputs/gaussian_splatting_variant_comparison")

for object_id in test_objects:
    root = output_root / object_id.replace("/", "_")
    complete = all(
        (root / f"{view_count}views" / variant / backend / "summary.json").is_file()
        for view_count in views
        for variant in variants
        for backend in backends
    )
    if complete:
        print(f"SKIP {object_id} (already complete)", file=sys.stderr)
    else:
        print(object_id)
PY
)

for object_id in "${test_objects[@]}"; do
  python3 scripts/evaluate_gaussian_splatting_variants.py \
    --object-id "$object_id" \
    --views 1 2 3 5 10 \
    --backend both \
    --iterations 1500
done
```

Selection is deterministic: the command processes every object ID in
`splits.test` in manifest order. An object is omitted from the loop once all
requested view, variant, and backend summaries exist.

The runner discovers the five Phase 2 policies (`random`, `farthest`, `pun`,
`vggt`, and `oracle`) and the four distinct Phase 3 policies
(`vggt_independent_history`, `vggt_joint_history`,
`vggt_joint_pose_deepsets`, and `vggt_joint_token_attention`) from their
reconstruction CSV files. A duplicated independent-history control is run only
once. Each policy/view-count pair gets a fresh model; view counts do not
continue training from the preceding model.
Completed backend summaries are skipped individually on reruns unless `--force`
is given, so a failed `both` run resumes only its missing backend.
Use `--dry-run` to validate all histories and inspect the commands without
starting CUDA training.

Results for each object are written below
`outputs/gaussian_splatting_variant_comparison/<category>_<object>/`, where its
`index.html` links the 2DGS ground-truth overlays and the 2DGS/3DGS render
galleries for every variant and view count.

Both backends use the cached, camera-aligned VGGT points for initialization
and train against the selected RGB history with known NUM cameras. The 2DGS
backend renders median depth from all 48 canonical cameras, alpha-filters and
voxel-fuses those depths into `surface.ply`, and reports the existing
diameter-normalized Chamfer, accuracy, completeness, precision, recall, and
F-score metrics in `2dgs/summary.json` and `2dgs/metrics.csv`. It also writes
`comparison.png`, `comparison.ply`, and a self-contained
`comparison_interactive.html` that overlays the fused 2DGS surface with the
ground-truth sample or displays them side by side using the same world box.
It never applies ground-truth ICP. Treat this as a new evaluation candidate
until it has been validated on the complete cohort; it does not silently
replace the existing shared-VGGT reconstruction protocol.

The 3DGS backend is deliberately qualitative: it writes a checkpoint, a
self-contained `3dgs/turntable.html`, and
`3dgs/ground_truth_comparison.html`. The comparison synchronizes each
canonical 3DGS render with its corresponding ground-truth NUM RGB image,
supports side-by-side and opacity-overlay modes, and identifies training versus
held-out views. It does not report geometric metrics. This prevents raw
Gaussian centers—which are not a surface—from being compared with the
ground-truth mesh as though they were surface samples. The 2DGS backend also
writes a turntable for inspecting the representation used for evaluation.

Use `--backend 2dgs` or `--backend 3dgs` to run only one backend. Actual
training requires a CUDA-capable PyTorch environment; a missing CUDA runtime
or `gsplat` installation is reported before any outputs are written.

```bash
cd /content/cv3d-project

# Random
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml \
  --set experiment.name=phase2_random \
  --set 'phase2.evaluation.policies=[random]' \
  --set 'phase2.evaluation.reconstruction.policies=[random]'

# Farthest-view baseline
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml \
  --set experiment.name=phase2_farthest \
  --set 'phase2.evaluation.policies=[farthest]' \
  --set 'phase2.evaluation.reconstruction.policies=[farthest]'

# PUN
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml \
  --set experiment.name=phase2_pun \
  --set 'phase2.evaluation.policies=[pun]' \
  --set 'phase2.evaluation.reconstruction.policies=[pun]'

# Validation-selected VGGT head
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml \
  --set experiment.name=phase2_vggt \
  --set 'phase2.evaluation.policies=[vggt]' \
  --set 'phase2.evaluation.reconstruction.policies=[vggt]'

# Oracle upper bound
python3 scripts/evaluate_closed_loop.py \
  --config configs/experiments/phase2_closed_loop.yaml \
  --set experiment.name=phase2_oracle \
  --set 'phase2.evaluation.policies=[oracle]' \
  --set 'phase2.evaluation.reconstruction.policies=[oracle]'
```

### Phase 3

The checked-in history dataset is ready for the configured Phase 3 runs. Do
not regenerate it when resuming, because doing so can change its identity and
invalidate the matching feature cache and training checkpoint.

Train the independent and capacity-matched joint controls, then evaluate both
together on the shared fixed-history and closed-loop protocols:

```bash
cd /content/cv3d-project

python3 scripts/train_history.py \
  --config configs/experiments/phase3_independent.yaml

python3 scripts/train_joint_history.py \
  --config configs/experiments/phase3_joint.yaml

python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml
```

That controlled evaluation covers both `vggt_independent_history` and
`vggt_joint_history`; they are intentionally evaluated together because the
reported one-step metrics are paired by history.

This downstream comparison reports the same visibility curves and applies the
same cached VGGT reconstruction evaluator to both history policies. Run the
configured Phase 2 evaluation first so its summary is available as the
external baseline reference. That reference includes all five Phase 2 policies;
the geometric policies and privileged oracle are reported alongside the learned
PUN and VGGT baselines.

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

To resume any interrupted Phase 3 evaluation, append `--resume` to its command.
For expressive variants, retain the same `--joint-checkpoint` and
`--experiment-name` arguments when resuming.

Generate the combined Phase 2/3 coverage and reconstruction plots:

```bash
cd /content/cv3d-project
python3 scripts/plot_all_policy_coverage.py
python3 scripts/plot_all_policy_reconstruction.py
```

By default, both plotters merge the standalone `phase2_random`,
`phase2_farthest`, `phase2_pun`, `phase2_vggt`, and `phase2_oracle` seed-0
runs. Repeat `--phase2-run` to select a different set of Phase 2 runs.

The plotters recursively discover Phase 3 evaluation runs below
`outputs/phase3` and include every policy from runs whose
`metrics/phase3_completion.json` status is `complete`. Incomplete evaluations
are excluded; the reconstruction plot additionally requires that run's
`metrics/reconstruction_curves.csv`. To select runs explicitly instead, repeat
`--phase3-run`, for example
`--phase3-run outputs/phase3/controlled_pose_deepsets/seed_0`.

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
  data/cache/visibility
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
        data/cache/visibility
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
  data/cache/visibility
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
