# Frozen-feature next-best-view study

This repository implements the phased project described in
[`project_overview.md`](project_overview.md). Development is currently limited
to Phase 1 (the single-image feature probe). For local work, follow Steps 1–7
in order. The final section restates the full remote-GPU path as one sequential
Colab workflow.

## Steps 1–4: setup and dataset verification

### Environment and infrastructure

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Initialize a reproducible Phase 1 run directory from the checked-in config:

```bash
python3 scripts/init_experiment.py \
  --config configs/experiments/phase1.yaml
```

This records the resolved configuration, seed, environment metadata, and Git
commit under `outputs/phase1/infrastructure/seed_0/`. It does not train a model.

Generated files follow two stable roots: run artifacts use
`outputs/<phase>/<experiment>/seed_<n>/`, while reusable downloads and caches
use `data/cache/{models,features,sources}/`. Workflow step numbers are never
used as directory names.

Run the infrastructure tests with:

```bash
python3 -m unittest discover -s tests -v
```

The canonical PUN-compatible 48-anchor definition now lives in
`src/nbv/geometry/anchors_v1.csv`.

### NUM dataset loader

Place the [official PUN NUM dataset](https://github.com/ZhangLab-DeepNeuroCogLab/PUN)
at `data/NUM` (or pass another root). The loader reads the released layout:

```text
NUM/<category_id>/<object_id>/images/viewpoint_<anchor>_offset_phi_0.png
NUM/<category_id>/<object_id>/uncertainties/viewpoint_<anchor>_offset_phi_0.json
```

The checked-in `data/splits/num_v1.json` manifest prevents different views of
one ShapeNet object from crossing splits. It preserves PUN's held-out car/chair
categories and sorted 90/10 object boundary; validation uses the final one
ninth of each training pool at object level because PUN's image-level random
validation split would leak objects.

Regenerate-and-compare the frozen manifest with:

```bash
python3 scripts/prepare_num_split.py \
  --data-root data/NUM \
  --output data/splits/num_v1.json
```

Inspect one sample and create a standalone PUN-style polar uncertainty-map SVG
with the input image embedded alongside it, plus a self-contained rotatable 3D
sphere in the same canonical run directory:

```bash
python3 scripts/inspect_num_sample.py \
  --data-root data/NUM \
  --split train \
  --split-manifest data/splits/num_v1.json \
  --target PSNR \
  --output outputs/phase1/num_sample/seed_0/figures/num_sample.svg
```

To view the interactive HTML through a local web server, run this command from
the repository root:

```bash
python3 -m http.server 8000
```

Then open the following address in a browser:

```text
http://localhost:8000/outputs/phase1/num_sample/seed_0/figures/num_sample_3d.html
```

Stop the server with `Ctrl+C`.

The UMap is source-relative: its center is the current view and its edge is the
opposite view. The 48 HEALPix targets are displayed as nearest-anchor regions,
without smoothing between unsupported viewpoints. PSNR and SSIM are inverted
when normalized so that brighter colors consistently mean higher uncertainty;
MSE and LPIPS increase directly with uncertainty. Use
`--uncertainty-direction` to specify the convention for a custom target.

The loader preserves official Phase 1 targets exactly; it does not reinterpret
PSNR/SSIM/MSE/LPIPS arrays as Phase 2 surface-gain utilities. The common metric
library is available in `nbv.eval`.

## Step 5: frozen feature smoke tests

Step 5 implements the shared frozen-feature interface and the supervised
ImageNet ViT-B/16, DINOv2, and VGGT extractors. It does **not** implement a
predictor, loss, training loop, or feature cache yet. Consequently, the useful
run at this checkpoint is a one-image extraction smoke test.

ImageNet ViT-B/16 is the lightest setup check and can run on CPU (the first run
downloads its weights):

```bash
python3 scripts/inspect_features.py \
  --backbone imagenet_vit \
  --data-root data/NUM \
  --device cpu
```

DINOv2 also works on CPU, although a CUDA GPU is faster. Its first run downloads
the official torch.hub source and weights:

```bash
python3 scripts/inspect_features.py \
  --backbone dinov2 \
  --data-root data/NUM \
  --device auto
```

VGGT is a 1B-parameter model, so use a Colab GPU rather than a CPU-only local
machine. Install the official package in the same environment first:

```bash
git clone https://github.com/facebookresearch/vggt.git data/cache/sources/vggt
python3 -m pip install -e data/cache/sources/vggt
python3 scripts/inspect_features.py \
  --backbone vggt \
  --data-root data/NUM \
  --device cuda
```

The VGGT smoke test processes each NUM image as a one-frame sequence, which is
the required independent single-image behavior for Phase 1. It does not perform
the joint multi-view processing reserved for Phase 3.

## Step 6: lightweight probe-head overfit check

Step 6 adds only the small fixed-size-feature MLP, masked Huber regression, and
a deterministic tiny-subset training command. Frozen features are extracted
once into memory, the backbone is released, and only the probe head is optimized.
Ranking loss, full-split training, feature-cache files, experiment sweeps, and
evaluation are added by Step 7 below.

Feature inputs are selected independently for each backbone under
`probe.feature_selection`. Each value is an ordered list; multiple components
are concatenated before the prediction head:

| Backbone | Available components |
|---|---|
| ImageNet ViT | `pooled_patch`, `max_pooled_patch`, `cls_token` |
| DINOv2 | `pooled_patch`, `max_pooled_patch`, `cls_token`, `pooled_register` |
| VGGT | `pooled_patch`, `max_pooled_patch`, `pooled_camera`, `pooled_register` |

`pooled_patch` is mean pooling. Camera and register sequences are also mean
pooled. Selecting `pooled_register` for DINOv2 requires a model variant that
actually exposes register tokens, such as `dinov2_vitb14_reg`. The default
configuration uses only `pooled_patch` for all three backbones.

With those defaults, ImageNet ViT-B/16 and DINOv2 ViT-B/14 produce
768-dimensional probe inputs, whereas VGGT-1B produces 1024-dimensional
inputs. The head infers this width and maps it through the shared
128-dimensional hidden layer to 48 outputs. Thus the architecture is shared,
but its capacity is not strictly matched: the 768-dimensional heads have
106,160 parameters and the VGGT head has 139,440. Combining components further
increases the input width and parameter count. A final controlled comparison
must account for these differences; strict capacity matching is outside this
Step 6 overfit check.

Run the required tiny-subset check with the supervised ImageNet ViT:

```bash
python scripts/train_probe.py \
  --config configs/experiments/phase1_probe_tiny.yaml
```

The command returns a nonzero status unless it reaches the configured relative
loss reduction. It writes `best.pt` and `tiny_overfit.json` under
`outputs/phase1/probe_tiny/seed_0/`. To check another frozen backbone while
keeping the same head and training setup, override only the backbone:

```bash
python scripts/train_probe.py \
  --config configs/experiments/phase1_probe_tiny.yaml \
  --set probe.backbone=vggt \
  --set 'probe.feature_selection.vggt=[pooled_camera,pooled_patch]' \
  --set probe.device=cuda \
  --set probe.extraction_batch_size=1
```

## Step 7: Phase 1 experiment runner

Step 7 adds the configured single-image sweep path and stops at the 48-anchor
prediction head. One command extracts or reuses frozen features, trains the
same lightweight head for every configured feature variant, selects the best
validation checkpoint, evaluates it on the object-disjoint test split, and
writes a comparison table:

```bash
python scripts/run_phase1.py \
  --config configs/experiments/phase1_sweep.yaml
```

The default sweep compares mean-pooled patches and classification tokens for
both ImageNet ViT and DINOv2. For VGGT it compares mean-pooled patches,
max-pooled patches, the camera token, mean-pooled register tokens, and one
camera-plus-mean-patch combination. VGGT has no classification token; its
camera token is the model-specific global-token alternative. Variants using
the same backbone are extracted in one pass, so all five VGGT variants share
the same VGGT forward per image batch. Cache files are fingerprinted from the
backbone settings, preprocessing, layer, pooling, split manifest, target, and
sample IDs. They live under `data/cache/features/` by default.

The configured loss is:

```text
Huber + ranking_weight * pairwise_ranking
```

Set `probe.training.ranking_weight=0.0` to run pure Huber regression. PSNR and
SSIM are automatically interpreted as lower-is-more-uncertain for ranking
metrics; MSE and LPIPS are interpreted as higher-is-more-uncertain. A custom
target must set `probe.target_direction` explicitly.

Results are written under
`outputs/phase1/backbone_sweep/seed_0/`. The run contains the resolved config
and environment metadata, while each `variants/<name>/` directory contains its
best head checkpoint, training history, summary, and per-sample test metrics.
The common table is emitted as `metrics/comparison.csv`,
`metrics/comparison.json`, and `metrics/comparison.md`.

This step deliberately does not implement PUN integration, qualitative
visualization, policies, geometry, closed-loop evaluation, or multi-view
processing.

## Complete Google Colab workflow from VS Code

Use this section in order when running the GPU workflow remotely. Connect a VS
Code notebook with the official Colab extension, then open a remote terminal
with `Colab: Open Terminal`. Unless a step says otherwise, run its commands in
that terminal rather than in the local VS Code terminal.

### 1. Select Python 3.12

Check the remote runtime before installing anything:

```bash
python --version
```

If it already reports Python 3.12, keep the current runtime. Otherwise, in VS
Code disconnect the current server with `Colab: Remove Server`, select the
notebook kernel again, choose `Colab` and `New Colab Server`, and select a GPU
runtime whose Python version is 3.12 (for example, runtime `2026.07`). In the
Colab web interface, the equivalent setting is under **Runtime > Change runtime
type > Runtime version**. Reconnect and verify with `python --version`.

Changing the runtime recreates the Colab VM and deletes files under `/content`,
so select the runtime before cloning the repository or extracting the dataset.
Files saved in mounted Google Drive are not deleted.

### 2. Clone and install the project

The Colab VM cannot use the SSH key or SSH agent on the local machine. Install
GitHub CLI and authenticate through its browser/device flow:

```bash
sudo apt-get update
sudo apt-get install -y gh
gh auth login
```

Choose `GitHub.com`, `HTTPS`, authentication for Git operations, and browser
login. Open the URL shown by `gh`, enter its one-time code, and then clone and
install the project:

```bash
cd /content
gh repo clone CarlDavidRe/cv3d-project
cd cv3d-project
python -m pip install -r requirements.txt
python -m pip install -e .
```

Authentication and the clone under `/content` are lost when the Colab runtime
is recycled, so repeat this step in a new runtime. Commit and push source-code
changes before recycling a runtime if they need to be retained.

### 3. Upload and mount the NUM dataset

Colab cannot directly mount a directory from the local computer. In a **local**
terminal at the repository root, archive and verify the dataset:

```bash
cd data
tar -czf NUM.tar.gz NUM
ls -lh NUM.tar.gz
tar -tzf NUM.tar.gz | head
```

Do not use VS Code's `Upload to Colab` action for the NUM archive. That action
uses the extension's file API and large archives may exceed its memory, request,
or timeout limits. It remains useful for small files only.

In a browser, create `MyDrive/cv3d-datasets` in Google Drive and upload
`data/NUM.tar.gz` there. In VS Code, run
`Colab: Mount Google Drive to Server` from the command palette and execute the
cell it creates. Back in the Colab terminal, verify the archive and temporary
disk space:

```bash
ls -lh /content/drive/MyDrive/cv3d-datasets/NUM.tar.gz
df -h /content
```

### 4. Extract the dataset to temporary storage

Extract the archive onto Colab's faster temporary disk, then verify the NUM
directory:

```bash
mkdir -p /content/cv3d-project/data
tar -xzf /content/drive/MyDrive/cv3d-datasets/NUM.tar.gz \
  -C /content/cv3d-project/data
ls /content/cv3d-project/data/NUM
```

Reading the archive once and extracting it into `/content` is normally faster
than training against thousands of small files directly on Drive.

### 5. Run tests and frozen-feature smoke tests

Keep generated files in a semantic Phase 1 run directory until the workflow is
complete. Workflow step numbers are not used as directory names. First restore
the common cache so pretrained models and external sources can be reused. If
the Drive cache does not exist yet, skip the `rsync` command:

```bash
cd /content/cv3d-project
mkdir -p data/cache
rsync -av /content/drive/MyDrive/cv3d-project/data/cache/ data/cache/

mkdir -p outputs/phase1/feature_smoke/seed_0/metrics
set -o pipefail
python -m unittest discover -s tests -v 2>&1 \
  | tee outputs/phase1/feature_smoke/seed_0/tests.log

python scripts/inspect_features.py \
  --backbone imagenet_vit \
  --data-root data/NUM \
  --device cuda \
  | tee outputs/phase1/feature_smoke/seed_0/metrics/imagenet_vit.json
```

For VGGT, install the official repository in the runtime and run its smoke
test:

```bash
if [ ! -d /content/cv3d-project/data/cache/sources/vggt/.git ]; then
  git clone https://github.com/facebookresearch/vggt.git \
    /content/cv3d-project/data/cache/sources/vggt
fi
python -m pip install -e /content/cv3d-project/data/cache/sources/vggt

cd /content/cv3d-project
python scripts/inspect_features.py \
  --backbone vggt \
  --data-root data/NUM \
  --device cuda \
  | tee outputs/phase1/feature_smoke/seed_0/metrics/vggt.json
```

On Python 3.12, VGGT's NumPy requirement should install from a wheel. If pip
instead downloads `numpy-1.26.4.tar.gz` and spends a long time building it,
cancel the install and run:

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: "numpy==1.26.4"
python -m pip install --no-build-isolation \
  -e /content/cv3d-project/data/cache/sources/vggt
```

Model checkpoints are downloaded only when they are missing from
`data/cache/models`.

### 6. Run the probe-head overfit check

Run the Step 6 check after the smoke tests. Its checkpoint and metrics are
written under `outputs/phase1/probe_tiny/seed_0/`:

```bash
cd /content/cv3d-project
python scripts/train_probe.py \
  --config configs/experiments/phase1_probe_tiny.yaml
```

Use the VGGT override shown in Step 6 above if that is the backbone being
checked.

### 7. Run the Phase 1 comparison

After the tiny overfit check succeeds, run the complete configured backbone
comparison. On a fresh Colab VM, mount Google Drive and restore the previously
saved frozen-feature files into the repository cache first:

```bash
cd /content/cv3d-project
mkdir -p data/cache/features

if [ -d /content/drive/MyDrive/cv3d-project/data/cache/features ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/data/cache/features/ \
    data/cache/features/
else
  echo "No saved feature cache found; features will be extracted again."
fi
```

The trailing slashes copy the contents of the Drive feature directory into the
local `data/cache/features/` directory without creating an extra nested
`features` directory. Then run the compute-intensive Step 7 command:

```bash
cd /content/cv3d-project
python scripts/run_phase1.py \
  --config configs/experiments/phase1_sweep.yaml
```

The command checks compatible files under `data/cache/features/` before it
loads a backbone. It trains directly from cached variants and extracts only
the missing variants, sharing the frozen-backbone forward when several missing
variants use the same backbone. This is the default because
`probe.feature_cache.rebuild` is `false`. Compatible restored entries are reused
automatically; missing features are calculated by the runner.

### 8. Copy generated files to Google Drive

This is the final step before disconnecting or recycling the runtime. Copy the
entire output tree to persistent Drive storage and inspect the transferred
files:

```bash
mkdir -p /content/drive/MyDrive/cv3d-project/outputs
rsync -av /content/cv3d-project/outputs/ \
  /content/drive/MyDrive/cv3d-project/outputs/

mkdir -p /content/drive/MyDrive/cv3d-project/data/cache
rsync -av /content/cv3d-project/data/cache/ \
  /content/drive/MyDrive/cv3d-project/data/cache/

find /content/drive/MyDrive/cv3d-project/outputs -type f | sort
```

Colab's `/content` storage is temporary. The final copy preserves test logs,
feature summaries, checkpoints, metrics, and any other repository outputs in
`MyDrive/cv3d-project/outputs`; the second copy preserves feature caches,
pretrained-model downloads, and cached external sources under the same
`data/cache` name used locally. Run the copy after feature extraction finishes
so only complete cache files are persisted.
