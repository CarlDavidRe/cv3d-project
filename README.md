# Frozen-feature next-best-view study

This repository implements the phased project described in
[`project_overview.md`](project_overview.md). Phase 1 is stable. The first
Phase 2 infrastructure component—deterministic ground-truth surface sampling
and canonical-anchor visibility caching—is also available. The learned
policies and closed-loop evaluator remain future work.

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
http://localhost:8000/outputs/phase1/num_sample_3d.html
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

## Phase 2: surface visibility cache

The NUM download contains RGB images and uncertainty targets, but not its
ShapeNet meshes. Place the matching ShapeNetCore.v2 release at
`data/ShapeNetCore.v2`, retaining this layout:

```text
ShapeNetCore.v2/<category_id>/<object_id>/models/model_normalized.obj
```

### Download the correct ShapeNet release

Use the official gated
[`ShapeNet/ShapeNetCore`](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
dataset on Hugging Face. Its dataset card identifies it as **ShapeNetCore v2**
and provides the original per-category ZIP archives containing
`model_normalized.obj`. Do not substitute ShapeNetCore v1 or the separate
GLB/GLTF conversions: the PUN object IDs and this repository's mesh loader
expect the v2 OBJ layout above.

ShapeNet access is licensed for approved research/educational use. Before
downloading:

1. Sign in to Hugging Face and open the official dataset page linked above.
2. Review the terms, provide your real full name, PI/advisor, and affiliation,
   and request access. Approval is controlled by ShapeNet, not this project.
3. Install the current Hugging Face CLI and authenticate locally. Use a
   read-only user token if the browser login flow asks for one:

```bash
python3 -m pip install --upgrade huggingface_hub
hf auth login
hf auth whoami
```

The frozen NUM split uses only the following 13 ShapeNet synsets. Downloading
these archives is sufficient for every object referenced by
`data/splits/num_v1.json` and avoids downloading unrelated ShapeNet classes:

```bash
mkdir -p data/downloads/shapenetcore

hf download ShapeNet/ShapeNetCore \
  02691156.zip \
  02828884.zip \
  02933112.zip \
  02958343.zip \
  03001627.zip \
  03211117.zip \
  03636649.zip \
  03691459.zip \
  04090263.zip \
  04256520.zip \
  04379243.zip \
  04401088.zip \
  04530566.zip \
  --repo-type dataset \
  --local-dir data/downloads/shapenetcore
```

To download all ShapeNetCore v2 categories instead, use the following command.
The official repository is approximately 24 GB before extraction, so confirm
that substantially more free disk space is available first.

```bash
hf download ShapeNet/ShapeNetCore \
  --repo-type dataset \
  --include '*.zip' \
  --local-dir data/downloads/shapenetcore
```

Extract the downloaded category archives directly into the configured mesh
root:

```bash
mkdir -p data/ShapeNetCore.v2

for archive in data/downloads/shapenetcore/*.zip; do
  unzip -q -n "$archive" -d data/ShapeNetCore.v2
done
```

After extraction, verify a NUM object and the expected OBJ layout:

```bash
test -f \
  data/ShapeNetCore.v2/02691156/10155655850468db78d106ce0a280f87/models/model_normalized.obj \
  && echo "ShapeNetCore.v2 layout verified"
```

If that command fails, inspect the ZIP structure with
`unzip -l data/downloads/shapenetcore/02691156.zip | head`; the directory passed
as `paths.mesh_root` must be the directory immediately containing synset
folders such as `02691156/`, not a parent download/cache directory.

The defaults in `configs/experiments/phase2_visibility.yaml` reproduce the
official PUN generation geometry: normalized OBJ scale 2.0, camera radius
2.73, 30-degree pinhole field of view, near/far 1.2/4.0, and the existing
canonical 48-anchor order. The depth cache renders at 256×256 by default for
more stable depth consistency than the released 64×64 RGB images; this is a
configurable cache parameter and is recorded in metadata.

Generate one object and a debug SVG:

```bash
python3 scripts/precompute_visibility.py \
  --object 02691156/10155655850468db78d106ce0a280f87 \
  --debug-anchor 0
```

Generate a deterministic subset or complete configured split:

```bash
python3 scripts/precompute_visibility.py --split test --limit 10
python3 scripts/precompute_visibility.py --split test
```

Compatible `.npz` caches are skipped. A metadata or mesh-checksum mismatch is
reported as an error; pass `--overwrite` only when intentionally rebuilding.
All relevant parameters support the normal repeatable `--set KEY=VALUE`
overrides. Cache files are loaded without knowledge of the rendering backend:

```python
from nbv.data import load_visibility_cache

cache = load_visibility_cache(
    "02691156/10155655850468db78d106ce0a280f87",
    cache_root="data/cache/visibility",
)
seen = cache.visibility[[0, 12]].any(axis=0)
coverage = seen.mean()
```

## Step 5: model smoke tests

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

PUN is an end-to-end 48-value predictor rather than a frozen feature extractor,
so it has a separate but equivalent one-image smoke command. This verifies the
official checkpoint checksum, official preprocessing, output shape and finite
values, and reports the unmasked MSE against the selected NUM target:

```bash
python3 scripts/inspect_pun.py \
  --data-root data/NUM \
  --device auto
```

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
| Raw RGB | `flattened_rgb` |
| ImageNet ViT | `pooled_patch`, `max_pooled_patch`, `cls_token` |
| DINOv2 | `pooled_patch`, `max_pooled_patch`, `cls_token`, `pooled_register` |
| VGGT | `pooled_patch`, `max_pooled_patch`, `pooled_camera`, `pooled_register` |

For the raw-RGB baseline, `flattened_rgb` is a backbone-free 16×16 adaptive
average RGB grid flattened to 768 values. `pooled_patch` is mean pooling.
Camera and register sequences are also mean pooled. Selecting
`pooled_register` for DINOv2 requires a model variant that actually exposes
register tokens, such as `dinov2_vitb14_reg`. The default tiny-probe
configuration uses only `pooled_patch` for its learned backbones.

The raw-RGB grid, ImageNet ViT-B/16, and DINOv2 ViT-B/14 produce
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
validation checkpoint, and then evaluates the official pretrained PUN/UPNet
checkpoint as the final sweep entry. All entries use the same object-disjoint
test records and common comparison table:

```bash
python scripts/run_phase1.py \
  --config configs/experiments/phase1_sweep.yaml
```

For a quick CPU-only end-to-end check on real NUM records before launching the
full GPU sweep, use:

```bash
python scripts/run_phase1.py \
  --config configs/experiments/phase1_sweep.yaml \
  --set experiment.name=phase1_runner_smoke \
  --set probe.device=cpu \
  --set probe.max_samples_per_split=8 \
  --set 'probe.variants=[{name: raw_rgb_16x16_mlp, backbone: raw_rgb, components: [flattened_rgb]}]' \
  --set 'probe.baselines=[{name: train_mean_map, type: train_mean_map}]' \
  --set probe.training.epochs=3 \
  --set probe.training.batch_size=4 \
  --set probe.training.patience=2 \
  --set probe.evaluation.batch_size=8
```

This smoke command validates the runner and artifact contract, but it is not a
reportable experiment and does not replace the configured pretrained-backbone
sweep.

To smoke-test the official PUN checkpoint and common evaluator on CPU without
retraining PUN, keep one minimal raw-RGB runner entry and replace the baselines:

```bash
python scripts/run_phase1.py \
  --config configs/experiments/phase1_sweep.yaml \
  --set experiment.name=pun_checkpoint_smoke \
  --set probe.device=cpu \
  --set probe.max_samples_per_split=2 \
  --set 'probe.variants=[{name: raw_rgb_smoke, backbone: raw_rgb, components: [flattened_rgb]}]' \
  --set 'probe.baselines=[{name: pun_upnet, type: pun}]' \
  --set probe.training.epochs=1 \
  --set probe.training.batch_size=2 \
  --set probe.training.patience=1 \
  --set probe.pun.batch_size=2 \
  --set probe.pun.num_workers=0
```

The default sweep starts with two controls: `train_mean_map` repeats the
per-anchor mean of valid training targets without using an image, while
`raw_rgb_16x16_mlp` feeds a cached 768-value coarse RGB grid to the shared MLP.
It then compares mean-pooled patches and classification tokens for both
ImageNet ViT and DINOv2. For VGGT it compares mean-pooled patches, max-pooled
patches, the camera token, mean-pooled register tokens, and one
camera-plus-mean-patch combination. VGGT has no classification token; its
camera token is the model-specific global-token alternative. Variants using
the same backbone are extracted in one pass, so all five VGGT variants share
the same VGGT forward per image batch. Cache files are fingerprinted from the
backbone settings, preprocessing, layer, pooling, split manifest, target, and
sample IDs. They live under `data/cache/features/` by default.

The final `pun_upnet` entry does not train PUN locally. It loads the official
released `vit_small_patch16_224_PSNR_250425172703/best_vit_regressor.pth`
checkpoint, whose UPNet architecture is a timm ViT-S/16 with its classifier
removed and a 48-output linear regressor. The runner uses the same deterministic
timm preprocessing as the official inference script. The release is pinned by
Google Drive file ID and SHA-256 in the sweep config; a missing checkpoint is
downloaded atomically to `data/cache/models/pun/`, while a checksum mismatch is
rejected.

PUN is evaluated last and receives exactly the same validation/test NUM images,
source-view mask, target orientation, normalized regret, Spearman, NDCG@5, and
Huber/ranking evaluator as the project methods. Its official unmasked map MSE
is also reported as `official_unmasked_mse_loss`. No optimizer, local training
split, or early stopping is used for this entry. The released checkpoint does
not include a machine-readable training manifest, so possible overlap between
its original training data and this project's held-out objects/categories is
recorded as a comparability limitation in `summary.json`.

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

The runner is resumable at the entry level. If a variant or baseline directory
already contains `best.pt`, `training_history.json`, `summary.json`, and
`test_per_sample.csv` matching the configured entry and target, its saved
metrics are reused and that entry is not trained or evaluated again. Incomplete
or incompatible artifact sets are recomputed.

The checked-in seed-0 artifacts predate PUN integration: they contain 41,760
training, 5,232 validation, and 14,400 test samples; one analytical baseline
plus ten locally trained variants; eleven artifact sets; and twenty-one
training SVGs. Rerun the current config to produce PUN as the twelfth and final
comparison row. Treat the single-seed result as preliminary until the final
comparison is repeated across the chosen report seeds.

Every learned variant's `training_history.json` contains an epoch-zero baseline
and one row per completed epoch. It records the batch-time
`optimization_train_loss` separately from the post-epoch, evaluation-mode
`train_loss`; the latter is directly comparable with `validation_loss` because
both use fixed end-of-epoch weights with dropout disabled. The history also
contains train/validation Huber and ranking loss components, learning rate, and
validation normalized regret, Spearman, and NDCG@5. The target-only mean-map
baseline has an empty history because it is not optimized.
The pretrained PUN entry also has an empty history because the sweep performs
inference only; its small `best.pt` is a checksum-pinned descriptor pointing to
the official checkpoint in the shared model cache.

The runner creates these dependency-free SVG diagnostics under
`figures/training/`:

```text
<variant>_losses.svg
<variant>_validation_metrics.svg
validation_loss_comparison.svg
```

The loss figure shows comparable evaluation-mode train and validation curves,
the separate batch-time optimization curve, and decomposed Huber/ranking
losses. The metrics figure shows validation ranking behavior. Each per-variant
figure marks the best epoch selected by validation loss and the last completed
epoch (identified as an early-stop point when applicable), while the comparison
figure overlays validation loss for all learned variants.

There is intentionally no test-loss-over-time curve. The test split is
evaluated only on the restored validation-selected checkpoint, and its final
loss/ranking metrics are stored in `summary.json` and the comparison table.
Use validation diagnostics—not test results—for early stopping and tuning. If
the Step 8 demo uses test examples, fix their sample IDs without inspecting
per-sample test results.

This step deliberately stops before policies, geometry, closed-loop
evaluation, or multi-view processing. Phase 2 must separately add the official
PUN history aggregation rule.

## Step 8: visualize a completed Phase 1 experiment

Pass the completed experiment directory as the positional CLI argument. With no
other arguments, the command discovers every complete saved variant (including
fixed baselines) and generates one plot per variant for the first deterministic
validation sample:

```bash
python scripts/visualize_phase1.py \
  outputs/phase1/backbone_sweep/seed_0
```

Use one or more repeatable `--variant` filters to generate only selected plots,
or choose a preselected validation sample explicitly:

```bash
python scripts/visualize_phase1.py \
  outputs/phase1/backbone_sweep/seed_0 \
  --variant vggt_max_pooled_patch \
  --variant raw_rgb_16x16_mlp \
  --split val \
  --sample-id 02691156/154146362c18b3c447fdda991f503a6b/0
```

For each learned variant, the command first looks for a compatible feature cache
containing that sample. Raw-RGB features can always be recreated in memory. To
complete the default all-variant run, every pretrained variant must have a
compatible cache; otherwise restore the caches from the training environment or
explicitly allow one in-memory backbone forward per missing feature with
`--extract-missing-features`. That option may require model checkpoints and a
suitable GPU, but it still does not write feature caches.

The only generated artifacts are self-contained SVGs under
`<experiment>/figures/predictions/<split>/<category>/<object>/<view>/<variant>.svg`
unless `--output-dir` is supplied. Each object/view has one directory containing
all generated variant SVGs. For example, the sample
`02691156/154146362c18b3c447fdda991f503a6b/1` stores its variants under
`figures/predictions/val/02691156/154146362c18b3c447fdda991f503a6b/1/`,
including `dinov2_pooled_patch.svg`.
The single-file `--output` option is available only when exactly one
`--variant` is selected. Each SVG contains:

- the input RGB image,
- ground-truth and predicted 48-anchor utility maps on one shared scale,
- an absolute utility-error map,
- the predicted and ground-truth top candidates,
- normalized regret, Spearman, NDCG@5, raw-target MAE, and top-five rankings.

The default uses validation index 0 for every variant, so comparisons use the
same reproducible example and are not selected by inspecting test results. The
CLI reads the saved config, checkpoints, summaries, dataset, and optional
feature caches; it does not train, change metrics, update checkpoints, or write
caches.

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

### 5. Run tests and model smoke tests

Keep generated files in a semantic Phase 1 run directory until the workflow is
complete. Workflow step numbers are not used as directory names. Restore only
the frozen-feature cache from Drive; model downloads and external-source caches
remain local to the runtime. If the Drive feature cache does not exist yet, the
conditional prints a message and continues:

```bash
cd /content/cv3d-project
mkdir -p data/cache/features
if [ -d /content/drive/MyDrive/cv3d-project/data/cache/features ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/data/cache/features/ \
    data/cache/features/
else
  echo "No saved feature cache found; features will be extracted again."
fi

mkdir -p outputs/phase1/feature_smoke/seed_0/metrics
set -o pipefail
python -m unittest discover -s tests -v 2>&1 \
  | tee outputs/phase1/feature_smoke/seed_0/tests.log

python scripts/inspect_features.py \
  --backbone imagenet_vit \
  --data-root data/NUM \
  --device cuda \
  | tee outputs/phase1/feature_smoke/seed_0/metrics/imagenet_vit.json

python scripts/inspect_pun.py \
  --data-root data/NUM \
  --device cuda \
  | tee outputs/phase1/feature_smoke/seed_0/metrics/pun_upnet.json
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
comparison. On a fresh Colab VM, mount Google Drive and manually restore both
previous experiment outputs and the saved frozen-feature files before starting
the runner:

```bash
cd /content/cv3d-project
mkdir -p outputs data/cache/features

if [ -d /content/drive/MyDrive/cv3d-project/outputs ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/outputs/ outputs/
else
  echo "No saved experiment outputs found; all variants will run."
fi

if [ -d /content/drive/MyDrive/cv3d-project/data/cache/features ]; then
  rsync -av /content/drive/MyDrive/cv3d-project/data/cache/features/ \
    data/cache/features/
else
  echo "No saved feature cache found; features will be extracted again."
fi
```

The trailing slashes copy directory contents without creating an extra nested
`outputs` or `features` directory. The runner reuses each complete restored
variant and trains only variants whose required result artifacts are missing or
incompatible. Then run the compute-intensive Step 7 command:

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
automatically; missing features are calculated by the runner. Inspect
`figures/training/validation_loss_comparison.svg` and each variant's loss and
validation-metric SVGs for underfitting, divergence, or a widening
train/validation gap.

### 8. Copy generated files to Google Drive

This is the final step before disconnecting or recycling the runtime. Copy the
entire output tree to persistent Drive storage and inspect the transferred
files:

```bash
mkdir -p /content/drive/MyDrive/cv3d-project/outputs
rsync -av /content/cv3d-project/outputs/ \
  /content/drive/MyDrive/cv3d-project/outputs/

mkdir -p /content/drive/MyDrive/cv3d-project/data/cache/features
rsync -av /content/cv3d-project/data/cache/features \
  /content/drive/MyDrive/cv3d-project/data/cache/features

find /content/drive/MyDrive/cv3d-project/outputs -type f | sort
```

Colab's `/content` storage is temporary. The final copy preserves test logs,
feature summaries, checkpoints, metrics, and any other repository outputs in
`MyDrive/cv3d-project/outputs`; the second copy preserves feature caches,
pretrained-model downloads, and cached external sources under the same
`data/cache` name used locally. Run the copy after feature extraction finishes
so only complete cache files are persisted.
