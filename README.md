# Frozen-feature next-best-view study

This repository implements the phased project described in
[`project_overview.md`](project_overview.md). Development is currently limited
to Phase 1 (the single-image feature probe).

## Setup

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

Run the infrastructure tests with:

```bash
python3 -m unittest discover -s tests -v
```

The canonical PUN-compatible 48-anchor definition now lives in
`src/nbv/geometry/anchors_v1.csv`.

## NUM dataset loader

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
sphere in `outputs/phase1/num_sample_3d.html`:

```bash
python3 scripts/inspect_num_sample.py \
  --data-root data/NUM \
  --split train \
  --split-manifest data/splits/num_v1.json \
  --target PSNR \
  --output outputs/phase1/num_sample.svg
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
git clone https://github.com/facebookresearch/vggt.git ../vggt
python3 -m pip install -e ../vggt
python3 scripts/inspect_features.py \
  --backbone vggt \
  --data-root data/NUM \
  --device cuda
```

The VGGT smoke test processes each NUM image as a one-frame sequence, which is
the required independent single-image behavior for Phase 1. It does not perform
the joint multi-view processing reserved for Phase 3.

### Google Colab from VS Code

Connect a VS Code notebook to a GPU using the official Colab extension, then
open a Colab terminal with the `Colab: Open Terminal` command. Commands in this
section run in that remote terminal, not in the local VS Code terminal.

#### Select Python 3.12

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

On Python 3.12, VGGT's NumPy requirement should install from a wheel. If pip
instead downloads `numpy-1.26.4.tar.gz` and spends a long time building it,
cancel the install and require the binary wheel explicitly:

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: "numpy==1.26.4"
python -m pip install --no-build-isolation -e /content/vggt
```

#### Clone this private repository with GitHub CLI

The Colab VM cannot use the SSH key or SSH agent on the local machine. Install
GitHub CLI and authenticate through its browser/device flow instead:

```bash
sudo apt-get update
sudo apt-get install -y gh
gh auth login
```

Choose `GitHub.com`, `HTTPS`, authentication for Git operations, and browser
login. Open the URL shown by `gh`, enter its one-time code, and then clone:

```bash
cd /content
gh repo clone CarlDavidRe/cv3d-project
cd cv3d-project

python -m pip install -r requirements.txt
python -m pip install -e .
```

Authentication and the clone under `/content` are lost when the Colab runtime
is recycled, so repeat these steps in a new runtime. Changes must be committed
and pushed before another runtime can clone them.

#### Make the local NUM dataset available

Colab cannot directly mount a directory from the local computer. Archive the
dataset locally from this repository and verify the archive before uploading:

```bash
cd data
tar -czf NUM.tar.gz NUM
ls -lh NUM.tar.gz
tar -tzf NUM.tar.gz | head
```

Do not use VS Code's `Upload to Colab` action for the NUM archive. That action
uses the extension's file API and large archives may exceed its memory, request,
or timeout limits. It remains useful for small files only.

In a web browser, create `MyDrive/cv3d-datasets` in Google Drive and upload
`data/NUM.tar.gz` there. In VS Code, run
`Colab: Mount Google Drive to Server` from the command palette and execute the
cell it creates. Verify both the archive and available temporary disk space in
the Colab terminal:

```bash
ls -lh /content/drive/MyDrive/cv3d-datasets/NUM.tar.gz
df -h /content
mkdir -p /content/drive/MyDrive/cv3d-project/outputs/step5
```

Extract the Drive archive onto Colab's faster temporary disk and verify the NUM
directory:

```bash
mkdir -p /content/cv3d-project/data
tar -xzf /content/drive/MyDrive/cv3d-datasets/NUM.tar.gz \
  -C /content/cv3d-project/data
ls /content/cv3d-project/data/NUM
```

Reading the archive once and extracting it into `/content` is normally faster
than training against thousands of small files directly on Drive. Run the test
suite and save its log to Drive:

```bash
cd /content/cv3d-project
set -o pipefail
python -m unittest discover -s tests -v 2>&1 | tee \
  /content/drive/MyDrive/cv3d-project/outputs/step5/tests.log
```

Use the extracted dataset for all backbone smoke tests. The current Step 5
script prints a JSON feature-shape summary rather than caching feature tensors,
so save that summary to Drive with `tee`:

```bash
cd /content/cv3d-project
python scripts/inspect_features.py \
  --backbone imagenet_vit \
  --data-root data/NUM \
  --device cuda \
  | tee /content/drive/MyDrive/cv3d-project/outputs/step5/imagenet_vit.json
```

For VGGT, install the official repository in the Colab runtime first:

```bash
git clone https://github.com/facebookresearch/vggt.git /content/vggt
python -m pip install -e /content/vggt

cd /content/cv3d-project
python scripts/inspect_features.py \
  --backbone vggt \
  --data-root data/NUM \
  --device cuda \
  | tee /content/drive/MyDrive/cv3d-project/outputs/step5/vggt.json
```

Any other command that writes files under the repository's temporary
`outputs/` directory can be synchronized to Drive before disconnecting:

```bash
mkdir -p /content/drive/MyDrive/cv3d-project/outputs/repository
rsync -av /content/cv3d-project/outputs/ \
  /content/drive/MyDrive/cv3d-project/outputs/repository/
```

Model checkpoints are downloaded only on their first use in each runtime.
Colab's `/content` storage is temporary; the commands above keep logs and
summaries under `MyDrive/cv3d-project/outputs`, where they survive runtime
recycling.
