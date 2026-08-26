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

### Google Colab

Select a GPU runtime, upload or clone this repository, and run these cells from
the project root:

```bash
!python -m pip install -r requirements.txt
!python -m pip install -e .
!python -m unittest discover -s tests -v
```

Mount Drive if the NUM dataset is stored there, and pass its absolute directory
to `--data-root`. For VGGT, install the official repository and run the smoke
test:

```bash
!git clone https://github.com/facebookresearch/vggt.git /content/vggt
!python -m pip install -e /content/vggt
!python scripts/inspect_features.py --backbone vggt \
    --data-root /content/drive/MyDrive/path/to/NUM --device cuda
```

Model checkpoints are downloaded only on the first run and use the runtime's
normal torch/huggingface caches. Colab's local storage is temporary, so point
those caches at Drive if you do not want to download the large VGGT checkpoint
again in a later session.
