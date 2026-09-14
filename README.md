# CV3D Next-Best-View Dashboard

This repository contains an interactive Streamlit dashboard for exploring the
CV3D next-best-view experiments and their saved results.

## Requirements

- Git
- Python 3.10 or newer, including the `venv` module
- `make`

A GPU is not required. The dashboard displays saved experiment results and does
not run model inference.

## Clone the repository

Using HTTPS:

```bash
git clone https://github.com/CarlDavidRe/cv3d-project.git
cd cv3d-project
```

If you use GitHub SSH authentication, clone with
`git@github.com:CarlDavidRe/cv3d-project.git` instead.

## Run the dashboard

From the repository root, run:

```bash
make dashboard
```

This single command creates `.venv`, installs every Python package needed by
the dashboard, and starts Streamlit. Open the local URL printed in the terminal,
normally <http://localhost:8501>.

Press `Ctrl+C` to stop the dashboard. Run `make dashboard` again whenever you
want to restart it.

## Dashboard data

No separate data-download command is required to run the dashboard. The clone
already includes all mandatory dashboard inputs:

- dataset split metadata and the 48 camera anchors;
- the compact Phase 1 prediction export;
- saved Phase 1, Phase 2, and Phase 3 results and rollout replays.

The large raw datasets are intentionally not required. Without them, the full
dashboard still runs; only raw RGB previews and ShapeNet mesh overlays may show
as unavailable.

To enable raw RGB previews, download and extract the optional public NUM dataset
with this **one executable command** from the repository root:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install --upgrade gdown && .venv/bin/gdown 1nuYhfQ7oN3ZPvNU0ViNfGsfgiwD6kHpS -O /tmp/NUMv2.zip && mkdir -p data && .venv/bin/python -m zipfile -e /tmp/NUMv2.zip data
```

After extraction, the images must be located at
`data/NUM/<category_id>/<object_id>/images/`.

ShapeNet mesh overlays are also optional. ShapeNetCore v2 is gated and cannot
be downloaded anonymously: request access and accept its terms on the
[official ShapeNetCore v2 page](https://huggingface.co/datasets/ShapeNet/ShapeNetCore),
then place the prepared ASCII PLY meshes at
`data/ShapeNetCore.v2/<category_id>/<object_id>/models/model_normalized.ply`.

For the longer training and Google Colab workflow, see
[README_manual.md](README_manual.md).

## Manual launch

If `make` is unavailable, the equivalent setup is:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dashboard.txt
.venv/bin/python -m streamlit run dashboard/app.py
```
