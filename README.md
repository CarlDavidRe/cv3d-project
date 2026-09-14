# CV3D Next-Best-View Dashboard

This repository contains an interactive Streamlit dashboard for exploring the
CV3D next-best-view experiments and their saved results.

## Requirements

- Git
- Python 3.10 or newer
- `make`

A GPU is not required. The dashboard displays saved experiment results and does
not run model inference.

## Minimal dashboard checkout

Use a shallow, blob-filtered sparse checkout so Git downloads only the source,
data, and saved-result trees used by the dashboard:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/CarlDavidRe/cv3d-project.git
cd cv3d-project
git sparse-checkout set --no-cone \
  '/*' \
  '!/*/' \
  '/dashboard/' \
  '/dashboard_data/' \
  '/dashboard_rendering_objects/' \
  '/data/splits/' \
  '/src/nbv/geometry/' \
  '/outputs/phase1/backbone_sweep/' \
  '/outputs/phase2/phase2_closed_loop_reconstruction/' \
  '/outputs/phase3/controlled_history_comparison_reconstruction/' \
  '/outputs/phase3/controlled_pose_deepsets/' \
  '/outputs/phase3/controlled_token_attention/' \
  '/outputs/gaussian_splatting_per_view_budget/' \
  '/outputs/gaussian_splatting_per_view_budget_alignment_repair/' \
  '!/outputs/phase3/controlled_history_comparison_reconstruction/**/joint_feature_cache/**' \
  '!/outputs/phase3/controlled_pose_deepsets/**/joint_feature_cache/**' \
  '!/outputs/phase3/controlled_token_attention/**/joint_feature_cache/**'
```

The non-cone patterns omit the Phase 3 `joint_feature_cache` artifacts, which
the dashboard does not use. Git still includes the root-level `Makefile`,
dependency list, and READMEs. For GitHub SSH authentication, replace the HTTPS
URL with `git@github.com:CarlDavidRe/cv3d-project.git`.

## Initialize and run the dashboard

From the repository root, initialize the dashboard once and then start it:

```bash
./init_dashboard.sh
make dashboard
```

The initialization script checks the sparse checkout and installs the dashboard
dependencies. `make dashboard` then starts Streamlit. Open the local URL printed
in the terminal, normally <http://localhost:8501>.

Press `Ctrl+C` to stop the dashboard. Run `make dashboard` again whenever you
want to restart it.

## Dashboard data

No separate data-download command is required to run the dashboard. The clone
already includes all mandatory dashboard inputs:

- dataset split metadata and the 48 camera anchors;
- 48 NUM RGB observations and one ShapeNet mesh for each of the 20 dashboard
  rendering objects, under `dashboard_data/`;
- the compact Phase 1 prediction export;
- saved Phase 1, Phase 2, and Phase 3 results and rollout replays.

The large raw datasets are intentionally not required. Dataset exploration and
rendering are limited to the allowlist in
[`dashboard_rendering_objects/README.md`](dashboard_rendering_objects/README.md).
