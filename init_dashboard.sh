#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repository_root"

for command_name in python3 make; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'error: required command not found: %s\n' "$command_name" >&2
    exit 1
  fi
done

python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else "Python 3.10 or newer is required.")'

required_paths=(
  dashboard/app.py
  dashboard/data/phase1_test_predictions.npz
  dashboard_data/NUM
  dashboard_data/ShapeNetCore.v2
  dashboard_rendering_objects/README.md
  data/splits/num_v1.json
  outputs/gaussian_splatting_per_view_budget
  outputs/gaussian_splatting_per_view_budget_alignment_repair
  outputs/phase1/backbone_sweep
  outputs/phase2/phase2_closed_loop_reconstruction
  outputs/phase3/controlled_history_comparison_reconstruction
  outputs/phase3/controlled_pose_deepsets
  outputs/phase3/controlled_token_attention
  requirements-dashboard.txt
  src/nbv/geometry/anchors_v1.csv
)

for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'error: sparse checkout is missing %s\n' "$required_path" >&2
    printf 'rerun the sparse-checkout command from README.md, then retry.\n' >&2
    exit 1
  fi
done

python3 -m venv .venv
if ! .venv/bin/python -c 'import altair, numpy, pandas, plotly, streamlit' 2>/dev/null; then
  .venv/bin/python -m pip install -r requirements-dashboard.txt
fi
.venv/bin/python -c 'import altair, numpy, pandas, plotly, streamlit'

printf '\nDashboard initialization complete. Start it with: make dashboard\n'
