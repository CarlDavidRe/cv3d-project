# Minimal Dashboard Data

This directory is the self-contained raw-data subset used by the Streamlit
dashboard. It includes only the objects listed in
[`dashboard_rendering_objects/README.md`](../dashboard_rendering_objects/README.md).

## Contents

- `NUM/`: 48 `viewpoint_<anchor>_offset_phi_0.png` observations per object,
  extracted from the local NUM v2 dataset. Training targets, example renders,
  and uncertainty JSON files are deliberately omitted.
- `ShapeNetCore.v2/`: one prepared ASCII `model_normalized.ply` mesh per object,
  extracted from the local ShapeNetCore v2 dataset.

There are 20 objects, 960 PNG observations, and 20 PLY meshes. The dashboard
does not load `data/cache/visibility`; its visibility values and acquisition
histories come from saved evaluation outputs, so copying the cache would only
increase the checkout.

Keep the category/object hierarchy unchanged. `dashboard/app.py` resolves the
assets directly from this directory.

## Archive

From the repository root:

```bash
tar -czf dashboard_data.tar.gz dashboard_data
```

To restore it, extract the archive at the repository root. The first asset path
should then be:

```text
dashboard_data/NUM/02691156/1628b65a9f3cd7c05e9e2656aff7dd5b/images/viewpoint_0_offset_phi_0.png
```

ShapeNetCore v2 has its own license and access terms. Preserve those terms when
redistributing or publishing the extracted meshes.
