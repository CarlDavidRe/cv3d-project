# NUM RGB / face-visibility camera alignment

Visibility generation and closed-loop evaluation use one camera convention
derived from PUN, with canonical 48-anchor IDs/directions and the existing
rasterized face-based `Vis`/`VisA` definitions. Prepared meshes are centered by
their bounding boxes before scaling. This project normalization is supported
by independent RGB/mesh checks on six validation objects; it is not a claim
of pixel-perfect agreement across the dataset.

## Calibration

1. `BlenderInterface` calls its focal-length setter with
   `(525 / 512) * resolution`. The setter converts this
   pixel focal length to the Blender lens value. For square RGB images the
   actual horizontal and vertical FOV is
   `2 * atan(512 / (2 * 525)) = 51.98948897809546°`.
2. `xyz2pose` uses `Vector(-position).to_track_quat('-Z', 'Y')`. Blender's
   tracking implementation generally aligns camera Y to the projection of
   **world Z** onto the image plane. Camera Y is the local up axis.

Sources are pinned to PUN revision
`aa6f8f4f12154854a4c1867209725c80475af102`:

- [RGB generator](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/aa6f8f4f12154854a4c1867209725c80475af102/fep_nbv/uncertainty_map_generation/single_rotation_generation.py)
- [Blender interface and actual focal-length assignment](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/aa6f8f4f12154854a4c1867209725c80475af102/08-vit-train/blender_utils/blender_interface.py)
- [Pixel focal length to Blender lens conversion](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/aa6f8f4f12154854a4c1867209725c80475af102/08-vit-train/blender_utils/util.py)
- [Camera tracking call](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/aa6f8f4f12154854a4c1867209725c80475af102/fep_nbv/utils/transform_viewpoints.py)
- [Blender tracking implementation](https://github.com/blender/blender/blob/v3.6.0/source/blender/blenlib/intern/math_rotation.c)

The mesh scale is **2.0**, and the camera radius is **2.73**. PUN
sets `object_world_matrix` to identity after importing the OBJ, then scales it.
The Blender OBJ importer stores axis conversion in `matrix_world`,
which this assignment replaces. The project first centers the prepared PLY
coordinates, then applies the uniform scale. Exact equivalence to the original
OBJ coordinates rendered for NUM has not been established. The pinned
`import_mesh()` has its centering operations commented out; our centering rule
is therefore a project normalization supported by the validation below, not
a claim that those source lines execute. See the
[importer](https://github.com/blender/blender-addons/blob/v3.6.0/io_scene_obj/import_obj.py).

## Pole convention and reference checks

At anchors 0 and 46, world Z has no projection onto the image plane. A tracking
quaternion can choose different 180° rolls depending on numerical residuals
and signed zeros. The released NUM RGB uses world +Y upward at both poles.
We explicitly set the rotations to identity at anchor 0 and
`diag(-1, 1, -1)` at anchor 46.

This is a documented stabilization to match the released data, rather than a
claim that every platform running the PUN source returns those pole values.
The independent local reference returned a 180°-rolled north pole that does
not match the downloaded RGB. Both alternatives were inspected for the three
fixed diagnostic objects; the released-data convention is retained.

The other 46 poses are checked against an independent reference generated from
PUN's HEALPix/rotation path, HEALPix 1.18.1 and Blender mathutils 3.3.0. All
rotation/translation entries agree within `1e-6`. The reference is saved in
`tests/fixtures/num_camera_reference.json`; neither reference package is a
project/runtime dependency. The implementation uses NumPy only.

`Anchor.camera_to_world()` implements this NUM pose. Visibility generation,
visibility diagnostics, and the closed-loop evaluator use
`anchor_camera_to_world()`, which validates the cache's convention and delegates
to the same implementation. Unsupported conventions are rejected.

## Mesh normalization and cache provenance

`prepare_visibility_mesh()` defines one mesh-only transform:

```text
center = (source_bounds_min + source_bounds_max) / 2
world_vertex = mesh_scale * (source_vertex - center)
```

The default `mesh_centering` is `bounding_box`, with `mesh_scale: 2.0`.
No RGB-derived offsets, rotations, per-view fitting, or per-object scale changes
are used. Face IDs and winding are preserved; face areas scale by the square
of the configured scale. Precomputation and alignment diagnostics share this
implementation. Debug SVGs receive the same prepared mesh, and the evaluator
uses its generated face masks with cameras looking at the centered origin.

The camera convention is
`opengl_blender_track_world_z_num_poles_pixel_centers`. Both Phase 2 configs
use `data/cache/visibility`. Schema-2 caches record the source mesh checksum,
`mesh_centering`, `mesh_scale`, and the actual 4×4 `mesh_to_world` transform,
as well as camera settings and rasterizer provenance. Precomputation validates
the transform; evaluation requires the configured centering convention.
Missing or incompatible centering metadata requires a rebuild, not relabeling.
Rollout fingerprints include the resulting geometry arrays and metadata.

Precompute the configured split, reusing compatible completed caches:

```bash
python scripts/precompute_visibility.py --split test
```

Use `--overwrite` when rebuilding incompatible entries. The three locally
available caches, their debug SVGs, and the Random/Oracle subset run have been
regenerated with centering. Full-split precomputation remains separate.

## Independent validation

The normalization rule was first identified using three diagnostic test
objects. Before adopting it, six different validation objects were selected
without inspecting their overlap: the lexicographically first object in each
of the first six sorted validation categories in `data/splits/num_v1.json`.
The camera settings and anchors `[0, 1, 12, 24, 46]` were fixed, including both
poles. None of these six objects belongs to the discovery cohort.

The comparison uses the same mesh, scale, camera, rasterization, and RGB
foreground threshold on both sides; only bounding-box centering differs.
Mean IoU improves from **51.29% to 86.79%** over the 30 validation views.
Every object's five-view mean improves:

| Category / object | Scaled source coordinates | Centered coordinates |
| --- | ---: | ---: |
| `02691156/154146362c18b3c447fdda991f503a6b` | 27.65% | 77.78% |
| `02828884/1ac6a3d5c76c8b96edccc47bf0dcf5d3` | 74.43% | 76.04% |
| `02933112/180154895560cd0cc59350d819542ec7` | 62.65% | 93.43% |
| `03211117/246f0b722dff7f16fb7ec8bd3f8b241d` | 60.51% | 90.85% |
| `03636649/1a5778ab63b3c558710629adc6653816` | 43.00% | 87.37% |
| `03691459/1e8aea643deed7cc94c70e7fd262be3` | 39.49% | 95.29% |

Reproduce the independent comparison:

```bash
python scripts/validate_visibility_alignment.py --split val --compare-source-mesh \
  --output-dir outputs/phase2/visibility_alignment/validation \
  --object 02691156/154146362c18b3c447fdda991f503a6b \
  --object 02828884/1ac6a3d5c76c8b96edccc47bf0dcf5d3 \
  --object 02933112/180154895560cd0cc59350d819542ec7 \
  --object 03211117/246f0b722dff7f16fb7ec8bd3f8b241d \
  --object 03636649/1a5778ab63b3c558710629adc6653816 \
  --object 03691459/1e8aea643deed7cc94c70e7fd262be3
```

`--split` verifies object membership. The report saves the split checksum,
source mesh/image checksums, applied transforms, and individual view metrics.
Validation does not build missing caches. The results support this convention
for the project; they do not establish exact original-OBJ equivalence or
all-object validation. The discovery bench's small IoU regression and residual
thin-surface differences remain part of that limitation.

## Current geometry and rollout checks

The three discovery objects also provide a cache/replay check:

```bash
python scripts/validate_visibility_alignment.py --split test \
  --object 02691156/1628b65a9f3cd7c05e9e2656aff7dd5b \
  --object 02828884/1b9ddee986099bb78880edc6251fa529 \
  --object 02933112/18d94e539b0ed30d105e720ebc569399
```

The report at `outputs/phase2/visibility_alignment/report.json` records mean
IoU **78.22%** across these 15 views. For present caches it requires exact
agreement with fresh face-ID renders, gain/coverage identities, zero acquired
gains, and order-independent unions. These are internal consistency checks
alongside the separate RGB comparison.

`outputs/phase2/random_oracle/seed_0` contains fresh Random/Oracle evaluation
on the three objects. Runs record
`geometry_validation_status: num_camera_bbox_centered`; completeness of the
requested cohort and full split are recorded separately. Learned-policy and
full-split results remain future work.

## Resolution, boundary, and centering diagnostics

The overlap diagnostic thresholds the original 64×64 RGB foreground at
`min(R,G,B) < 245`. The configured 256×256 face-ID silhouette is box-downsampled
to RGB resolution, counting any covered subpixel. This diagnostic mask is not
the face-based coverage metric. Thin surfaces, antialiasing, white surfaces,
and prepared-mesh differences can affect RGB/silhouette agreement.

The same command renders directly at RGB resolution, tests majority-covered
subpixels, and sweeps RGB thresholds 230, 245, and 254. It records areas,
bounding boxes, centroids, and symmetric nearest-boundary distances in RGB
pixels. Boundaries use four-neighbor connectivity; the distance summary pools
both directions and reports mean, 95th percentile, and fraction within one
pixel. Native resolution is an additional diagnostic, not a replacement for
the configured visibility rasterization.

A post-hoc translation search tests integer offsets up to eight pixels per
axis. Positive x means right and positive y means down. Translations that crop
foreground are excluded. Shared and per-view optima are recorded separately.
These fitted scores diagnose residual offsets; they never update production
geometry, caches, or policies and are not used as alignment validation scores.

Figures show RGB, configured overlay, native-resolution overlay, and the
native overlay after its best diagnostic shift. With `--compare-source-mesh`,
a fifth panel shows the scaled source coordinates before centering. Green is
agreement, red is RGB-only foreground, and blue is mesh-only foreground.
