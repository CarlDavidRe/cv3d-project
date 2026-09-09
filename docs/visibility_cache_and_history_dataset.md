# Visibility Cache and History Dataset

This document explains what the visibility cache represents and how it is used
to generate the Phase 3 history dataset's ground-truth surface-gain labels.

## Visibility cache

The visibility cache is a precomputed geometric lookup table for one object. It
records which mesh faces are visible from each of the 48 canonical camera
anchors. Precomputing this information avoids rasterizing the object's mesh
again for every sampled observation history.

Each cache contains:

- `face_visibility`: a Boolean array with shape `[48, number_of_faces]`.
  `face_visibility[j, f]` is true when face `f` is visible from anchor `j`.
- `face_areas`: the area of each triangular mesh face.
- `anchor_ids`: the canonical anchor ordering `[0, ..., 47]`.
- Metadata describing the object, mesh, camera, renderer, resolution,
  visibility definition, and other generation provenance.

The cache structure and validation are implemented in
`src/nbv/data/visibility_cache.py`.

### How face visibility is determined

The mesh is rendered once from each canonical anchor using a triangle-ID
z-buffer. Every covered pixel stores the ID of the nearest triangle. A face is
considered visible from an anchor if its ID wins the z-buffer for at least one
pixel. Faces hidden behind other geometry are therefore not marked visible.

This rasterization is implemented in `src/nbv/geometry/visibility.py` and
performed by `scripts/precompute_visibility.py`.

## Coverage definitions

For an acquired history of anchors

```text
H = {h_1, ..., h_t}
```

the already-observed faces are the union of the corresponding visibility
masks:

```text
seen[f] = OR(face_visibility[h, f] for h in H)
```

The project supports two ways of assigning a normalized weight to every face:

```text
weight[f] = 1 / number_of_faces                 # Vis
weight[f] = face_area[f] / total_mesh_area      # VisA
```

`Vis` treats every triangle equally. `VisA` weights each triangle by its share
of the complete mesh area. Coverage is the sum of the weights of all faces
observed by at least one acquired view:

```text
Coverage(H) = sum(weight[f] for f where seen[f])
```

The Phase 3 history configuration currently uses `vis_a`. Coverage and gain
calculations are implemented in `src/nbv/geometry/coverage.py`.

## Generating history-dataset targets

For every object and configured history length, the history builder samples a
deterministic, unique sequence of previously acquired anchors. For example:

```text
history = [3, 17, 29, 8]
```

The associated RGB images become the model input. The supervision is calculated
from the object's visibility cache rather than from those RGB images.

For every candidate anchor `j`, the builder identifies the faces visible from
that candidate that have not yet been observed:

```text
new_faces[j, f] = face_visibility[j, f] AND NOT seen[f]
```

The candidate's target is the normalized weight of those newly visible faces:

```text
target_surface_gain[j]
    = sum(weight[f] for f where new_faces[j, f])
    = Coverage(H union {j}) - Coverage(H)
```

This is calculated for all 48 candidates at once. In array form, the essential
operation is:

```python
((face_visibility & ~seen) * weights[None, :]).sum(axis=1)
```

For example:

```text
History sees faces:       {A, B, C}
Candidate 10 sees:        {B, C, D, E}
New faces from candidate: {D, E}
Candidate 10 target:      weight(D) + weight(E)
```

An already-acquired anchor has zero gain because every face visible from it is
already part of `seen`. Acquired anchors and globally invalid anchors are also
set to false in `valid_candidate_mask` so that they cannot be selected as the
next view.

The dataset generation path is implemented in
`src/nbv/data/history_dataset.py`, with `cache.candidate_gains(...)` producing
the 48-element target vector.

## Resulting sample

Each logical history sample contains:

```yaml
object_id: str
history_image_paths: str[history_length]
history_anchor_ids: int[history_length]
history_length: int
target_surface_gain: float[48]
valid_candidate_mask: bool[48]
visibility_cache_id: str
split: train|val|test
```

The RGB history is therefore the model input, while the visibility cache
provides the ground-truth geometric supervision.

## Reproducibility and consistency

Every cache has a SHA-256 fingerprint covering its arrays and complete metadata.
The history dataset records the fingerprint for every object and sample. This
ties the generated labels to the exact mesh, anchor ordering, rendering setup,
and visibility definition used to calculate them.

During generation, the builder also checks selected target vectors by explicitly
computing `Coverage(H union {j}) - Coverage(H)` for every candidate. It rejects
caches with incompatible metadata or inconsistent visibility definitions.

The full data flow is:

```text
ground-truth mesh
    -> render visibility from 48 anchors
    -> per-object visibility cache
    -> union visible-face masks for a sampled history
    -> measure unseen surface for each candidate
    -> 48-value target_surface_gain training label
```

