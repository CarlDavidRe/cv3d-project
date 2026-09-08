# Official PUN closed-loop adapter

The Phase 2 `pun` policy uses the released PSNR UPNet checkpoint without
retraining. Its model architecture and deterministic timm preprocessing are
shared with the completed Phase 1 PUN integration.

## Pinned official references

- Repository: <https://github.com/ZhangLab-DeepNeuroCogLab/PUN>
- Commit: `aa6f8f4f12154854a4c1867209725c80475af102`
- Policy source: [`our_policy_single.py`](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/aa6f8f4f12154854a4c1867209725c80475af102/fep_nbv/baseline/our_policy_single.py)
- Viewpoint source: [`generate_viewpoints.py`](https://github.com/ZhangLab-DeepNeuroCogLab/PUN/blob/aa6f8f4f12154854a4c1867209725c80475af102/fep_nbv/utils/generate_viewpoints.py)
- Release: `vit_small_patch16_224_PSNR_250425172703`
- Checkpoint SHA-256: `91b2065f7652aac0c84386d4af10cd0c1ae723c049c91e45ccc78722f70907ae`

The config also pins SHA-256 hashes for both source files inspected during the
integration. The official release has no machine-readable training manifest,
so overlap with this project's test objects or categories cannot be ruled out.

## Reproduced operations

For every acquired image, UPNet independently predicts one raw, source-relative
48-value PSNR map. The adapter then follows the official default `all/small`
path:

1. Rotate the canonical HEALPix grid with the official minimal Rodrigues
   rotation so local anchor zero points toward the acquired camera direction.
2. For every rollout candidate, take rotated anchors strictly within 30° and
   interpolate with normalized `exp(-angular_distance)` weights.
3. Min-max normalize each aligned history map only for the `small` filter.
4. For PSNR, suppress a candidate if any normalized map value is at least 0.9.
5. Multiply the aligned raw PSNR maps across the complete history without
   normalizing them.
6. Select the smallest surviving product. The common policy interface negates
   this product because its contract is higher-is-better.

The official exact-antipode behavior is retained: its rotation helper returns
the unrotated grid when the cross product is exactly zero, including both the
already-aligned and antipodal cases.

Raw UPNet maps are saved separately under
`prediction_maps/pun/<category>/<object>.npz`. Rollout `scores` remain the final
oriented aggregate and must not be described as predicted surface gains.

## Deliberate common-evaluator adaptations

The official script resamples 512 continuous spherical candidates at every
decision. The project evaluator instead requires the same fixed canonical 48
anchors for every policy. Therefore:

- interpolation and aggregation are reproduced on the canonical 48 candidates;
- the shared acquired/unavailable mask replaces continuous resampling as the
  repeat-prevention mechanism;
- ties choose the lowest valid canonical anchor ID;
- normalization for `small` uses the currently valid fixed candidates;
- if `small` removes every valid candidate, suppression is ignored for that
  step so the common evaluator can continue deterministically. Every such step
  is recorded in policy provenance.

These differences mean the adapter is a reproducible official-checkpoint PUN
policy under the project's common candidate/evaluation protocol, not a claim
of bit-identical reproduction of the official continuous 512-candidate
trajectory.
