# Phase 3 independent-history validation

Step 16 was validated on 2026-09-09 after completing the visibility cache and
direct surface-gain history dataset.

## Input audit

- History dataset ID:
  `dd7f802d5c8f15e3f47110f4d7a72402bc95673777b663a28889c0bc6aaf9a36`
- Coverage target: `vis_a`
- Objects: 870 train, 109 validation, 300 test (1,279 total)
- Histories: 34,800 train, 4,360 validation, 12,000 test (51,160 total)
- History lengths: 1, 2, 4, 6, and 8, with equal counts per length in each split
- Frozen feature width: 2,048 (`vggt_max_pooled_patch`)

Validation loaded every checksum-verified shard and resolved every history to
its split-specific feature cache. All 1,279 current visibility-cache
fingerprints matched the IDs stored in the history manifest. One history per
object was independently checked against explicit
`Coverage(history + candidate) - Coverage(history)` calculations for all 48
candidates; all checks passed.

## Completion run

The retained validation run is:

```text
outputs/phase3/independent_history_validation/seed_0
```

It used all generated histories with a deliberately bounded one-epoch budget
and batch size 512 because the validation environment exposed CPU-only
PyTorch. The default 100-epoch configuration remains unchanged for the later
full-budget experiment.

The validation-selected epoch-1 checkpoint contains 272,950 trainable
parameters and records:

| Split | Huber | Total loss | Regret | Spearman | NDCG@5 | Samples |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.002071 | 0.066698 | 0.3797 | 0.3423 | 0.6595 | 4,360 |
| Test | 0.001874 | 0.067549 | 0.4158 | 0.2971 | 0.6328 | 12,000 |

These values establish that the real-data training and evaluation path is
operational; they are not presented as a converged research result.

The validation-object rollout acquired five total views, reached absolute
`vis_a` coverage 0.7960 (reachable-normalized coverage 0.8542), and was replayed
successfully. Its policy provenance confirms independent single-image frozen
features, masked-mean history aggregation, direct surface-gain supervision,
and no cross-view interaction inside VGGT.

The complete repository test suite passed 156 tests after the run. This
includes tiny-set overfitting, variable-length padding, history permutation
invariance, the one-image VGGT sequence boundary, cache/live policy behavior,
checkpoint loading, and synthetic end-to-end rollout replay.
