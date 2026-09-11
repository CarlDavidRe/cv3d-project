# Phase 3 controlled experiment

Step 18 compares the validation-selected independent and joint history models
on the fixed object-disjoint test split. The only intended representation
difference is whether frozen VGGT processes each acquired image separately or
processes the complete acquired history jointly.

Run it in the same CUDA/VGGT environment used for Step 17:

```bash
python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml
```

The default configuration pins the retained Step 16 and Step 17 checkpoints by
SHA-256, uses the complete 300-object test split, starts every rollout at anchor
0, and stops at 10 acquired views. An interrupted run can continue with:

```bash
python3 scripts/evaluate_phase3.py \
  --config configs/experiments/phase3_controlled.yaml \
  --resume
```

## Control and leakage checks

Before evaluation, the runner verifies:

- identical direct-`VisA` history supervision, model dimensions, head capacity,
  feature component, VGGT model ID, input size, layer, and preprocessing;
- exact checkpoint and independent test-feature checksums;
- identical history identities, allowing only the already-audited repository-
  root relocation from `/content/cv3d-project` to the current checkout;
- object-disjoint test histories and closed-loop objects from that same test
  split;
- exact visibility-cache fingerprints shared by history labels and rollout
  evaluation;
- the same post-rollout VGGT point-cloud evaluator, ShapeNet surface target,
  alignment, filtering, and Chamfer/F-score settings used in Phase 2;
- cosine similarity of at least 0.999 between independent and joint features
  for length-1 histories. This empirical check covers effective frozen weights
  and preprocessing because the older independent cache has no weight digest;
- live joint closed-loop inference from acquired RGB only. Independent feature
  caches are never substituted for the joint forward.

The fixed-history joint vectors are materialized once in resumable shards.
Histories are grouped by real length, so padded images never enter VGGT. The
closed-loop policy still recomputes one complete joint VGGT forward for every
history state.

## Outputs and completion gate

New reconstruction-enabled runs use
`outputs/phase3/controlled_history_comparison_reconstruction/seed_0`. Its
`metrics/` directory
contains paired per-sample, aggregate, and history-length one-step tables;
closed-loop per-step, per-object, coverage, and policy-comparison tables;
shared-backend reconstruction per-object and curve tables; runtime/memory/
capacity profiling; external Phase 2 references; `summary.json`; and
`report.md`. Replayable rollouts and SVG coverage/reconstruction figures are
saved beside them. Exact ordered histories reuse `data/cache/reconstruction`
entries produced by Phase 2.

`metrics/phase3_completion.json` is the authoritative exit gate. It is
`complete` only for two policies on all 12,000 test histories and the complete
300-object fixed test cohort, with common evaluator, metrics, profiling,
figures, external references, feature equivalence, and a recorded conclusion.
Partial cohorts remain useful diagnostics but cannot freeze Phase 3.

Joint feature time is labeled `live_complete_precompute` only when no shards
existed before invocation. Resumed cache restoration is labeled separately and
must not be reported as live VGGT time. Closed-loop policy time is CUDA-
synchronized and excludes geometry evaluation and RGB acquisition.

## Current execution status

The implementation and its full synthetic run/resume/replay path are tested in
`tests/test_phase3_controlled.py`. The retained real Step 18 artifact predates
the reconstruction gate; this runtime has no CUDA device and no installed
`vggt` package with which to replace it. No reconstruction result should be
inferred from the old visibility metrics or cached independent features; the
GPU command above is required to produce the current-schema controlled result.
