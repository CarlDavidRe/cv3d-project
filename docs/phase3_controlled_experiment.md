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

The run directory is
`outputs/phase3/controlled_history_comparison/seed_0`. Its `metrics/` directory
contains paired per-sample, aggregate, and history-length one-step tables;
closed-loop per-step, per-object, coverage, and policy-comparison tables;
runtime/memory/capacity profiling; external Phase 2 references; `summary.json`;
and `report.md`. Replayable rollouts and SVG coverage/quality figures are saved
beside them.

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
`tests/test_phase3_controlled.py`. This checkout does not contain a real Step 18
result because its current runtime has no CUDA device and no installed `vggt`
package. No result should be inferred from the Step 17 validation metrics or
from cached independent features; the GPU command above is required to produce
the final controlled result.
