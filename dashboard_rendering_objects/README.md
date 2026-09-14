# Dashboard Rendering Objects

This file is the dashboard's object allowlist. The minimal dashboard checkout
contains the 20 test objects below: two objects from each of 10 ShapeNet
categories. Dataset exploration, rollout image grids, mesh overlays, and
reconstruction catalogs are all restricted to this list.

The required raw assets are checked in under [`dashboard_data`](../dashboard_data/README.md):

- `dashboard_data/NUM/<category_id>/<object_id>/images/` contains the 48 rendered
  NUM observations used by the dashboard;
- `dashboard_data/ShapeNetCore.v2/<category_id>/<object_id>/models/model_normalized.ply`
  contains the corresponding display mesh.

No full NUM or ShapeNet checkout is needed. Visibility-cache files are not part
of this bundle because the dashboard reads saved rollout and reconstruction
results and never loads those caches at runtime.

See [the policy-variant guide](../docs/policy_variants.md) for what each of the
nine policies does and how they form the 45 policy/view comparisons. See [the
training and loss guide](../docs/training_runs_and_losses.md) for the policy-head
runs and the separate 2DGS/3DGS fitting objectives.

See [the two 2DGS CPU fixes](../docs/2dgs_cpu_fixes.md) for the difference between
depth reevaluation and placement repair, their inputs/outputs, and execution order.

## Airplane (`02691156`)

- `1628b65a9f3cd7c05e9e2656aff7dd5b`
- `162ed8d0d989f3acc1ccec171a275967`

## Bench (`02828884`)

- `1b9ddee986099bb78880edc6251fa529`
- `1bace34d2c1dc49b3b5ee89f1f802f5a`

## Cabinet (`02933112`)

- `18d94e539b0ed30d105e720ebc569399`
- `18efc77e6837f7d311e76965808086c8`

## Car (`02958343`)

- `100c3076c74ee1874eb766e5a46fceab`
- `10716a366de708b8fac96522b26f7fd`

## Chair (`03001627`)

- `1006be65e7bc937e9141f9b58470d646`
- `1007e20d5e811b308351982a6e40cf41`

## Display/monitor (`03211117`)

- `26c4051b7dfbccf4afaac116abdd44e`
- `27107e057772be0d6b07917e9ad0834a`

## Loudspeaker (`03691459`)

- `21e46ca2f8bbd4df71187cb9cc8e1a`
- `221a981adf503875e17b9e33c097dbff`

## Rifle (`04090263`)

- `18e5827d2cfafd05d735fa1ab17311ec`
- `18fdbd5f5448e1eb9556d0a8c8dea494`

## Sofa (`04256520`)

- `165a78e3a2b705ef22c3a2386a9dfbe9`
- `1662f76e3762fc92413102507b68bcb5`

## Table (`04379243`)

- `1270e7980d2d69d293a790c6eb6d2ee5`
- `127d935d17cb36c8b0a3f25f5d8cb0f8`

## Evaluation status

The saved evaluation coverage varies by object and policy. The dashboard only
offers combinations for which it finds complete display artifacts.

## Minimal checkout

Use the sparse-clone instructions in the repository's [dashboard
README](../README.md#minimal-dashboard-checkout), then run `make dashboard`.
Those instructions select only the source, compact data, documentation, and
saved-result trees the dashboard reads.

The checked-in `dashboard_data` directory is already complete for the objects
above. It can also be archived directly for transfer:

```bash
tar -czf dashboard_data.tar.gz dashboard_data
```

Extract that archive at the repository root so the resulting path is
`dashboard_data/NUM/...`.
