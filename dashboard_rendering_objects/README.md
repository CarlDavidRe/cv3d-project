# Dashboard Rendering Objects

Gaussian-splatting evaluation summaries are available for 23 objects across
10 ShapeNet categories.

See [the policy-variant guide](../docs/policy_variants.md) for what each of the
nine policies does and how they form the 45 policy/view comparisons.
See [the training and loss guide](../docs/training_runs_and_losses.md) for the
policy-head runs and the separate 2DGS/3DGS fitting objectives.

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

The first five airplane objects have complete manifests for all 45
policy/view variants. The remaining objects currently have partial evaluation
artifacts.
