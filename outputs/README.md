# Generated outputs

Experiment artifacts use the identity `phase / experiment / seed`. Configured
runs store their resolved `config.yaml`, environment `metadata.json`, log,
metrics, checkpoints, and figures together. Standalone inspection and smoke-test
commands use the same layout.

Closed-loop runs with reconstruction enabled additionally write
`metrics/reconstruction_per_object.csv`, `metrics/reconstruction_curves.csv`,
reconstruction fields in their comparison/summary artifacts, and
`figures/closed_loop/reconstruction_curves.svg`. The expensive point clouds are
shared data caches under `data/cache/reconstruction`, not run outputs; preserve
that directory when moving or resuming experiments.
