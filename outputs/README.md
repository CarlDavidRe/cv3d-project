# Generated outputs

Experiment artifacts use the identity `phase / experiment / seed`. Configured
runs store their resolved `config.yaml`, environment `metadata.json`, log,
metrics, checkpoints, and figures together. Standalone inspection and smoke-test
commands use the same layout.

Workflow labels such as `step5` are documentation concepts and must not be used
as directory names.

Generated contents are intentionally ignored by Git; this file keeps the output
contract visible in a fresh checkout.
