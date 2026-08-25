"""Small, dependency-light YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is malformed."""


def load_config(
    path: str | Path, overrides: Sequence[str] | None = None
) -> dict[str, Any]:
    """Load, override, and validate an experiment YAML file.

    Overrides use ``section.key=value`` syntax. Values are parsed as YAML, so
    numbers and booleans keep their natural types.
    """

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError("The config root must be a mapping")

    config = deepcopy(dict(raw))
    for override in overrides or ():
        _apply_override(config, override)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate only the fields required by the current infrastructure."""

    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    experiment = _required_mapping(config, "experiment")
    for key in ("phase", "name"):
        if not isinstance(experiment.get(key), str) or not experiment[key].strip():
            raise ConfigError(f"experiment.{key} must be a non-empty string")
    if not isinstance(experiment.get("seed"), int) or isinstance(
        experiment.get("seed"), bool
    ):
        raise ConfigError("experiment.seed must be an integer")
    if not isinstance(experiment.get("deterministic"), bool):
        raise ConfigError("experiment.deterministic must be a boolean")

    paths = _required_mapping(config, "paths")
    for key in ("data_root", "output_root"):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ConfigError(f"paths.{key} must be a non-empty string")


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Save a resolved configuration with stable key ordering."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8"
    )


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"Override must have KEY=VALUE form: {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    if not dotted_key or any(not key for key in keys):
        raise ConfigError(f"Invalid override key: {dotted_key!r}")

    target: dict[str, Any] = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise ConfigError(f"Override path does not exist: {dotted_key!r}")
        target = child
    if keys[-1] not in target:
        raise ConfigError(f"Override key does not exist: {dotted_key!r}")

    try:
        target[keys[-1]] = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid override value in {override!r}") from exc
