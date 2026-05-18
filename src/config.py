"""Configuration helpers for the hotel no-show project."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def read_yaml(config_path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dictionary."""
    config_path = resolve_project_path(config_path)
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    return config


def write_yaml(config: dict[str, Any], config_path: str | Path) -> None:
    """Write one dictionary to a YAML file."""
    config_path = resolve_project_path(config_path)
    config_path.write_text(OmegaConf.to_yaml(config, sort_keys=False), encoding="utf-8")



def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and merge any included config files."""
    config_path = resolve_project_path(config_path)
    config = read_yaml(config_path)
    included_paths = config.pop("include", [])

    included_configs = [load_config(included_path) for included_path in included_paths]
    merged_config = OmegaConf.merge(*included_configs, config)
    config = OmegaConf.to_container(merged_config, resolve=True)

    return config
