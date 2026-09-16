"""Configuration loading.

Every threshold, path, seed and rule that affects a reported number lives in
config/config.yaml. This module loads it once, validates the parts the pipeline
depends on, and resolves paths against the repository root.

The discipline it enforces: if a number appears in the final report, it is
traceable to a named configuration version. Nothing that changes a result may
be hard-coded in src/ or in a notebook. That is the operational form of the
reproducibility claim in Section 3.10.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "repo_root",
    "resolve_path",
    "load_config",
    "load_leakage_register",
    "set_global_seed",
]

_REQUIRED_SECTIONS = {"meta", "seed", "paths", "sample", "partition", "leakage"}


def repo_root() -> Path:
    """Absolute path to the repository root, inferred from this file's location.

    src/utils/config.py -> parents[0]=utils, [1]=src, [2]=repo root.
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the repository root.

    Config files hold relative paths so the project works from any checkout
    location. This turns them into absolute paths at use time.

    Args:
        relative: A path as written in config.yaml.

    Returns:
        Absolute path. Not created, and not required to exist.
    """
    return repo_root() / relative


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the master configuration.

    Validation is deliberately shallow: it checks the sections the pipeline
    cannot run without, and the one invariant that is easy to break by hand
    (partition shares summing to one). Deeper validation belongs with the code
    that consumes each section.

    Args:
        path: Path to the YAML config. Defaults to config/config.yaml.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If the config file is absent.
        ValueError: If a required section is missing, or partition shares do
            not sum to 1.0.
    """
    config_path = Path(path) if path is not None else repo_root() / "config" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    missing = _REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"config missing required sections: {sorted(missing)}")

    partition = config["partition"]

    legacy = partition.get("legacy_out_of_time")
    if legacy is not None:
        total = legacy["train_share"] + legacy["validation_share"] + legacy["test_share"]
        if not np.isclose(total, 1.0):
            raise ValueError(f"partition shares must sum to 1.0, got {total}")

    rolling = partition.get("rolling_origin")
    if rolling is not None:
        folds = rolling.get("folds") or []
        if not folds:
            raise ValueError("partition.rolling_origin.folds must not be empty")
        for index, fold in enumerate(folds):
            absent = {"train_end", "evaluate_end"} - set(fold)
            if absent:
                raise ValueError(f"fold {index} missing required keys: {sorted(absent)}")

    return config


def load_leakage_register(path: str | Path | None = None) -> dict[str, Any]:
    """Load the leakage exclusion register (Table 3.4).

    Args:
        path: Path to the register YAML. Defaults to config/leakage_register.yaml.

    Returns:
        The parsed register.

    Raises:
        FileNotFoundError: If the register is absent.
        ValueError: If the register has no 'exclusions' section, which would
            silently admit every post-origination field.
    """
    register_path = (
        Path(path) if path is not None else repo_root() / "config" / "leakage_register.yaml"
    )
    if not register_path.is_file():
        raise FileNotFoundError(f"leakage register not found: {register_path}")

    with register_path.open("r", encoding="utf-8") as handle:
        register: dict[str, Any] = yaml.safe_load(handle)

    if not register.get("exclusions"):
        raise ValueError(
            "leakage register has no 'exclusions' section; refusing to proceed, "
            "since an empty register would admit every post-origination field"
        )
    return register


def set_global_seed(seed: int) -> None:
    """Seed the global random sources the pipeline touches.

    Call once at the start of any script that produces a reported number.
    Library-specific seeding (xgboost, lightgbm, catboost, sklearn splitters)
    is passed explicitly at construction using this same value, because those
    libraries do not all read the global numpy state.

    Args:
        seed: The seed from config.yaml.
    """
    random.seed(seed)
    np.random.seed(seed)
