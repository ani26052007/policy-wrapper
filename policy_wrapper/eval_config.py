"""``config/eval.yaml``: the one place the evaluation is configured.

Sections:
  ``client:``  the Arena-side policy (observation and action translation, scheduler). Its keys are
               the fields of ``Gr00tClosedloopPolicyCfg``; an unknown key is an error. Relative
               paths are relative to the repo root (the directory that holds ``config/``).
  ``server:``  the GR00T policy server (``policy_wrapper.serve``).
  ``env:``     the Arena environment and its settings (read by the evaluation harness).
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from policy_wrapper.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
from policy_wrapper.utils.io_utils import convert_yaml_value

EVAL_CONFIG_PATH = Path("config") / "eval.yaml"
SECTIONS = ("client", "server", "env")


def repo_root(eval_config_path: str | Path) -> Path:
    """The repo root of an eval config at ``<root>/config/eval.yaml``."""
    return Path(eval_config_path).resolve().parent.parent


def read_eval_config(eval_config_path: str | Path) -> dict[str, Any]:
    """The whole eval config as a mapping with the three sections."""
    path = Path(eval_config_path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    assert isinstance(data, dict), f"{path} must hold a mapping"
    unknown = set(data) - set(SECTIONS)
    assert not unknown, f"{path}: unknown sections {sorted(unknown)}; expected {list(SECTIONS)}"
    for section in SECTIONS:
        assert isinstance(data.get(section), dict), f"{path}: missing or empty section {section!r}"
    return data


def load_client_config(eval_config_path: str | Path) -> Gr00tClosedloopPolicyCfg:
    """Build the closed-loop policy config from the eval config's ``client:`` section."""
    raw = read_eval_config(eval_config_path)["client"]
    root = repo_root(eval_config_path)
    field_types = {f.name: f.type for f in fields(Gr00tClosedloopPolicyCfg)}
    unknown = set(raw) - set(field_types)
    if unknown:
        raise ValueError(f"{eval_config_path}: unknown client keys {sorted(unknown)}")
    values: dict[str, Any] = {}
    for name, value in raw.items():
        if name.endswith("_path") and value is not None and not Path(value).is_absolute():
            value = str(root / value)
        values[name] = convert_yaml_value(field_types[name], value)
    return Gr00tClosedloopPolicyCfg(**values)
