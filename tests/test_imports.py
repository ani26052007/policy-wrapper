"""Every module imports without Isaac Sim, Isaac Lab-Arena or Isaac-GR00T installed."""

from __future__ import annotations

import importlib
import pkgutil
import sys

import policy_wrapper


def test_every_module_imports_and_uses_the_local_policy_base_without_arena():
    names = [m.name for m in pkgutil.walk_packages(policy_wrapper.__path__, "policy_wrapper.")
             if not m.name.endswith("gr1_arms_only_data_config")]
    for name in names:
        importlib.import_module(name)
    from policy_wrapper import _arena

    assert not _arena.ARENA_AVAILABLE
    assert not [m for m in sys.modules if m.split(".")[0] in ("isaaclab", "isaaclab_arena", "isaacsim")]
    assert "policy_wrapper.serve" in names and "policy_wrapper.instrument" in names
