"""Arena's policy base classes, or minimal stand-ins when Arena cannot be imported.

``isaaclab_arena.policy`` imports Isaac Lab when its package is imported, so ``PolicyBase`` is only
importable inside Isaac Sim's Python. There the classes below are Arena's own and
``register_policy`` registers the policy with Arena's registry. Elsewhere (unit tests) they are
stand-ins with the same constructor and defaults, and ``register_policy`` leaves the class as is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

try:
    from isaaclab_arena.assets.register import register_policy
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

    ARENA_AVAILABLE = True
except ImportError:
    ARENA_AVAILABLE = False

    @dataclass
    class PolicyCfg:
        """Mark a typed policy configuration."""

    PolicyCfgT = TypeVar("PolicyCfgT", bound=PolicyCfg)

    class PolicyBase(ABC, Generic[PolicyCfgT]):
        """Runtime behaviour shared by policies (same surface as Arena's ``PolicyBase``)."""

        def __init__(self, config: PolicyCfgT):
            self.config = config

        @abstractmethod
        def get_action(self, env: Any, observation: Any) -> Any:
            """Return the action for ``observation``."""

        def reset(self, env_ids: Any = None) -> None:
            pass

        def close(self) -> None:
            pass

        def set_task_description(self, task_description: str | None) -> str:
            self.task_description = task_description
            return self.task_description

        def has_length(self) -> bool:
            return False

        def length(self) -> int | None:
            return None

        @property
        def is_remote(self) -> bool:
            return False

    def register_policy(cls):
        """Return ``cls`` unchanged: there is no Arena registry to add it to."""
        return cls


__all__ = ["ARENA_AVAILABLE", "PolicyBase", "PolicyCfg", "register_policy"]
