# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for action scheduling strategies."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Callable

from policy_wrapper.instrument import TaggedAction


class ActionScheduler(ABC):
    """Translates raw action outputs into per-step actions.

    The policy calls ``get_action(fetch_action_fn)`` at every environment step.
    The scheduler controls when to query the action and how to derive a single
    action from one or more action outputs.
    """

    @abstractmethod
    def get_tagged_action(
        self,
        fetch_action_tensor_fn: Callable[[], torch.Tensor],
        hold_action: torch.Tensor | None = None,
        obs_step: torch.Tensor | None = None,
    ) -> TaggedAction:
        """Return one action per env for the current timestep, with its tag.

        Args:
            fetch_action_tensor_fn: Callable that queries the action and returns an action tensor.
            hold_action: Optional ``(num_envs, action_dim)`` tensor used by schedulers
                that need a fallback action for envs waiting on others. Schedulers that don't
                need it accept ``None`` and ignore the argument so callers can use a uniform call site.
            obs_step: Optional ``(num_envs,)`` step of the current observation. A chunk fetched now is
                tagged with it; ``None`` leaves the tags at -1.

        Returns:
            ``TaggedAction``: the per-step action tensor of shape ``(num_envs, action_dim)``, the
            observation step of each action's chunk and the action's index in it.
        """
        ...

    def get_action(
        self,
        fetch_action_tensor_fn: Callable[[], torch.Tensor],
        hold_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one action per env for the current timestep (``get_tagged_action`` without the tag).

        Returns:
            Per-step action tensor of shape ``(num_envs, action_dim)``.
        """
        return self.get_tagged_action(fetch_action_tensor_fn, hold_action).action
