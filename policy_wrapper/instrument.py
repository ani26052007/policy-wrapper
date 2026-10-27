"""Per-episode instrumentation of the policy client.

Disabled in this version: every hook is a no-op and nothing is written.
"""

from __future__ import annotations

from typing import Any


class Instrument:
    """No-op instrumentation hooks."""

    enabled = False

    def on_policy_input(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chunk(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_reset(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass
