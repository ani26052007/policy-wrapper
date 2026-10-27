"""The policy's sampling seed for one inference request.

GR00T samples the action head's noise from the global torch RNG and seeds nothing itself, so the
server seeds torch before every request. The client chooses the seed and sends it as
``options["seed"]``: ``request_seed(config seed, number of resets so far, request index since the
last reset)``. The same episode order and observation sequence therefore get the same seeds, also
across server restarts, and the seed is known on the client side. ``serve.py`` uses the same
function with its own counters when a request carries no seed.
"""

from __future__ import annotations

import hashlib

MAX_SEED = 2**31 - 1


def request_seed(base_seed: int, episode: int, request: int) -> int:
    """A seed in ``[0, 2**31 - 1]`` determined by its three integer arguments."""
    digest = hashlib.sha256(f"{int(base_seed)}:{int(episode)}:{int(request)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % MAX_SEED
