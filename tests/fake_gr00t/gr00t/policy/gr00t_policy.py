"""Stand-in ``Gr00tPolicy``: returns a fixed GR1 action chunk; ``info["noise"]`` is a torch draw made
after the server seeded torch, so tests can see which seed was used."""

import json
from pathlib import Path

import numpy as np
import torch

GROUP_SIZES = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6}
HORIZON = 16


def fixed_chunk(num_envs: int) -> dict:
    """Group ``g``, step ``t``, joint ``j``: ``100 * t + j + 0.5 * env``, plus 1000 per group index."""
    out = {}
    for gi, (group, size) in enumerate(GROUP_SIZES.items()):
        t = np.arange(HORIZON, dtype=np.float32)[None, :, None]
        j = np.arange(size, dtype=np.float32)[None, None, :]
        env = np.arange(num_envs, dtype=np.float32)[:, None, None]
        out[group] = (1000.0 * gi + 100.0 * t + j + 0.5 * env).astype(np.float32)
    return out


class _ActionHead:
    num_inference_timesteps = 4


class _Model:
    def __init__(self):
        self.action_head = _ActionHead()


class _Processor:
    def __init__(self, statistics):
        self.statistics = statistics
        self.set_statistics_calls = []

    def set_statistics(self, statistics, override=False):
        self.set_statistics_calls.append((statistics, override))
        if override:
            self.statistics = statistics


class Gr00tPolicy:
    def __init__(self, embodiment_tag, model_path, *, device, strict=True):
        self.embodiment_tag, self.model_path, self.device, self.strict = embodiment_tag, model_path, device, strict
        stats = Path(model_path) / "statistics.json"
        self.model = _Model()
        self.processor = _Processor(json.loads(stats.read_text()) if stats.is_file() else {})
        self.observations = []
        self.seen_seeds = []
        self.resets = 0

    def get_action(self, observation, options=None):
        self.observations.append(observation)
        self.seen_seeds.append(torch.initial_seed())
        num_envs = len(observation["language"][next(iter(observation["language"]))])
        return fixed_chunk(num_envs), {"noise": float(torch.rand(1).item())}

    def reset(self, options=None):
        self.resets += 1
        return {}

    def get_modality_config(self):
        return {}
