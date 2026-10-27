"""Per-episode instrumentation of the policy client: what the evaluation harness reads back as evidence.

The policy calls five hooks and nothing else:

  ``steps``                         per-env step counters (the observation step a new chunk is tagged with)
  ``on_request(observation, sent, camera_names)``  the observation dict exactly as it goes to the server
  ``on_reply(actions)``             the server's action dict exactly as it came back
  ``on_step(observation, tagged)``  the action exactly as it is returned to ``env.step``; returns it
  ``on_reset(env_ids)`` / ``close()``  episode boundaries

Everything is logged at the client's two boundaries, so a change anywhere inside the client shows up in
the logs instead of being hidden by them:

* **Frames.** ``on_request`` summarises the video arrays of the dict sent to the server (after every
  preprocessing step) as ``policy_input``, and the environment's camera frame they came from (before
  any preprocessing) as ``render``. One sample per env at most every ``frame_every`` steps. The summary
  has the fields of the router's ``FrameStats`` (``frame_stats``).
* **Actions.** ``on_reply`` stores each env's chunk exactly as the server returned it, with the step of
  the observation it was computed from.
* **Timing.** Every action carries a tag (``TaggedAction``): the step of the observation its chunk was
  computed from, set when the chunk arrives, and its index in that chunk. The scheduler and anything
  that buffers actions pass the tag along with the action. ``on_step`` logs
  ``(obs_step, applied_step, chunk_offset)`` with ``applied_step`` = the env's step counter at the moment
  the action is returned to ``env.step``. Delay added anywhere between the server's reply and
  ``env.step`` therefore shows as ``applied_step - obs_step - chunk_offset > 0``. Held actions (an env
  waiting for the others in a synced batch) have ``obs_step = -1`` and are not logged as step pairs.
* **Observation keys.** ``on_step`` also records the listed observation keys (dotted paths such as
  ``policy.left_eef_pos``) at every step, as given by the environment.

Output: one JSON file per finished episode, ``<dir>/env<env_id>_ep<episode_in_env>.json``. An episode
ends at ``on_reset`` for its env or at ``close``; an env reset before its first step starts no episode.
``start_index`` numbers episodes in the order they take their first step (envs in id order within a
step).

Configuration comes from the environment of the process, set by the evaluation harness, never from
``config/eval.yaml``: ``VLA_INSTRUMENT_DIR`` (unset: disabled), ``VLA_INSTRUMENT_FRAME_EVERY``,
``VLA_INSTRUMENT_PROFILE_BUCKETS``, ``VLA_INSTRUMENT_OBS_KEYS`` (comma-separated). Disabled, the hooks
are ``NullInstrument``'s: no counters, no statistics, no files, and ``on_step`` returns the action.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ENV_DIR = "VLA_INSTRUMENT_DIR"
ENV_FRAME_EVERY = "VLA_INSTRUMENT_FRAME_EVERY"
ENV_PROFILE_BUCKETS = "VLA_INSTRUMENT_PROFILE_BUCKETS"
ENV_OBS_KEYS = "VLA_INSTRUMENT_OBS_KEYS"
FORMAT = "policy_wrapper.instrument/1"
DEFAULT_FRAME_EVERY = 1
DEFAULT_PROFILE_BUCKETS = 16


@dataclass(frozen=True)
class TaggedAction:
    """One action per env and where each came from.

    action:       (num_envs, action_dim)
    obs_step:     (num_envs,) int64, the step of the observation the action's chunk was computed from;
                  -1 for an env that holds its state instead of a policy action
    chunk_offset: (num_envs,) int64, the action's index in that chunk
    """

    action: torch.Tensor
    obs_step: torch.Tensor
    chunk_offset: torch.Tensor


# --------------------------------------------------------------------------- #
# frame statistics
# --------------------------------------------------------------------------- #


def frame_stats(frame: Any, buckets: int = DEFAULT_PROFILE_BUCKETS) -> dict[str, Any]:
    """Summary of one HWC image, as the router's ``FrameStats.to_dict``.

    ``channel_means`` are in the array's own channel order. ``row_profile`` / ``col_profile`` are the
    mean intensity (mean over channels) of ``buckets`` equal bands of rows / columns, top to bottom and
    left to right. A pixel is saturated when every channel is at the dtype's maximum (255 for uint8,
    1.0 for floats). ``missing_texture_fraction`` is not measured here and stays 0.
    """
    a = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {a.shape}")
    h, w, c = a.shape
    if h < buckets or w < buckets:
        raise ValueError(f"image {a.shape} is smaller than {buckets} profile buckets")
    top = float(np.iinfo(a.dtype).max) if np.issubdtype(a.dtype, np.integer) else 1.0
    x = a.astype(np.float64)
    intensity = x.mean(axis=2)
    return {
        "shape": [int(h), int(w), int(c)],
        "channel_means": [float(m) for m in x.reshape(-1, c).mean(axis=0)],
        "min_value": float(x.min()),
        "max_value": float(x.max()),
        "row_profile": [float(b.mean()) for b in np.array_split(intensity, buckets, axis=0)],
        "col_profile": [float(b.mean()) for b in np.array_split(intensity, buckets, axis=1)],
        "saturated_fraction": float((x >= top).all(axis=2).mean()),
        "missing_texture_fraction": 0.0,
    }


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #


class NullInstrument:
    """Instrumentation switched off: every hook does nothing."""

    enabled = False
    steps = None

    def on_request(self, observation: Any, sent: Any, camera_names: Sequence[str]) -> None:
        pass

    def on_reply(self, actions: Any) -> None:
        pass

    def on_step(self, observation: Any, tagged: TaggedAction) -> torch.Tensor:
        return tagged.action

    def on_reset(self, env_ids: Any = None) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass
class _Episode:
    episode_in_env: int
    start_index: int | None = None
    num_steps: int = 0
    frames: list = field(default_factory=list)
    step_pairs: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    observations: dict = field(default_factory=dict)


def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _dotted(d: Mapping, dotted: str) -> Any:
    node: Any = d
    for part in dotted.split("."):
        node = node[part]
    return node


def _env_list(env_ids: Any, num_envs: int) -> list[int]:
    if env_ids is None:
        return list(range(num_envs))
    if isinstance(env_ids, slice):
        return list(range(num_envs))[env_ids]
    return [int(e) for e in _to_numpy(env_ids).reshape(-1)]


class Instrument:
    """Instrumentation switched on: logs every hook and writes one JSON file per finished episode."""

    enabled = True

    def __init__(self, num_envs: int, out_dir: str | Path, *, frame_every: int = DEFAULT_FRAME_EVERY,
                 profile_buckets: int = DEFAULT_PROFILE_BUCKETS, obs_keys: Sequence[str] = ()) -> None:
        if frame_every < 1 or profile_buckets < 1:
            raise ValueError("frame_every and profile_buckets must be >= 1")
        self.num_envs = num_envs
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frame_every = frame_every
        self.profile_buckets = profile_buckets
        self.obs_keys = tuple(obs_keys)
        self.steps = torch.zeros(num_envs, dtype=torch.int64)
        self._last_frame: list[int | None] = [None] * num_envs
        self._request_steps: np.ndarray | None = None
        self._episodes = [_Episode(0) for _ in range(num_envs)]
        self._next_start = 0
        self.written: list[Path] = []

    def on_request(self, observation: Mapping[str, Any], sent: Mapping[str, Any], camera_names: Sequence[str]) -> None:
        steps = self.steps.numpy().copy()
        self._request_steps = steps
        video = sent["video"]
        pairs = list(zip(camera_names, video))
        for e in range(self.num_envs):
            last = self._last_frame[e]
            if last is not None and steps[e] - last < self.frame_every:
                continue
            self._last_frame[e] = int(steps[e])
            for sim_camera, key in pairs:
                self._episodes[e].frames.append({
                    "step": int(steps[e]), "camera": key, "sim_camera": sim_camera,
                    "policy_input": frame_stats(_to_numpy(video[key])[e, -1], self.profile_buckets),
                    "render": frame_stats(_to_numpy(observation["camera_obs"][sim_camera][e]), self.profile_buckets),
                })

    def on_reply(self, actions: Mapping[str, Any]) -> None:
        steps = self._request_steps if self._request_steps is not None else self.steps.numpy().copy()
        self._request_steps = None
        arrays = {k: _to_numpy(v) for k, v in actions.items()}
        for e in range(self.num_envs):
            self._episodes[e].chunks.append({"obs_step": int(steps[e]),
                                             "raw": {k: a[e].tolist() for k, a in arrays.items()}})

    def on_step(self, observation: Mapping[str, Any], tagged: TaggedAction) -> torch.Tensor:
        obs_step = _to_numpy(tagged.obs_step).reshape(-1)
        offset = _to_numpy(tagged.chunk_offset).reshape(-1)
        steps = self.steps.numpy()
        values = {k: _to_numpy(_dotted(observation, k)) for k in self.obs_keys}
        for e in range(self.num_envs):
            ep = self._episodes[e]
            if ep.start_index is None:
                ep.start_index = self._next_start
                self._next_start += 1
            if obs_step[e] >= 0:
                ep.step_pairs.append([int(obs_step[e]), int(steps[e]), int(offset[e])])
            for k, v in values.items():
                ep.observations.setdefault(k, []).append(np.asarray(v[e], dtype=np.float64).reshape(-1).tolist())
            ep.num_steps += 1
        self.steps += 1
        return tagged.action

    def on_reset(self, env_ids: Any = None) -> None:
        for e in _env_list(env_ids, self.num_envs):
            ep = self._episodes[e]
            if ep.num_steps == 0:
                continue
            self._write(e, ep, "reset")
            self._episodes[e] = _Episode(ep.episode_in_env + 1)
            self.steps[e] = 0
            self._last_frame[e] = None

    def close(self) -> None:
        for e, ep in enumerate(self._episodes):
            if ep.num_steps:
                self._write(e, ep, "close")
                self._episodes[e] = _Episode(ep.episode_in_env + 1)

    def _write(self, env_id: int, ep: _Episode, ended_by: str) -> None:
        record = {"format": FORMAT, "env_id": env_id, "episode_in_env": ep.episode_in_env,
                  "start_index": ep.start_index, "num_steps": ep.num_steps, "ended_by": ended_by,
                  "frame_every": self.frame_every, "profile_buckets": self.profile_buckets,
                  "obs_keys": list(self.obs_keys), "frames": ep.frames, "step_pairs": ep.step_pairs,
                  "chunks": ep.chunks, "observations": ep.observations}
        path = self.out_dir / f"env{env_id:04d}_ep{ep.episode_in_env:05d}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, path)
        self.written.append(path)


def make_instrument(num_envs: int, environ: Mapping[str, str] | None = None) -> Instrument | NullInstrument:
    """The hooks for this process: ``Instrument`` when ``VLA_INSTRUMENT_DIR`` is set, else ``NullInstrument``."""
    env = os.environ if environ is None else environ
    out_dir = env.get(ENV_DIR)
    if not out_dir:
        return NullInstrument()
    keys = [k.strip() for k in env.get(ENV_OBS_KEYS, "").split(",") if k.strip()]
    return Instrument(num_envs, out_dir, frame_every=int(env.get(ENV_FRAME_EVERY, DEFAULT_FRAME_EVERY)),
                      profile_buckets=int(env.get(ENV_PROFILE_BUCKETS, DEFAULT_PROFILE_BUCKETS)), obs_keys=keys)
