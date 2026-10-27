# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Gr00tRemoteClosedloopPolicy.

Exercises the real obs/action translation pipeline and mocks only the native PolicyClient -- the
wire boundary to a remote GR00T server -- so no server is required.

Verifies:
- the obs dict sent to the server has the expected language/video/state structure;
- the action dict returned by the server is converted to a tensor of the
  expected (num_envs, action_chunk_length, action_dim) shape;
- ``reset`` propagates to the client;
- a failing ping during construction raises ``ConnectionError``.

Re-pointed at ``policy_wrapper`` and the GR1 pilot (``config/eval.yaml``: 54-DoF state, 26-DoF
policy joints, 36-DoF action, one 512x512 POV camera, 16-step chunks). The ``gr00t`` modules are the
stand-ins in ``tests/fake_gr00t``; Arena's policy registry is not available, so the registration
test checks the config type and the scheduler selected by the eval config.
"""

from __future__ import annotations

import numpy as np
import sys
import torch
from types import ModuleType
from typing import Any

import pytest

from policy_wrapper import scheduling as action_scheduling
from policy_wrapper.policy import gr00t_remote_closedloop_policy as gr00t_policy
from tests.helpers import EVAL_YAML, eval_variant

NUM_ENVS = 2
ORIGINAL_HEIGHT = 512
ORIGINAL_WIDTH = 512
NUM_CHANNELS = 3
NUM_SIM_JOINTS = 54  # GR1 54-DoF state
ACTION_HORIZON = 16
ACTION_CHUNK_LENGTH = 16
EXPECTED_ACTION_DIM = 36  # GR1 36-DoF action joint space
CAMERA = "robot_pov_cam_rgb"
LANGUAGE_KEY = "annotation.human.action.task_description"

# Joint group sizes from gr00t_26dof_joint_space.yaml (must match the YAML).
POLICY_GROUP_SIZES = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
}


# ----------------------------- fixtures ------------------------------ #


@pytest.fixture
def policy_config_yaml():
    """Return the repo's eval config, whose client section drives obs/action translation."""
    return str(EVAL_YAML)


@pytest.fixture
def synthetic_observation():
    """Env-shaped torch obs: one POV camera + 54 joint positions per env."""
    rgb = torch.randint(0, 255, (NUM_ENVS, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS), dtype=torch.uint8)
    joint_pos = torch.randn(NUM_ENVS, NUM_SIM_JOINTS, dtype=torch.float32)
    return {
        "camera_obs": {CAMERA: rgb},
        "policy": {"robot_joint_pos": joint_pos},
    }


def _make_action_response(num_envs: int, horizon: int) -> dict[str, np.ndarray]:
    """Synthetic GR00T-server response in the format ``build_gr00t_action_np`` expects."""
    response: dict[str, np.ndarray] = {}
    for group, size in POLICY_GROUP_SIZES.items():
        response[group] = np.random.randn(num_envs, horizon, size).astype(np.float32)
    return response


class _FakePolicyClient:
    """Minimal stand-in for ``gr00t.policy.server_client.PolicyClient``."""

    def __init__(self, *args, ping_ok: bool = True, **kwargs):
        self.init_kwargs = kwargs
        self._ping_ok = ping_ok
        self.last_observation: dict[str, Any] | None = None
        self.get_action_calls = 0
        self.reset_called = False
        self.options: list[dict[str, Any] | None] = []

    def ping(self) -> bool:
        return self._ping_ok

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
        self.last_observation = observation
        self.options.append(options)
        self.get_action_calls += 1
        # Match the real PolicyClient return signature: (action_dict, latency_or_meta).
        return _make_action_response(NUM_ENVS, ACTION_HORIZON), None

    def reset(self):
        self.reset_called = True


@pytest.fixture
def fake_client_factory(monkeypatch):
    """Install a fake native PolicyClient and expose each constructed client."""
    created: list[_FakePolicyClient] = []

    def factory(ping_ok: bool = True):
        def _ctor(*args, **kwargs):
            client = _FakePolicyClient(*args, ping_ok=ping_ok, **kwargs)
            created.append(client)
            return client

        server_client_module = ModuleType("gr00t.policy.server_client")
        server_client_module.PolicyClient = _ctor
        monkeypatch.setitem(sys.modules, "gr00t.policy.server_client", server_client_module)
        return created

    return factory


def _build_policy(policy_config_yaml: str, scheduler: str = "chunk", tmp_path=None):
    if scheduler != "chunk":
        policy_config_yaml = str(eval_variant(tmp_path, client={"scheduler": scheduler}))
    cfg = gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=policy_config_yaml,
        policy_device="cpu",  # keep the test portable; remote policy does no GPU compute
        num_envs=NUM_ENVS,
        remote_host="unused",
        remote_port=0,
        remote_api_token=None,
    )
    return gr00t_policy.Gr00tRemoteClosedloopPolicy(cfg)


# ------------------------------- tests ------------------------------- #


def test_gr00t_client_imports():
    """Import the real lightweight GR00T client API."""
    from gr00t.policy.server_client import MsgSerializer, PolicyClient

    assert MsgSerializer is not None
    assert PolicyClient is not None


def test_observation_sent_to_server_has_expected_structure(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """The dict passed to ``PolicyClient.get_action`` matches GR00T's expected
    language/video/state layout for the GR1 arms-only modality config."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    policy._get_action_chunk(synthetic_observation, [CAMERA])

    assert len(clients) == 1
    sent = clients[0].last_observation
    assert sent is not None and clients[0].get_action_calls == 1

    # language: one [task] entry per env
    assert "language" in sent
    lang = sent["language"][LANGUAGE_KEY]
    assert lang == [["pick up the brown box"]] * NUM_ENVS

    # video: one ego view, shaped (N, 1, H, W, C) and uint8
    assert "video" in sent and "ego_view" in sent["video"]
    ego = sent["video"]["ego_view"]
    assert ego.shape == (NUM_ENVS, 1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS)
    assert ego.dtype == np.uint8

    # state: one (N, 1, dof_in_group) entry per modality state key
    expected_state_keys = {"left_arm", "right_arm", "left_hand", "right_hand"}
    assert set(sent["state"].keys()) == expected_state_keys
    for key in expected_state_keys:
        arr = sent["state"][key]
        assert arr.ndim == 3
        assert arr.shape[0] == NUM_ENVS
        assert arr.shape[1] == 1
        assert arr.shape[2] == POLICY_GROUP_SIZES[key]


def test_action_response_is_translated_to_correct_tensor_shape(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """Server's grouped joint dict is reassembled into a (N, chunk, action_dim) tensor."""
    fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    action = policy._get_action_chunk(synthetic_observation, [CAMERA])

    assert isinstance(action, torch.Tensor)
    # _get_action_chunk asserts action_tensor.shape[1] >= action_chunk_length, but
    # the full horizon is returned; pin both dims explicitly.
    assert action.shape == (NUM_ENVS, ACTION_HORIZON, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_reset_propagates_to_client(policy_config_yaml, fake_client_factory):
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    assert clients[0].reset_called is False
    policy.reset()
    assert clients[0].reset_called is True


def test_construction_fails_when_server_unreachable(policy_config_yaml, fake_client_factory):
    fake_client_factory(ping_ok=False)
    with pytest.raises(ConnectionError):
        _build_policy(policy_config_yaml)


def test_registration_builds_typed_config_and_scheduler(policy_config_yaml, fake_client_factory, tmp_path):
    """The typed config is the policy's generic parameter; the eval config selects the scheduler."""
    fake_client_factory(ping_ok=True)
    assert gr00t_policy.Gr00tRemoteClosedloopPolicy.__orig_bases__[0].__args__ == (
        gr00t_policy.Gr00tRemoteClosedloopPolicyCfg,
    )
    policy = gr00t_policy.Gr00tRemoteClosedloopPolicy(
        gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
            policy_config_yaml_path=str(eval_variant(tmp_path, client={"scheduler": "synced_batch"})),
            policy_device="cpu",
            num_envs=NUM_ENVS,
            remote_host="unused",
            remote_port=0,
        )
    )

    assert isinstance(policy.config, gr00t_policy.Gr00tRemoteClosedloopPolicyCfg)
    assert isinstance(policy._chunking_state, action_scheduling.SyncedBatchActionScheduler)


# ---------------- scheduler-switch tests ---------------- #


@pytest.mark.parametrize("scheduler_cls_name", ["ActionChunkScheduler", "SyncedBatchActionScheduler"])
def test_get_action_returns_correct_shape_for_each_scheduler(
    policy_config_yaml, synthetic_observation, fake_client_factory, scheduler_cls_name, tmp_path
):
    """Both ActionChunkScheduler and SyncedBatchActionScheduler should drive the policy
    end-to-end and produce one (num_envs, action_dim) action per get_action call."""
    scheduler_cls = getattr(action_scheduling, scheduler_cls_name)
    scheduler = "synced_batch" if scheduler_cls_name == "SyncedBatchActionScheduler" else "chunk"

    fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, scheduler=scheduler, tmp_path=tmp_path)
    assert isinstance(policy._chunking_state, scheduler_cls)
    policy.set_task_description("pick up the brown box")

    action = policy.get_action(env=None, observation=synthetic_observation)

    assert isinstance(action, torch.Tensor)
    assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_synced_batch_holds_joint_position_for_env_after_partial_reset(
    policy_config_yaml, synthetic_observation, fake_client_factory, tmp_path
):
    """After resetting a single env, SyncedBatchActionScheduler should hold that env on
    the joint-position-derived hold_action while the others continue stepping their chunk."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, scheduler="synced_batch", tmp_path=tmp_path)
    policy.set_task_description("pick up the brown box")

    # Step 1: every env needs a chunk → exactly one fetch.
    policy.get_action(env=None, observation=synthetic_observation)
    assert clients[0].get_action_calls == 1

    # Reset env 1 only; env 0 still has its chunk.
    policy.reset(env_ids=torch.tensor([1]))

    action = policy.get_action(env=None, observation=synthetic_observation)
    # No new chunk fetched (env 0 not yet exhausted, so .all() is False).
    assert clients[0].get_action_calls == 1
    expected_hold = policy._extract_hold_action(synthetic_observation)
    torch.testing.assert_close(action[1], expected_hold[1])
