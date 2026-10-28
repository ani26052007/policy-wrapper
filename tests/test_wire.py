"""The client against a ZMQ REP server returning a fixed chunk (``serve.SeededPolicy`` over the
stand-in ``Gr00tPolicy``): the observation dict as it arrives on the wire, chunk replay through the
scheduler, and per-request seeding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from policy_wrapper.policy.gr00t_remote_closedloop_policy import (
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyCfg,
)
from policy_wrapper.seeding import request_seed
from policy_wrapper.serve import SeededPolicy
from tests.helpers import INSTRUCTION, ServerThread, fixed_eval_yaml

NUM_ENVS = 2
CHUNK = 16
GROUPS = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6}
CONFIG_SEED = 10            # Gr00tClosedloopPolicyCfg.seed default; the pilot eval config does not set it


def make_policy():
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return SeededPolicy(Gr00tPolicy(None, "unused", device="cpu"), seed_per_request=True)


def observation(step: int = 0) -> dict:
    g = torch.Generator().manual_seed(step)
    return {
        "camera_obs": {"robot_pov_cam_rgb": torch.randint(0, 255, (NUM_ENVS, 512, 512, 3), dtype=torch.uint8, generator=g)},
        "policy": {"robot_joint_pos": torch.randn(NUM_ENVS, 54, generator=g)},
    }


def client(port: int) -> Gr00tRemoteClosedloopPolicy:
    policy = Gr00tRemoteClosedloopPolicy(Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=str(fixed_eval_yaml()), policy_device="cpu", num_envs=NUM_ENVS,
        remote_host="127.0.0.1", remote_port=port))
    policy.set_task_description(None)
    return policy


def get_actions(server: ServerThread, n: int):
    """``n`` steps from a fresh client, then a reset and one more step."""
    policy = client(server.port)
    actions = [policy.get_action(None, observation(i)) for i in range(n)]
    policy.reset()
    actions.append(policy.get_action(None, observation(n)))
    policy.close()
    return actions


def requests(server: ServerThread, endpoint: str) -> list[dict]:
    return [r for r in server.server.requests if r.get("endpoint") == endpoint]


def test_the_observation_on_the_wire_has_the_gr00t_layout():
    with ServerThread(make_policy) as server:
        policy = client(server.port)
        obs = observation()
        policy.get_action(None, obs)
        policy.close()
        sent = server.server.policy.policy.observations[0]
    assert set(sent) == {"language", "video", "state"}
    assert sent["language"] == {"annotation.human.action.task_description": [[INSTRUCTION]] * NUM_ENVS}
    ego = sent["video"]["ego_view"]
    assert ego.shape == (NUM_ENVS, 1, 512, 512, 3) and ego.dtype == np.uint8
    assert {k: v.shape for k, v in sent["state"].items()} == {k: (NUM_ENVS, 1, n) for k, n in GROUPS.items()}
    # left_shoulder_pitch_joint is state index 12 in the 54-DoF space and first in the policy's left_arm
    np.testing.assert_allclose(sent["state"]["left_arm"][:, 0, 0], obs["policy"]["robot_joint_pos"][:, 12].numpy())


def test_the_client_fetches_one_chunk_per_chunk_length_and_after_a_reset():
    with ServerThread(make_policy) as server:
        actions = get_actions(server, CHUNK + 1)
        n_get = len(requests(server, "get_action"))
    assert n_get == 3                    # steps 0-15, step 16, and the step after the reset
    assert all(a.shape == (NUM_ENVS, 36) for a in actions)


def test_every_request_carries_its_seed_and_the_server_seeds_torch_with_it():
    with ServerThread(make_policy) as server:
        get_actions(server, CHUNK + 1)
        sent = [r["data"]["options"]["seed"] for r in requests(server, "get_action")]
        used = [s.policy.seen_seeds for s in [server.server.policy]][0]
    assert sent == [request_seed(CONFIG_SEED, 0, 0), request_seed(CONFIG_SEED, 0, 1), request_seed(CONFIG_SEED, 1, 0)]
    assert len(set(sent)) == 3 and used == sent


def test_the_seed_sequence_is_reproducible_across_servers_and_reset_restarts_it():
    def run():
        with ServerThread(make_policy) as server:
            policy = client(server.port)
            for i in range(CHUNK * 2):
                policy.get_action(None, observation(i))
            policy.reset()
            policy.get_action(None, observation(0))
            policy.close()
            return [r["data"]["options"]["seed"] for r in requests(server, "get_action")]

    first, second = run(), run()
    assert first == second
    assert first[2] == request_seed(CONFIG_SEED, 1, 0) != first[0]


def test_seeded_policy_uses_the_request_seed_or_its_own_counters():
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    def expected_noise(seed: int) -> float:
        torch.manual_seed(seed)
        return float(torch.rand(1).item())

    obs = {"language": {"k": [["x"]]}}
    p = SeededPolicy(Gr00tPolicy(None, "unused", device="cpu"), seed_per_request=True)
    _, info = p.get_action(obs, {"seed": 123})
    assert info["seed"] == 123 and info["noise"] == pytest.approx(expected_noise(123))
    _, info = p.get_action(obs, None)
    assert info["seed"] == request_seed(0, 0, 1) and info["noise"] == pytest.approx(expected_noise(info["seed"]))
    p.reset()
    _, info = p.get_action(obs, {})
    assert info["seed"] == request_seed(0, 1, 0)
    off = SeededPolicy(Gr00tPolicy(None, "unused", device="cpu"), seed_per_request=False)
    assert off.get_action(obs, {"seed": 5})[1]["seed"] is None
