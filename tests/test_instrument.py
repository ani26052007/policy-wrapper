"""Instrumentation (``policy_wrapper.instrument``): frame statistics, the two logging boundaries, action
tags through the scheduler and through a buffer, episode files, and the disabled path."""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import pytest
import torch

from policy_wrapper import instrument as I
from policy_wrapper.instrument import (Instrument, NullInstrument, TaggedAction, content_hash, frame_stats,
                                       make_instrument)
from policy_wrapper.policy import gr00t_core
from policy_wrapper.policy.gr00t_remote_closedloop_policy import (
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyCfg,
)
from policy_wrapper.scheduling import ActionChunkScheduler, SyncedBatchActionScheduler
from tests.helpers import EVAL_YAML, ServerThread, eval_variant
from tests.test_wire import make_policy

NUM_ENVS = 2
CHUNK = 16
CAMERA, VIDEO_KEY = "robot_pov_cam_rgb", "ego_view"


# ----------------------------- frame statistics ----------------------------- #

def gradient_image(h: int = 64, w: int = 48) -> np.ndarray:
    """Red bright everywhere, green rising top to bottom, blue rising left to right (uint8 RGB)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[..., 2] = np.linspace(0, 120, w, dtype=np.uint8)[None, :]
    return img


def test_frame_stats_channel_means_tell_rgb_from_bgr():
    rgb = gradient_image()
    s, b = frame_stats(rgb), frame_stats(rgb[..., ::-1])
    assert s["shape"] == [64, 48, 3]
    assert s["channel_means"][0] == pytest.approx(200.0) and s["channel_means"][0] > s["channel_means"][2]
    assert b["channel_means"] == pytest.approx(s["channel_means"][::-1])
    assert (s["min_value"], s["max_value"]) == (0.0, 255.0)


def test_frame_stats_profiles_show_a_flip():
    img = gradient_image()
    s = frame_stats(img, buckets=8)
    assert len(s["row_profile"]) == len(s["col_profile"]) == 8
    assert s["row_profile"] == sorted(s["row_profile"])               # brighter towards the bottom
    assert frame_stats(img[::-1], buckets=8)["row_profile"] == pytest.approx(s["row_profile"][::-1])
    assert frame_stats(img[:, ::-1], buckets=8)["col_profile"] == pytest.approx(s["col_profile"][::-1])
    assert frame_stats(img[:, ::-1], buckets=8)["row_profile"] == pytest.approx(s["row_profile"])


def test_frame_stats_saturation_and_bucket_count_do_not_depend_on_resolution():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:8] = 255
    assert frame_stats(img)["saturated_fraction"] == pytest.approx(0.25)
    assert frame_stats(img.astype(np.float32) / 255.0)["saturated_fraction"] == pytest.approx(0.25)
    big = np.kron(gradient_image(), np.ones((4, 4, 1), dtype=np.uint8))
    assert len(frame_stats(big)["row_profile"]) == len(frame_stats(gradient_image())["row_profile"]) == 16
    with pytest.raises(ValueError):
        frame_stats(np.zeros((8, 8, 3), dtype=np.uint8))


# ----------------------------- the client, end to end ----------------------- #

def observation(step: int) -> dict:
    g = torch.Generator().manual_seed(step)
    return {
        "camera_obs": {CAMERA: torch.randint(0, 255, (NUM_ENVS, 512, 512, 3), dtype=torch.uint8, generator=g)},
        "policy": {"robot_joint_pos": torch.randn(NUM_ENVS, 54, generator=g),
                   "left_eef_pos": torch.full((NUM_ENVS, 3), float(step))},
    }


def client(port: int, config=EVAL_YAML, cls=Gr00tRemoteClosedloopPolicy) -> Gr00tRemoteClosedloopPolicy:
    policy = cls(Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=str(config), policy_device="cpu", num_envs=NUM_ENVS,
        remote_host="127.0.0.1", remote_port=port))
    policy.set_task_description(None)
    return policy


@pytest.fixture
def instrumented(tmp_path, monkeypatch):
    out = tmp_path / "instrument"
    monkeypatch.setenv(I.ENV_DIR, str(out))
    monkeypatch.setenv(I.ENV_OBS_KEYS, "policy.left_eef_pos")
    return out


def run_episode(policy, steps: int, start: int = 0) -> list[torch.Tensor]:
    return [policy.get_action(None, observation(start + t)) for t in range(steps)]


def episodes(out) -> dict:
    return {p.name: json.loads(p.read_text()) for p in sorted(out.glob("*.json"))}


def test_disabled_instrumentation_writes_nothing_and_returns_the_same_actions(tmp_path, monkeypatch):
    monkeypatch.delenv(I.ENV_DIR, raising=False)
    assert isinstance(make_instrument(2), NullInstrument)
    calls, hashed = [], []
    real, real_hash = I.frame_stats, I.content_hash
    monkeypatch.setattr(I, "frame_stats", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(I, "content_hash", lambda *a, **k: hashed.append(1) or real_hash(*a, **k))
    with ServerThread(make_policy) as server:
        policy = client(server.port)
        assert policy._instrument.steps is None
        off = run_episode(policy, CHUNK + 3)
        policy.close()
        assert calls == [] and hashed == [] and not list(tmp_path.rglob("*.json"))
        monkeypatch.setenv(I.ENV_DIR, str(tmp_path / "on"))
        policy = client(server.port)
        on = run_episode(policy, CHUNK + 3)
        policy.close()
    assert calls and hashed and len(list((tmp_path / "on").glob("*.json"))) == NUM_ENVS
    assert all(torch.equal(a, b) for a, b in zip(off, on))


def test_frames_are_the_observation_exactly_as_the_server_received_it_and_the_env_frame_it_came_from(instrumented):
    with ServerThread(make_policy) as server:
        policy = client(server.port)
        run_episode(policy, CHUNK + 1)
        policy.close()
        received = server.server.policy.policy.observations
    eps = episodes(instrumented)
    assert sorted(eps) == ["env0000_ep00000.json", "env0001_ep00000.json"]
    for e in range(NUM_ENVS):
        ep = eps[f"env{e:04d}_ep00000.json"]
        assert [f["step"] for f in ep["frames"]] == [0, CHUNK]           # one sample per request
        for f, sent in zip(ep["frames"], received):
            assert (f["camera"], f["sim_camera"]) == (VIDEO_KEY, CAMERA)
            assert f["policy_input"] == frame_stats(sent["video"][VIDEO_KEY][e, -1])
            assert f["render"] == frame_stats(observation(f["step"])["camera_obs"][CAMERA][e])


def test_content_hash_depends_on_bytes_dtype_and_shape_only():
    img = gradient_image()
    assert content_hash(img) == content_hash(img.copy()) == content_hash(torch.from_numpy(img))
    assert content_hash(img).startswith("blake2b-128:") and len(content_hash(img)) == len("blake2b-128:") + 32
    changed = img.copy()
    changed[0, 0, 0] ^= 1
    assert content_hash(changed) != content_hash(img)
    assert content_hash(img[::-1]) != content_hash(img)                  # same bytes multiset, other order
    assert content_hash(img.astype(np.int16)) != content_hash(img)
    assert content_hash(img.reshape(48, 64, 3)) != content_hash(img)


def test_every_image_sent_is_hashed_at_every_request_next_to_the_frame_stats(instrumented, monkeypatch):
    monkeypatch.setenv(I.ENV_FRAME_EVERY, str(3 * CHUNK))               # frame stats at the first request only
    with ServerThread(make_policy) as server:
        policy = client(server.port)
        run_episode(policy, 2 * CHUNK + 1)
        policy.close()
        received = server.server.policy.policy.observations
    assert len(received) == 3
    for e in range(NUM_ENVS):
        ep = episodes(instrumented)[f"env{e:04d}_ep00000.json"]
        assert len(ep["frames"]) == 1 and len(ep["chunks"]) == 3
        for c, sent in zip(ep["chunks"], received):
            frames = sent["video"][VIDEO_KEY][e]
            assert c["sent_hashes"] == {VIDEO_KEY: [content_hash(f) for f in frames]}
        assert ep["frames"][0]["policy_input_hash"] == content_hash(received[0]["video"][VIDEO_KEY][e, -1])
        assert len({c["sent_hashes"][VIDEO_KEY][-1] for c in ep["chunks"]}) == 3   # fresh frames: all differ


def test_a_stale_frame_reused_by_the_client_shows_as_a_repeated_hash(instrumented):
    with ServerThread(make_policy) as server:
        policy = client(server.port)
        for t in range(2 * CHUNK + 1):
            policy.get_action(None, observation(0))                     # the same camera frame every step
        policy.close()
    ep = episodes(instrumented)["env0000_ep00000.json"]
    assert len({c["sent_hashes"][VIDEO_KEY][-1] for c in ep["chunks"]}) == 1 and len(ep["chunks"]) == 3


def test_raw_actions_are_logged_exactly_as_the_server_returned_them(instrumented):
    from gr00t.policy.gr00t_policy import fixed_chunk

    with ServerThread(make_policy) as server:
        policy = client(server.port)
        run_episode(policy, CHUNK + 1)
        policy.close()
    chunk = fixed_chunk(NUM_ENVS)
    for e in range(NUM_ENVS):
        ep = episodes(instrumented)[f"env{e:04d}_ep00000.json"]
        assert [c["obs_step"] for c in ep["chunks"]] == [0, CHUNK]
        for c in ep["chunks"]:
            assert list(c["raw"]) == list(chunk)
            for k, v in chunk.items():
                np.testing.assert_array_equal(np.asarray(c["raw"][k], dtype=np.float32), v[e])


def test_step_pairs_follow_the_chunk_with_no_extra_delay(instrumented):
    with ServerThread(make_policy) as server:
        policy = client(server.port)
        run_episode(policy, 2 * CHUNK + 4)
        policy.close()
    pairs = episodes(instrumented)["env0000_ep00000.json"]["step_pairs"]
    assert pairs[:CHUNK + 1] == [[0, t, t] for t in range(CHUNK)] + [[CHUNK, CHUNK, 0]]
    assert {a - o - k for o, a, k in pairs} == {0} and len(pairs) == 2 * CHUNK + 4


class DelayedPolicy(Gr00tRemoteClosedloopPolicy):
    """The client with a 3-step action buffer between the scheduler and ``env.step``: it returns each
    action 3 steps after the scheduler produced it, and holds (tag -1) until the buffer has filled."""

    DELAY = 3

    def _next_action(self, observation):
        if not hasattr(self, "_buffer"):
            self._buffer = deque()
        self._buffer.append(super()._next_action(observation))
        if len(self._buffer) <= self.DELAY:
            hold = self._extract_hold_action(observation)
            return TaggedAction(hold, torch.full((self.num_envs,), -1), torch.zeros(self.num_envs, dtype=torch.int64))
        return self._buffer.popleft()


def test_a_delay_buffer_shows_as_a_constant_extra_delay_of_three(instrumented):
    with ServerThread(make_policy) as server:
        policy = client(server.port, cls=DelayedPolicy)
        run_episode(policy, 2 * CHUNK + 4)
        policy.close()
    pairs = episodes(instrumented)["env0000_ep00000.json"]["step_pairs"]
    assert pairs[0] == [0, 3, 0] and pairs[CHUNK] == [CHUNK, CHUNK + 3, 0]
    assert {a - o - k for o, a, k in pairs} == {3}
    assert len(pairs) == 2 * CHUNK + 4 - DelayedPolicy.DELAY          # the held steps are not policy actions


def test_a_preprocessing_change_inside_the_client_shows_in_the_policy_input_stats(instrumented, tmp_path, monkeypatch):
    def frames(config=EVAL_YAML):
        with ServerThread(make_policy) as server:
            policy = client(server.port, config)
            run_episode(policy, 1)
            policy.close()
        return episodes(instrumented)["env0000_ep00000.json"]["frames"][0]

    nominal = frames()
    assert nominal["policy_input"] == nominal["render"]                  # same size, no conversion: identical
    swapped = frames(eval_variant(tmp_path, client={"bgr_conversion": True}))
    assert swapped["render"] == nominal["render"]
    assert swapped["policy_input"]["channel_means"] == pytest.approx(nominal["render"]["channel_means"][::-1])
    original = gr00t_core.resize_rgb_for_policy
    monkeypatch.setattr(gr00t_core, "resize_rgb_for_policy",
                        lambda *a, **k: [np.ascontiguousarray(f[:, ::-1]) for f in original(*a, **k)])
    flipped = frames()
    assert flipped["policy_input"]["row_profile"] == pytest.approx(nominal["render"]["row_profile"][::-1])


def test_episodes_split_at_reset_and_observation_keys_are_recorded(instrumented):
    with ServerThread(make_policy) as server:
        policy = client(server.port)
        policy.reset()                                   # before any step: starts no episode
        run_episode(policy, 5)
        policy.reset(torch.tensor([1]))                  # env 1 finishes after 5 steps
        run_episode(policy, 4, start=5)
        policy.close()                                   # env 0 (9 steps) and env 1 (4 steps)
    eps = episodes(instrumented)
    assert sorted(eps) == ["env0000_ep00000.json", "env0001_ep00000.json", "env0001_ep00001.json"]
    e0, e1a, e1b = eps["env0000_ep00000.json"], eps["env0001_ep00000.json"], eps["env0001_ep00001.json"]
    assert [(e["num_steps"], e["ended_by"], e["start_index"]) for e in (e0, e1a, e1b)] == [
        (9, "close", 0), (5, "reset", 1), (4, "close", 2)]
    assert e1b["step_pairs"][0] == [0, 0, 0]              # the new episode's chunk is tagged at its own step 0
    assert e0["observations"]["policy.left_eef_pos"] == [[float(t)] * 3 for t in range(9)]
    assert e1b["observations"]["policy.left_eef_pos"] == [[float(t)] * 3 for t in range(5, 9)]


# ----------------------------- schedulers ----------------------------------- #

def scripted_fetch(num_envs: int, horizon: int, dim: int = 2):
    calls = []

    def fetch():
        calls.append(1)
        return torch.arange(horizon, dtype=torch.float).repeat(num_envs, 1)[..., None].expand(num_envs, horizon, dim).clone()
    return fetch, calls


def test_the_chunk_scheduler_tags_each_env_with_its_own_fetch_step():
    s = ActionChunkScheduler(num_envs=2, action_chunk_length=4, action_horizon=4, action_dim=2, device="cpu")
    fetch, _ = scripted_fetch(2, 4)
    tags = []
    for step in range(6):
        if step == 2:
            s.reset([1])                                  # env 1 refetches at step 2
        t = s.get_tagged_action(fetch, obs_step=torch.full((2,), step))
        tags.append((t.obs_step.tolist(), t.chunk_offset.tolist()))
    assert tags == [([0, 0], [0, 0]), ([0, 0], [1, 1]), ([0, 2], [2, 0]), ([0, 2], [3, 1]),
                    ([4, 2], [0, 2]), ([4, 2], [1, 3])]
    assert s.get_action(fetch).shape == (2, 2)            # the untagged API is unchanged


def test_the_synced_batch_scheduler_tags_held_envs_with_minus_one():
    s = SyncedBatchActionScheduler(num_envs=2, action_chunk_length=3, action_horizon=3, action_dim=2, device="cpu")
    fetch, calls = scripted_fetch(2, 3)
    hold = torch.full((2, 2), -7.0)
    t0 = s.get_tagged_action(fetch, hold, torch.zeros(2, dtype=torch.int64))
    s.reset([0])                                          # env 0 waits for env 1
    t1 = s.get_tagged_action(fetch, hold, torch.ones(2, dtype=torch.int64))
    assert t0.obs_step.tolist() == [0, 0] and t1.obs_step.tolist() == [-1, 0] and t1.chunk_offset.tolist() == [0, 1]
    assert t1.action[0].tolist() == [-7.0, -7.0] and len(calls) == 1


def test_an_instrument_without_steps_leaves_the_tags_unset():
    s = ActionChunkScheduler(num_envs=1, action_chunk_length=2, action_horizon=2, action_dim=1, device="cpu")
    fetch, _ = scripted_fetch(1, 2, 1)
    assert s.get_tagged_action(fetch, obs_step=NullInstrument.steps).obs_step.tolist() == [-1]
    with pytest.raises(ValueError):
        Instrument(1, "unused", frame_every=0)
