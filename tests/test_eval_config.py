"""``config/eval.yaml``: the pilot values, and the client section as the client's only config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_wrapper.eval_config import load_client_config, read_eval_config, repo_root
from policy_wrapper.policy.gr00t_core import resize_rgb_for_policy
from policy_wrapper.serve import load_server_section
from tests.helpers import EVAL_YAML, REPO, TEST_CONFIG, eval_variant

def test_the_repo_eval_config_has_the_three_sections_and_every_client_and_server_key_loads():
    raw = read_eval_config(EVAL_YAML)
    assert set(raw) == {"client", "server", "env"} and repo_root(EVAL_YAML) == REPO
    cfg = load_client_config(EVAL_YAML)
    for key, value in raw["client"].items():
        if key.endswith("_path"):
            assert Path(getattr(cfg, key)) == REPO / value                # relative to the repo root
    assert load_server_section(EVAL_YAML).checkpoint == raw["server"]["checkpoint"]
    assert {"task", "embodiment", "object", "control_hz", "actuator_overrides", "render"} <= set(raw["env"])


def test_the_client_config_comes_from_the_client_section(fixed_yaml):
    cfg = load_client_config(fixed_yaml)
    client = TEST_CONFIG["client"]
    assert cfg.language_instruction == client["language_instruction"]
    assert (cfg.action_horizon, cfg.action_chunk_length, cfg.scheduler, cfg.bgr_conversion) == (16, 16, "chunk", False)
    assert cfg.target_image_size == (512, 512, 3) and cfg.pov_cam_name_sim == ["robot_pov_cam_rgb"]
    assert Path(cfg.policy_joints_config_path) == REPO / client["policy_joints_config_path"]
    assert Path(cfg.modality_config_path) == REPO / client["modality_config_path"]


def test_an_unknown_client_key_or_section_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown client keys"):
        load_client_config(eval_variant(tmp_path / "a", client={"replan_steps": 8}))
    cfg = dict(TEST_CONFIG)
    cfg["extra"] = {}
    (tmp_path / "b" / "config").mkdir(parents=True)
    (tmp_path / "b" / "config" / "eval.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match="unknown sections"):
        read_eval_config(tmp_path / "b" / "config" / "eval.yaml")


def test_the_client_config_still_validates_its_values(tmp_path):
    with pytest.raises(AssertionError, match="action_chunk_length"):
        load_client_config(eval_variant(tmp_path, client={"action_chunk_length": 32}))


def test_bgr_conversion_swaps_the_channels_before_resizing():
    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    frames[..., 0] = 10
    frames[..., 2] = 200
    same = resize_rgb_for_policy([frames], (4, 4, 3))[0]
    swapped = resize_rgb_for_policy([frames], (4, 4, 3), bgr_conversion=True)[0]
    assert same[0, 0, 0].tolist() == [10, 0, 200] and swapped[0, 0, 0].tolist() == [200, 0, 10]
    assert swapped.flags["C_CONTIGUOUS"]
