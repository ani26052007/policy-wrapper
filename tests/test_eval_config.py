"""``config/eval.yaml``: the pilot values, and the client section as the client's only config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_wrapper.eval_config import load_client_config, read_eval_config, repo_root
from policy_wrapper.policy.gr00t_core import resize_rgb_for_policy
from tests.helpers import EVAL_YAML, REPO, eval_variant

PILOT = {
    "client": {
        "language_instruction": "Place the sauce bottle on the top shelf of the fridge, and close the fridge door.",
        "embodiment_tag": "GR1", "action_horizon": 16, "action_chunk_length": 16, "scheduler": "chunk",
        "modality_config_path": "policy_wrapper/embodiments/gr1/gr1_arms_only_data_config.py",
        "policy_joints_config_path": "policy_wrapper/embodiments/gr1/gr00t_26dof_joint_space.yaml",
        "action_joints_config_path": "policy_wrapper/embodiments/gr1/36dof_joint_space.yaml",
        "state_joints_config_path": "policy_wrapper/embodiments/gr1/54dof_joint_space.yaml",
        "task_mode_name": "gr1_tabletop_manipulation", "pov_cam_name_sim": "robot_pov_cam_rgb",
        "original_image_size": [512, 512, 3], "target_image_size": [512, 512, 3], "bgr_conversion": False},
    "server": {
        "checkpoint": "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge",
        "checkpoint_revision": "TODO_PIN", "embodiment_tag": "GR1", "denoising_steps": 4,
        "seed_per_request": True, "norm_stats_override": None},
    "env": {
        "task": "put_item_in_fridge_and_close_door", "embodiment": "gr1_joint", "object": "ranch_dressing_hope_robolab",
        "control_hz": 50, "actuator_overrides": {}, "render": {}},
}


def test_the_eval_config_holds_exactly_the_pilot_values():
    assert read_eval_config(EVAL_YAML) == PILOT
    assert repo_root(EVAL_YAML) == REPO


def test_the_client_config_comes_from_the_client_section_with_repo_relative_paths():
    cfg = load_client_config(EVAL_YAML)
    assert cfg.language_instruction == PILOT["client"]["language_instruction"]
    assert (cfg.action_horizon, cfg.action_chunk_length, cfg.scheduler, cfg.bgr_conversion) == (16, 16, "chunk", False)
    assert cfg.target_image_size == (512, 512, 3) and cfg.pov_cam_name_sim == ["robot_pov_cam_rgb"]
    assert Path(cfg.policy_joints_config_path) == REPO / PILOT["client"]["policy_joints_config_path"]
    assert Path(cfg.modality_config_path) == REPO / PILOT["client"]["modality_config_path"]


def test_an_unknown_client_key_or_section_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown client keys"):
        load_client_config(eval_variant(tmp_path / "a", client={"replan_steps": 8}))
    cfg = yaml.safe_load(EVAL_YAML.read_text())
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
