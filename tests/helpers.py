"""Shared test helpers: a fixed test eval config and variants of it, free ports, a policy server in a thread.

The tests run on ``fixed_eval_config``, a config written for the tests, not on the repo's ``config/eval.yaml``:
that file is the evaluation's setting and changes with it; the tests check the mechanism."""

from __future__ import annotations

import copy
import functools
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[1]
EVAL_YAML = REPO / "config" / "eval.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


INSTRUCTION = "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
TEST_CONFIG = {
    "client": {
        "language_instruction": INSTRUCTION, "embodiment_tag": "GR1", "action_horizon": 16, "action_chunk_length": 16,
        "scheduler": "chunk", "modality_config_path": "policy_wrapper/embodiments/gr1/gr1_arms_only_data_config.py",
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


def eval_variant(tmp_path: Path, **sections: dict[str, Any]) -> Path:
    """``TEST_CONFIG`` written to ``tmp_path/config/eval.yaml`` with section keys replaced; the client's
    relative paths point into this repo."""
    cfg = copy.deepcopy(TEST_CONFIG)
    for key, value in list(cfg["client"].items()):
        if key.endswith("_path") and value:
            cfg["client"][key] = str(REPO / value)
    for section, values in sections.items():
        cfg[section].update(values)
    path = tmp_path / "config" / "eval.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path



@functools.lru_cache(maxsize=None)
def fixed_eval_yaml() -> Path:
    """``TEST_CONFIG`` as a file, written once per test session."""
    return eval_variant(Path(tempfile.mkdtemp(prefix="policy-wrapper-tests-")))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    """``gr00t.policy.server_client.PolicyServer`` built and run inside its own thread (a ZMQ socket
    must stay in the thread that uses it). Stopped with the ``kill`` endpoint."""

    def __init__(self, make_policy: Callable[[], Any]):
        self.port = free_port()
        self.ready = threading.Event()
        self.server = None
        self.thread = threading.Thread(target=self._run, args=(make_policy,), daemon=True)

    def _run(self, make_policy):
        from gr00t.policy.server_client import PolicyServer

        self.server = PolicyServer(make_policy(), host="127.0.0.1", port=self.port)
        self.ready.set()
        self.server.run()

    def __enter__(self) -> "ServerThread":
        self.thread.start()
        assert self.ready.wait(10), "server did not start"
        return self

    def __exit__(self, *exc) -> None:
        from gr00t.policy.server_client import PolicyClient

        client = PolicyClient(host="127.0.0.1", port=self.port, timeout_ms=5000)
        client.kill_server()
        self.thread.join(10)
        client.socket.close(linger=0)
        client.context.term()
        assert not self.thread.is_alive()
