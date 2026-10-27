"""``serve.py``: the server section, checkpoint resolution, ``build_policy`` hooks, and the server
process end to end (stand-in ``gr00t``; no model, no GPU)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

from policy_wrapper.serve import ServerSection, SeededPolicy, build_policy, load_server_section, resolve_checkpoint
from tests.conftest import FAKE_GR00T
from tests.helpers import EVAL_YAML, FIXTURES, REPO, eval_variant, free_port

CKPT = FIXTURES / "checkpoint"


def section(**kw) -> ServerSection:
    base = load_server_section(EVAL_YAML)
    return ServerSection(**{**base.__dict__, **kw})


def test_the_server_section_is_the_pilot_and_is_read_strictly(tmp_path):
    s = load_server_section(EVAL_YAML)
    assert s == ServerSection("nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge",
                              "TODO_PIN", "GR1", 4, True, None)
    with pytest.raises(AssertionError, match="unknown server keys"):
        load_server_section(eval_variant(tmp_path / "a", server={"num_inference_timesteps": 8}))
    cfg = yaml.safe_load(EVAL_YAML.read_text())
    del cfg["server"]["seed_per_request"]
    (tmp_path / "b" / "config").mkdir(parents=True)
    (tmp_path / "b" / "config" / "eval.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match="missing server keys"):
        load_server_section(tmp_path / "b" / "config" / "eval.yaml")
    with pytest.raises(AssertionError, match="denoising_steps"):
        section(denoising_steps=0)


def test_the_checkpoint_resolves_from_a_dir_a_relative_path_or_the_local_cache_only(tmp_path, monkeypatch):
    assert resolve_checkpoint(section(), REPO, CKPT) == CKPT
    assert resolve_checkpoint(section(checkpoint=str(CKPT)), REPO) == CKPT
    assert resolve_checkpoint(section(checkpoint="tests/fixtures/checkpoint"), REPO) == REPO / "tests/fixtures/checkpoint"
    calls = []
    snap = tmp_path / "snap"
    (snap / "ranch_bottle_into_fridge").mkdir(parents=True)

    def snapshot_download(repo_id, **kw):
        calls.append((repo_id, kw))
        return str(snap)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    assert resolve_checkpoint(section(), REPO) == snap / "ranch_bottle_into_fridge"
    assert resolve_checkpoint(section(checkpoint_revision="abc123"), REPO) == snap / "ranch_bottle_into_fridge"
    assert calls == [
        ("nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task",
         {"revision": None, "allow_patterns": ["ranch_bottle_into_fridge/*"], "local_files_only": True}),
        ("nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task",
         {"revision": "abc123", "allow_patterns": ["ranch_bottle_into_fridge/*"], "local_files_only": True})]
    with pytest.raises(AssertionError, match="not a directory"):
        resolve_checkpoint(section(), REPO, tmp_path / "missing")


def test_build_policy_sets_denoising_steps_and_keeps_the_checkpoint_statistics():
    p = build_policy(section(denoising_steps=8), REPO, device="cpu", checkpoint_dir=CKPT)
    assert isinstance(p, SeededPolicy) and p.seed_per_request
    inner = p.policy
    assert inner.model.action_head.num_inference_timesteps == 8
    assert inner.model_path == str(CKPT) and inner.device == "cpu" and inner.embodiment_tag.name == "GR1"
    assert inner.processor.set_statistics_calls == []
    assert inner.processor.statistics == json.loads((CKPT / "statistics.json").read_text())


def test_norm_stats_override_replaces_the_statistics(tmp_path):
    override = {"gr1": {"action": {"left_arm": {"min": [-2.0] * 7, "max": [2.0] * 7}}}}
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(override))
    p = build_policy(section(norm_stats_override=str(path)), REPO, device="cpu", checkpoint_dir=CKPT)
    assert p.policy.processor.set_statistics_calls == [(override, True)]
    assert p.policy.processor.statistics == override
    rel = build_policy(section(norm_stats_override="tests/fixtures/checkpoint/statistics.json"), REPO,
                       device="cpu", checkpoint_dir=CKPT)
    assert rel.policy.processor.set_statistics_calls[0][1] is True


def test_the_server_process_answers_ping_and_get_action_and_stops_on_kill(tmp_path):
    from gr00t.policy.server_client import PolicyClient

    port = free_port()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(FAKE_GR00T), str(REPO)])}
    proc = subprocess.Popen([sys.executable, "-m", "policy_wrapper.serve", "--config", str(EVAL_YAML),
                             "--host", "127.0.0.1", "--port", str(port), "--device", "cpu",
                             "--checkpoint-dir", str(CKPT)],
                            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=1000)
    try:
        deadline = time.monotonic() + 30
        while not client.ping():
            assert proc.poll() is None, proc.stdout.read()
            assert time.monotonic() < deadline, "server did not answer ping"
            client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=1000)
        action, info = client.get_action({"language": {"k": [["x"], ["x"]]}}, {"seed": 7})
        assert action["left_arm"].shape == (2, 16, 7) and info["seed"] == 7
        assert client.reset() == {}
        client.kill_server()
        assert proc.wait(10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert "policy_wrapper.serve: nvidia/GN1.6" in proc.stdout.read()
