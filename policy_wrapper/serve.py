"""GR00T policy server for this repo: ``Gr00tPolicy`` configured from ``config/eval.yaml`` ``server:``.

    python -m policy_wrapper.serve --config config/eval.yaml --host 127.0.0.1 --port 5555 \
        [--device cuda] [--checkpoint-dir <local checkpoint dir>]

``server:`` keys:
  ``checkpoint``           Hugging Face id ``<org>/<repo>[/<subfolder>]`` or a local directory
                           (absolute, or relative to the repo root).
  ``checkpoint_revision``  the Hub revision of ``checkpoint``.
  ``embodiment_tag``       a GR00T ``EmbodimentTag`` name, e.g. ``GR1``.
  ``denoising_steps``      flow-matching steps of the action head
                           (``model.action_head.num_inference_timesteps``).
  ``seed_per_request``     seed torch before every ``get_action`` (below).
  ``norm_stats_override``  null, or a ``statistics.json`` (relative to the repo root) that replaces
                           the checkpoint's normalisation statistics.

The checkpoint is never downloaded here: ``--checkpoint-dir``, a local path, or the local Hugging
Face cache (``local_files_only``) must already hold it.

Seeding: GR00T draws the action head's noise from the global torch RNG and seeds nothing. With
``seed_per_request`` the server calls ``torch.manual_seed`` before each request, with
``options["seed"]`` when the client sends one (``policy_wrapper.seeding``), otherwise with
``request_seed(0, resets, requests since the last reset)`` from its own counters, which ``reset``
restarts. The seed used is returned in the reply's info dict.

Modality config: GR00T's own server ignores ``--modality-config-path`` when it is given a model
path; the policy uses the modality configs stored in the checkpoint's processor. This server has no
such option. ``client.modality_config_path`` is read by the Arena-side client only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

from policy_wrapper.eval_config import EVAL_CONFIG_PATH, read_eval_config, repo_root
from policy_wrapper.seeding import request_seed

UNPINNED = "TODO_PIN"


@dataclass(frozen=True)
class ServerSection:
    checkpoint: str
    checkpoint_revision: str
    embodiment_tag: str
    denoising_steps: int
    seed_per_request: bool
    norm_stats_override: Optional[str]

    def __post_init__(self) -> None:
        assert self.checkpoint, "server.checkpoint is empty"
        assert isinstance(self.denoising_steps, int) and self.denoising_steps > 0, "server.denoising_steps must be > 0"
        assert isinstance(self.seed_per_request, bool), "server.seed_per_request must be true or false"


def load_server_section(eval_config_path: str | Path) -> ServerSection:
    raw = read_eval_config(eval_config_path)["server"]
    names = [f.name for f in fields(ServerSection)]
    unknown, missing = set(raw) - set(names), set(names) - set(raw)
    assert not unknown, f"{eval_config_path}: unknown server keys {sorted(unknown)}"
    assert not missing, f"{eval_config_path}: missing server keys {sorted(missing)}"
    return ServerSection(**raw)


def resolve_checkpoint(section: ServerSection, root: Path, checkpoint_dir: str | Path | None = None) -> Path:
    """The local directory that holds the checkpoint's files."""
    if checkpoint_dir is not None:
        path = Path(checkpoint_dir)
        assert path.is_dir(), f"--checkpoint-dir {path} is not a directory"
        return path
    local = Path(section.checkpoint) if Path(section.checkpoint).is_absolute() else root / section.checkpoint
    if local.is_dir():
        return local
    parts = section.checkpoint.split("/")
    assert len(parts) >= 2, f"server.checkpoint {section.checkpoint!r} is neither a directory nor a Hub id"
    repo_id, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    from huggingface_hub import snapshot_download

    # VERIFY(V-ckpt-cache): a checkpoint downloaded by R2 resolves from the cache with local_files_only.
    snapshot = snapshot_download(
        repo_id,
        revision=None if section.checkpoint_revision == UNPINNED else section.checkpoint_revision,
        allow_patterns=[f"{subfolder}/*"] if subfolder else None,
        local_files_only=True,
    )
    path = Path(snapshot) / subfolder
    assert path.is_dir(), f"checkpoint {section.checkpoint!r} is not in the local Hugging Face cache"
    return path


class SeededPolicy:
    """Wraps a GR00T policy for ``PolicyServer``: seeds torch per request when configured."""

    def __init__(self, policy: Any, seed_per_request: bool):
        self.policy = policy
        self.seed_per_request = seed_per_request
        self.num_resets = 0
        self.num_requests = 0

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
        seed = None
        if self.seed_per_request:
            import torch

            if options and options.get("seed") is not None:
                seed = int(options["seed"])
            else:
                seed = request_seed(0, self.num_resets, self.num_requests)
            torch.manual_seed(seed)
        self.num_requests += 1
        action, info = self.policy.get_action(observation, options)
        return action, {**(info or {}), "seed": seed}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.num_resets += 1
        self.num_requests = 0
        return self.policy.reset(options)

    def get_modality_config(self):
        return self.policy.get_modality_config()


def build_policy(
    section: ServerSection,
    root: Path,
    *,
    device: str = "cuda",
    checkpoint_dir: str | Path | None = None,
) -> SeededPolicy:
    """``Gr00tPolicy`` from the server section, with denoising steps, statistics and seeding applied."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    model_dir = resolve_checkpoint(section, root, checkpoint_dir)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag[section.embodiment_tag.upper()],
        model_path=str(model_dir),
        device=device,
    )
    action_head = policy.model.action_head
    assert hasattr(action_head, "num_inference_timesteps"), "the action head has no num_inference_timesteps"
    action_head.num_inference_timesteps = section.denoising_steps
    if section.norm_stats_override is not None:
        stats_path = Path(section.norm_stats_override)
        if not stats_path.is_absolute():
            stats_path = root / stats_path
        with open(stats_path, encoding="utf-8") as f:
            policy.processor.set_statistics(json.load(f), override=True)
    return SeededPolicy(policy, section.seed_per_request)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(EVAL_CONFIG_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args(argv)

    section = load_server_section(args.config)
    policy = build_policy(section, repo_root(args.config), device=args.device, checkpoint_dir=args.checkpoint_dir)

    from gr00t.policy.server_client import PolicyServer

    server = PolicyServer(policy=policy, host=args.host, port=args.port)
    print(f"policy_wrapper.serve: {section.checkpoint} on {args.host}:{args.port}", flush=True)
    server.run()


if __name__ == "__main__":
    main()
