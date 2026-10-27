"""Shared test helpers: eval-config variants, free ports, a policy server in a thread."""

from __future__ import annotations

import socket
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[1]
EVAL_YAML = REPO / "config" / "eval.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def eval_variant(tmp_path: Path, **sections: dict[str, Any]) -> Path:
    """A copy of ``config/eval.yaml`` under ``tmp_path`` with section keys replaced; the client's
    relative paths point back into this repo."""
    cfg = yaml.safe_load(EVAL_YAML.read_text())
    for key, value in list(cfg["client"].items()):
        if key.endswith("_path") and value:
            cfg["client"][key] = str(REPO / value)
    for section, values in sections.items():
        cfg[section].update(values)
    path = tmp_path / "config" / "eval.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


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
