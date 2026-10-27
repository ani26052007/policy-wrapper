# policy-wrapper

The GR00T N1.6 client used by Isaac Lab-Arena's experiment runner, and the policy server it
talks to, configured by one file: `config/eval.yaml` (`client:`, `server:`, `env:`).

- Client: `policy_wrapper.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy`
  (Arena `policy.type`; `policy_config_yaml_path` is `config/eval.yaml`).
- Server: `python -m policy_wrapper.serve --config config/eval.yaml --host 127.0.0.1 --port 5555`
  (needs Isaac-GR00T and a local copy of the checkpoint).

Tests (no GPU, no Isaac Sim, no GR00T install): `python -m pytest -q`.

See NOTICE for the files that come from Isaac Lab-Arena.
