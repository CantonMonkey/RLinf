# RLinf v0.3 R2R environment/config staging

This is an isolated patch set installed on 2026-09-24 in the A800 private
`repos/RLinf-v0.3` checkout at `0505431899574619da86f551bad70b71e0ea2177`.
The installed file mapping and validation state are in `VLN_A800_INSTALL.md`.
The existing PointNav registry file has SHA256
`058b738a916bbe0dc5cd11c24faf228115f49801e79cdd57918ebebab0b7f3c6`.
`envs_init_vln.py` is a copy of that exact file with one `vln` enum and one
lazy-import branch added. Applying a complete-file copy requires checking the
destination hash and diff again; do not overwrite subsequent remote changes.

| Local staged file | Destination relative to `repos/RLinf-v0.3/` |
| --- | --- |
| `envs_init_vln.py` | `rlinf/envs/__init__.py` |
| `vln_init.py` | `rlinf/envs/vln/__init__.py` |
| `vln_r2r_env.yaml` | `examples/embodiment/config/env/vln_r2r_env.yaml` |
| `vln_policy.yaml` | `examples/embodiment/config/model/vln_policy.yaml` |
| `vln_r2r_ppo_smoke.yaml` | `examples/embodiment/config/vln_r2r_ppo_smoke.yaml` |
| `vln_r2r_ppo_eval.yaml` | `examples/embodiment/config/vln_r2r_ppo_eval.yaml` |

The environment implementation is separately staged as `vln_env.py` for
`rlinf/envs/vln/vln_env.py`. The service is separately staged under
`rlinf-nav/scripts/habitat_r2r_service.py`. Model registry and implementation
have their own staging map in `VLN_REGISTRATION.md`. This config expects those
parts plus a VLN-aware evaluation entrypoint before a Ray run.

`vln_r2r_ppo_smoke` selects R2R train episode file index 4852 and
`val_unseen` index 630, on different MP3D scenes. It uses 64×64 RGB and the
original 200-position R2R instruction token vector with vocabulary size 2504.
The environment protocol requests a 3.0 m Success threshold. The eight-step
train/validation budget only checks rollout, update, and checkpoint mechanics.
The YAML explicitly provides the private Habitat-Sim Python/library paths and
Mesa EGL vendor; `VLNEnv` applies these only to its simulator child process,
leaving the RLinf policy worker's CUDA environment intact.
`vln_r2r_ppo_eval` evaluates one held-out `val_unseen` episode with the
500-step Habitat VLN horizon, loads matching `runner.resume_dir` and
`runner.ckpt_path`, and asks RLinf's `RecordVideo` wrapper to write a six-fps
MP4 from the `rgb` observation. Actual metric/video output is pending an
integrated run. The shared `/cache/navigation-example` assets remain read-only.

Focused local check:

```powershell
cd C:\path\to\navigation-project\rlinf-nav
python -B -m unittest discover -s tests -p test_vln_env_registry_config_staged.py -v
```

This check parses YAML and tests the lazy registry branch. Both configs have
also composed in the installed checkout with CUDA hidden, and the real
train/val_unseen Habitat environment bridge has passed separately. Ray/FSDP,
checkpoint restoration and output video remain unverified.
