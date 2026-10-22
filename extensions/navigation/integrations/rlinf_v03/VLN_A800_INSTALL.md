# A800 private RLinf v0.3 R2R installation

Installed on 2026-09-24 in the **private** checkout
`/srv/navigation/repos/RLinf-v0.3`, based on commit
`0505431899574619da86f551bad70b71e0ea2177`. The existing PointNav
changes in that checkout were retained. R2R/MP3D data under
`/cache/navigation-example` are only read by the new Habitat service.

The staged source files live under
`/srv/navigation/repos/rlinf-nav/integrations/rlinf_v03`.
The following 13 staged-to-installed SHA256 pairs matched at the initial installation. Later smoke-config, worker, and UUID-gate revisions are documented below:

| Staged filename | Installed path relative to `RLinf-v0.3/` |
| --- | --- |
| `config_vln.py` | `rlinf/config.py` |
| `models_init_vln.py` | `rlinf/models/__init__.py` |
| `huggingface_worker_vln.py` | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| `envs_init_vln.py` | `rlinf/envs/__init__.py` |
| `vln_init.py` | `rlinf/envs/vln/__init__.py` |
| `vln_env.py` | `rlinf/envs/vln/vln_env.py` |
| `vln_policy.py` | `rlinf/models/embodiment/vln_policy.py` |
| `embodied_io_struct_vln.py` | `rlinf/data/embodied_io_struct.py` |
| `train_embodied_agent.py` | `examples/embodiment/train_embodied_agent.py` |
| `vln_r2r_env.yaml` | `examples/embodiment/config/env/vln_r2r_env.yaml` |
| `vln_policy.yaml` | `examples/embodiment/config/model/vln_policy.yaml` |
| `vln_r2r_ppo_smoke.yaml` | `examples/embodiment/config/vln_r2r_ppo_smoke.yaml` |
| `vln_r2r_ppo_eval.yaml` | `examples/embodiment/config/vln_r2r_ppo_eval.yaml` |

Before replacing existing destinations, their SHA256 values matched the
recorded private-checkout inputs in `VLN_REGISTRATION.md`,
`VLN_ENV_CONFIG_STAGING.md`, and the local pre-VLN PointNav entrypoint:
`config.py` `bef76609…`, `models/__init__.py` `b6920944…`, HF rollout
`480d3797…`, env registry `058b738a…`, `embodied_io_struct.py`
`c21c2144…`, trainer `9dace8f3…`. The trainer's previous PointNav version is
preserved at `repos/rlinf-nav/integrations/rlinf_v03/train_embodied_agent_pre_vln.py`
with the same `9dace8f3…` hash. `git diff --check` passed after installation.

Both `vln_r2r_ppo_smoke` and `vln_r2r_ppo_eval` composed with resolved Hydra
interpolations in the installed checkout while CUDA was hidden. The composed
train config selects train index 4852; the evaluation config selects
`val_unseen` index 630, 500 maximum steps, and RLinf RGB MP4 recording.
Config composition does not start Ray, CUDA, Habitat, PPO, or recording.
The pre-existing PointNav node preflight was adjusted to accept the exact
PointNav or VLN-extended registry/entrypoint variants. Re-running it after
installation returned `static_preflight_ok`, verified its 12 bridge files,
and recomposed the PointNav train/val split checks with CUDA hidden; see
`artifacts/integrations/rlinf_v03_runtime/vln_install_pointnav_preflight_20260924.json`
at project root.

The real A800 Habitat service and VLN environment bridge were separately
validated on both selected R2R splits before checkout installation. Their
evidence is in `artifacts/vln_r2r_policy_boundary_20260924/` and
`artifacts/vln_r2r_env_bridge_20260924/` at project root. The service uses the
private Mesa build in a child process and verifies the 3.0 m STOP success
threshold. These checks establish the observation and episode protocol, not
learned navigation or a Ray/FSDP update.

After installation, a bounded A800 CPU run used the actual R2R train index
4852 observation to exercise the installed environment/model registry,
`EnvOutput.to_dict()` with exactly `rgb`, `instruction_ids`, and
`instruction_mask`, the installed HuggingFace rollout `predict` function body,
behavior log-probability/value replay, finite policy gradients, and one sampled
action passed to real Habitat. The sampled `turn_right` action returned a finite
reward; the service exited cleanly. The test unwraps `Worker.timer` to call the
installed function body without a Ray worker. The source, result, exact hashes,
and reproduction command are in
`artifacts/vln_cpu_installed_integration_20260924/README.md` at project root.
The successful pass used Hydra composition and targeted field checks. Full
`validate_cfg()` starts a local Ray instance as a side effect, so it was not
used in the final CPU pass; the incidental Ray session from an earlier failed
attempt had no surviving processes when checked.

A separate fixed-seed **untrained** `VLNPolicy` drove real Habitat R2R
`val_unseen` index 630 (episode ID 631) for six actions followed by STOP.
Private CPU Mesa rendered seven 64×64 RGB frames into an H.264 MP4. The video,
action trace, manifest, and random-initialization weights were pulled to
Windows with four matching remote/local SHA256 pairs; FFmpeg decoded every
frame locally. Success and SPL were both zero. The result and reproduction
command are in `artifacts/vln_video_rollout_20260924/README.md` at project
root; the playable file is
`artifacts/vln_video_rollout_20260924/val_unseen_episode630_untrained/habitat_rgb_rollout.mp4`.
This validates recorded visual rollout and transfer, not learning or a trained
policy evaluation.

## One-sample GPU smoke configuration follow-up

On `ebk180`, a PointNav GPU smoke completed the trainer path but a first-step
STOP left one valid transition. With sample-standard-deviation advantage
normalization enabled, its advantage and gradient norm became NaN and FSDP did
not make a usable optimizer update (`runs/pointnav-gpu-gfuvud0v`). An earlier
attempt (`runs/pointnav-gpu-298xcx3y`) had a Ray actor import failure. After
the numerical configuration change, another attempt
(`runs/pointnav-gpu-kpd2bhmr`) hit a separate PortLockManager SIGSEGV while
importing Ray State API/colorful, before the numerical fix could be judged.
The subsequent 4090 private `runs/pointnav-gpu-0sjdh9h3` retry completed an
actual bounded PointNav GPU/FSDP one-step update, with reported gradient norm
3.746 and actor optimizer state at step 2 with nonzero moments. That is a 4090
PointNav result, not an A800 or VLN training result.

Only the private A800 **smoke** YAMLs `pointnav_ppo_smoke.yaml` and
`vln_r2r_ppo_smoke.yaml` were synchronized from the local sources with
`algorithm.normalize_advantages: false`; ordinary training configurations
were not changed. Before replacement, each A800 staging file matched its
installed counterpart. The four original files were backed up in the private
`/srv/navigation/runs/smoke-config-nan-fix-20260924/`
directory. Afterwards both PointNav copies had SHA256 `47a060077fca…` and
both VLN copies `22bebf795125…`, matching the local sources. CPU-only Hydra
composition of both installed configs resolved their interpolations and
confirmed `normalize_advantages=false` with train/val or train/val_unseen
roles; the PointNav node preflight returned `static_preflight_ok`. Detailed
hashes and local check outputs are in the project-root
[configuration follow-up](../../../artifacts/vln_gpu_smoke_config_fix_20260924/README.md).
The `cluster.py` adjustment was subsequently installed in both A800 private
staging and RLinf checkout, after the matching prior copies (SHA256
`00c9fe55012e69b6656dd71db366bf3933f7bc338142d916ff61f575d257fea3`)
were backed up in private `runs/cluster-import-fix-20260924/`. Both current
copies match the local SHA256
`59d68e0e7b692d31431e8a2d09414e9fc816fc052b10ba9b92b014bc395e9491`.
CPU import with CUDA hidden succeeded and left Ray uninitialized;
`git diff --check`, the PointNav static preflight, and VLN Hydra composition
passed. The exact checks are in the project-root
[A800 cluster import record](../../../artifacts/a800_cluster_import_fix_20260924/README.md).
The installed RLinf `worker.py` still imports `ray.util.state` at module load,
so this CPU check alone did not prove every actor import path was free of it.
The later bounded A800 Ray runs and their outcomes are recorded below.

## Bounded A800 GPU acceptance

The initial 13-file installation was followed by two private worker revisions.
`gpu_uuid_gate.py` is installed at `rlinf/utils/gpu_uuid_gate.py`,
`fsdp_actor_worker_gpu_uuid.py` at
`rlinf/workers/actor/fsdp_actor_worker.py`, and the rollout worker was updated
to call the same gate. The final installed worker source SHA256 is
`e2ee2a879907775c0a285bde83439e081ad44c32e57910686be5bb0668ce6c14`.
The gate checks the allowed physical ordinal, one visible logical CUDA device,
and the driver-reported physical UUID before making a CUDA context. It is
opt-in for these bounded runs. The train/eval preflights verified the staged
sources against their installed counterparts; the final VLN eval preflight
lists 15 matching installed files.

A repaired PointNav Ray/FSDP run,
`runs/pointnav-a800-gpu-train-14hh4yyc`, completed trainer exit 0,
`Global Step 1/1`, and grad norm 2.848 using real Habitat train index 50.
Its actor checkpoint passed strict DCP/full-weight comparison; optimizer
state reached step 2 with nonzero moments. The wrapper's postflight checked
GPU ordinal 0 instead of 7, so that run's `result.json` reports `error`
despite trainer exit 0. The wrapper was corrected before the independent
`runs/pointnav-a800-gpu-eval-r_9o20vp` run. That `only_eval` exited 0 and
recorded val indices 50–54 / episode IDs 0–4, all Success/SPL zero and
path length zero. The PointNav run used a GPU 7 allow-list and resource
preflight, but did not directly log actor/rollout physical UUID.

The first VLN run with the actual-worker UUID gate,
`runs/vln-a800-gpu-train-_dy_1bw_`, exposed a second one-sample STOP issue:
although the smoke YAML said `normalize_advantages=false`,
`EmbodiedFSDPActor` had not passed that setting to
`calculate_adv_and_returns`. GAE and the gradient were NaN, so trainer
exit 0 and a saved checkpoint do **not** count as a valid update.
The installed actor revision now explicitly passes
`normalize_advantages` in those GAE keyword arguments. The later bounded
`runs/vln-a800-gpu-train-nquddujf` used real R2R train index 4852,
completed `Global Step 1/1`, and produced finite one-step STOP advantage
-0.0064, actor grad norm 0.016, and policy loss 0.0064. Its actor PID
2038464 and rollout PID 2038527 both logged `ok=true`,
`CUDA_VISIBLE_DEVICES=7`, logical `cuda:0`, physical UUID
`GPU-22222222-2222-2222-2222-222222222222`, with no context created by
the gate. The checkpoint DCP and `full_weights.pt` match on all 21 model
tensors; 21 optimizer state entries reached step 2 and all 42 first/second
moment tensors are finite and nonzero. The train episode failed after an
immediate STOP; this is a valid optimizer-update acceptance, not evidence
of learned navigation.

Independent `runs/vln-a800-gpu-eval-7j_32guy` loaded that exact
`global_step_1` checkpoint. CPU preflight strictly compared DCP with full
weights (SHA256
`230098662c44f46131539286ee1b33c99133769ea19d95a2732447eb1661894b`)
before Ray start. Eval actor and rollout workers separately passed the
same GPU 7 UUID gate, and the trainer exited 0. The per-episode JSON
identifies `val_unseen` index 630, episode ID 631, a scene disjoint from
the train split; Success=0, SPL=0, path length=0, distance to goal=5.035 m,
and collisions=0. The H.264 64×64 MP4 has 501 frames and decoded fully
on Windows. Its remote/local SHA256 is
`cf1498f5bf005f57102ff0522877754178c040703de20dcac559249d73a99ec4`.
Frame count is not the count of effective navigation moves. Decoded-frame
hashes show only three distinct pictures in those 501 frames; frames 0-249
are identical, as are frames 250-499. RLinf did not emit an action trace for
this eval. A separate strict replay of the same checkpoint SHA256 and the
same val_unseen episode recorded `stop` at step 1, `terminated=true`, an
unchanged RGB hash, and path length 0 m. Its
[`action_trace.jsonl`](../../../artifacts/a800_gpu_20260924/vln_trained_action_diagnostic/vln-trained-action-diagnostic-val630-20260924/action_trace.jsonl)
and [`manifest.json`](../../../artifacts/a800_gpu_20260924/vln_trained_action_diagnostic/vln-trained-action-diagnostic-val630-20260924/manifest.json)
document that diagnostic. The original 64x64 video is consequently blurry
and almost static; upscaling does not add scene detail. Both final VLN
train and eval wrappers returned
`trainer_exit_zero`; their targeted Ray process-group cleanup left no
members, private Ray temp directories were removed, and the selected
ports had no listeners. Their postflight GPU 7 UUID checks also matched.

A separate [native 512x512 Habitat demo](../../../artifacts/a800_gpu_20260924/scripted_hd_demo/vln-scripted-hd-0310q0n3/watch_hd_demo.html)
uses 24 fixed scripted actions in the same validation scene: 0.50 m actual
travel, zero collisions, and 25 distinct source frames. Its
[`manifest.json`](../../../artifacts/a800_gpu_20260924/scripted_hd_demo/vln-scripted-hd-0310q0n3/manifest.json)
records the source resolution and video digest. No policy was loaded for
that demo; it verifies offline rendering quality and visible simulation
motion, not learned navigation.

A fixed untrained greedy baseline,
`runs/vln-baseline-greedy-val630-500-20260924`, used the same
`val_unseen` index 630, episode ID 631, and 500-step cap under private
CPU Mesa. It likewise
had Success/SPL=0, path length=0, distance=5.035 m, and zero collisions.
Its 501-frame video was pulled to Windows, SHA256-matched, and fully
decoded. One-episode equality does not establish learning improvement.
The project-root [A800 GPU acceptance record](../../../artifacts/a800_gpu_20260924/README.md)
links the exact train,
checkpoint, eval JSON, video, and baseline artifacts.

This acceptance used CPU Mesa for Habitat RGB rendering and A800 GPU 7
for Ray/FSDP policy execution and training. It does not verify A800
NVIDIA EGL, multi-environment visual throughput, long training, or
navigation success.

To recheck config composition on A800 without starting training, run as
`navcloud` with the private RLinf Python:

```bash
BASE=/srv/navigation
CUDA_VISIBLE_DEVICES= PYTHONPATH="$BASE/repos/RLinf-v0.3" \
  "$BASE/envs/rlinf-v03-cu118/bin/python" \
  "$BASE/repos/RLinf-v0.3/examples/embodiment/train_embodied_agent.py" \
  --cfg job --resolve --config-name vln_r2r_ppo_smoke
```

Remaining research and scaling gates: repeat over a meaningful train horizon
and multiple disjoint `val_unseen` episodes; compare against fixed untrained
and other appropriate baselines using Success/SPL, collisions, and trajectories.
The trained-policy and baseline videos are engineering evidence only:
neither policy moved toward the goal in the selected episode. The baseline
trace shows 500 right turns and time-limit truncation; the trained RLinf
eval did not retain action IDs. Check GPU
availability before each new run; the shared Pod does not reserve GPU 7.
A800 NVIDIA EGL still requires a separate graphics-context and Habitat
rendering test if GPU-based simulation rendering is needed.
