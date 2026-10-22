# RLinf v0.3 PointNav bridge

> Historical staging and implementation notes. For the current public export,
> start with [the extension README](../../README.md). Machine values below are
> deployment examples; external run evidence is not included. Regenerate frozen
> source hashes after configuring a deployment. The copy maps below describe
> particular integration paths and are not a universal installer.

This integration targets upstream RLinf commit
`0505431899574619da86f551bad70b71e0ea2177` in the isolated A800 checkout
`/srv/navigation/repos/RLinf-v0.3`. The pinned `b85c071`
checkout is untouched. RLinf runs in its own Python 3.10 environment; the
Habitat-Lab 0.2.4 process runs in the existing Python 3.9 `streamvln` environment.
The policy receives only the two-number PointGoal observation, scalar rewards,
and 4 categorical actions. An explicit `diagnostic` request can return pose
and goal for offline trace rendering; normal RLinf reset/step responses contain
no privileged geometry. The default bridge service uses no visual sensors. An
opt-in CPU train service records RGB/depth outside the policy input.

## Exact file mapping

The following files under this local `integrations/rlinf_v03/` directory are
complete patched copies for that one upstream commit. On A800, the same
staging files live under
`/srv/navigation/repos/rlinf-nav/integrations/rlinf_v03/`.

| Staged file | Destination relative to `RLinf-v0.3/` |
| --- | --- |
| `envs_init.py` | `rlinf/envs/__init__.py` |
| `pointnav_init.py` | `rlinf/envs/pointnav/__init__.py` |
| `pointnav_env.py` | `rlinf/envs/pointnav/pointnav_env.py` |
| `mlp_init.py` | `rlinf/models/embodiment/mlp_policy/__init__.py` |
| `mlp_policy.py` | `rlinf/models/embodiment/mlp_policy/mlp_policy.py` |
| `train_embodied_agent.py` | `examples/embodiment/train_embodied_agent.py` |
| `streamvln_p8_successor_protocol.py` | `rlinf/workers/streamvln_p8_successor_protocol.py` |
| `streamvln_p8_successor_workers.py` | `rlinf/workers/streamvln_p8_successor_workers.py` |
| `streamvln_p8_successor_actor.py` | `rlinf/workers/streamvln_p8_successor_actor.py` |
| `streamvln_p8_successor_policy.py` | `rlinf/workers/streamvln_p8_successor_policy.py` |
| `streamvln_chunk_ppo.py` | `rlinf/workers/streamvln_chunk_ppo.py` |
| `streamvln_executor_contract.py` | `rlinf/workers/streamvln_executor_contract.py` |
| `streamvln_p8_successor_action.py` | `rlinf/workers/streamvln_p8_successor_action.py` |
| `streamvln_p8_successor_executor.py` | `rlinf/workers/streamvln_p8_successor_executor.py` |
| `streamvln_p8_successor_policy.yaml` | `examples/embodiment/config/model/streamvln_p8_successor_policy.yaml` |
| `streamvln_p8_successor_bc_smoke.yaml` | `examples/embodiment/config/streamvln_p8_successor_bc_smoke.yaml` |
| `pointnav_env.yaml` | `examples/embodiment/config/env/pointnav_env.yaml` |
| `pointnav_ppo_smoke.yaml` | `examples/embodiment/config/pointnav_ppo_smoke.yaml` |
| `pointnav_ppo_eval_64.yaml` | `examples/embodiment/config/pointnav_ppo_eval_64.yaml` |
| `nvidia_gpu.py` | `rlinf/scheduler/hardware/accelerators/nvidia_gpu.py` |
| `cluster.py` | `rlinf/scheduler/cluster/cluster.py` |
| `test_gpu_allowlist.py` | `tests/unit_tests/test_gpu_allowlist.py` |

The service itself is `scripts/pointnav_service.py` in `rlinf-nav/`, with
transition tests in `tests/test_pointnav_service.py`. The `pyproject.toml` and
`uv.lock` changes for Torch 2.6/cu118 are maintained separately by the runtime
workstream and are outside this bridge patch.

To reapply from the A800 staging directory, first confirm the upstream commit:

```bash
BASE=/srv/navigation/repos
test "$(git -C "$BASE/RLinf-v0.3" rev-parse HEAD)" = 0505431899574619da86f551bad70b71e0ea2177
SRC="$BASE/rlinf-nav/integrations/rlinf_v03"
DST="$BASE/RLinf-v0.3"
mkdir -p "$DST/rlinf/envs/pointnav"
cp "$SRC/envs_init.py" "$DST/rlinf/envs/__init__.py"
cp "$SRC/pointnav_init.py" "$DST/rlinf/envs/pointnav/__init__.py"
cp "$SRC/pointnav_env.py" "$DST/rlinf/envs/pointnav/pointnav_env.py"
cp "$SRC/mlp_init.py" "$DST/rlinf/models/embodiment/mlp_policy/__init__.py"
cp "$SRC/mlp_policy.py" "$DST/rlinf/models/embodiment/mlp_policy/mlp_policy.py"
cp "$SRC/train_embodied_agent.py" "$DST/examples/embodiment/train_embodied_agent.py"
cp "$SRC/streamvln_p8_successor_protocol.py" "$DST/rlinf/workers/streamvln_p8_successor_protocol.py"
cp "$SRC/streamvln_p8_successor_workers.py" "$DST/rlinf/workers/streamvln_p8_successor_workers.py"
cp "$SRC/streamvln_p8_successor_actor.py" "$DST/rlinf/workers/streamvln_p8_successor_actor.py"
cp "$SRC/streamvln_p8_successor_policy.py" "$DST/rlinf/workers/streamvln_p8_successor_policy.py"
cp "$SRC/streamvln_chunk_ppo.py" "$DST/rlinf/workers/streamvln_chunk_ppo.py"
cp "$SRC/streamvln_executor_contract.py" "$DST/rlinf/workers/streamvln_executor_contract.py"
cp "$SRC/streamvln_p8_successor_action.py" "$DST/rlinf/workers/streamvln_p8_successor_action.py"
cp "$SRC/streamvln_p8_successor_executor.py" "$DST/rlinf/workers/streamvln_p8_successor_executor.py"
cp "$SRC/streamvln_p8_successor_policy.yaml" "$DST/examples/embodiment/config/model/streamvln_p8_successor_policy.yaml"
cp "$SRC/streamvln_p8_successor_bc_smoke.yaml" "$DST/examples/embodiment/config/streamvln_p8_successor_bc_smoke.yaml"
cp "$SRC/pointnav_env.yaml" "$DST/examples/embodiment/config/env/pointnav_env.yaml"
cp "$SRC/pointnav_ppo_smoke.yaml" "$DST/examples/embodiment/config/pointnav_ppo_smoke.yaml"
cp "$SRC/pointnav_ppo_eval_64.yaml" "$DST/examples/embodiment/config/pointnav_ppo_eval_64.yaml"
cp "$SRC/nvidia_gpu.py" "$DST/rlinf/scheduler/hardware/accelerators/nvidia_gpu.py"
cp "$SRC/cluster.py" "$DST/rlinf/scheduler/cluster/cluster.py"
cp "$SRC/test_gpu_allowlist.py" "$DST/tests/unit_tests/test_gpu_allowlist.py"
git -C "$DST" status --short
git -C "$DST" diff -- rlinf/envs/__init__.py rlinf/models/embodiment/mlp_policy examples/embodiment/train_embodied_agent.py rlinf/scheduler/hardware/accelerators/nvidia_gpu.py rlinf/scheduler/cluster/cluster.py
```

Run the copy commands as `navcloud`; all isolated checkout and staging files
should remain owned by `navcloud:ma-group`.

## Explicit A800 GPU mapping

An earlier one-step PPO probe finished and wrote a checkpoint, but it did not
establish GPU 7 isolation. The Ray head was started with
`CUDA_VISIBLE_DEVICES=7 --num-gpus=1`, yet RLinf discovered all eight host
GPUs and generated `CUDA_VISIBLE_DEVICES=0` for actor/rollout/env workers.
A concurrent `nvidia-smi` snapshot showed a new process on GPU 0; its host PID
could not be tied to a container PID. Treat that probe as a mechanical PPO
check only.

The isolated v0.3 patch adds `RLINF_ALLOWED_PHYSICAL_GPUS`. Set the same value
on the Ray head and RLinf driver, for example `7` once GPU 7 is assigned and
idle. RLinf then exposes one *logical* GPU to its placement system and maps
logical rank `0` to CUDA ordinal `7` in each worker's
`CUDA_VISIBLE_DEVICES`. It rejects malformed lists, out-of-range logical
ranks, node/driver disagreement, and worker environment overrides. Internally
the worker keeps logical rank `0` for Cluster indexing. The equivalent
`MUJOCO_EGL_DEVICE_ID=7` is generated, but EGL device ordering still needs an
actual graphics-container validation before visual Habitat work.
The current PointNav Ray bridge starts Habitat with no visual sensors and
`create_renderer=False`. A physically assigned single GPU with `compute,utility`
is sufficient for its CUDA actor/rollout smoke; NVIDIA EGL is a separate gate
only for GPU RGB/depth. On A800-SXM4-80GB, a `graphics` container setting alone
does not establish that a usable NVIDIA EGL context or Habitat frame exists.

This variable identifies a CUDA ordinal in the container. Before another
PPO run, compare that ordinal with the intended GPU's UUID/PCI bus ID from
inside the same container, and make a non-training Ray worker probe of
`CUDA_VISIBLE_DEVICES`, `torch.cuda.device_count()` and the selected GPU UUID.
The CPU checks below prove mapping and fail-closed logic only; they do not
prove which physical board a future process will open. Do not rely on
`CUDA_VISIBLE_DEVICES=7` at the Ray head alone for RLinf worker isolation.
An isolated CPU-only Ray probe with `--num-gpus=0` confirmed that the real
`_RemoteNodeProbe` received the job-level allow-list `7` and enumerated one
RLinf accelerator. A `num_gpus=0` worker actor received
`VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=7` and reported logical device
`[0]`. The saved result is
`artifacts/integrations/rlinf_v03_runtime/ray_allowlist_probe_result.json` in
the parent project. This probe did not initialize CUDA or verify GPU UUID.
Start training or evaluation only in a genuinely single-GPU container, or
after a non-training Ray worker probe confirms its CUDA device UUID matches
the assigned idle GPU. Keep the allow-list identical on the Ray head and
RLinf driver for either operation.

## Behavior and limits

- `0=STOP`, `1=MOVE_FORWARD`, `2=TURN_LEFT`, `3=TURN_RIGHT`. The actor outputs
  integer action tensors `[batch, 1, 1]` with Categorical log probabilities.
- PointGoal is the official polar distance/heading vector `[batch, 2]`.
  Reward is geodesic progress plus `10` on actual Habitat STOP success,
  minus `0.01` per step and `0.2` for a collision. A premature STOP terminates
  without success; 64-step expiration truncates. The service has an optional
  `--premature-stop-penalty` (default `0`) for controlled training diagnostics;
  it applies only to unsuccessful STOP, preserving the reference reward at
  default settings. Success uses Habitat's
  strict distance `< 0.2 m` and SPL measure.
- After early STOP the service freezes the terminal observation and emits
  reward zero and no repeated done. It repeats only the terminal summary so
  RLinf can log it at the fixed rollout's last step; its loss mask ignores
  actions after the first done.
- Habitat-Lab 0.2.4 segfaulted during a scene switch in the 100-episode val
  audit. Every simulator process enforces exactly one scene. The 64-step eval
  config cycles official val indices `50..54`, all in `van-gogh-room`.
- `pointnav_ppo_smoke` has eight steps for a cheap mechanical PPO check. The
  corrected config explicitly uses train indices `50,51` in one scene and val
  index `50` in another. The service ready/reset handshake rejects a different
  split, data file or episode selection. The eight-step budget cannot establish
  navigation success or SPL. `pointnav_ppo_eval_64` runs five 64-step episodes.
  Supply both `runner.resume_dir=.../global_step_N` and
  `runner.ckpt_path=.../global_step_N/actor/model_state_dict/full_weights.pt`
  from the same checkpoint. The evaluation entry checks that these paths match
  before Ray starts. Run the CPU content gate with
  `node_preflight.py --compose-check --eval-resume-dir` before using the emitted
  `eval_argv`: it strictly loads the PointNav MLP, compares actor DCP model
  tensors exactly with the full weights, and rejects nonfinite tensors. RLinf
  v0.3 uses `resume_dir` to restore actor state and
  loads the complete model weights from `ckpt_path` in both actor and rollout
  workers. `only_eval` has no rollout weight syncer, so this entry does not
  call `update_rollout_weights()`.

The earlier one-step Ray/FSDP probe used the **old** bridge: RLinf's train
phase actually loaded Habitat val episode 0 and its eval phase loaded val
episode 1. Its checkpoint is a protocol fixture, not a train-split result;
the later val 0–4 aggregate includes an episode used in that update. The
current split fix and A800 CPU-only verification are recorded in
`artifacts/integrations/rlinf_v03_runtime/RAY_SPLIT_FIX_20260924.md` in the
parent project. The corrected Ray/FSDP path still needs a new run on a GPU
whose availability, memory headroom and actual training-worker UUID have been
checked. A shared Pod can host a bounded smoke; a stable allocation is
preferable for longer runs.

## Minimal checks and launch

From local Windows project root:

```powershell
python -m unittest discover -s tests -v
```

From A800 as `navcloud`:

```bash
BASE=/srv/navigation
RL="$BASE/envs/rlinf-v03-cu118/bin/python"
env -u PYTHONPATH "$RL" -B "$BASE/repos/rlinf-nav/integrations/rlinf_v03/bridge_smoke.py"
env -u PYTHONPATH "$RL" -B "$BASE/repos/rlinf-nav/integrations/rlinf_v03/policy_smoke.py"
env -u PYTHONPATH "$RL" -B "$BASE/repos/rlinf-nav/integrations/rlinf_v03/config_smoke.py"
cd "$BASE/repos/RLinf-v0.3"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= "$RL" -m pytest -q tests/unit_tests/test_gpu_allowlist.py
```

The allow-list unit suite passed all 12 cases in the isolated RLinf environment;
Ruff, Python compilation and `git diff --check` also passed.

`config_validate_smoke.py` also passed for the smoke and 64-step eval configs.
It calls RLinf `validate_cfg`, which starts a local Ray instance; run it only
when a cluster-level check is appropriate. In this check Ray's dashboard
reported a `MetricsHead` startup error, although validation exited 0. A
one-step PPO functional probe subsequently reached Global Step 1/1 and saved
a checkpoint, but its Habitat train phase used val episode 0 under the old
service default. The physical GPU placement was not verified. A future PPO run
needs a verified usable GPU window and the explicit allow-list on both Ray head
and trainer; the actual training-worker UUID must be checked.

After a training checkpoint exists, evaluate five episodes:

```bash
: "${TARGET_CUDA_ORDINAL:?set to the CUDA ordinal verified by the worker UUID probe}"
: "${CHECKPOINT_DIR:?set to a checkpoint from a new train-split Ray run}"
WEIGHTS="$CHECKPOINT_DIR/actor/model_state_dict/full_weights.pt"
env -u PYTHONPATH RLINF_ALLOWED_PHYSICAL_GPUS="$TARGET_CUDA_ORDINAL" \
  CUDA_VISIBLE_DEVICES="$TARGET_CUDA_ORDINAL" \
  "$RL" "$BASE/repos/RLinf-v0.3/examples/embodiment/train_embodied_agent.py" \
  --config-name pointnav_ppo_eval_64 \
  runner.resume_dir="$CHECKPOINT_DIR" runner.ckpt_path="$WEIGHTS"
```

Set the same `RLINF_ALLOWED_PHYSICAL_GPUS` on the Ray head before this command.
`CHECKPOINT_DIR` must be a `global_step_N` directory containing nonempty
`actor/model_state_dict/full_weights.pt` and actor DCP metadata/shards. Run
`node_preflight.py --compose-check --eval-resume-dir "$CHECKPOINT_DIR"`
first; it strictly loads the complete PointNav model on CPU, converts actor
DCP in a cleaned temporary directory, checks exact tensor equality and finite
values, then emits both matching Hydra overrides in `eval_argv` with
`eval_checkpoint_content_verified=true`. Missing Torch/RLinf or mismatched
checkpoint content stops preflight before an evaluation command is emitted.
This check does not start Ray or use CUDA.
In `only_eval`, the bridge now keeps the worker's per-episode tensors before
RLinf reduces them to means. It requires exactly one result for each configured
official val file index and writes
`runner.logger.log_path/experiment_name/eval_episodes_step_<N>_<id>.json`
before logging the usual Success/SPL summary. The JSON contains each file
index, its official episode ID, scene, metrics, checkpoint path, and the
aggregate. The bridge checks the service's reset and step episode identity;
this output still requires a live Ray evaluation to establish GPU execution.
`bridge_smoke.py` has verified real Habitat train 50/51 and val 50–54
reset/step/STOP/frozen transition shapes on A800 with CUDA hidden; the val
episode IDs reported by Habitat are `0,1,2,3,4` within van-gogh-room, while
their **file indices** are 50–54. `policy_smoke.py` has verified finite
Categorical policy gradients. These checks do not establish PPO convergence.

## Single-process CPU on-policy PPO

`train_pointnav_ppo_cpu.py` is a bounded reference trainer for the official
Habitat **train** split. It defaults to no rendering and can opt into private
CPU RGB/depth rendering with `--render-rgbd`. It samples real transitions
through the stdio service, uses RLinf v0.3 `MLPPolicy` and
`compute_ppo_actor_critic_loss`, computes GAE with explicit STOP, timeout and
rollout-horizon boundaries, performs Adam updates, and strictly reloads
checkpoint state dictionaries. It uses neither Ray/FSDP nor CUDA and must not
be reported as a distributed RLinf GPU run. Policy actions see only two-float
PointGoal; pose and goal coordinates stay outside the observation.

The official test assets contain duplicate train/val entries: val indices
`0..49` match train `0..49` by complete episode content and start/goal
geometry. The trainer rejects any selected train episode matching any val
episode. Its tested train selections `50,51`, `50..149` and `5000..5099`
pass that gate. Val `50..99` are disjoint from the entire train file, but are
in the same scene as train `5000..5099`; this is held-out-episode evaluation,
not unseen-scene generalization. The audit is
`artifacts/pointnav_train_split_20260924/split_overlap_audit.json` in the
parent project.

Minimal CPU run from A800, using a **new** output directory:

```bash
BASE=/srv/navigation
runuser -u navcloud -- env -u PYTHONPATH -u RAY_ADDRESS \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$BASE/envs/rlinf-v03-cu118/bin/python" -B \
  "$BASE/repos/rlinf-nav/integrations/rlinf_v03/train_pointnav_ppo_cpu.py" \
  --habitat-python /opt/conda/envs/streamvln/bin/python \
  --output-dir "$BASE/runs/pointnav-ppo-v03/cpu-train50-51-new" \
  --episode-indices 50,51 --updates 1 --rollout-steps 32 \
  --max-episode-steps 64 --ppo-epochs 2 --minibatch-size 16 \
  --checkpoint-every 1 --max-total-steps 32 --threads 1 --seed 1234
```

The completed smoke has 32 real steps, one PPO update, nine completed train
episodes, zero successes, and strict-loadable initial/final weights. The
separate 8,192-step and short-distance 2,048-step runs also had zero train
successes and converged toward premature STOP. A controlled short-distance
rerun with `--premature-stop-penalty 1.0` used the same seed and other
hyperparameters; `--premature-stop-penalty 0.0` reproduced the original
trace and final weights byte-for-byte after the code change. The extra STOP
cost did not yield a held-out success. See the parent project's
`artifacts/integrations/rlinf_v03_runtime/CPU_PPO_RESULTS.md` for data,
paired baselines and hashes. Do not select a checkpoint by training reward
alone.

### Opt-in real RGB/depth training rollout

`--render-rgbd` selects `scripts/habitat_rgbd_train_service.py` in the private
Habitat-Sim/Lab 0.2.4 environment. The service renders real RGB and metric
depth during **each PPO train-split reset and step**, sends the same PointGoal,
reward and terminal/metric fields to the trainer, and saves images, pose and
depth only in offline artifacts. The trainer checks every visual service
transition against its own `train_trace.jsonl` and requires a complete frame
sequence for this bounded run. It clears CUDA, runs Mesa llvmpipe on CPU, and
does not start Ray/FSDP. The policy never reads the rendered pixels or pose.

The completed A800 run is
`$BASE/runs/pointnav-ppo-v03/cpu-train50-51-rgbd-ppo1x32-20260924`. It used
official train indices `50,51`, seed `1234`, 32 steps, one PPO update cycle,
nine resets/completed episodes and zero train successes. Its 41 visual events
contain nine resets and 32 steps, each with a contiguous recorded frame.
The 41-frame, 512×256 H.264 RGB/depth video decoded fully. The run's PPO trace
and both `full_weights.pt` and `training.pt` at updates 0 and 1 have the same
SHA256 as the no-render 32-step run; in particular, the post-update weights
hash to `ee6ba99f22488d147b395d47f3e5e0b44c270c607b57fc6b0a9f32300f6f2aa0`.
The local evidence is
[`../../../artifacts/integrations/rlinf_v03_runtime/cpu-train50-51-rgbd-ppo1x32-20260924/`](../../../artifacts/integrations/rlinf_v03_runtime/cpu-train50-51-rgbd-ppo1x32-20260924/),
including `visual_rollout/manifest.json`, `visual_rollout/trace.jsonl`,
`visual_rollout/rgb_depth.mp4`, metric-depth NumPy frames, and checkpoints.

Repeat with a **new** output directory; the wrapper pins the exact two Python
environments, train IDs, seed, 32-step budget and 64-frame recording limit:

```bash
BASE=/srv/navigation
OUT="$BASE/runs/pointnav-ppo-v03/cpu-train50-51-rgbd-ppo1x32-repeat"
runuser -u navcloud -- bash \
  "$BASE/repos/rlinf-nav/scripts/run_a800_visual_cpu_ppo_smoke.sh" "$OUT"
```

The wrapper must execute as `navcloud`; when already logged in as that user,
invoke it with `bash` directly. An existing `OUT` is rejected. This is a
rendered rollout feeding a real RLinf v0.3 PPO update, but its policy remains
PointGoal-only and its zero-success outcome does not show learned navigation.

For a custom checkpoint, `eval_checkpoint_cpu.py` requires
`--checkpoint` and its `--expected-checkpoint-sha256`. A custom val range
also requires matching `--episode-start`, `--episodes`, `--max-steps`,
`--baseline-trace`, `--baseline-summary`, and `--output`; it verifies
the baseline trace hash and every initial episode state. Existing defaults
remain val `0..4` with a 64-action cap for the original step-1 fixture.

## CPU checkpoint rollout diagnostic

`eval_checkpoint_cpu.py` in this staging directory defaults to the exact
`global_step_1/actor/model_state_dict/full_weights.pt` checkpoint (SHA256
`80c1e8ff3296fdf6f6b4dd32a47efc4a85301152429f1ce475437f33a88cef7c`).
It loads the four-action MLP on CPU, calls
`predict_action_batch(mode="eval")` for deterministic argmax or
`predict_action_batch(mode="train")` for fixed-seed categorical sampling,
and sends each action to the existing Python 3.9 PointNav stdio service.
`CUDA_VISIBLE_DEVICES` is cleared before Torch import; no Ray or renderer is
started. The JSONL records every true service reward and metric response.

Reproduce on A800 as `navcloud`, from the pinned RLinf checkout:

```bash
BASE=/srv/navigation
cd "$BASE/repos/RLinf-v0.3"
RL="$BASE/envs/rlinf-v03-cu118/bin/python"
SCRIPT="$BASE/repos/rlinf-nav/integrations/rlinf_v03/eval_checkpoint_cpu.py"
env -u PYTHONPATH -u RAY_ADDRESS CUDA_VISIBLE_DEVICES= "$RL" -B "$SCRIPT" --mode argmax
env -u PYTHONPATH -u RAY_ADDRESS CUDA_VISIBLE_DEVICES= "$RL" -B "$SCRIPT" --mode sampled --torch-seed 7
```

Both runs used official val episode indices 0–4 and at most 64 actions per
episode; all five initial PointGoal vectors, geodesic distances, IDs and
scene IDs matched the existing random-seed-7 baseline. The checkpoint source
version is RLinf commit `0505431899574619da86f551bad70b71e0ea2177`
plus the `policy_source_sha256` and `policy_config` saved in each summary.
This old checkpoint was updated using Habitat val episode 0, so the aggregate
over val 0–4 includes a training episode and is only a protocol diagnostic.

| Diagnostic | Actions | STOP episodes | Success | Mean SPL | Mean final geodesic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checkpoint argmax | 8 | 5 | 0/5 | 0 | 9.464 m |
| Checkpoint sampled, Torch seed 7 | 25 | 5 | 0/5 | 0 | 9.606 m |
| Existing random seed 7 | 164 | 3 | 0/5 | 0 | 10.221 m |

The argmax policy called STOP immediately in episodes 0–3 and after four
actions in episode 4. The sampled rollout used 3–7 actions per episode and
also stopped prematurely. This is a one-update checkpoint diagnostic on only
five episodes; the distance numbers do not establish policy quality or a
benefit over random. Neither run validates RGB rendering or video.

Remote traces and summaries are under
`$BASE/runs/pointnav-ppo-v03/cpu_eval5_64/`. Identical copies with verified
SHA256 are in the local parent project's
`artifacts/integrations/rlinf_v03_runtime/cpu_eval5_64/` directory. The
summaries contain the trace SHA256, checkpoint SHA256, policy source SHA256,
per-episode outcomes, and the paired random baseline reference.

### Offline checkpoint pose video

The service now accepts a separate `{"cmd":"diagnostic"}` request after
reset/step. It returns current agent position, xyzw orientation, and goal
position. `eval_checkpoint_cpu.py --record-poses` writes those fields only into
the offline JSONL trace. `choose_action` still receives just PointGoal, and
normal RLinf reset/step responses remain unchanged. With `--node-root`, the
evaluator derives its checkout, checkpoint, baseline, service, data, and output
paths from that root; pass `--habitat-python` explicitly on a second node.

On A800, both argmax and seed-7 sampled evaluations were rerun on CPU with
`--record-poses` and separate `*_with_poses.jsonl` output paths. After removing
the diagnostic field, their 13 and 30 JSONL rows respectively are identical
to the earlier evaluations. The summaries report 8 and 25 actions, five
premature STOPs each, and zero success. The new remote traces/summaries were
copied to the local `cpu_eval5_64/` artifact directory with matching SHA256.

The sampled run used this CPU-only command; substitute `--mode argmax` and a
different `--output` for the deterministic run:

```bash
BASE=/srv/navigation
runuser -u navcloud -- env -u PYTHONPATH -u RAY_ADDRESS CUDA_VISIBLE_DEVICES= \
  "$BASE/envs/rlinf-v03-cu118/bin/python" -B \
  "$BASE/repos/rlinf-nav/integrations/rlinf_v03/eval_checkpoint_cpu.py" \
  --mode sampled --torch-seed 7 --record-poses \
  --output "$BASE/runs/pointnav-ppo-v03/cpu_eval5_64/actor_step1_sampled_seed7_with_poses.jsonl"
```

Render a local top-down video from a copied trace:

```powershell
python .\rlinf-nav\scripts\render_pointnav_trace.py `
  --input .\artifacts\integrations\rlinf_v03_runtime\cpu_eval5_64\actor_step1_sampled_seed7_with_poses.jsonl `
  --episode-index 4 `
  --output .\artifacts\integrations\rlinf_v03_runtime\cpu_eval5_64\actor_step1_sampled_seed7_ep4_pose_trace.mp4 `
  --png .\artifacts\integrations\rlinf_v03_runtime\cpu_eval5_64\actor_step1_sampled_seed7_ep4_pose_trace.png
```

The sampled episode 4 travels 0.50 m before a premature STOP. Its local H.264
video is 880×520, 19 frames, 3.167 s, SHA256
`f6699330a56a94ca5ab6961b36388aeda526e2f4c01452f3b5edb3fb1cdf4c92`.
The matching argmax episode 4 video has 17 frames and SHA256
`1a4f6d78fb3feb2ff91fdc5732ea20000241c34e5c6799f2e369d8ade8861cd2`.
`ffprobe` identified both as H.264 at 880×520, and `ffmpeg` decoded both fully
without errors. The drawing contains only logged agent pose, goal and metrics;
it has no scene geometry, obstacle map, or Habitat RGB/depth camera image.

## Checkpoint-driven Habitat RGB-D episode

`eval_checkpoint_rgbd_cpu.py` defaults to the same SHA-pinned step-1 checkpoint in
the private RLinf Python 3.10 environment and launches
`scripts/habitat_rgbd_policy_service.py` in the private Habitat Python 3.9
environment. The latter renders the official val episode 4 with CPU EGL/Mesa
llvmpipe. The policy function receives only the live two-float PointGoal;
RGB/depth and privileged pose are written by the simulator to offline files.
It does not initialize CUDA or Ray. The client checks the first 19 seeded
samples of earlier official episodes against their saved PointGoal trace to
continue the exact Torch sampling stream, then selects each episode 4 action
from the new visual simulator response. An independent verifier checks all
25 actions, logits, and selected log probabilities.

The result is 6 steps and 7 real camera/depth frames: right, left, right,
forward, forward, STOP; 0.500016 m traveled; Success 0/SPL 0. Initial state,
all actions, PointGoal transitions, reward, and per-step metrics match the
earlier no-renderer episode 4. The video was fully decoded after copying to
Windows; the nine result-file hashes match A800. This is a rollout/video
pipeline proof for an almost untrained policy, not evidence of navigation
quality or RGB-conditioned control. The complete log, hashes, manifest,
metric-depth array, MP4, scripts and exact commands are in
`artifacts/integrations/rlinf_v03_runtime/checkpoint_rgbd_ep4/README.md`.

Run again into an unused directory:

```bash
BASE=/srv/navigation
runuser -u navcloud -- env -u PYTHONPATH -u RAY_ADDRESS CUDA_VISIBLE_DEVICES= \
  "$BASE/envs/rlinf-v03-cu118/bin/python" -B \
  "$BASE/repos/rlinf-nav/integrations/rlinf_v03/eval_checkpoint_rgbd_cpu.py" \
  --output-dir "$BASE/runs/pointnav-ppo-v03/checkpoint-rgbd-ep4-sampled-seed7-repeat"
```

### Train-split CPU PPO checkpoint to real RGB-D video

The same visual path was also run with the **post-update** checkpoint from the
single-process CPU PPO smoke above: official train episodes `50,51`, 32 real
steps, one RLinf v0.3 PPO loss/Adam update, checkpoint
`cpu-train50-51-ppo1x32-20260924/checkpoints/update_0001/full_weights.pt`
(SHA256 `ee6ba99f22488d147b395d47f3e5e0b44c270c607b57fc6b0a9f32300f6f2aa0`).
Its prior no-render sampled evaluation of official val episodes `0..4` uses
Torch seed 7. The evaluator verifies the prior trace hash, checkpoint SHA and
seed, advances the first 19 sampled actions, then drives val episode 4 live
through the private Habitat RGB/depth service. It fails if the initial state,
six selected actions, PointGoal transitions, or Habitat rewards/metrics differ
from that prior evaluation. `verify_checkpoint_rgbd_alignment.py` separately
replays all 25 actions and compares logits and selected log probabilities to
within `1e-7`.

The strict A800 run is at
`$BASE/runs/pointnav-ppo-v03/cpu-train50-51-ppo1x32-update1-rgbd-ep4-strict-20260924`;
the copied evidence is under the parent project's
`artifacts/integrations/rlinf_v03_runtime/cpu_train_update1_rgbd_ep4_strict_20260924/`.
It recorded six actions and seven real RGB/depth frames. The policy stopped
early: Success 0 and SPL 0. This demonstrates that a checkpoint produced by
the bounded train-split CPU PPO run can drive a rendered val episode and
produce an offline video. The policy still consumes only PointGoal, not image
pixels or depth; this run proves neither visual learning nor navigation
quality. CUDA and Ray were disabled, so it is not a GPU PPO run.

To reproduce in a **new** output directory, use the matching no-render
evaluation as the sampling reference:

```bash
BASE=/srv/navigation
TRAIN="$BASE/runs/pointnav-ppo-v03/cpu-train50-51-ppo1x32-20260924"
OUT="$BASE/runs/pointnav-ppo-v03/cpu-train50-51-ppo1x32-update1-rgbd-ep4-repeat"
SCRIPT="$BASE/repos/rlinf-nav/integrations/rlinf_v03"
CKPT="$TRAIN/checkpoints/update_0001/full_weights.pt"
SHA=ee6ba99f22488d147b395d47f3e5e0b44c270c607b57fc6b0a9f32300f6f2aa0
runuser -u navcloud -- env -u PYTHONPATH -u RAY_ADDRESS \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$BASE/envs/rlinf-v03-cu118/bin/python" -B "$SCRIPT/eval_checkpoint_rgbd_cpu.py" \
  --checkpoint "$CKPT" --expected-checkpoint-sha256 "$SHA" --torch-seed 7 \
  --previous-trace "$TRAIN/eval_val0-4_update1_seed7.jsonl" \
  --previous-summary "$TRAIN/eval_val0-4_update1_seed7.summary.json" \
  --output-dir "$OUT"
runuser -u navcloud -- env -u PYTHONPATH -u RAY_ADDRESS \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$BASE/envs/rlinf-v03-cu118/bin/python" -B "$SCRIPT/verify_checkpoint_rgbd_alignment.py" \
  --checkpoint "$CKPT" --expected-checkpoint-sha256 "$SHA" --torch-seed 7 \
  --prior-trace "$TRAIN/eval_val0-4_update1_seed7.jsonl" --output-dir "$OUT"
```

### Visually recorded holdout episode 50

The evaluator also accepts `--episode-index` inside the selected no-render
reference range. A800 used the same genuine train-50/51 CPU PPO checkpoint and
the sampled val-50–54 trace to drive official **val 50** live through private
Mesa RGB/depth. Val 50 is disjoint from the entire official train file under
the recorded episode audit, though it shares a scene with training episodes.
The no-render val 50 trace begins the seed-7 sampling stream, so zero earlier
actions need replay. Reset, six actions, PointGoal, rewards and metrics matched
the no-render trace; the six-step run ended in premature STOP, Success 0/SPL 0.
Seven RGB/depth frames were encoded and transferred. Evidence and SHA256 values
are in the parent project's
`artifacts/integrations/rlinf_v03_runtime/cpu-train50-51-rgbd-val50-holdout-20260924/README.md`.
This establishes a visually recorded held-out **episode** evaluation path, not
navigation success, visual-input learning or unseen-scene generalization.

```bash
BASE=/srv/navigation
TRAIN="$BASE/runs/pointnav-ppo-v03/cpu-train50-51-ppo1x32-20260924"
REF="$BASE/runs/pointnav-ppo-v03/cpu-train50-51-val50-54-holdout-20260924"
OUT="$BASE/runs/pointnav-ppo-v03/cpu-train50-51-rgbd-val50-holdout-repeat"
SHA=ee6ba99f22488d147b395d47f3e5e0b44c270c607b57fc6b0a9f32300f6f2aa0
runuser -u navcloud -- env -u PYTHONPATH -u RAY_ADDRESS CUDA_VISIBLE_DEVICES= \
  "$BASE/envs/rlinf-v03-cu118/bin/python" -B \
  "$BASE/repos/rlinf-nav/integrations/rlinf_v03/eval_checkpoint_rgbd_cpu.py" \
  --checkpoint "$TRAIN/checkpoints/update_0001/full_weights.pt" \
  --expected-checkpoint-sha256 "$SHA" --torch-seed 7 --episode-index 50 \
  --previous-trace "$REF/eval_val50-54_update1_seed7.jsonl" \
  --previous-summary "$REF/eval_val50-54_update1_seed7.summary.json" \
  --output-dir "$OUT"
```

## Second-node path preflight

The staged Hydra YAML retains its verified A800 defaults. On a 4090 Linux/WSL
node, run `node_preflight.py` with that node's **actual** workspace and Python
paths before attempting RLinf. It is a read-only check: it confirms the pinned
RLinf commit, SHA equality of all 12 staged/installed bridge files, official
small test assets, service, and two Python executables. With `--compose-check`,
it prints a complete smoke argv with explicit Hydra overrides for both train
and eval environments, plus run and video directories. It emits `eval_argv`
only when `--eval-resume-dir` points inside the isolated node root to a
`global_step_N` directory containing nonempty
`actor/model_state_dict/full_weights.pt`, actor DCP metadata and a nonempty
DCP shard. Using the node's RLinf Python on CPU, it also strictly loads the
PointNav MLP from full weights, converts the trusted actor DCP to a temporary
CPU file, and requires identical finite model tensors in both states. The
temporary converted file is removed. Only then does it compose the evaluation
config with matching
`runner.resume_dir` and `runner.ckpt_path` overrides. This matches the pinned
RLinf actor and rollout initialization paths; JSON reports
`eval_checkpoint_content_verified=true`. It never imports Habitat, starts Ray,
opens CUDA, or creates a renderer.

```bash
NODE_ROOT=/path/to/isolated/rlinf-navigation-rl
HABITAT_PY=/path/to/habitat-python
RLINF_PY="$NODE_ROOT/envs/rlinf-v03-cu118/bin/python"
"$RLINF_PY" -B "$NODE_ROOT/repos/rlinf-nav/integrations/rlinf_v03/node_preflight.py" \
  --node-root "$NODE_ROOT" --habitat-python "$HABITAT_PY" \
  --rlinf-python "$RLINF_PY" --compose-check > "$NODE_ROOT/node_preflight.json"
```

If the checkout, data, or environment lives elsewhere, pass the explicit
`--rlinf-checkout`, `--asset-root`, `--service-script`, `--staging-dir`, `--run-root`, or
`--video-root` option. Output directories must remain inside the isolated node
root. After training has produced a checkpoint, rerun with
`--eval-resume-dir /path/to/global_step_N` to obtain `eval_argv`;
without it, `eval_argv` is null because the evaluation entry rejects a missing
checkpoint. A800 passed the content gate first with this local script streamed
through SSH standard input, then with the installed private staged script,
against the old trusted `global_step_1` fixture with CUDA hidden. That checkpoint's train
phase used Habitat val 0, so the check proves checkpoint format and consistency,
not an independent navigation result. No Ray/GPU evaluation was run.
These argument arrays are not proof of runtime readiness.
`--compose-check` resolves both Hydra configs and checks
explicit train/val splits, selected-episode separation, and both scenes in
the RLinf Python environment without starting Ray. Without it, preflight
intentionally omits runnable argv.
With `--eval-resume-dir`, preflight checks checkpoint content on CPU; without
that option it does no checkpoint load. GPU identity/ownership, CUDA, Habitat
RGB/depth and the full worker path each need a live check before launch. Do not
use the A800 YAML defaults unmodified on another node.

## StreamVLN categorical G2 overlay

This overlay keeps the small `vln_policy` route intact and adds the frozen
four-action StreamVLN model behind a process boundary. It targets the same
pinned upstream RLinf commit,
`0505431899574619da86f551bad70b71e0ea2177`.

| Staged file | Destination relative to `RLinf-v0.3/` |
| --- | --- |
| `config_vln.py` | `rlinf/config.py` |
| `models_init_vln.py` | `rlinf/models/__init__.py` |
| `huggingface_worker_vln.py` | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| `embodied_io_struct_vln.py` | `rlinf/data/embodied_io_struct.py` |
| `envs_init_vln.py` | `rlinf/envs/__init__.py` |
| `vln_env.py` | `rlinf/envs/vln/vln_env.py` |
| `streamvln_init.py` | `rlinf/envs/streamvln/__init__.py` |
| `streamvln_env.py` | `rlinf/envs/streamvln/streamvln_env.py` |
| `streamvln_categorical_init.py` | `rlinf/models/embodiment/streamvln_categorical/__init__.py` |
| `streamvln_policy.py` | `rlinf/models/embodiment/streamvln_categorical/streamvln_policy.py` |
| `streamvln_policy_server.py` | `rlinf/models/embodiment/streamvln_categorical/streamvln_policy_server.py` |
| `streamvln_context.py` | `rlinf/models/embodiment/streamvln_categorical/streamvln_context.py` |
| `streamvln_replay.py` | `rlinf/models/embodiment/streamvln_categorical/streamvln_replay.py` |
| `streamvln_policy.yaml` | `examples/embodiment/config/model/streamvln_policy.yaml` |
| `streamvln_env.yaml` | `examples/embodiment/config/env/streamvln_env.yaml` |
| `streamvln_frozen_eval.yaml` | `examples/embodiment/config/streamvln_frozen_eval.yaml` |
| `eval_streamvln.py` | `examples/embodiment/eval_streamvln.py` |

The service also needs
`scripts/habitat_r2r_service.py` and
`scripts/habitat_r2r_rgb_instruction_smoke.py`. The complete staged rollout
worker already depends on the existing `gpu_uuid_gate.py`; copy that file to
`rlinf/utils/gpu_uuid_gate.py` when starting from a clean upstream worktree.

Deployment paths live in explicit CLI arguments or OmegaConf environment
inputs. No Windows checkout path is embedded in the runtime modules.

G2 verification used a detached clean worktree at
`/srv/navigation/repos/RLinf-v0.3-streamvln-g2` and an
independent Git-bundle clone of StreamVLN commit
`85cda3bc4574953cb9eee467c4d3448f8ef228f9`. It ran train episodes 1189 and
1191 from different MP3D scenes as one batch through the exact unwrapped
`MultiStepRolloutWorker.predict` implementation for two decisions. Both
sessions advanced from physical step 0 to 4 independently. Maximum
sample/current-weight replay error was `6.258487701416016e-07`; reversing the
two fixed replay rows before restore and repeating predict after replay both
gave `0.0` error. The configured guard was `0.03`.

This entry is frozen and non-differentiable by design:
`actor_handshake.replay_differentiable=false` and `ppo_started=false`.
It proves model/env registration, observation isolation, action acknowledgement,
four-action physical execution, fixed tensor replay and side-effect-free
restore. It does not prove FSDP/LoRA gradients, a value head, chunk-aware GAE,
weight synchronization or navigation quality. Those belong to G3 and later.

The tracked evidence index is
`artifacts/streamvln_g2_20260926/README.md` in the parent repository.

## P8 one-shot A800 parity

`run_streamvln_p8_a800_parity.py` compares the verified clean-e48 evaluator
methods with the official P8 adapter on deterministic synthetic frames. It
requires explicit `--source-root`, `--import-root`, `--checkpoint`,
`--tokenizer`, `--vision`, `--manifest`, `--device cuda:N`, `--local-rank`, and
`--output` arguments. The formal release anchor, manifest, source, and artifact
bytes are verified before Torch, Transformers, NumPy, PIL, or upstream model
imports. The frozen manifest must reproduce canonical SHA-256
`09a344903944942210a342f5defe8708523ecd74be98ec6be9b896334da471ac`;
any mismatch exits nonzero without replacing an existing receipt. The run never
constructs Habitat or reads a dataset or split.

The accepted one-shot run is
`/srv/navigation/runs/p8_successor_parity_20260929_91db1ca`.
It used clean e48 and the original released artifacts on physical A800 7,
returned status `pass` for preprocessing, initial generation, continuation and
history/reset, and returned process-visible GPU memory to `0 MiB`. Receipt file SHA-256 is
`ef322a0ac58909714d0cbbdfbc4e2b005789cb66d023d5a926dcfb3543b5cc3c`.
This closes synthetic released-checkpoint parity, not Habitat or navigation.

## P8 successor policy-wiring bootstrap

`streamvln_p8_successor_policy.py` is the shared RLinf-synchronized policy used
by rollout sampling and actor recomputation.  Its policy routes are exactly
KEEP_ORIGINAL, WAYPOINT and STOP.  WAYPOINT is a conditional bounded 2D
distribution with its transform Jacobian included in the likelihood;
RECOVER_PARENT is not a sampled route.  Replay carries a fixed tensor-only
versioned context, route candidate mask, sampled joint log-probability and
separate expert targets.  The actor optionally adds train-only route CE plus
WAYPOINT-masked NLL/regression.  Until a BC-ready buffer is loaded or a BC
backward pass marks it ready, rollout is forced to the safe KEEP branch.

The single-slot `streamvln_p8_successor_bc_smoke.yaml` uses the released
official action as an explicit KEEP expert target.  This is a policy wiring
smoke, not evidence that STOP or WAYPOINT behavior has been behavior-cloned.
The release-anchor preflight runs before config validation can create placement
objects, and full factory/model readiness still runs before the explicit
training Cluster.  The trainer module itself imports Torch at module import, so
the current guarantee is before weights, successor model/factory construction,
and Cluster launch rather than before the Python Torch import.

The production worker pair now transports an identity-bound, history-only
causal topology snapshot into the shared policy.  A compiled waypoint executes
one primitive at a time, and only the closed macro's real executed prefix
updates topology once.  Collision-triggered RECOVER_PARENT remains an internal
executor override to a real parent and fails closed at the root; it never enters
the policy likelihood.  Reset rotates the episode token and clears topology,
planner, and pending-prefix state.  The remaining execution constraint is that
the fixed physical-step rollout horizon must land on a macro boundary; a
variable waypoint prefix crossing that boundary is rejected.  These wiring and
state-machine checks do not establish map quality or a successful Torch/CUDA
training run. The A800 parity receipt now passes; live Habitat executor ceiling,
BC readiness and a bounded successor worker/policy CUDA smoke remain gated.

## Original StreamVLN differentiable PPO (G3)

`streamvln_ppo_policy.py` and `run_streamvln_ppo_g3.py` implement the first
bounded differentiable policy update. The source contract pins clean upstream
StreamVLN commit `e48f6ff7e9201d93aae003e8f64b04c00cec13bc` and rejects a memory
adapter or secondary post-training route. The policy uses original causal
action likelihoods, exact language-only LoRA on Qwen layers 24-27 `q_proj` and
`v_proj`, and a small value head. Vision, projector, embeddings, LM head and
the base language weights remain frozen.

The runner is a one-decision correctness harness. It invokes RLinf's
`FSDPModelManager`, executes a real Habitat R2R chunk, uses duration-aware
returns/GAE, audits gradient ownership and parameter changes, writes the LoRA
and value checkpoint, and starts a fresh frozen process to verify exact
checkpoint loading and replay. FSDP-qualified PEFT names are canonicalized
before saving; the guard requires 16 adapter tensors / 360,448 parameters and
zero tensor-load error. See
`artifacts/streamvln_g3_20260926/README.md` in the parent project for the
accepted A800 evidence and hashes.

This harness is not a multi-episode trainer or benchmark. Use its contracts as
the G4 foundation, then measure fixed held-out navigation SR/SPL/NE separately.
## Bounded multi-scene StreamVLN PPO (G4)

`run_streamvln_ppo_g4.py` extends the G3 correctness path to a fixed manifest of
8 train scenes and 8 disjoint `val_unseen` scenes. It supports causal continued
sampling, one state value per decision, duration-aware GAE across two decisions,
one PPO update per train episode, a canonical LoRA/value checkpoint, and two
fresh-process greedy evaluations with matched episode order and step budget.

The accepted bounded run changed all 16 LoRA tensors and both value tensors, and
all source, ratio, gradient, frozen-parameter and adapter-load guards passed.
Frozen and RL then produced identical actions and metrics on the eight fixed
unseen episodes: SR/SPL `0`, mean NE `7.213194 m`. This is an operational
multi-scene RL/evaluation result and a negative navigation result, not evidence
of convergence or improvement. The tracked evidence index is
`artifacts/streamvln_g4_20260926/README.md` in the parent project.
## G5 topological waypoint policy

The G5 overlay adds the following staged runtime files:

| Staged file | Destination relative to RLinf-v0.3 |
| --- | --- |
| streamvln_topology.py | scripts/streamvln_topology.py |
| streamvln_topological_policy.py | scripts/streamvln_topological_policy.py |
| run_streamvln_topology_g5.py | scripts/run_streamvln_topology_g5.py |
| streamvln_g5_experiment.json | scripts/streamvln_g5_experiment.json |

It also extends habitat_r2r_service.py with a train-only expert_waypoint command,
vln_env.py with strict expert-response validation, and streamvln_env.py with a
per-primitive collision trace. Policy observations remain RGB, official
instruction tensors and action history; goal coordinates, geodesic distance
and reference paths are not policy inputs.

The accepted bounded run freezes original StreamVLN and trains a 1,003,592
parameter high-level head for a local waypoint plus NAVIGATE/STOP/RECOVER. It
compares RL-only, BC-only and BC+RL on the same fixed eight val_unseen
episodes. All FSDP, ratio, gradient, checkpoint and data-isolation guards pass,
but all variants have SR/SPL 0. BC-only moves but collides; BC+RL reduces
collisions by turning rather than learning recovery. See the parent project's
STREAMVLN_G5_EXPERIMENT_PLAN.md and
artifacts/streamvln_g5_20260926/README.md. This runner is a bounded research
harness, not a converged navigation policy.

## StreamVLN G6 recovery overlay

G6 adds a bounded collision-recovery layer and stable batched high-level PPO on top of the G5 topology/waypoint head. The original StreamVLN source and release checkpoint remain read-only; only the 1,003,592-parameter high-level head is trainable.

| Staged file | Purpose |
| --- | --- |
| `streamvln_recovery.py` | Online topology recovery state, collision trigger, bounded backtrack compiler, and waypoint execution error |
| `streamvln_recovery_policy.py` | Balanced three-mode BC and per-mode accuracy for NAVIGATE/STOP/RECOVER |
| `run_streamvln_recovery_g6.py` | Train-only oracle diagnostic, balanced warm-start, batched PPO, independent checkpoint evaluation, and guards |
| `streamvln_g6_experiment.json` | Fixed 8-scene train/eval protocol and G6 optimizer/KL thresholds |

The accepted run is `streamvln_g6_recovery_20260927_v3`. All guards passed. On the fixed 8-scene `val_unseen` set, deterministic recovery reduced balanced-BC collisions from 7.25 to 3.50 without improving SR/SPL; batched PPO then changed NE from 8.1185 m to 7.9464 m and collisions from 3.50 to 3.375, again with SR/SPL 0. Treat this as an engineering and behavior result, not a navigation-success result. See the repository root [G6 experiment record](../../../STREAMVLN_G6_EXPERIMENT_PLAN.md).
## StreamVLN G7 train-only upper-bound overlay

| Staged file | Purpose |
| --- | --- |
| streamvln_candidates.py | Goal-free local/topology candidate generation, train-only oracle selector, and phased recovery compiler |
| run_streamvln_candidates_g7.py | Fixed train-only upper-bound comparison and hard G8 gates |
| streamvln_g7_experiment.json | Reused fixed split manifest with a G7 stage identity |

G7 loaded no policy model and ran no val_unseen episodes. Candidate geometry met its coverage/error gates, but candidate SR was 0.5 versus the continuous plain oracle 0.625; candidate plus recovery was 0.375. G8 remains blocked. See the repository root G7 experiment record.
## StreamVLN G7.1 adaptive execution overlay

| Staged file | Purpose |
| --- | --- |
| streamvln_env.py | Optional execute_count prefix while retaining the fixed four-action tensor and padded outputs |
| streamvln_candidates_g71.py | Dense multiscale candidates, adaptive compiler, and short-lived collision direction feasibility |
| run_streamvln_candidates_g71.py | Train-only fixed4/adaptive/blacklist factorial and hard G8 gates |
| streamvln_g71_experiment.json | Reused fixed manifest with a G7.1 stage identity |

All three G7.1 variants reached 6/8 train-only oracle success. The strict G8 aggregate gate remained false; G7.2 will finalize expanded candidate geometry before any learned scorer.
## StreamVLN G7.2 expanded geometry final gate

| Staged file | Purpose |
| --- | --- |
| `run_streamvln_candidates_g72.py` | Expanded fixed4 train-only oracle and hard G8 gates |
| `streamvln_g72_experiment.json` | Fixed split manifest with G7.2 stage identity |

G7.2 loaded no model and ran zero `val_unseen` episodes. Coverage reached 0.949 and p90 quantization error 0.131 m, but SR/SPL fell to 0.625 and collision increased to 3.750. The expanded nominal waypoint bank is not aligned with the fixed-four-primitive endpoint; G8 remains blocked. See the repository root G7.2 experiment record.

## StreamVLN G7.3 executor-aligned candidates

| Staged file | Purpose |
| --- | --- |
| `streamvln_candidates_g73.py` | Enumerate, simulate and deduplicate exact fixed4 action endpoints |
| `run_streamvln_candidates_g73.py` | Fixed train-only executor-aligned oracle and unchanged G8 gates |
| `streamvln_g73_experiment.json` | Reused fixed manifest with G7.3 stage identity |

G7.3 loaded no model and ran zero `val_unseen` episodes. Exact endpoint binding restored SR/SPL to 0.75, but short-horizon coverage and collision interruption remained limiting; aggregate collision was 4.125. G8 remains blocked. See the repository root G7.3 experiment record.

## StreamVLN G7.4 exact macros and collision interruption

| Staged file | Purpose |
| --- | --- |
| `streamvln_candidates_g74.py` | Exact 2–8 primitive phased macro endpoints and world-heading collision memory |
| `run_streamvln_candidates_g74.py` | Short/macro crossed with plain/interrupt execution under one physical budget |
| `streamvln_g74_experiment.json` | Reused fixed manifest with G7.4 stage identity |

G7.4 loaded no model and ran zero `val_unseen` episodes. Macro plus interruption passed coverage, error, SR/SPL, progress and STOP gates; collision was 1.875 versus the 1.375 gate. G8 remains blocked. See the repository root G7.4 experiment record.

## StreamVLN G7.5 angular-sector final safety gate

| File | Responsibility |
| --- | --- |
| `streamvln_candidates_g75.py` | World-heading angular sector shield with fixed two-decision cooldown |
| `run_streamvln_candidates_g75.py` | Paired ±15/±30-degree train-only diagnostic and predeclared selection rule |
| `streamvln_g75_experiment.json` | Fixed G7.5 manifest and source contract |

G7.5 loaded no model, trained nothing and ran zero `val_unseen` episodes. The ±15-degree sector kept geometry quality but missed collision and navigation gates; the ±30-degree sector met collision while failing coverage, p90 and SR/SPL. No variant was selected, G8 remains blocked, and the next stage is observation-conditioned local passability rather than more angular-sector tuning. See the repository root G7.5 experiment record.

## StreamVLN G7.6 observation-conditioned passability

| File | Responsibility |
| --- | --- |
| `scripts/habitat_r2r_passability_service.py` | Train-only simulator branch labels with exact pose restoration |
| `vln_env.py::probe_candidates` | Strict evaluator-only probe schema and provenance validation |
| `streamvln_passability_g76.py` | Geometry-only and RGB+geometry models, fixed threshold grid and metrics |
| `run_streamvln_passability_g76.py` | 6/2 scene split, collection, training and one frozen-threshold navigation gate |
| `streamvln_g76_experiment.json` | Fixed episode set and scene-disjoint passability split |

G7.6 collected 1,650 labeled macro candidates with zero `val_unseen` episodes. Current 64x64 RGB did not outperform geometry-only on train-dev, and the navigation gate regressed to SR/SPL 0.50 without reducing collision. G8 remains blocked. The next stage changes observation timing to causal turn-observe-translate.

## StreamVLN G7.11 explicit geometry controller gate

| File | Responsibility |
| --- | --- |
| `streamvln_geometry_g711.py` | Frozen metric-depth projection, controlled-motion severity and joint controller gates |
| `scripts/habitat_r2r_geometry_controller_service.py` | Exact-restored open-loop, endpoint, causal-depth and privileged-navmesh physical variants |
| `vln_env.py::probe_geometry_controller` | Strict 128-row evaluator-only schema with a bounded G7.11 communication timeout |
| `run_streamvln_geometry_g711.py` | G7.10 anchor reproduction, frozen G7.8 reference, matched metrics and routing |
| `streamvln_g711_experiment.json` | Frozen source/data/camera/footprint/controller/gate contract |

The accepted A800 run completed 128 rows and 1,024 physical repeats with all guards passing. Causal depth retained 85.75% of primitives but reduced severe exposure only 33.33%; privileged navmesh removed severe exposure but retained 45.43% of primitives and completed 72.13% of free/grazing macros. Both joint gates failed. G7.12 will test a severity-aligned privileged oracle contract before any new causal geometry controller. G8 remains blocked.

## StreamVLN G7.12 target-aligned privileged-oracle contract audit

| File | Responsibility |
| --- | --- |
| `streamvln_oracle_g712.py` | Frozen G7.10 prefix-severity recomputation, longest admissible prefix and joint gates |
| `run_streamvln_oracle_g712.py` | Accepted G7.11 anchor validation, exact-prefix reuse, metrics and routing |
| `streamvln_g712_experiment.json` | Frozen input hashes, tolerances, denominators and gate contract |

The accepted audit reused all 128 G7.11 rows and 256 exact recorded open-loop repeats. It required no new physical replay. The severity-aligned oracle reduced severe exposure from 39 to 0, retained 83.33% of primitives, preserved 100% free/grazing completion and progress, and selected nonzero motion for every row. This is a retrospective privileged ceiling, not a causal controller. G7.13 will test fixed pre-action ground-support observability before any controller or navigation work; G8 remains blocked.

## StreamVLN G7.13 causal ground-support observability audit

| File | Responsibility |
| --- | --- |
| `streamvln_ground_support_g713.py` | Frozen pair reconstruction, expected-ground features, AUROC and gate calculations |
| `scripts/habitat_r2r_ground_support_service.py` | Exact-restored pre-critical metric-depth capture without executing the critical primitive |
| `vln_env.py::probe_ground_support` | Strict finite schema validation for the evaluator-only service response |
| `run_streamvln_ground_support_g713.py` | Accepted-anchor validation, repeated capture, feature isolation and result routing |
| `streamvln_g713_experiment.json` | Frozen 26-pair, camera, feature and gate contract |

The accepted A800 run contains 52 unique observations and 104 exact-restored snapshots. All protocol guards passed, but the fixed primary feature failed every signal gate: overall AUROC `0.4763`, scene/state minima `0.35/0.4615`, strict paired wins `2/26`, and equal zero medians. The level forward camera did not image the immediate next-primitive ground. This representation is closed; G7.14 tests visibility from one geometry-derived downward visual view before any controller or navigation work. G8 remains blocked.

## StreamVLN G7.14 immediate-footprint visibility ceiling

| File | Responsibility |
| --- | --- |
| `streamvln_visibility_g714.py` | Fixed pitch derivation, 48-ray target geometry, depth partition and visibility gates |
| `scripts/habitat_r2r_visibility_service.py` | Pitched-depth capture, active sensor extrinsics and exact restoration |
| `vln_env.py::probe_immediate_footprint_visibility` | Strict evaluator-only response and metric-depth validation |
| `run_streamvln_visibility_g714.py` | G7.13 anchor checks, bounded collection, repeat/extrinsic guards and routing |
| `streamvln_g714_experiment.json` | Frozen 26 cases, pitch, spatial bins and gates |

The accepted A800 run produced 26 rows and 52 exact snapshots. All 16 protocol guards and all seven visibility gates passed. Each observation contained 48 finite, unclipped target rays with at least four rays per 3×3 bin; actual extrinsics matched within `8e-8`. This is a discrete visibility result only. G7.15 must test executed outcomes on disjoint train scenes before any controller or navigation work; G8 remains blocked.

## StreamVLN G7.15 executed-outcome discrimination

| File | Responsibility |
| --- | --- |
| `streamvln_outcome_g715.py` | Frozen equal-weight visual risk, AUROC and deterministic one-to-one matching |
| `scripts/habitat_r2r_outcome_service.py` | Action-preceding pitched depth, exactly one critical forward primitive and exact restoration |
| `vln_env.py::probe_immediate_footprint_outcomes` | Strict train-only response, geometry and one-action protocol validation |
| `run_streamvln_outcome_g715.py` | G7.14 anchor checks, six-scene collection, G7.10 label, gates and routing |
| `streamvln_g715_experiment.json` | Frozen scenes, states, 32 cases, score, label, matching and support/separation gates |

The accepted v2 run produced 384 rows, 768 exact snapshot/outcomes and 29 deterministic matched pairs. Protocol integrity passed, but class support and outcome separation failed. The fixed score is closed; no controller, navigation, `val_unseen` or G8 route is opened. G7.16 defines the lower-layer sensing/action-execution boundary instead of tuning this representation.

## StreamVLN G7.16 causal executor contract

| File | Responsibility |
| --- | --- |
| `streamvln_executor_contract.py` | Strict high-level command, per-primitive feedback, chunk aggregation, reward and privileged-data contracts |
| `streamvln_env.py` | Preserve the StreamVLN observation schema while returning post-action path, collision, blockage and reward feedback in `info` |
| `vln_env.py` | Pass causal cumulative path length and collision count from Habitat after every primitive |
| `streamvln_topology.py` | Integrate measured path increments and keep collision separate from fixed-ratio blockage |
| `run_streamvln_contract_g716.py` | Two-scene exact-repeat, reward, recovery and zero-`val_unseen` smoke |
| `streamvln_g716_contract.json` | Frozen sources, action scripts, schema, blacklist and gates |

The accepted A800 run produced four exact scene-repeat records, 12 chunks and 36 executed primitives with zero reward-accounting error. All contract gates passed. This confirms only readiness for G7.17, a bounded train-only high-level RL plumbing smoke. No model was loaded or updated, recovery was compiled but not executed, and no navigation, controller, robot, `val_unseen` or G8 claim is opened.
