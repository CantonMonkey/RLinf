# A800 R2R VLN observation protocol (draft for the active Goal)

This contract connects the private Habitat 0.2.4 Python 3.9 process to the
RLinf v0.3 Python 3.10 environment worker. It is a new R2R path; the existing
PointNav bridge and its results remain separate. The first runnable target is
one 64×64 RGB environment using existing read-only R2R/MP3D assets.

## Policy-visible observation

Only the following tensors may enter the policy and PPO trajectory:

| Key | Type and shape at one environment | Meaning |
| --- | --- | --- |
| `rgb` | `uint8 [1, 64, 64, 3]` | Current real Habitat RGB frame; resolution is configurable. |
| `instruction_ids` | `int64 [1, 200]` | R2R instruction token IDs from the shared 2,504-word dataset vocabulary, padded to the recorded length. |
| `instruction_mask` | `bool [1, 200]` | True for non-PAD instruction tokens (`PAD_INDEX=0`). |

The Habitat service may transport RGB as base64 in its JSONL stdio response,
but the RLinf environment must decode it to the tensor above before returning
  the observation. A ready handshake must identify the requested split, source
dataset, episode index/ID, scene, vocabulary hash and success threshold.
Train and val_unseen vocabulary content has been checked identical. Reset and step responses
must repeat the episode identity and reject a changed scene or instruction.

The policy must never receive PointGoal, target coordinates, `reference_path`,
privileged pose, geodesic distance or the episode's goal as an observation.
The Habitat service may use goal/path data for reward and evaluator metrics;
terminal `success`, `spl`, path length and collision counts go to logging only.
The R2R evaluator uses an explicit 3.0 m STOP Success/SPL threshold, following
the VLN-CE task convention. The assets and Habitat version here differ from
that published VLN-CE benchmark, so local scores are not official benchmark scores.
Metric depth can be recorded with RGB for debugging, but is not a policy input
in the first RGB+instruction baseline.

## Dataset and runtime boundary

- R2R train and val_unseen assets are referenced read-only from the existing
  A800 cache. The selected first episodes are train index 4852 / ID 4862 and
  val_unseen index 630 / ID 631. Their scenes are disjoint; hashes and paths
  are recorded in `artifacts/vln_asset_preflight_20260924.json` at project root.
- Start one Habitat process per selected scene. The prior Habitat 0.2.4
  PointNav audit encountered a native crash on a cross-scene transition.
- On-policy sampling, checkpoint restore and val_unseen inference must log the
  policy version, data split, episode identity, action trace and metrics.
- The first short PPO update is an engineering smoke. Success/SPL and a fixed
  control are needed before making any claim that the policy learned VLN.

This document specifies the intended interface. A real R2R camera reset/step
and a separate RGB+instruction JSONL observation probe have passed on A800;
the standalone RLinf policy has passed CPU tests. The JSONL service, integrated
policy rollout and GPU training are not yet validated end to end.
