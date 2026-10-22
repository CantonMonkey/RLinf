# Navigation extension

Navigation development for the CantonMonkey RLinf fork. This directory contains
413 files imported from the committed `rlinf-nav/` source at
`b9c2af0ef32f41f6065896ee5d6dbf4e494ce169` (`navrlt-lightnav-v1`).
The RLinf runtime baseline is `0505431899574619da86f551bad70b71e0ea2177`.
See [EXPORT_PROVENANCE.json](EXPORT_PROVENANCE.json) for the **initial import**
digests at commit `3f02d5d`; later edits are tracked by Git, not by rewriting that
historical receipt. Current source packages receive a separate `SOURCE_STAGE.json`.
Publication checks and historical replay limitations are recorded in
[EXPORT_VALIDATION.md](EXPORT_VALIDATION.md).

The P8 public unit-test packaging defect is repaired. See
[P8 hashes and migration](P8_HASH_AND_MIGRATION.md) for the cause, test boundary
and current Navigation-RLT source staging commands.

For shared branch updates, private runtime overlays and per-experiment code
identity, see [the remote workflow](REMOTE_WORKFLOW.md).

## Contents

| Directory | Purpose |
| --- | --- |
| `navrl/` | Navigation contracts, deterministic grid environment, policies and adapters |
| `navrl/stack/` | Shared collection, execution, returns, PPO and checkpoint interfaces |
| `scripts/` | Habitat services, render/rollout tools and portable mock smoke commands |
| `integrations/rlinf_v03/` | Staged RLinf modules, PointNav/VLN, StreamVLN, LightNav and Navigation-RLT runners, configs and manifests |
| `tests/` | Contract, storage, protocol, model and integration tests; requirements vary |

## Quick local checks

Run from this directory with Python 3.10+; these commands use the standard library:

```bash
python -B -m unittest discover -s tests -p test_grid_mock.py
python -B -m unittest discover -s tests -p test_stack_core.py
python -B -m unittest discover -s tests -p test_stack_legacy_task.py
python -B -m unittest discover -s tests -p test_stack_ppo.py
python -B -m unittest discover -s tests -p test_serve_artifacts.py
python -B -m unittest discover -s tests -p test_navigation_rlt_storage_guard.py
python -B -m scripts.smoke_env --policy oracle --episodes 10 --seed 7
```

The mock checks deterministic transitions and navigation contracts. It does not
establish GPU training, real RGB rendering or navigation benchmark performance.
Full test discovery includes optional dependencies such as NumPy, PyTorch, Hydra,
Ray and Habitat; some historical evidence tests require external run artifacts.

## Deploying an integration

1. Choose a specific execution path and its matching dependency environments.
   RLinf and Habitat may require separate Python environments. Models, datasets
   and scene assets are provisioned separately under their own licenses.
2. Read the [staged integration notes](integrations/rlinf_v03/README.md) and the
   corresponding runner. Files here are **staged source**, not automatically
   registered in the upstream `rlinf/` package. Legacy PointNav/VLN copy mappings
   do not install the entire newer StreamVLN/LightNav/Navigation-RLT stack.
3. Configure paths, users, GPU identity, node address, ports and storage before
   launching. Examples use `/srv/navigation`, `/opt/conda`,
   `/cache/navigation-example`, `/obs/navigation-example`, `navcloud`/`navgpu`,
   documentation-only `192.0.2.*` addresses, placeholder GPU UUIDs and
   `navigation.storage.example.invalid:/`. The `ebk180` profile is a retained
   legacy label for a Btrfs workstation; its example device path must also match
   the actual deployment. Keep all identity, capacity and mount checks enabled.
4. Older scripts expect a project layout with `repos/rlinf-nav`, `repos/RLinf-v0.3`,
   `envs`, `data` and `runs`. Point each runner's paths at the actual deployment;
   the public `extensions/navigation` layout is a source layout. For configuration
   smoke scripts set `RLINF_CHECKOUT` to the selected installed runtime checkout.
5. Stage Navigation-RLT source with `scripts.stage_navigation_rlt_code` as
   documented in [the migration guide](P8_HASH_AND_MIGRATION.md). Configure the
   target machine and generate a new run manifest before execution. New run
   receipts must come from the new run; historical receipts cannot be relabeled.
   Frozen P8 manifests remain historical and are not generic launch templates.
6. Start with the selected runner's bounded smoke and verify rollout, actor
   update, weight synchronization and re-sampling on the target hardware.

## Scope of this publication

The export contains the committed navigation source and test fixtures. Original
planning-repository history, machine credentials, uncommitted patches, training
outputs, datasets and weights are not included. Historical Markdown files retain
implementation context; links to external run evidence may be unavailable here,
and historical results are not a fresh validation of this public configuration.

The publication check covers local contract tests and syntax. It does not rerun
A800/4090 jobs or establish a benchmark result. Development starts on the fork's
`navigation-vln` branch; upstream `main` and deployed machines are unaffected.
