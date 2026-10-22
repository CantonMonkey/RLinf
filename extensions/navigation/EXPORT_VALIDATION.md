# Publication checks — 2026-10-22

Environment: Windows, Python 3.11.15. These checks concern the public source
export, not the deployed A800/4090 environments.

## Initial import checks at `3f02d5d`

- 42 tests across `test_grid_mock`, `test_stack_core`,
  `test_stack_legacy_task`, `test_stack_ppo`, `test_serve_artifacts` and
  `test_navigation_rlt_storage_guard`.
- Mock oracle rollout: 10/10 episodes successful, zero collisions.
- 21 tests passed across G714, G716, G717, G718, G719, G719-v2 and P8 parity.
  Seven assertions requiring omitted historical evidence were explicitly skipped;
  source/contract assertions remain active.
- All 328 Python source files parsed successfully.
- No known private-key, GitHub-token, AWS-key or Slack-token patterns found in
  the exported text. This limited scan is not a general security certification.

## P8 packaging correction and migration checks

- The initial import's three Feature-v2 errors were caused by using transformed
  deployment templates as fixtures for the original frozen experiment identity.
  Independent, fixed-hash synthetic test fixtures now cover the public code.
- The original six Feature-v2 tests, sixteen adjacent causal tests and fifteen
  readiness tests pass. Six additional fixture-boundary tests confirm byte and
  semantic checks, context restoration, reproducible fixture generation, and
  production rejection before provider/output access.
- Nine staging tests run: eight pass; one real-symlink creation test is skipped
  because this Windows account lacks the required privilege. Sixteen storage
  guard and three source-inventory tests also pass: **70 passed, 1 skipped**.
- Real-source staging and verification succeeded for 45 files. The source-only
  receipt does not certify deployment readiness or training outcomes.
- All seven changed Python files pass Ruff 0.14.3 checks with preview rules and
  formatting. Historical production runner/manifest anchors were not modified.
- Root cause, remaining historical replay requirements and migration commands:
  [P8_HASH_AND_MIGRATION.md](P8_HASH_AND_MIGRATION.md).

## Remaining validation limits

- Historical P8 production entrypoints still require the original matching
  manifest, source, receipts and checkpoint. Synthetic fixtures never authorize
  a production run; LightNav / Navigation-RLT do not depend on these P8 gates.
- `test_streamvln_g715_staging` was syntax checked but not executed because
  pytest is absent from the validation environment.
- Full test discovery, whole-repository pre-commit/lint, GPU training, Habitat rendering
  and performance evaluation were not run for this export. Heavy runtime
  dependencies, datasets and weights were not installed locally.
- Historical evidence links in older notes refer to artifacts omitted from this
  public source tree. Use the extension README for publication scope.

Configure the intended deployment and create its own run manifest before live
jobs. Receipts must be produced by that run. Preserve identity, storage and
digest checks; do not treat historical receipts as current evidence.
