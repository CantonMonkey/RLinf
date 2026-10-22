# Publication checks — 2026-10-22

Environment: Windows, Python 3.11.15. These checks concern the public source
export, not the deployed A800/4090 environments.

## Passed

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

## Known limitations

- `test_streamvln_p8_successor_feature_v2_ab`: three tests pass and three error
  with `v5 readiness manifest canonical digest drifted`. The public configuration
  substitutions change historical manifest bytes. Frozen digest checks remain
  active; this publication does not authorize or validate a new P8 experiment.
- `test_streamvln_g715_staging` was syntax checked but not executed because
  pytest is absent from the validation environment.
- Full test discovery, upstream pre-commit/lint, GPU training, Habitat rendering
  and performance evaluation were not run for this export. Heavy runtime
  dependencies, datasets and weights were not installed locally.
- Historical evidence links in older notes refer to artifacts omitted from this
  public source tree. Use the extension README for publication scope.

Configure the intended deployment and regenerate its manifests through the
appropriate experiment workflow before running live jobs. Preserve identity,
storage and digest checks; do not treat historical receipts as current evidence.
