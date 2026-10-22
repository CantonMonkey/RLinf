# Synthetic P8 unit-test contract

These fixtures exercise the public source and validators with a simulated
authorization scenario. They are **unit-test inputs, not experimental receipts,
checkpoint promotion, historical evidence, or permission to launch training**.

`readiness.json`, `causal.json` and `feature_v2_ab.json` preserve the protocol,
episodes, train-only restrictions, budgets, architecture and acceptance thresholds
of the tests. Their implementation lists bind the reviewed public source. Receipt
and candidate identity anchors are explicitly synthetic. `profile.json` pins the
exact fixture bytes; `p8_public_fixture.py` independently pins the profile digest.

Tests validate these fixed files before temporarily injecting only the listed
identity anchors. No hash function or validator is mocked. The existing
fake-provider test stages actual source and passes real implementation validation.
All injections are restored, including on exceptional exits. Outside the test
context, unchanged production entrypoints reject this bundle and the public
deployment templates before calling a provider or creating output.

## Updating after an intentional source change

From `extensions/navigation`, generate into a **new** directory:

```bash
python -B tests/build_public_p8_fixtures.py --output /path/to/new-fixtures
```

Review the JSON/source diff, replace the four fixture JSON files deliberately,
and update `PROFILE_SHA256` in `tests/p8_public_fixture.py` to the printed digest.
Run both `test_streamvln_p8_successor_feature_v2_ab.py` and
`test_p8_public_fixture.py`. Regeneration is never performed during tests and
never changes production `EXPECTED_*` values or historical experiment manifests.
