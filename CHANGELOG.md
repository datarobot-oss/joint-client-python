# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are cut with `task release`, which uses Commitizen to read
[Conventional Commits](https://www.conventionalcommits.org/) since the last
release tag, infer the next SemVer bump, prepend the new section here,
commit the bump, and create an annotated tag.

## v0.17.0 (2026-11-25)

### Feat

- Keep working with every newer deployment of the same schema_version by passing advertised query modes, condition kinds, return modes, time index modes, and decoding strategies the client does not know instead of refusing the deployment at its health check, which now refuses only another schema_version, a model version mismatch, or a missing or malformed field, while the condition, table, and expert-hint gates still refuse a call before sending and the service refuses any other unserved mode; read precision, tail_map, max_request_bytes, and sample_row_draws in HealthMetadata and requested and allowed in StructuredError, each None from a deployment that does not send it; ship the runner's OpenAPI document as package data, copied by task contract:sync from the runner release RUNNER_VERSION names and returned by openapi_document(), and add unknown_contract_terms(document), which names every health field, error field, error code, or advertised mode of a service's document that the client does not read; capture every fixture from a running service with task fixtures:capture, which checks each answer against the document, replaces versions and the checkpoint path by placeholders, and runs the leak guard, keep one fixture directory per service, add request fixtures for the two missing condition modes, a two-column interval condition, and the four refusals, read the numbers tests assert from the captured fixtures, and fail test_wire_contract.py when a fixture is not a body the document describes or the document lists a term the client does not read; split contract.py into a contract package grouped by concern; and document the compatibility rule, the new fields and functions, and how to follow a runner release in the README, the API reference, AGENTS.md, and CONTRIBUTING.md
- Send tables with JointFMClient.predict_table, build_table_payload, and build_table_payload_from_dataframe, which list context rows and query rows and name the cells to predict in requested_cells whatever their columns' roles, and parse their answers into TableMeanResult, TableSampleResult, TableQuantileResult, and TableLogProbResult, refusing a deployment without the unordered time index mode before the round trip (require_table_support), accept health metadata that advertises a subset of KNOWN_TIME_INDEX_MODES and KNOWN_QUERY_MODES, the complete query mode included, instead of requiring the exact lists, so the client no longer refuses the health of a deployment that answers tables or completions, test both with fixtures captured from a running service, and document tables in the README's Predictive Modes section and the API reference

### Fix

- Stop refusing a forecast plan with fewer columns than min_series in plan_forecast_columns, because min_series is the fewest columns a training sample carried and limits no request, so any width from one column up to max_series plans, and say so in DataGenerationCapabilities and the API reference

## v0.16.2 (2026-11-21)

## v0.16.1 (2026-11-20)

### Feat

- Average permutation feature importance over n_repeats shuffles per feature and demonstrate it on a synthetic history with a known driver and a placebo column

## v0.16.0 (2026-11-09)

### Feat

- add stats task to display lines added and removed by commits over various timeframes

## v0.15.0 (2026-10-29)

## v0.14.0 (2026-10-22)

### Feat

- Updating CUDA (#35)

## v0.13.1 (2026-10-21)

## v0.13.0 (2026-10-16)

### Feat

- Speak service schema v6, add leak guard (#23)

### Fix

- Isolate tests from the calling git process's environment (#24)

## v0.12.0 (2026-10-13)

### Feat

- Update health payload and pool client to use instance_id and new time index encoding
- Add a local instance pool selector and name pool peers by instance_id (#22)

## v0.11.0 (2026-10-06)

### Feat

- Split pool sample forecasts across all endpoints and drop the single-request utilization stage (#20)

## v0.10.0 (2026-10-06)

### Feat

- Return forecasts as NumPy arrays and add a concurrency-aware maximum-utilization check (#19)

## v0.9.0 (2026-09-24)

### Feat

- Let each condition name its own positions and move the envelope to schema v5 (#18)
- Add the log_prob return mode and move the envelope to schema v4 (#17)

## v0.8.0 (2026-09-22)

### Feat

- Conditional queries and schema v3 (#15)

## v0.7.0 (2026-09-08)

### Feat

- Model and schema update (#14)

## v0.6.0 (2026-09-08)

### Feat

- add permutation feature importance to JointFMClient (#12)

## v0.5.0 (2026-09-08)

### Feat

- Improved version management (#13)

## v0.4.1 (2026-08-27)

### Fix

- enhance service health notebook with detailed topology descripti… (#11)
- update README and Taskfile (#9)

## v0.4.0 (2026-08-26)

### Feat

- add CLAUDE.md to document agent instructions reference
- specify uv version in CI and publish workflows, update required version in pyproject.toml

### Fix

- update AGENTS.md to clarify agent command execution and configuration comment guidelines

## v0.3.0 (2026-08-17)

### Feat

- add support for decoding strategies in health metadata (#6)

### Fix

- ensure chat editing settings are preserved in VSCode configuration

## v0.2.2 (2026-08-01)

## v0.2.1 (2026-07-28)

### Fix

- increase timeout and retry settings for transport configuration (#3)

## v0.2.0 (2026-07-28)

### Feat

- add .env file creation step in setup task

### Fix

- clarify guidelines on handling destructive git operations and temporary identifiers
- update notebook kernel display name to match version format

## v0.1.0 (2026-07-06)

### Feat

- enhance release process with pre-bump hooks and improved checks
- add GitHub Actions workflow for publishing to PyPI and authors file
- introducing commitizen chore: introducing ruff format

### Fix

- remove default_sample_count from HealthMetadata and related tests
- add copyright notice
- no unnecessary typo exceptions
- pre-commit install type

## v0.0.1 (2026-06-26)
