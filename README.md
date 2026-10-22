<p align="center">
  <a href="https://github.com/datarobot/joint-client-python">
    <img src="https://raw.githubusercontent.com/datarobot/joint-client-python/main/.github/datarobot_logo.avif" width="600px" alt="DataRobot Logo"/>
  </a>
</p>
<h3 align="center">JointFM Python Client</h3>

<p align="center">
  <a href="https://www.datarobot.com/">Homepage</a>
  ·
  <a href="https://pypi.org/project/jointfm-client/">PyPI</a>
  ·
  <a href="https://github.com/datarobot/joint-client-python/blob/main/docs/api-reference.md">API reference</a>
  ·
  <a href="https://docs.datarobot.com/en/docs/get-started/troubleshooting/general-help.html">Support</a>
</p>

<p align="center">
  <a href="https://github.com/datarobot/joint-client-python/actions/workflows/ci.yml">
    <img src="https://github.com/datarobot/joint-client-python/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/jointfm-client/">
    <img src="https://img.shields.io/pypi/v/jointfm-client" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/jointfm-client/">
    <img src="https://img.shields.io/pypi/pyversions/jointfm-client" alt="Python versions">
  </a>
  <a href="https://github.com/datarobot/joint-client-python/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/datarobot/joint-client-python" alt="License">
  </a>
</p>

`jointfm-client` is the Python client for JointFM, DataRobot's forecasting foundation model. Given a running JointFM service, it builds and validates forecast requests, checks that the service speaks the same contract, and returns forecasts as NumPy arrays or pandas frames. It also installs a `jointfm-client` command for health checks and CSV forecasts.

> **Status: alpha.** The API and the service contract can still change. While the version is `0.x`, a breaking change ships as a minor version bump, so pin a single minor series (`jointfm-client~=0.N.0`) if you need stability.

## Before You Start: A JointFM Service

This package is only the client. It contains no model and no weights: every call goes to a running JointFM service, so you need one before the first forecast works. The client connects to either kind:

- **A service you run yourself**, on your own machine or GPU host. Point the client at its base URL with `JOINTFM_LOCAL_BASE_URL`. No DataRobot account is needed.
- **A JointFM deployment hosted on DataRobot.** Point the client at it with `DATAROBOT_ENDPOINT`, `DATAROBOT_API_TOKEN`, and `JOINTFM_DEPLOYMENT_ID`.

Running the server, deploying it, and obtaining model weights are covered by the JointFM inference server project, which is not public yet. This README will link to it once it is published.

## Quick Start

### Install

Python 3.11 or newer is required.

```bash
pip install jointfm-client
# or, in a uv project:
uv add jointfm-client
```

The pandas helpers and the example notebooks need the optional extra: `pip install "jointfm-client[notebooks]"`.

Verify the install:

```bash
python -c "import jointfm_client; print(jointfm_client.__version__)"
```

### Configure A Service

Create a `.env` file in your working directory, or set the same values in your shell. [.env.sample](.env.sample) documents every option. For a service you run yourself:

```dotenv
JOINTFM_LOCAL_BASE_URL=http://127.0.0.1:8080
JOINTFM_SCHEMA_VERSION=v6
```

For a deployment hosted on DataRobot:

```dotenv
DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
DATAROBOT_API_TOKEN=<token>
JOINTFM_SCHEMA_VERSION=v6
JOINTFM_DEPLOYMENT_ID=<deployment-id>
# Or: JOINTFM_DEPLOYMENT_IDS=<deployment-id-1>,<deployment-id-2>
# Optional drift-detection pin; the SDK discovers the model version from /healthz when unset:
# JOINTFM_MODEL_VERSION=jointfm-inference:<image-version>+ckpt.<checkpoint-version>
```

Check that the client can reach the service and that its metadata matches the configured schema and model pins:

```bash
jointfm-client health
```

### Run The First Forecast

This minimal example uses row dictionaries plus explicit column metadata, so it does not require pandas:

```python
from jointfm_client import (
	ColumnSpec,
	DataFrameSchema,
	JointFMClient,
	build_ordinal_query_times,
)

history_rows = [
	{"t": 0, "sales": 10.0},
	{"t": 1, "sales": 12.0},
	{"t": 2, "sales": 13.5},
	{"t": 3, "sales": 15.0},
]
schema = DataFrameSchema(
	columns=(ColumnSpec(name="sales", modality="numeric", role="target"),),
	time_index_mode="ordinal",
	time_column="t",
)

client = JointFMClient.from_env()
result = client.forecast_mean(
	history_rows,
	schema=schema,
	query_times=build_ordinal_query_times([row["t"] for row in history_rows], periods=3),
	requested_columns=("sales",),
)

print(result.to_pandas_tidy())
```

Use `forecast_samples(...)` for sampled trajectories or `forecast_quantiles(...)` with `quantiles=(0.1, 0.5, 0.9)` for quantile surfaces. For pandas inputs, install the optional extra and pass a `DataFrame` to `forecast(...)` with `target_columns` and `requested_columns` set explicitly.

## Package Contract

- Distribution package: `jointfm-client`
- Import namespace: `jointfm_client`
- Supported Python: `>=3.11`
- Current SDK package version: `0.14.0`
- Current JointFM service schema: `schema_version="v6"`

The public API shape is a synchronous low-level `JointFMClient` with `health()`, `health_instances()`, and `predict(payload)` methods plus high-level `forecast(...)`, `forecast_mean(...)`, `forecast_samples(...)`, and `forecast_quantiles(...)` helpers. The SDK is not a proxy service; callers use it as a local Python library that talks to the hosted or local JointFM endpoint.

SDK package versions are standard Python distribution versions: `[project].version` in `pyproject.toml` is the single declared value, and `jointfm_client.__version__` reports it back from the installed distribution metadata. JointFM `schema_version`, `image_version`, `model_version`, and `checkpoint_version` are service compatibility identifiers carried in configuration, health metadata, requests, and responses. They are not SDK package versions, and changing a deployment pin does not by itself require changing the SDK package version.

See [docs/api-reference.md](docs/api-reference.md) for the checked-in API reference covering public classes, functions, exceptions, environment variables, and payload fields.

## Service Contract

The DataRobot-hosted prediction URL is built from the DataRobot API v2 endpoint and deployment ID:

```python
from urllib.parse import urljoin

service_base_url = DATAROBOT_ENDPOINT.rstrip("/") + "/"
predict_url = urljoin(
		service_base_url,
		f"deployments/{deployment_id}/predictionsUnstructured",
)
```

The direct local service exposes `GET /healthz` and `POST /predict`.

## Configuration And Authentication

Structured SDK defaults live in `jointfm_client.configuration.JointFMConfig` and are mirrored in the checked-in `config.sample.yaml`. Copy `config.sample.yaml` to `config.yaml` and change only the fields needed for your deployment or transport defaults. `JointFMClient.from_env()` and `load_settings()` read `config.yaml` by default, then layer `.env` values over it, then layer process environment variables or the supplied `env` mapping over both. Explicit Python arguments such as `timeout=` and `retry_config=` still override YAML transport defaults.

`JointFMClient.from_env()` and `load_settings()` resolve `JOINTFM_SCHEMA_VERSION` and exactly one service selector from that layered configuration. Hosted options include `JOINTFM_DEPLOYMENT_ID` or load-balanced `JOINTFM_DEPLOYMENT_IDS` (comma-separated same-checkpoint peers; mutually exclusive with other selectors). `JOINTFM_MODEL_VERSION` is optional: when unset the SDK discovers the model version from `/healthz` on first use, and when set the SDK validates it against `/healthz` as a drift-detection guard. Local options are `JOINTFM_LOCAL_BASE_URL` or load-balanced `JOINTFM_LOCAL_BASE_URLS` (comma-separated same-checkpoint local services; mutually exclusive with other selectors). Hosted selectors also require `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN`; the local selectors do not use DataRobot credentials. Missing credentials, missing schema version, malformed credentials, unsupported schema versions, missing selectors, and multiple selectors raise `JointFMConfigurationError`.

`DATAROBOT_ENDPOINT` must be a normalized HTTPS DataRobot API v2 URL ending in `/api/v2`; the SDK stores it without a trailing slash. `DATAROBOT_API_TOKEN` must be non-empty and whitespace-free. The token is excluded from `JointFMSettings` repr output.

Required `.env` entries for hosted SDK calls are `DATAROBOT_ENDPOINT`, `DATAROBOT_API_TOKEN`, `JOINTFM_SCHEMA_VERSION`, and exactly one hosted selector from the list below. Required `.env` entries for local REST calls are `JOINTFM_LOCAL_BASE_URL` and `JOINTFM_SCHEMA_VERSION`. `JOINTFM_MODEL_VERSION` may be set in `.env` to pin a specific deployment artifact (the SDK then hard-errors on mismatch with `/healthz`); leave it unset to let the SDK use whatever version the deployment currently advertises. `.env` is the right place for these pins when using `from_env()` because they describe the selected JointFM service rather than a package-wide default. They are not secrets, and callers can still override them with process environment variables. Optional live DataRobot smoke tests additionally read `DATAROBOT_DEPLOYMENT_ID` from `.env` and use it as the hosted deployment ID for the `deployments/{deployment_id}/predictionsUnstructured` route.

Example deployment configuration:

```yaml
deployment:
	datarobot_endpoint: https://app.datarobot.com/api/v2
	datarobot_api_token: <token>
	schema_version: v6
	deployment_id: <deployment-id>
	# Optional model-version pin; the SDK discovers it from /healthz when unset:
	# model_version: jointfm-inference:<image-version>+ckpt.<checkpoint-version>
transport:
	timeout:
		connect_seconds: 10.0
		read_seconds: 120.0
	retry:
		max_attempts: 20
		backoff_seconds: 2
```

Equivalent `.env` deployment configuration:

```dotenv
DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
DATAROBOT_API_TOKEN=<token>
JOINTFM_SCHEMA_VERSION=v6
JOINTFM_DEPLOYMENT_ID=<deployment-id>
# Optional drift-detection pin; the SDK discovers the model version from /healthz when unset:
# JOINTFM_MODEL_VERSION=jointfm-inference:<image-version>+ckpt.<checkpoint-version>
```

Equivalent configuration for a JointFM inference service you run yourself:

```dotenv
JOINTFM_LOCAL_BASE_URL=http://127.0.0.1:8080
JOINTFM_SCHEMA_VERSION=v6
# Optional drift-detection pin; the SDK discovers the model version from /healthz when unset:
# JOINTFM_MODEL_VERSION=jointfm-inference:<image-version>+ckpt.<checkpoint-version>
```

Choose exactly one service selector:

- `JOINTFM_DEPLOYMENT_ID`: builds `DATAROBOT_ENDPOINT.rstrip("/") + "/"` plus `deployments/{deployment_id}/predictionsUnstructured`
- `JOINTFM_DEPLOYMENT_IDS`: comma-separated hosted deployment IDs (≥2 unique, same checkpoint) for round-robin load balancing; mutually exclusive with other selectors
- `JOINTFM_DEPLOYMENT_URL`: appends `/predictionsUnstructured` to a hosted deployment URL
- `JOINTFM_PREDICT_URL`: uses a full hosted prediction URL ending in `/predictionsUnstructured`
- `JOINTFM_DEPLOYMENT_TARGET` with `JOINTFM_PULUMI_OUTPUTS_PATH`: resolves a named target from saved Pulumi outputs JSON, preferring `deployment_id`, then `deployment_url`, then `predict_url`
- `JOINTFM_LOCAL_BASE_URL`: builds direct local `GET /healthz` and `POST /predict` URLs without DataRobot authentication
- `JOINTFM_LOCAL_BASE_URLS`: comma-separated local service base URLs (≥2 unique after normalization, same checkpoint), for example one instance per GPU on distinct ports, load-balanced exactly like `JOINTFM_DEPLOYMENT_IDS` and without DataRobot authentication; mutually exclusive with other selectors

Pulumi deployment discovery is explicit and file-backed. Export stack outputs to a JSON object keyed by target name, then set `JOINTFM_DEPLOYMENT_TARGET` to the key and `JOINTFM_PULUMI_OUTPUTS_PATH` to that JSON file:

```json
{
	"my-target": {
		"deployment_id": "<deployment-id>"
	}
}
```

For each target, the SDK accepts exactly one of these output fields: `deployment_id`, `deployment_url`, or `predict_url`. `deployment_id` is preferred because it lets the SDK build both the hosted health URL and the `deployments/{deployment_id}/predictionsUnstructured` URL from the configured `DATAROBOT_ENDPOINT`. `deployment_url` is normalized and extended with `/healthz` and `/predictionsUnstructured`. `predict_url` is accepted when the full hosted prediction URL has already been discovered; the SDK derives the owning deployment URL from it for health checks.

Hosted prediction calls use the same authorization scheme as the notebook helper:

```python
{
	"Authorization": f"Bearer {DATAROBOT_API_TOKEN}",
	"Accept": "*/*",
	"Content-Type": "application/json;charset=UTF-8",
}
```

The SDK still decodes hosted prediction responses as JSON; the broad `Accept` value avoids hosted unstructured prediction content negotiation failures before the deployment body is returned.

Direct local URL helpers are used by the local service selector: `build_local_health_url("http://localhost:8080")` returns `/healthz`, and `build_local_predict_url("http://localhost:8080")` returns `/predict`.

Hosted deployments are probed through the prediction route: `health()` POSTs `{"request_type": "health"}` to the resolved `predict_url`, because the DataRobot deployment gateway proxies only the unstructured prediction route. `JointFMClient.health(cache=True)` stores typed `HealthMetadata` only when the caller asks for caching, and `JointFMClient.refresh_health()` fetches a fresh copy.

Each endpoint's health payload describes only that endpoint. With a pool (`JOINTFM_DEPLOYMENT_IDS` or `JOINTFM_LOCAL_BASE_URLS`), the client probes every configured peer by POSTing `{"request_type": "health"}` to its prediction route, which local services answer as well, and aggregates locally: `health()` returns consensus metadata whose `max_sample_count` is the **minimum** reachable cap (the sample-batch size), while `health_instances()` returns one `InstanceHealth` per configured instance, named by its `instance_id` (the deployment ID, or the local base URL), plus the **sum** of reachable caps (one request per instance; each instance's `max_concurrent_requests` says how many such requests it serves at once), a compact `topology` / `topology_label` (for example `2x5000` or `1x7000, 1x3000`), and errors for unavailable peers.

## CLI Workflows

The package installs a `jointfm-client` command. It reads `.env` by default, accepts `--dotenv <path>` for another file, and accepts `--no-dotenv` when the process environment should be the only source.

Validate credentials, resolve the deployment, probe health, and print non-secret service metadata plus per-instance availability and sample topology:

```bash
jointfm-client health
```

The health command includes consensus `service` metadata, an `instances` list (available/unavailable, per-instance sample cap and request concurrency, errors), `topology` (for example `1x7000, 1x3000`), overall `max_sample_count` (sum of reachable caps), and non-secret `deployment` settings when configured.

Submit one low-level JSON request file and write the JSON response file:

```bash
jointfm-client predict request.json response.json
```

Forecast from CSV history and write tidy forecast rows as CSV:

```bash
jointfm-client forecast-csv history.csv forecast.csv \
	--query-times 2,3,4 \
	--target-column target \
	--return-mode mean
```

`forecast-csv` supports `--time-index-mode ordinal|continuous_float|absolute_datetime`, `--time-column`, repeated `--target-column`, repeated `--requested-column`, `--return-mode mean|samples|quantiles`, `--n-samples`, `--quantiles`, and `--seed`. The output is the same tidy shape returned by the Python result helpers.

Check that the deployment serves the largest request its health envelope advertises:

```bash
jointfm-client max-utilization
```

`max-utilization` fills every request slot at once with a seeded synthetic sample forecast of `max_series` target columns, an `n_input`-row history, an `n_output`-step horizon, and `max_sample_count` samples: each reachable endpoint (every pool instance with `JOINTFM_DEPLOYMENT_IDS` or `JOINTFM_LOCAL_BASE_URLS`) receives as many simultaneous maximal requests as its `max_concurrent_requests` admits. It prints the burst's wall-clock time and a final `VERDICT: PASS` or `VERDICT: FAIL` line. It exits with status 1 on FAIL: when a request raises a service or transport error or returns the wrong sample count, or when an endpoint fails the health probe. Retries are disabled for this command, so a request that fails under load is reported instead of retried. Each run uses real GPU time on the deployment. From a clone of this repository, `task max-utilization` runs the same command, and `notebooks/max_utilization.ipynb` shows the burst timing as a table.

## Notebook Workflows

Checked-in example notebooks live under `notebooks/`. Every example starts with:

```python
from jointfm_client import bootstrap_notebook

bootstrap_notebook(add_src_root=True)
```

The notebooks run from a clone of this repository. Run `task setup` there first (see [CONTRIBUTING.md](CONTRIBUTING.md#local-setup)) so VS Code can select the registered `Python (joint-client-python)` notebook kernel backed by the clone's `.venv`.

The bootstrap helper resolves the nearest src-layout Python project root, switches the working directory there, and prepends that project's local `src` tree during development. The examples cover hosted health checks, a maximum-utilization check, low-level JSON prediction, mean forecasts, sample forecasts, quantile forecasts, conditional forecasts (one conditional read as a mean, as draws, as quantiles inside a bounded band, and as the log density of observed values, plus the ranking of candidate conditions), pandas/NumPy result conversion, and CSV forecast workflows. They use `.env.sample` placeholders and checked-in fixture payloads; no real tokens or deployment IDs are stored in notebooks.

The current forecast request contract is:

- `schema_version`: exactly `"v6"`, configured as `JOINTFM_SCHEMA_VERSION` for `from_env()` clients
- `model_version`: exact model version advertised by `/healthz` or otherwise selected by the caller. Optional for `from_env()` clients: when `JOINTFM_MODEL_VERSION` is unset the SDK reads it from `/healthz` on first use; when set it acts as a drift-detection pin
- `query_mode`: `"forecast"` for the unconditional forecast, or `"condition"` for the forecast given conditions on some columns; the high-level helpers set it from whether a `condition` was passed
- `return_mode`: one of `"mean"`, `"samples"`, or `"quantiles"`
- `time_index_mode`: one of `"ordinal"`, `"continuous_float"`, or `"absolute_datetime"`
- `time_column`: required for `"absolute_datetime"`, and used for ordered ordinal or continuous histories when supplied
- `query_times`: non-empty future forecast times only
- `requested_columns`: optional column names or integer column indices, with duplicates rejected
- `n_samples`: positive sample count for sampled forecasts and quantile estimation. When `return_mode="samples"` exceeds the `max_sample_count` advertised by the deployment's health metadata, `forecast_samples(...)` splits the request into capped prediction batches up front and returns one merged `SampleForecastResult`. With a pool (`JOINTFM_DEPLOYMENT_IDS` or `JOINTFM_LOCAL_BASE_URLS`), every sample request is split evenly across all reachable endpoints, whatever its size, and each endpoint serves at most one of its batches at a time. Batch `i` uses seed `seed + i`, so a seeded result depends on how many endpoints were reachable.
- `condition`: required with `query_mode="condition"` and forbidden otherwise. One `EqualityCondition` or `IntervalCondition`, or a list of them, each covering the future positions its `query_time_indices` names; sent on the wire as the `conditions` list, see below.

### Conditional Queries

The `condition` query mode asks for the forecast *given* something about some of its columns. Each condition names the future positions it covers by index into `query_times` through `query_time_indices`, and covers every position when that is `None`. Two conditions on the same column must cover disjoint positions; conditions on different columns may share positions, which is how one position mixes both kinds:

- `EqualityCondition(column, value, query_time_indices=None)` pins the column to a finite value. It stays nameable in `requested_columns` and reads back the value the request supplied, so a scenario answer lines up column for column with an unconditioned one.
- `IntervalCondition(column, lower=None, upper=None, query_time_indices=None)` confines the column to a range; `None` leaves that side open, and at least one side must be bounded. The column stays readable, and what comes back is its distribution inside the range.

The response answers every entry of `query_times`. A position no condition covers carries the unconditioned forecast, and that is exact: the model draws each future position from its own joint, independently of the others, so a condition relates columns to each other within a position and never reaches another one. At every covered position at least one column must stay unconditioned, because what the position reads out is the conditional distribution of those columns. `requested_columns` chooses what the response carries independently of that, defaulting to every declared column in declared order. Pass one condition or a list of them to `forecast(...)`, `forecast_mean(...)`, `forecast_samples(...)`, `forecast_quantiles(...)`, or `forecast_log_prob(...)`:

```python
from jointfm_client import EqualityCondition, IntervalCondition

result = client.forecast_mean(
	history,
	query_times=query_times,
	requested_columns=["portfolio_nav", "realized_volatility"],
	columns=plan.columns,
	condition=[
		EqualityCondition(column="equity_index_level", value=4780.0),
		IntervalCondition(
			column="treasury_10y_yield", lower=0.041, upper=0.045, query_time_indices=[0]
		),
	],
)
print(result.plausibility)
```

Whether a deployment can condition depends on the mounted checkpoint's head. `/healthz` advertises `condition` in `supported_query_modes` and the kinds it answers in `supported_condition_kinds` (empty when the mode is absent). The client checks that advertisement before sending, so a deployment that cannot condition is refused with `UnsupportedServiceContractError` rather than after a paid round trip; `require_condition_support(metadata, condition)` exposes the same check.

A condition response carries a `plausibility` block: `equality_log_density` is the log density the model assigns to the pinned values and `region_log_probability` the log probability it gives the interval region *given the pinned values at the same position*, so the two add to the plausibility of the whole condition set, each `None` when the request carried no condition of that kind. Over several covered positions each is the sum of the per-position values, exact because positions are independent. They separate a confident answer from one conditioned on something the model finds implausible; the service reports them and never refuses on them. `diagnostics.condition_draws` counts the draws behind sampled outputs, and `diagnostics.interval_estimator` reports the numerical accounting (`points`, `effective_sample_size`) when some position bounds more than one column and its region probability had to be estimated; across several estimated positions it describes the one with the smallest effective sample size.

Column descriptors support the server fields `name`, `modality`, `role`, `nullable`, `vocabulary_size`, `level_count`, `mapping`, `lower_bound`, `upper_bound`, `time_value_kind`, `time_value_scale_seconds`, `time_value_use_local_normalized_time`, `time_value_calendar_id`, `time_value_timezone`, and `traits`. `traits` carries expert hints, such as `{"jump_process": "present"}`. The deployment validates them against its trait vocabulary and refuses any field it does not know, and the client sends a hint only to a deployment whose `/healthz` lists the trait in `supported_metadata_traits`.

Forecast values come back as read-only float64 NumPy arrays (`result.samples`, `result.mean`, each quantile surface's `values`); NumPy is a core dependency. DataFrame helpers and the notebook examples are available through one optional extra that pulls in `pandas`:

```bash
uv add "jointfm-client[notebooks]"
```

Use `build_forecast_payload_from_dataframe(...)` when history is already in a pandas DataFrame. It can accept explicit `ColumnSpec` objects or infer basic numeric, categorical, ordinal, count, binary, and time-valued columns from the DataFrame plus role, mapping, nullable, and bounds hints. The helper emits `history_rows` in the same order as the service frame builder: `time_column` first when present, followed by the ordered modeled columns. `build_forecast_payload_from_arrays(...)` provides the same request path for two-dimensional NumPy-like arrays when callers already have array values and column metadata. `build_datetime_query_times(...)`, `build_ordinal_query_times(...)`, `build_continuous_query_times(...)`, and `validate_forecast_horizon(...)` perform local future-horizon validation before the SDK sends the request.

Successful forecast responses preserve `schema_version`, `image_version`, `model_version`, `checkpoint_version`, `head`, `query_mode`, `return_mode`, `outputs`, `plausibility`, and `diagnostics`. Structured service errors use this shape:

```json
{
	"schema_version": "v6",
	"errors": [
		{
			"code": "VALIDATION_ERROR",
			"message": "request field explanation",
			"field": "request_field"
		}
	]
}
```

Known error codes are `VALIDATION_ERROR`, `UNSUPPORTED_HEAD_QUERY_COMBINATION`, `UNSUPPORTED_RETURN_MODE`, `SCHEMA_VERSION_MISMATCH`, `MODEL_VERSION_MISMATCH`, `INPUT_SIZE_EXCEEDED`, and `INTERNAL_ERROR`.

## Compatibility Policy

The SDK supports only `schema_version="v6"`. `validate_service_metadata()` checks `/healthz` metadata and raises typed compatibility errors before prediction if the service advertises a different schema, an unexpected model version, mode capabilities outside the recorded service contract, or an unsupported `decoding_strategy`. Return modes and time-index modes must match the SDK's lists exactly. Query modes and condition kinds are derived by the service from the mounted head, so a deployment may advertise fewer of them than the SDK knows; it must advertise at least one query mode and nothing the SDK does not know.

Callers should pass an expected `model_version` when they already know which deployment artifact they intend to use. A mismatch is treated as a hard compatibility error rather than silently downgrading, guessing, or retrying another model.

High-level forecast helpers build the same validated request payloads as `build_forecast_payload(...)` and return `ForecastResponse`. Row-list inputs require an explicit `DataFrameSchema` or `ColumnSpec` sequence, while pandas DataFrame inputs can use the DataFrame adapter inference options. `predict(payload)` remains the low-level JSON method and requires the payload to include `model_version`.

## Support

- **Bugs and questions:** open an issue at [datarobot/joint-client-python](https://github.com/datarobot/joint-client-python/issues).
- **Security issues:** do not open a public issue; follow [SECURITY.md](SECURITY.md).
- **DataRobot platform help:** see [DataRobot support](https://docs.datarobot.com/en/docs/get-started/troubleshooting/general-help.html).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the commit format, and how releases are cut.

## License

Apache License 2.0. See [LICENSE](LICENSE).
