# JointFM Python SDK API Reference

This reference covers the supported public Python surface exported by `jointfm_client`. Names that start with an underscore are implementation details and are not part of the package contract.

## Client Entry Point

| Name | Purpose |
| --- | --- |
| `JointFMClient` | Synchronous client for hosted or local JointFM endpoints. Use `from_env()` for `.env` and `config.yaml` backed hosted settings, `health()` for consensus typed service metadata, `health_instances()` for per-instance probe results and pooled sample topology, `endpoint_client(instance_id)` for an independent single-endpoint client bound to one configured instance, `predict(payload)` for low-level JSON prediction, `forecast(...)` for validated tabular forecasts, and the `forecast_mean(...)`, `forecast_samples(...)`, `forecast_quantiles(...)`, and `forecast_log_prob(...)` convenience methods for typed forecast results. `forecast_log_prob(...)` is the one that asks about values the caller already holds: it takes `query_rows`, one observed row per entry of `query_times` carrying every declared column, and returns their log density under the model's joint. Each forecast method accepts `condition=`, one `EqualityCondition` or `IntervalCondition` or a list of them, which switches the request to the `condition` query mode and, before anything is sent, checks that the deployment's health metadata advertises the mode and every condition kind the conditions use. `predict_table(history, query_rows=..., requested_cells=...)` sends a table instead of a forecast: context rows, query rows, and the cells to predict, as a pandas `DataFrame` pair or as row mappings with `columns`, and returns a `TableResponse`; before anything is sent it checks that the deployment's health metadata lists `unordered` in `supported_time_index_modes`. `health()` probes `GET /healthz` for a single local service and POSTs `{"request_type": "health"}` to `predict_url` for hosted DataRobot deployments, because the DataRobot deployment gateway only proxies the unstructured prediction route, and for every peer of a pool. `feature_importance(...)` runs permutation feature importance: one baseline `forecast_samples` call plus `n_repeats` (default 5) per feature column, each shuffling the column anew, returning a list of `{"feature", "mean", "distance"}` dicts, each holding that feature's absolute forecast-mean shift and centered squared 2-Wasserstein distance, averaged over the shuffles and indexed by target and horizon. |

`JointFMClient.from_env()` loads `config.yaml`, optional `.env` values, and process environment variables. The read-only `timeout` and `retry_config` properties return the `JointFMTimeoutConfig` and `JointFMRetryConfig` the client was built with. `JointFMClient.health(cache=True)` caches health metadata only when requested. `JointFMClient.health_instances()` returns the same probe as a `HealthInstances` object: one `InstanceHealth` per configured instance (including failures), `max_sample_count` as the sum of reachable caps (one request per instance; each instance's `max_concurrent_requests` says how many such requests it serves at once), and `topology` / `topology_label` grouping those caps (unavailable peers are listed but excluded from the sum and topology). Each endpoint's health payload describes only that endpoint; the client aggregates by calling each configured peer. `health()` still exposes the minimum reachable `max_sample_count`, which is the sample-batch cap used by forecast helpers, and the minimum reachable `max_concurrent_requests`. `JointFMClient.endpoint_client(instance_id)` returns a new client bound to one configured instance (a pool instance, or the client's own `settings.instance_id`) with this client's timeout and retry policy but its own HTTP session and health cache, so several such clients can send requests to the same instance at once; a local pool instance becomes a `JOINTFM_LOCAL_BASE_URL` client and a hosted one a `JOINTFM_DEPLOYMENT_ID` client. It raises `JointFMConfigurationError` for any other instance ID. `JointFMClient.predict(payload)` requires `payload["model_version"]`; high-level forecast helpers resolve the configured model version when the caller does not pass one explicitly. When `forecast_samples(...)` requests an explicit `n_samples`, the client learns the deployment's `max_sample_count` from health metadata before the first prediction, splits oversized requests into capped prediction batches, and returns one merged `SampleForecastResult`. A pool (`JOINTFM_DEPLOYMENT_IDS` or `JOINTFM_LOCAL_BASE_URLS`) splits every sample request evenly across all routable endpoints, at least one sample per batch, because downloading the response dominates a sample request and downloads from separate endpoints overlap; each endpoint serves at most one batch at a time. Batch `i` uses seed `seed + i`, so a seeded result depends on the number of routable endpoints. Clients configured without a reachable health route fall back to discovering the cap from the structured service error.

## Contract Classes

| Name | Purpose |
| --- | --- |
| `ColumnSpec` | Describes one modeled request column. Fields are `name`, `modality`, `role`, `nullable`, `vocabulary_size`, `level_count`, `mapping`, `lower_bound`, `upper_bound`, `time_value_kind`, `time_value_scale_seconds`, `time_value_use_local_normalized_time`, `time_value_calendar_id`, `time_value_timezone`, and `traits`. |
| `DataFrameSchema` | Describes tabular history layout. Fields are `columns`, `time_index_mode`, `time_column`, `time_scale_seconds`, `use_local_normalized_time`, `calendar_id`, and `timezone`. |
| `ForecastRequestMetadata` | Holds `schema_version`, `model_version`, `query_mode` (`forecast` or `condition`), and `return_mode` for one forecast request. |
| `ForecastRequest` | Validated request object that combines metadata, schema, history rows, query times, requested columns, sample or quantile controls, `seed`, an optional `condition` (one condition or a list), and the optional `query_rows` a scored request supplies, then emits a JSON-compatible payload with `to_payload()`. `query_mode="condition"` and a `condition` must appear together, as must `return_mode="log_prob"` and `query_rows`, and both are validated against the request's own schema and `query_times` before any round trip. |
| `EqualityCondition` | One column pinned to a finite `value` at the positions `query_time_indices` names (indices into `query_times`, `None` for every position). The column may be named in `requested_columns` like any other, and reads back the pinned value the request supplied. |
| `IntervalCondition` | One column confined to `[lower, upper]` at the positions `query_time_indices` names (`None` for every position); `None` leaves a side open, at least one side must be bounded, and `lower < upper`. The column stays readable and the response describes its distribution inside the range. |
| `Condition` | Type alias for `EqualityCondition \| IntervalCondition`. Every `condition=` parameter takes one of them or a sequence of them. |
| `ConditionPlausibility` | What the model thinks of the conditions it was given: `equality_log_density` of the pinned values and `region_log_probability` of the interval region given the pinned values at the same position, each `None` when the request carried no condition of that kind. Reported by the service and never refused on. |
| `IntervalEstimator` | Numerical accounting (`points`, `effective_sample_size`) behind a region probability that had to be estimated, which happens when some position bounds more than one column. Across several estimated positions it describes the one with the smallest effective sample size. |
| `HealthMetadata` | Typed service-health payload with service status, schema and model versions, checkpoint metadata, device, head, `decoding_strategy`, advertised query modes, `supported_condition_kinds` (empty when the deployment cannot condition), return modes, time-index modes, time-index encoding, `max_sample_count`, `max_concurrent_requests` (how many requests the container serves at once; a further request waits for a free slot), an optional `data_generation` block carrying advertised capacity limits, `supported_metadata_traits` (the expert-hint traits the mounted model reads; empty when the field is absent), and the optional `precision`, `tail_map`, `max_request_bytes`, and `sample_row_draws`, each `None` from a deployment that does not send it. The container exposes it on `GET /healthz` for direct local access and as the response to `POST {"request_type": "health"}` on the unstructured prediction route for DataRobot-hosted deployments. Each endpoint reports only its own capabilities. |
| `InstanceHealth` | One configured instance's probe outcome: `instance_id` (the deployment ID of a hosted instance, the normalized base URL of a local one), optional `metadata` (`HealthMetadata` when reachable), and optional `error` when the peer was skipped. |
| `HealthInstances` | Client aggregation of `health_instances()`: `instances` (one `InstanceHealth` per configured instance), `max_sample_count` (sum of reachable caps, one request per instance), `topology` as `(count, cap)` pairs sorted by descending cap, and `topology_label` such as `2x5000` or `1x7000, 1x3000`. Unavailable peers stay in `instances` but are omitted from the sum and topology. |
| `DataGenerationCapabilities` | Optional service-health block describing the deployed checkpoint's data-generation capacity. Fields are `min_series`, `max_series`, `t_input`, `t_output`, `n_input`, and `n_output`. `max_series` bounds the total number of series in one request, features and targets counted together — there is no separate per-role cap. `min_series` is the fewest series a training sample carried; it limits no request. |
| `UtilizationStage` | The burst of `run_max_utilization_check`: `name`, the total requested `n_samples`, the `request_count` of maximal requests sent at once, `wallclock_seconds` (an upper bound on each request's duration), and `errors`, one entry per request that ended in a JointFM error or a wrong sample count. `passed` is true without errors. |
| `MaxUtilizationReport` | Result of `run_max_utilization_check`: the `envelope` the requests were sized from, `decoding_strategy`, `topology_label`, `request_slots` (the sum of `max_concurrent_requests` over reachable endpoints), `unavailable_instances`, `read_timeout_seconds`, `health_seconds`, and the `concurrent_saturation` stage. `passed` requires every endpoint to be reachable and the stage to pass; `format_lines()` renders the report ending in a `VERDICT:` line. |
| `ForecastPlan` | Validated forecast plan returned by `plan_forecast_columns`. Fields are `columns` (ordered `ColumnSpec` tuple), `feature_columns`, `target_columns` (both as the caller declared them), and `requested_columns` (the caller's target list). |
| `StructuredError` | One structured JointFM service error with `code`, `message`, optional `field`, and, on an `INPUT_SIZE_EXCEEDED` error, the `requested` size and the `allowed` limit (`None` otherwise and from a service that does not send them). |
| `ForecastDiagnostics` | Response diagnostics containing `history_rows`, `horizon_count`, optional `seed`, and, on condition responses, `condition_draws` and an optional `interval_estimator`. |
| `QuantileForecast` | One quantile surface with `quantile` and `values`, a read-only float64 array with axis order `(horizon, column)` that `to_numpy()` returns. |
| `ForecastOutputs` | Legacy nested view of parsed output arrays, including `query_times`, `requested_columns`, and exactly one of `mean`, `samples`, `quantiles`, or `log_prob`. A response whose `return_mode` no parser handles is refused rather than read as another mode's shape. |
| `LogProbScores` | One `outputs.log_prob` block: per-horizon `values` and `nll_values` as read-only float64 arrays, plus the `total`, `mean`, `nll_total`, and `nll_mean` summaries. The summaries are redundant, so parsing recomputes them from `values` and rejects a payload that disagrees with itself. A log density is joint over the scored columns, so the block has no column axis. |
| `ForecastResponse` | Shared base for parsed forecast results. It preserves schema, image, model, checkpoint, head, mode, query-time, requested-column, diagnostic, and error metadata, plus `plausibility` (`None` on forecast responses). Every forecast value block is a read-only float64 NumPy array, the one representation for every column modality and both model generations: categorical and ordinal values stay class and level indices, as the service sends them. Values must be finite numbers, with one exception: in a column the request declared `nullable`, `null` is the observed missing state and parses to NaN. A `null` anywhere else, strings, booleans, ragged nesting, and wrong shapes are refused. Results compare by identity, because arrays define no single truth value for `==`; compare their arrays instead, for example with `numpy.array_equal`. |
| `MeanForecastResult` | Parsed mean forecast. `mean` is a read-only float64 array with axis order `(horizon, column)`, which `to_numpy()` returns; pandas helpers return tidy or wide frames. |
| `SampleForecastResult` | Parsed sample forecasts. `samples` is a read-only float64 array with axis order `(sample, horizon, column)`, which `to_numpy()` returns; batched sample requests are concatenated along the sample axis. Pandas helpers return tidy or wide frames. |
| `QuantileForecastResult` | Parsed quantile forecasts. `to_numpy()` stacks the surfaces into a new float64 array with axis order `(quantile, horizon, column)`, `quantile_levels` exposes the ordered levels, and pandas helpers return tidy or wide frames. |
| `LogProbResult` | Parsed log densities of the values the request supplied; it carries no forecast values. `to_numpy()` returns `(horizon,)`, `to_pandas_tidy()` one row per scored position, and `to_pandas_wide()` the one-row request summary, since a log density has no column axis to widen. |
| `TableRequest` | Validated table request: `metadata` (a `ForecastRequestMetadata` whose `query_mode` is `forecast`), `columns`, the context rows `history_rows`, the query rows `query_rows`, `requested_cells`, and the `n_samples`, `quantiles`, and `seed` controls. It refuses what the deployment would refuse about the rows and requested cells: a row missing a declared column or holding an undeclared one, a requested cell outside its row list or in an undeclared column, a repeated index, a request of no cell, a query row that requests none, and a requested cell that holds a value outside `log_prob` or none under it. `to_payload()` writes `time_index_mode="unordered"`. |
| `RequestedCells` | Type alias for the `requested_cells` mapping: row list (`history_rows` or `query_rows`) to column name to 0-based row indices. |
| `TableRows` | The rows a table answer covers: `history_rows` and `query_rows` indices, each ascending, context rows first in the answer; `labels` gives `(row list, index)` per answer row. |
| `TableResponse` | Shared base for parsed table answers, with schema, image, model, checkpoint, head, return mode, `rows`, `requested_columns` (the declared columns that hold a requested cell, in declared order), `requested_cells` (a read-only boolean `(row, column)` array, true where the request asked for a cell), `diagnostics`, and `errors`. `from_payload(payload, request_payload=...)` needs the request, because only the request says which answer cells hold values; it refuses an answer whose rows, columns, sizes, or versions differ from the request's, a value in a cell not requested, and a `null` in a requested cell of a column the request did not declare `nullable`. Every value array holds NaN in the cells not requested. |
| `TableMeanResult` | Parsed table means: `mean` with axis order `(row, column)`, which `to_numpy()` returns. `to_pandas_tidy()` has one row per requested cell with `row_list`, `row`, `requested_column`, and `value`; `to_pandas_wide()` one row per answer row with one column per requested column. |
| `TableSampleResult` | Parsed table draws: `samples` with axis order `(sample, row, column)`; the cells of one row in one draw belong together. Table samples are not split across requests. Pandas helpers lead with `sample`. |
| `TableQuantileResult` | Parsed table quantiles: one `QuantileForecast` per level whose `values` have axis order `(row, column)`; `to_numpy()` stacks them as `(quantile, row, column)`, and pandas helpers lead with `quantile`. |
| `TableLogProbResult` | Parsed log densities of the values held in the requested cells, one per answer row, joint over that row's requested cells. `to_numpy()` returns `(row,)`, `to_pandas_tidy()` one row per answer row, and `to_pandas_wide()` the one-row summary. |

## Configuration Classes

| Name | Purpose |
| --- | --- |
| `JointFMSettings` | Validated hosted or local service settings: optional normalized DataRobot endpoint, optional secret token, health and prediction URLs, service selector, schema pin, optional load-balanced `instances` pool, model pin, and optional selector details. `is_local` tells local selectors from hosted ones; `instance_id` names the single target as a pool instance would (local base URL or deployment ID). The API token is excluded from `repr`. |
| `JointFMConfig` | Top-level structured configuration loaded from defaults, YAML, and explicit overrides. |
| `PathConfig` | Default local file names for `config.yaml`, `config.sample.yaml`, and `.env`. |
| `EnvironmentVariableConfig` | Environment variable names consumed by settings loading. |
| `HostedDeploymentConfig` | Optional service target values layered below `.env` and process environment variables. |
| `TransportConfig` | Default HTTP transport settings: timeouts, retry policy, response excerpt length, retryable methods, DataRobot request-id headers, and user-agent header. |
| `TimeoutConfig` | YAML-backed connect and read timeout values. |
| `RetryConfig` | YAML-backed retry attempts, backoff, and retryable HTTP status codes. |
| `ForecastConfig` | Default forecast controls used by client and adapter helpers. |
| `ForecastCsvConfig` | Defaults used by the `forecast-csv` CLI command. |
| `CLIConfig` | Command line defaults, including `.env` path and CSV forecast defaults. |

## Transport Classes

| Name | Purpose |
| --- | --- |
| `JSONTransport` | Protocol for test transports or alternate JSON transports. It requires `get_json(url)` and `post_json(url, payload)`. |
| `JointFMHTTPTransport` | `requests.Session` backed JSON transport with configured headers, user agent, timeouts, retries, response decoding, request-id capture, and typed SDK exceptions. Response bodies are parsed from their bytes with orjson, which rejects the non-standard `NaN` and `Infinity` tokens. The text excerpt an error carries is built only when the error is raised, so a large successful response is never decoded to text. Request bodies stay encoded by `requests` with `allow_nan=False`, so a non-finite value in a request fails instead of being sent. |
| `JointFMTimeoutConfig` | Runtime connect and read timeout settings consumed by `requests`. |
| `JointFMRetryConfig` | Runtime retry policy converted to the `urllib3` retry object used by `requests` adapters. |

## Exceptions

All SDK-specific exceptions inherit from `JointFMError`.

| Name | Raised When |
| --- | --- |
| `JointFMConfigurationError` | Local settings, credentials, URLs, deployment selection, or request configuration are invalid. |
| `JointFMCapacityError` | Forecast inputs exceed the deployed checkpoint's advertised data-generation capacity, the service-health payload does not advertise a `data_generation` block, or a request body exceeds the `max_request_bytes` the deployment advertises (the smallest across a pool's instances), raised before any byte is uploaded with a message naming both sizes and the request's column, history-row, and query-time counts. A deployment that does not advertise the cap gets no such check. |
| `JointFMTransportError` | Base class for transport failures. |
| `JointFMRequestEncodingError` | A request payload cannot be JSON encoded, including a `NaN` or infinite value, which JSON cannot spell; it is raised once, before any attempt. |
| `JointFMRequestError` | A network request fails before a usable response is received. |
| `JointFMResponseError` | Base class for malformed or unsuccessful service responses. It preserves `status_code`, `response_body_excerpt`, optional `datarobot_request_id`, and structured JointFM errors. |
| `JointFMResponseDecodeError` | A service response is empty or whitespace-only, is not strict JSON (including a body with `NaN` or `Infinity` tokens), or is not a JSON object. |
| `JointFMHTTPStatusError` | The service returns an HTTP error status. |
| `JointFMInputSizeExceededError` | A `JointFMHTTPStatusError` for an `INPUT_SIZE_EXCEEDED` answer: a body above `max_request_bytes` (HTTP 413), more samples than `max_sample_count`, or more columns than `max_series`. Its `field`, `requested`, and `allowed` name the refused input, its size, and the service's limit (`None` when the service does not send them). It is never retried. |
| `JointFMServiceError` | A response body contains non-empty JointFM `errors`, including the case where HTTP status unexpectedly succeeded. |
| `JointFMCompatibilityError` | Base class for fail-fast service compatibility failures. |
| `UnsupportedSchemaVersionError` | The service or response advertises a schema version other than `v6`. |
| `UnsupportedModelVersionError` | The service or response model version differs from the configured or requested version. |
| `UnsupportedServiceContractError` | The service-health payload omits or malforms a required field, or a request needs a capability the deployment does not advertise: a condition request to a deployment without the `condition` mode or one of the request's condition kinds, a table to a deployment without `unordered`, or an expert hint for a trait it does not read. |

## Public Functions

| Name | Purpose |
| --- | --- |
| `load_configuration(config_path=..., overrides=...)` | Load structured SDK configuration from defaults, optional YAML, and explicit overrides. |
| `load_settings(env=..., dotenv_path=..., config_path=..., config=...)` | Load and validate hosted DataRobot or direct local REST settings from configuration, `.env`, and environment values. |
| `normalize_datarobot_endpoint(value)` | Validate and normalize a DataRobot API v2 endpoint ending in `/api/v2`. |
| `validate_datarobot_api_token(value)` | Validate a non-empty, whitespace-free token without exposing it. |
| `normalize_deployment_id(value)` | Validate a deployment ID as one non-empty path segment. |
| `normalize_hosted_deployment_url(value)` | Validate a hosted deployment URL ending in `/deployments/{deployment_id}`. |
| `normalize_hosted_predict_url(value)` | Validate a hosted prediction URL ending in `/predictionsUnstructured`. |
| `normalize_local_service_base_url(value)` | Validate a direct local JointFM service base URL. |
| `build_hosted_deployment_url(datarobot_endpoint, deployment_id)` | Build `.../deployments/{deployment_id}` from the DataRobot endpoint. |
| `build_hosted_predict_url(datarobot_endpoint, deployment_id)` | Build the hosted `predictionsUnstructured` URL from the DataRobot endpoint and deployment ID. |
| `build_hosted_predict_url_from_deployment_url(deployment_url)` | Append `/predictionsUnstructured` to a normalized hosted deployment URL. |
| `deployment_id_from_hosted_deployment_url(deployment_url)` | Extract a deployment ID from a hosted deployment URL. |
| `deployment_url_from_hosted_predict_url(predict_url)` | Return the owning hosted deployment URL for a hosted prediction URL. |
| `build_local_health_url(service_base_url)` | Build a direct local service `/healthz` URL. |
| `build_local_predict_url(service_base_url)` | Build a direct local service `/predict` URL. |
| `build_datarobot_prediction_headers(api_token)` | Build hosted prediction headers: bearer authorization, broad accept header, and JSON content type. |
| `build_forecast_payload(...)` | Build a validated JSON-compatible forecast payload from explicit schema, history rows, query times, and return-mode controls. |
| `build_table_payload(...)` | Build a validated JSON-compatible table payload from explicit columns, context rows, query rows, requested cells, and return-mode controls. |
| `openapi_document()` | Return the OpenAPI 3.1 document of the inference runner release this SDK speaks, as the package ships it: every request, response, health, and error field of `POST /predict` and `GET /healthz`, and one example request per query and return mode. Each call returns a fresh mapping. |
| `unknown_contract_terms(document)` | List, sorted, what an OpenAPI document of a JointFM service holds that this SDK does not read: a health field `HealthMetadata` lacks, an error field `StructuredError` lacks, an error code outside `STRUCTURED_ERROR_CODES`, an advertised mode outside the SDK's vocabulary, or an enumerated health field the SDK neither checks nor passes through. Empty when the SDK reads everything the document lists. |
| `validate_service_metadata(metadata, expected_model_version=None)` | Validate the service-health metadata against the supported schema version and the expected model when supplied, and require a non-empty `decoding_strategy` and each advertised capability list as an array of non-empty strings: `supported_query_modes`, `supported_return_modes`, and `supported_time_index_modes` non-empty, `supported_condition_kinds` possibly empty. Values outside what the SDK knows (`KNOWN_QUERY_MODES`, `SUPPORTED_CONDITION_KINDS`, `SUPPORTED_RETURN_MODES`, `KNOWN_TIME_INDEX_MODES`, `SUPPORTED_DECODING_STRATEGIES`) pass, so the client keeps working with a newer deployment of the same schema version. |
| `require_metadata_trait_support(metadata, columns)` | Raise `UnsupportedServiceContractError` when a request column descriptor carries a trait that `HealthMetadata.supported_metadata_traits` does not list. The forecast helpers call it before sending a hinted request. |
| `require_condition_support(metadata, condition)` | Raise `UnsupportedServiceContractError` when `HealthMetadata` does not advertise the `condition` query mode or one of the kinds `condition` uses. The forecast helpers call it before sending a condition request. |
| `require_table_support(metadata)` | Raise `UnsupportedServiceContractError` when `HealthMetadata.supported_time_index_modes` lacks `unordered`, as on every JointFM 1 deployment. `predict_table(...)` calls it before sending. |
| `resolve_conditions(condition)` | Normalize one condition or a sequence of them into a tuple, raising `ValueError` for an empty sequence, an entry that is not a condition, or two conditions on one column at a shared position (`query_time_indices=None` overlaps every other condition on its column). |
| `infer_column_specs_from_dataframe(frame, ...)` | Infer ordered `ColumnSpec` objects from a pandas `DataFrame` and explicit role, modality, mapping, nullability, time-value, and bounds hints. |
| `dataframe_to_history_rows(frame, schema)` | Convert a pandas `DataFrame` into server-compatible `history_rows`. |
| `arrays_to_history_rows(values, columns=..., ...)` | Convert a two-dimensional NumPy-like array plus column metadata into `history_rows`. |
| `build_forecast_payload_from_dataframe(frame, ...)` | Build a validated forecast payload from a pandas `DataFrame`. |
| `build_forecast_payload_from_arrays(values, ...)` | Build a validated forecast payload from a two-dimensional NumPy-like array. |
| `build_table_payload_from_dataframe(history, query_rows, ...)` | Build a validated table payload from a context-row and a query-row `DataFrame`, indexing `requested_cells` by row position. Without `columns` it infers the descriptors from both frames together; a column is nullable only when listed in `nullable_columns` or missing a value in a cell the request does not ask for, and a missing requested cell is sent as `null`. |
| `build_datetime_query_times(history_times, periods=..., frequency=...)` | Build future absolute-datetime query times from ordered history times. |
| `build_ordinal_query_times(history_times, periods=..., step=1)` | Build future ordinal query times from ordered history times. |
| `build_continuous_query_times(history_times, periods=..., step=...)` | Build future continuous-float query times from ordered history times. |
| `validate_forecast_horizon(history_times, query_times, time_index_mode=...)` | Validate that query times are future, increasing, and encoded for the selected time-index mode. |
| `resolve_notebook_project_root(start_dir=None)` | Resolve the nearest src-layout Python project root for notebooks started inside a repository tree. |
| `bootstrap_notebook(add_src_root=False)` | Switch notebook working directory to the project root and optionally prepend the local `src` directory. |
| `run_max_utilization_check(client, *, seed=DEFAULT_UTILIZATION_SEED)` | Fill every request slot at once with the largest sample forecast the deployment's `data_generation` envelope admits (all `max_series` columns as targets, `n_input` history rows, `n_output` query times): each reachable endpoint receives `max_concurrent_requests` simultaneous requests for its own `max_sample_count`, each through its own `endpoint_client`. Returns a `MaxUtilizationReport`. A JointFM error during the burst is recorded in the stage, not raised. Raises `JointFMConfigurationError` unless the client was built with `JointFMRetryConfig(max_attempts=1)`, and `JointFMCapacityError` when health metadata has no `data_generation` block. |
| `plan_forecast_columns(*, health, feature_columns, target_columns, history_length, query_times_length)` | Build a `ForecastPlan` from `HealthMetadata`. Caps `history_length` at `data_generation.n_input` and `query_times_length` at `n_output` (both upper bounds; smaller requests are allowed), checks the combined feature and target count against `data_generation.max_series` (any count from one up fits below it), and rejects duplicate column names. Raises `JointFMCapacityError` when any bound is exceeded or required metadata is missing. |

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATAROBOT_ENDPOINT` | Hosted calls | HTTPS DataRobot API v2 endpoint, normalized without a trailing slash and required to end in `/api/v2`. |
| `DATAROBOT_API_TOKEN` | Hosted calls | Non-empty, whitespace-free API token used in the hosted bearer authorization header. |
| `JOINTFM_SCHEMA_VERSION` | Hosted calls | Request schema pin. The SDK supports only `v6`. |
| `JOINTFM_MODEL_VERSION` | Optional | Exact JointFM deployment model version expected from the service-health payload and prediction responses. When unset, the SDK discovers the version from the service-health payload on first use; when set, a mismatch is a hard compatibility error. |
| `JOINTFM_DEPLOYMENT_ID` | One selector | Deployment ID used to build hosted health and prediction URLs. |
| `JOINTFM_DEPLOYMENT_IDS` | One selector | Comma-separated hosted deployment IDs for round-robin load balancing (at least two unique IDs). Mutually exclusive with other selectors. Peers must share `model_version` and `checkpoint_version`. `health()` uses the minimum reachable `max_sample_count` as the sample-batch cap; `health_instances()` sums reachable caps, one request per instance, and reports topology. Sample forecasts are split evenly across all reachable instances. |
| `JOINTFM_DEPLOYMENT_URL` | One selector | Hosted deployment URL; the SDK derives the `/predictionsUnstructured` route from it and reuses that route for health probes. |
| `JOINTFM_PREDICT_URL` | One selector | Full hosted prediction URL ending in `/predictionsUnstructured`; the SDK derives the owning deployment URL. |
| `JOINTFM_DEPLOYMENT_TARGET` | One selector with outputs path | Key in a saved Pulumi outputs JSON file. |
| `JOINTFM_PULUMI_OUTPUTS_PATH` | With target selector | JSON file containing target outputs with exactly one of `deployment_id`, `deployment_url`, or `predict_url`. |
| `JOINTFM_LOCAL_BASE_URL` | One selector | Direct local JointFM REST service base URL. The SDK calls `GET /healthz` and `POST /predict` without DataRobot authorization. |
| `JOINTFM_LOCAL_BASE_URLS` | One selector | Comma-separated local service base URLs for round-robin load balancing (at least two unique URLs after normalization), each validated like `JOINTFM_LOCAL_BASE_URL` and reached without DataRobot authorization. Mutually exclusive with other selectors. Routing, failover, the health gate, and the sample split behave as for `JOINTFM_DEPLOYMENT_IDS`; each instance is named by its base URL. |
| `DATAROBOT_DEPLOYMENT_ID` | Optional live tests | Hosted deployment ID used only by the optional live smoke test so normal CI does not call DataRobot accidentally. |

Set exactly one selector among `JOINTFM_DEPLOYMENT_ID`, `JOINTFM_DEPLOYMENT_IDS`, `JOINTFM_DEPLOYMENT_URL`, `JOINTFM_PREDICT_URL`, `JOINTFM_DEPLOYMENT_TARGET`, `JOINTFM_LOCAL_BASE_URL`, and `JOINTFM_LOCAL_BASE_URLS`.

## Payload Fields

### Request Type Discriminator

Every container request body carries an optional top-level `request_type` field that selects the request kind. The container handles `"health"` before any schema or model-version validation so callers can fetch the typed `HealthMetadata` payload without first knowing the deployed version pins.

| Value | Description |
| --- | --- |
| `"predict"` (default when omitted) | Standard forecast request; the rest of the fields in the table below are required. |
| `"health"` | Minimal probe; the container returns the same payload it serves on `GET /healthz`. The SDK uses this for hosted DataRobot deployments because DataRobot's deployment gateway only proxies the unstructured prediction route. |

The string literals are exposed as `PREDICT_REQUEST_TYPE`, `HEALTH_REQUEST_TYPE`, and the validated tuple `SUPPORTED_REQUEST_TYPES`.

### Forecast Request

| Field | Required | Description |
| --- | --- | --- |
| `request_type` | Optional | One of `"predict"` (default) or `"health"`. Forecast requests omit this field or set it to `"predict"`. |
| `schema_version` | Yes | Must be `"v6"`. |
| `model_version` | Yes | Exact deployed model version expected by the caller. |
| `query_mode` | Yes | `"forecast"` for the unconditional forecast or `"condition"` for the forecast given the request's `conditions`. |
| `return_mode` | Yes | One of `"mean"`, `"samples"`, `"quantiles"`, or `"log_prob"`, covered by the `forecast_mean`, `forecast_samples`, `forecast_quantiles`, and `forecast_log_prob` helpers. |
| `time_index_mode` | Yes | One of `"ordinal"`, `"continuous_float"`, or `"absolute_datetime"`. |
| `columns` | Yes | Non-empty array of column descriptors for modeled columns. |
| `history_rows` | Yes | Non-empty array of history row objects in the declared schema. |
| `query_times` | Yes | Non-empty future forecast horizon values. Absolute datetimes are encoded timezone-stably. |
| `time_column` | For absolute datetime, optional otherwise | Name of the history time column. It must not duplicate a modeled column name. |
| `requested_columns` | Optional | Output column names or integer indices, answered in the order given. Duplicates are rejected. Any declared column may be named, whatever its `role` and whatever a condition says about it. Defaults to every declared column in declared order, so a condition response and a forecast response over the same schema line up column for column. Under `return_mode="log_prob"` an explicit list must name every declared column in declared order, because the score is joint over them. |
| `query_rows` | `log_prob` mode | Observed rows to score, one per entry of `query_times`, each carrying a value for every declared column. Required by `return_mode="log_prob"` and rejected for every other mode. A row that contradicts a condition — a pinned column given another value, or a bounded column outside its range — is refused by the service instead of scored. |
| `n_samples` | Optional | Positive sample count for answers that sample; the deployment's `max_sample_count` when omitted. A JointFM 1 deployment answers means and quantiles exactly, so there it sizes `samples` and the draws behind answers at positions an interval condition covers. Oversized sample forecasts are batched automatically against the cap advertised in health metadata; a pool splits every sample forecast evenly across its reachable instances. |
| `quantiles` | Quantiles mode | Quantile levels in `(0, 1)`, required for `return_mode="quantiles"`. |
| `seed` | Optional | Integer random seed for the service's draws. The same request with the same seed returns the same answer while the deployment serves the same weights (`checkpoint_version`) from the same image (`image_version`) on the same `device` and at the same numeric precision. `requested_columns` only selects among the draws; changing anything else, `n_samples` included, can change every draw. When a sample forecast is batched, batch *i* is sent with `seed + i`, so a seeded result also depends on the batching: the deployment's `max_sample_count` and, for a pool, the number of reachable instances. Two seeds closer together than the number of batches share batches, so space the seeds of independent repetitions by at least that number. |
| `conditions` | With `query_mode="condition"` | Non-empty list of `{"column", "kind": "equality", "value", "query_time_indices"}` or `{"column", "kind": "interval", "lower", "upper", "query_time_indices"}` entries, `null` marking an open bound. `query_time_indices` is a list of distinct indices into `query_times`, or `null` for every position. Two conditions on one column must cover disjoint positions, and every covered position must leave at least one column unconditioned. Forbidden with any other query mode. |
| `time_scale_seconds` | Optional | Positive scale for continuous time indexes. |
| `use_local_normalized_time` | Optional | Whether the service should use local normalized time features. |
| `calendar_id` | Optional | Calendar identifier, defaulting to `pandas-default`. |
| `timezone` | Optional | Time zone for absolute datetime handling. |

### Table Request

A table request carries `schema_version`, `model_version`, `query_mode` (`"forecast"`), `return_mode`, `columns`, `n_samples`, `quantiles`, and `seed` as a forecast request does, and refuses `time_column`, `query_times`, `requested_columns`, `use_local_normalized_time`, `calendar_id`, and `timezone`.

| Field | Required | Description |
| --- | --- | --- |
| `time_index_mode` | Yes | `"unordered"`. |
| `history_rows` | Yes | Non-empty array of context rows, each holding every declared column. |
| `query_rows` | Yes | Non-empty array of query rows, each holding every declared column. |
| `requested_cells` | Yes | Per row list (`history_rows`, `query_rows`), an object mapping a declared column name to the 0-based indices of the rows whose cell in that column is requested. Any cell may be requested, whatever its column's `role`; every query row requests at least one. A requested cell holds `null`, or under `log_prob` the value it scores. |

### Column Descriptor

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Unique modeled column name. |
| `modality` | Yes | One of `numeric`, `categorical`, `ordinal`, `binary`, `count`, or `time_value`. |
| `role` | Optional | One of `target`, `known_dynamic`, `past_dynamic`, `static`, or `feature`; omitted values default to `feature`. |
| `nullable` | Optional | Whether null values are accepted for the column. |
| `vocabulary_size` | Categorical metadata | Positive category vocabulary size. |
| `level_count` | Ordinal metadata | Positive number of ordered levels. |
| `mapping` | Categorical or ordinal metadata | Mapping from raw values to integer category or level IDs. |
| `lower_bound` | Numeric metadata | Optional finite lower bound. Read only for `samples`: the service replaces a draw outside a bound by a fresh draw, across all columns of its position at once, so the draws follow the joint distribution restricted to the bounds; a bound that draws still miss after the service's redraws fails the request. Means, quantiles, and scores ignore bounds. |
| `upper_bound` | Numeric metadata | Optional finite upper bound; must not be less than `lower_bound`. Applied as `lower_bound` is. |
| `time_value_kind` | Time-value columns | One of `continuous_float` or `absolute_datetime` when `modality="time_value"`. |
| `time_value_scale_seconds` | Time-value columns | Positive scale for time-valued numeric data. |
| `time_value_use_local_normalized_time` | Time-value columns | Whether local normalized time features are used. |
| `time_value_calendar_id` | Time-value columns | Calendar identifier, defaulting to `pandas-default`. |
| `time_value_timezone` | Time-value columns | Time zone for absolute time-valued columns. |
| `traits` | Expert hints | Optional `{trait_name: value}` mapping of metadata traits the caller asserts about the column, such as `{"jump_process": "present"}`. The deployment owns the vocabulary and refuses unknown names, invalid values, and traits that do not apply to the column's modality; omitted traits stay `unknown`. A hint is sent only to a deployment whose `supported_metadata_traits` lists it. |

### Forecast Response

| Field | Description |
| --- | --- |
| `schema_version` | Response schema, expected to be `"v6"`. |
| `image_version` | Service image version that produced the response. |
| `model_version` | Model version that produced the response. |
| `checkpoint_version` | Checkpoint version that produced the response. |
| `head` | Forecast head used by the service. |
| `query_mode` | Response query mode, matching the request: `"forecast"` or `"condition"`. |
| `return_mode` | Response return mode matching the request. |
| `outputs.query_times` | Forecast horizon values preserved from the request, on condition responses too: a position no condition covers carries the unconditioned forecast. |
| `outputs.rows` | `null` on a forecast; a service that answers no tables leaves it out. On a table, the answer rows as `{"history_rows": [...], "query_rows": [...]}`: the rows that carry a requested cell, context rows first. A table's `outputs.query_times` is `null`, and every value array keeps a `(row, column)` layout with `null` in each cell not requested. |
| `outputs.requested_columns` | Output columns in response order. |
| `outputs.mean` | Mean values with axis order `(horizon, column)` when `return_mode="mean"`. |
| `outputs.samples` | Sample values with axis order `(sample, horizon, column)` when `return_mode="samples"`. |
| `outputs.quantiles` | Quantile surfaces with axis order `(quantile, horizon, column)` when `return_mode="quantiles"`. |
| `diagnostics.history_rows` | Number of history rows processed. |
| `diagnostics.horizon_count` | Number of forecast horizon steps returned; on a table, the number of answer rows. |
| `diagnostics.seed` | Optional seed used by the service. |
| `outputs.log_prob` | `log_prob` responses only: per-horizon `values` and `nll_values` with their `total`, `mean`, `nll_total`, and `nll_mean` summaries. |
| `diagnostics.condition_draws` | Condition responses only: number of draws behind sampled outputs. Batched sample requests report the merged count. |
| `diagnostics.interval_estimator` | Condition responses only, and only when some position bounds more than one column: `points` and `effective_sample_size` of the numerical region-probability estimate, describing the estimated position with the smallest effective sample size. |
| `plausibility` | `null` on forecast responses. On condition responses an object with `equality_log_density` (log density of the pinned values) and `region_log_probability` (log probability of the interval region, conditional on the pinned values at the same position), each `null` when the request carried no condition of that kind. Over several covered positions each is the sum of the per-position values. |
| `errors` | Structured service errors. Non-empty arrays raise typed SDK exceptions. |

### Health Metadata

| Field | Description |
| --- | --- |
| `status` | Service status string. |
| `schema_version` | Advertised schema version. The SDK requires `v6`. |
| `image_version` | Running service image version. |
| `model_version` | Running model version. |
| `checkpoint_version` | Loaded checkpoint version. |
| `checkpoint_path` | Loaded checkpoint path reported by the service. |
| `device` | Device used by inference. |
| `precision` | Numeric precision every model forward runs at: `fp32`, `bf16`, or `fp16`. `None` from a deployment that does not send it. |
| `generation` | JointFM generation of the mounted checkpoint: `jointfm-1` or `jointfm-2`. Required; a deployment that does not report it fails to parse. |
| `head` | Active forecast head. |
| `tail_map` | Per-column tail map the head's forecast applies, `none` when it applies none. `None` from a deployment that does not send it. |
| `decoding_strategy` | Horizon decoding mode advertised by the mounted model. The SDK knows `SUPPORTED_DECODING_STRATEGIES`: `parallel_dense`, `parallel_scalable`, and `autoregressive`; a newer strategy passes through as advertised. Parallel strategies decode every horizon in one pass; `autoregressive` rolls horizons sequentially. |
| `supported_query_modes` | Non-empty list of the query modes the deployment serves, derived from the mounted model; the SDK knows `forecast`, `condition`, and `complete`, and keeps any other as advertised. `complete` means the deployment fills cells of an ordered request's own history rows; the SDK reads it in the health check but sends only `forecast` and `condition` requests. |
| `supported_condition_kinds` | Condition kinds the deployment answers, empty when `condition` is not advertised; the SDK knows `equality` and `interval`. Condition requests are refused locally when a condition uses a kind that is missing here. |
| `supported_return_modes` | Non-empty list of the return modes the deployment serves; the SDK knows `mean`, `samples`, `quantiles`, and `log_prob`. The deployment refuses an unserved one with `VALIDATION_ERROR`. |
| `supported_time_index_modes` | Non-empty list of the time-index modes the deployment serves, derived from the mounted model; the SDK knows `ordinal`, `continuous_float`, `absolute_datetime`, and `unordered`. `unordered` means the deployment answers table requests. |
| `time_index_encoding` | Time-index encoding advertised by the service. |
| `supported_metadata_traits` | Expert-hint traits the mounted model reads; empty when it reads none. A deployment that does not send the field predates hints and parses as supporting none. Forecasts with a `traits` entry outside this list are refused locally. |
| `max_sample_count` | Maximum sample-count budget the service accepts in a single prediction. The client reads it during health probes and batches oversized sample requests locally, so the service never has to reject them. |
| `max_request_bytes` | Largest request body, in bytes, the service reads; a larger one is refused unread with `INPUT_SIZE_EXCEEDED`. The client refuses a larger body with `JointFMCapacityError` before uploading it. `None` from a deployment that does not send it. |
| `sample_row_draws` | JointFM 2 deployments: answered rows times draws one sampling pass holds; a request needing more is drawn in several passes, and a `seed` repeats an answer only under the same value. `None` from a deployment that does not send it. |
| `max_concurrent_requests` | How many prediction requests the container serves at once; a further request waits for a free slot. `run_max_utilization_check` fills every slot; the forecast helpers send at most one request per endpoint at a time. |
| `data_generation` | Optional capability block describing the deployed checkpoint's advertised data-generation capacity. Absent on legacy checkpoints; present payloads expose `min_series`, `max_series`, `t_input`, `t_output`, `n_input`, and `n_output`. |

## Docstring Enforcement

Ruff is configured in `pyproject.toml` to require module, class, method, and function docstrings with `D100`, `D101`, `D102`, and `D103`. The repository pre-commit configuration runs `uv run ruff check .`, so `task pre-commit` fails if public documentation strings are missing.