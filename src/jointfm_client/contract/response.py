# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parsed forecast responses, one result type per return mode, checked against their request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from jointfm_client.exceptions import (
    UnsupportedModelVersionError,
    UnsupportedSchemaVersionError,
)
from jointfm_client.contract.frames import (
    _require_pandas_module,
    _tidy_frame,
    _wide_frame,
)
from jointfm_client.contract.outputs import (
    ConditionPlausibility,
    ForecastDiagnostics,
    ForecastOutputs,
    LogProbScores,
    QuantileForecast,
)
from jointfm_client.contract.request import (
    _require_quantiles,
)
from jointfm_client.contract.service_errors import (
    StructuredError,
    _raise_for_response_errors,
    _structured_error_tuple,
)
from jointfm_client.contract.values import (
    _optional_float,
    _optional_positive_int,
    _optional_string,
    _require_mapping,
    _require_member,
    _require_sequence,
    _require_string,
)
from jointfm_client.contract.vocabulary import (
    QueryMode,
    ReturnMode,
    SCHEMA_VERSION,
    SUPPORTED_QUERY_MODES,
    SUPPORTED_RETURN_MODES,
)


@dataclass(frozen=True, slots=True, eq=False)
class ForecastResponse:
    """Shared metadata preserved on every parsed forecast result.

    Results compare by identity: their values are NumPy arrays, which define
    no single truth value for ``==``. Compare the arrays themselves, for
    example with ``numpy.array_equal``.
    """

    schema_version: str
    image_version: str
    model_version: str
    checkpoint_version: str
    head: str
    query_mode: QueryMode
    return_mode: ReturnMode
    query_times: tuple[Any, ...]
    requested_columns: tuple[str, ...]
    diagnostics: ForecastDiagnostics
    errors: tuple[StructuredError, ...]
    plausibility: ConditionPlausibility | None

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy outputs view for compatibility with nested accessors."""
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        """Return the parsed forecast values as a NumPy array."""
        raise NotImplementedError

    def to_pandas_tidy(self) -> Any:
        """Return a tidy pandas DataFrame for the parsed forecast values."""
        raise NotImplementedError

    def to_pandas_wide(self) -> Any:
        """Return a wide pandas DataFrame for the parsed forecast values."""
        raise NotImplementedError

    @staticmethod
    def raise_for_errors(payload: Mapping[str, Any]) -> None:
        """Raise a typed SDK exception when one response carries JointFM errors."""
        _raise_for_response_errors(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        request_payload: Mapping[str, Any] | None = None,
    ) -> "ForecastResponse":
        """Parse one forecast response into the concrete result class for its mode."""
        _raise_for_response_errors(payload)

        expectations = _forecast_response_expectations(request_payload)
        schema_version = _require_string(
            payload.get("schema_version"), field="schema_version"
        )
        if schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                "Unsupported JointFM schema_version: "
                f"expected {SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        if (
            expectations.schema_version is not None
            and schema_version != expectations.schema_version
        ):
            raise ValueError(
                "forecast response schema_version mismatch: "
                f"expected {expectations.schema_version!r}, got {schema_version!r}"
            )

        model_version = _require_string(
            payload.get("model_version"), field="model_version"
        )
        if (
            expectations.model_version is not None
            and model_version != expectations.model_version
        ):
            raise UnsupportedModelVersionError(
                "Unsupported JointFM model_version: "
                f"expected {expectations.model_version!r}, got {model_version!r}"
            )

        query_mode = _require_member(
            payload.get("query_mode"),
            field="query_mode",
            supported_values=SUPPORTED_QUERY_MODES,
        )
        if (
            expectations.query_mode is not None
            and query_mode != expectations.query_mode
        ):
            raise ValueError(
                "forecast response query_mode mismatch: "
                f"expected {expectations.query_mode!r}, got {query_mode!r}"
            )

        return_mode = _require_member(
            payload.get("return_mode"),
            field="return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        )
        if (
            expectations.return_mode is not None
            and return_mode != expectations.return_mode
        ):
            raise ValueError(
                "forecast response return_mode mismatch: "
                f"expected {expectations.return_mode!r}, got {return_mode!r}"
            )

        outputs_payload = _require_mapping(payload.get("outputs"), field="outputs")
        outputs = ForecastOutputs.from_payload(
            outputs_payload,
            return_mode=return_mode,
            expected_horizon_count=expectations.horizon_count,
            expected_requested_column_count=expectations.requested_column_count,
            expected_sample_count=expectations.n_samples,
            expected_quantiles=expectations.quantiles,
            nullable_columns=expectations.nullable_columns,
        )
        diagnostics = ForecastDiagnostics.from_payload(
            _require_mapping(payload.get("diagnostics"), field="diagnostics")
        )
        if diagnostics.horizon_count != len(outputs.query_times):
            raise ValueError(
                "diagnostics.horizon_count mismatch: "
                f"expected {len(outputs.query_times)}, got {diagnostics.horizon_count}"
            )
        if (
            expectations.query_times is not None
            and outputs.query_times != expectations.query_times
        ):
            raise ValueError(
                "forecast response query_times mismatch: "
                f"expected {list(expectations.query_times)!r}, got {list(outputs.query_times)!r}"
            )
        if (
            expectations.requested_columns is not None
            and outputs.requested_columns != expectations.requested_columns
        ):
            raise ValueError(
                "forecast response requested_columns mismatch: "
                f"expected {list(expectations.requested_columns)!r}, "
                f"got {list(outputs.requested_columns)!r}"
            )
        if return_mode == "samples" and outputs.samples is not None:
            _validate_sample_bounds(
                outputs.samples,
                outputs.requested_columns,
                request_payload,
            )

        shared_fields = {
            "schema_version": schema_version,
            "image_version": _require_string(
                payload.get("image_version"), field="image_version"
            ),
            "model_version": model_version,
            "checkpoint_version": _require_string(
                payload.get("checkpoint_version"),
                field="checkpoint_version",
            ),
            "head": _require_string(payload.get("head"), field="head"),
            "query_mode": query_mode,
            "return_mode": return_mode,
            "query_times": outputs.query_times,
            "requested_columns": outputs.requested_columns,
            "diagnostics": diagnostics,
            "errors": _structured_error_tuple(payload.get("errors")),
            "plausibility": (
                None
                if payload.get("plausibility") is None
                else ConditionPlausibility.from_payload(
                    _require_mapping(payload.get("plausibility"), field="plausibility")
                )
            ),
        }

        if return_mode == "mean":
            assert outputs.mean is not None
            return MeanForecastResult(mean=outputs.mean, **shared_fields)
        if return_mode == "samples":
            assert outputs.samples is not None
            return SampleForecastResult(samples=outputs.samples, **shared_fields)
        if return_mode == "quantiles":
            assert outputs.quantiles is not None
            return QuantileForecastResult(quantiles=outputs.quantiles, **shared_fields)
        assert outputs.log_prob is not None
        return LogProbResult(log_prob=outputs.log_prob, **shared_fields)


@dataclass(frozen=True, slots=True, eq=False)
class MeanForecastResult(ForecastResponse):
    """Parsed mean forecast values with shared metadata and conversion helpers.

    ``mean`` is a read-only float64 array with axis order ``(horizon, column)``.
    """

    mean: np.ndarray

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for mean forecasts."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            mean=self.mean,
        )

    def to_numpy(self) -> np.ndarray:
        """Return the read-only values with axis order ``(horizon, column)``."""
        return self.mean

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``query_time``, ``requested_column``, and ``value``."""
        return _tidy_frame(
            self.mean[np.newaxis],
            query_times=self.query_times,
            requested_columns=self.requested_columns,
        )

    def to_pandas_wide(self) -> Any:
        """Return a wide DataFrame with one row per forecast horizon."""
        return _wide_frame(
            self.mean[np.newaxis],
            query_times=self.query_times,
            requested_columns=self.requested_columns,
        )


@dataclass(frozen=True, slots=True, eq=False)
class SampleForecastResult(ForecastResponse):
    """Parsed sampled forecast trajectories with shared metadata and conversions.

    ``samples`` is a read-only float64 array with axis order
    ``(sample, horizon, column)``.
    """

    samples: np.ndarray

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for sampled forecasts."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            samples=self.samples,
        )

    def to_numpy(self) -> np.ndarray:
        """Return the read-only values with axis order ``(sample, horizon, column)``."""
        return self.samples

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``sample``, ``query_time``, and one value per column."""
        return _tidy_frame(
            self.samples,
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            leading_name="sample",
            leading_labels=range(self.samples.shape[0]),
        )

    def to_pandas_wide(self) -> Any:
        """Return a wide DataFrame with one row per sample and forecast horizon."""
        return _wide_frame(
            self.samples,
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            leading_name="sample",
            leading_labels=range(self.samples.shape[0]),
        )


@dataclass(frozen=True, slots=True, eq=False)
class QuantileForecastResult(ForecastResponse):
    """Parsed quantile forecast surfaces with shared metadata and conversions."""

    quantiles: tuple[QuantileForecast, ...]

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for quantile forecasts."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            quantiles=self.quantiles,
        )

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Return the ordered quantile levels preserved from the service payload."""
        return tuple(entry.quantile for entry in self.quantiles)

    def to_numpy(self) -> np.ndarray:
        """Return a new array with axis order ``(quantile, horizon, column)``."""
        return np.stack([entry.values for entry in self.quantiles])

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``quantile``, ``query_time``, and one value per column."""
        return _tidy_frame(
            self.to_numpy(),
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            leading_name="quantile",
            leading_labels=self.quantile_levels,
        )

    def to_pandas_wide(self) -> Any:
        """Return a wide DataFrame with one row per quantile and forecast horizon."""
        return _wide_frame(
            self.to_numpy(),
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            leading_name="quantile",
            leading_labels=self.quantile_levels,
        )


@dataclass(frozen=True, slots=True, eq=False)
class LogProbResult(ForecastResponse):
    """Parsed log densities of observed values, with shared metadata.

    This result carries no forecast values, because a scored request supplies
    them itself as ``query_rows``: what comes back is how plausible the model
    finds them. Each score is joint over the scored columns, so the conversions
    have a horizon axis and no column axis.
    """

    log_prob: LogProbScores

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for scored requests."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            log_prob=self.log_prob,
        )

    def to_numpy(self) -> np.ndarray:
        """Return the read-only log densities with axis order ``(horizon,)``."""
        return self.log_prob.values

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``query_time``, ``log_prob``, and ``nll``."""
        pandas_module = _require_pandas_module()
        rows = [
            {"query_time": query_time, "log_prob": score, "nll": negated}
            for query_time, score, negated in zip(
                self.query_times,
                self.log_prob.values,
                self.log_prob.nll_values,
                strict=True,
            )
        ]
        return pandas_module.DataFrame.from_records(rows)

    def to_pandas_wide(self) -> Any:
        """Return the one-row request summary over the scored horizons.

        The other result classes widen the column axis, which a log density does
        not have. The wide view is therefore the request-level summary the
        service reports, and the per-horizon scores stay in the tidy view.
        """
        pandas_module = _require_pandas_module()
        return pandas_module.DataFrame.from_records(
            [
                {
                    "total": self.log_prob.total,
                    "mean": self.log_prob.mean,
                    "nll_total": self.log_prob.nll_total,
                    "nll_mean": self.log_prob.nll_mean,
                }
            ]
        )


@dataclass(frozen=True, slots=True)
class _ForecastResponseExpectations:
    schema_version: str | None = None
    model_version: str | None = None
    query_mode: QueryMode | None = None
    return_mode: ReturnMode | None = None
    query_times: tuple[Any, ...] | None = None
    horizon_count: int | None = None
    requested_columns: tuple[str, ...] | None = None
    requested_column_count: int | None = None
    n_samples: int | None = None
    quantiles: tuple[float, ...] | None = None
    nullable_columns: frozenset[str] = frozenset()


def _forecast_response_expectations(
    request_payload: Mapping[str, Any] | None,
) -> _ForecastResponseExpectations:
    if request_payload is None:
        return _ForecastResponseExpectations()
    if not isinstance(request_payload, Mapping):
        raise ValueError("request_payload must be a JSON object")

    query_times_value = request_payload.get("query_times")
    query_times = (
        tuple(_require_sequence(query_times_value, field="request_payload.query_times"))
        if query_times_value is not None
        else None
    )
    requested_columns: tuple[str, ...] | None = None
    requested_column_count: int | None = None
    requested_columns_value = request_payload.get("requested_columns")
    if requested_columns_value is not None:
        requested_column_values = _require_sequence(
            requested_columns_value,
            field="request_payload.requested_columns",
        )
        requested_column_count = len(requested_column_values)
        if all(
            isinstance(value, str) and value != "" for value in requested_column_values
        ):
            requested_columns = tuple(requested_column_values)

    query_mode_value = request_payload.get("query_mode")
    return_mode_value = request_payload.get("return_mode")
    quantiles_value = request_payload.get("quantiles")
    return _ForecastResponseExpectations(
        schema_version=_optional_string(
            request_payload.get("schema_version"),
            field="request_payload.schema_version",
        ),
        model_version=_optional_string(
            request_payload.get("model_version"),
            field="request_payload.model_version",
        ),
        query_mode=None
        if query_mode_value is None
        else _require_member(
            query_mode_value,
            field="request_payload.query_mode",
            supported_values=SUPPORTED_QUERY_MODES,
        ),
        return_mode=None
        if return_mode_value is None
        else _require_member(
            return_mode_value,
            field="request_payload.return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        ),
        query_times=query_times,
        horizon_count=None if query_times is None else len(query_times),
        requested_columns=requested_columns,
        requested_column_count=requested_column_count,
        n_samples=_optional_positive_int(
            request_payload.get("n_samples"),
            field="request_payload.n_samples",
        ),
        quantiles=None
        if quantiles_value is None
        else tuple(_require_quantiles(quantiles_value)),
        nullable_columns=_nullable_column_names(request_payload),
    )


def _nullable_column_names(request_payload: Mapping[str, Any]) -> frozenset[str]:
    """Return the names of the columns the request declares nullable."""
    columns_value = request_payload.get("columns")
    if columns_value is None:
        return frozenset()
    names: set[str] = set()
    for index, column in enumerate(
        _require_sequence(columns_value, field="request_payload.columns")
    ):
        mapping = _require_mapping(column, field=f"request_payload.columns[{index}]")
        if mapping.get("nullable") is True:
            names.add(
                _require_string(
                    mapping.get("name"), field=f"request_payload.columns[{index}].name"
                )
            )
    return frozenset(names)


def _validate_sample_bounds(
    samples: np.ndarray,
    requested_columns: Sequence[str],
    request_payload: Mapping[str, Any] | None,
) -> None:
    if request_payload is None:
        return

    bounds_by_column = _sample_bounds_by_column_name(request_payload)
    if not bounds_by_column:
        return

    for requested_column_index, requested_column in enumerate(requested_columns):
        bounds = bounds_by_column.get(requested_column)
        if bounds is None:
            continue
        lower_bound, upper_bound = bounds
        column_values = samples[:, :, requested_column_index]
        if lower_bound is not None:
            _raise_for_first_bound_violation(
                column_values < lower_bound,
                column_values,
                requested_column_index=requested_column_index,
                description=f"lower_bound for column {requested_column!r}",
                comparison="<",
                bound=lower_bound,
            )
        if upper_bound is not None:
            _raise_for_first_bound_violation(
                column_values > upper_bound,
                column_values,
                requested_column_index=requested_column_index,
                description=f"upper_bound for column {requested_column!r}",
                comparison=">",
                bound=upper_bound,
            )


def _raise_for_first_bound_violation(
    violations: np.ndarray,
    column_values: np.ndarray,
    *,
    requested_column_index: int,
    description: str,
    comparison: str,
    bound: float,
) -> None:
    """Raise for the first ``(sample, horizon)`` cell flagged in ``violations``."""
    if not violations.any():
        return
    sample_index, horizon_index = (int(index) for index in np.argwhere(violations)[0])
    value = float(column_values[sample_index, horizon_index])
    raise ValueError(
        f"outputs.samples[{sample_index}][{horizon_index}][{requested_column_index}] "
        f"violates requested {description}: {value} {comparison} {bound}"
    )


def _sample_bounds_by_column_name(
    request_payload: Mapping[str, Any],
) -> dict[str, tuple[float | None, float | None]]:
    columns_value = request_payload.get("columns")
    if columns_value is None:
        return {}

    columns = _require_sequence(columns_value, field="request_payload.columns")
    bounds_by_column: dict[str, tuple[float | None, float | None]] = {}
    for column_index, column_value in enumerate(columns):
        field = f"request_payload.columns[{column_index}]"
        column = _require_mapping(column_value, field=field)
        column_name = _require_string(column.get("name"), field=f"{field}.name")
        lower_bound = _optional_float(
            column.get("lower_bound"),
            field=f"{field}.lower_bound",
        )
        upper_bound = _optional_float(
            column.get("upper_bound"),
            field=f"{field}.upper_bound",
        )
        if lower_bound is None and upper_bound is None:
            continue
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError(
                f"{field}.lower_bound must be less than or equal to upper_bound"
            )
        bounds_by_column[column_name] = (lower_bound, upper_bound)
    return bounds_by_column
