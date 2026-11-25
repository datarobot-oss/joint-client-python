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

"""Column descriptors and the forecast request this SDK builds, validates, and serializes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any

from jointfm_client.contract.conditions import (
    Condition,
    EqualityCondition,
    _covers,
    resolve_conditions,
)
from jointfm_client.contract.values import (
    _optional_float,
    _optional_int,
    _optional_mapping,
    _optional_member,
    _optional_positive_float,
    _optional_positive_int,
    _optional_string,
    _require_bool,
    _require_mapping,
    _require_member,
    _require_sequence,
    _require_string,
)
from jointfm_client.contract.vocabulary import (
    ColumnModality,
    ColumnRole,
    DEFAULT_CALENDAR_ID,
    QueryMode,
    ReturnMode,
    SCHEMA_VERSION,
    SUPPORTED_COLUMN_MODALITIES,
    SUPPORTED_COLUMN_ROLES,
    SUPPORTED_QUERY_MODES,
    SUPPORTED_RETURN_MODES,
    SUPPORTED_TIME_INDEX_MODES,
    SUPPORTED_TIME_VALUE_KINDS,
    TimeIndexMode,
    TimeValueKind,
)


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """JSON-facing descriptor for one JointFM request column."""

    name: str
    modality: ColumnModality
    role: ColumnRole = "feature"
    nullable: bool = False
    vocabulary_size: int | None = None
    level_count: int | None = None
    mapping: Mapping[str | int, int] | None = None
    lower_bound: float | int | None = None
    upper_bound: float | int | None = None
    time_value_kind: TimeValueKind | None = None
    time_value_scale_seconds: float | int | None = None
    time_value_use_local_normalized_time: bool = False
    time_value_calendar_id: str = DEFAULT_CALENDAR_ID
    time_value_timezone: str | None = None
    traits: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate column metadata against the service contract.

        ``traits`` are expert hints, ``{trait_name: value}``. Only their shape
        is checked here; the deployment owns the trait vocabulary and refuses
        unknown names, values outside a trait's value space, and traits that do
        not apply to the column's modality.
        """
        _require_string(self.name, field="columns.name")
        _require_member(
            self.modality,
            field="columns.modality",
            supported_values=SUPPORTED_COLUMN_MODALITIES,
        )
        _require_member(
            self.role,
            field="columns.role",
            supported_values=SUPPORTED_COLUMN_ROLES,
        )
        _require_bool(self.nullable, field="columns.nullable")
        _optional_positive_int(self.vocabulary_size, field="columns.vocabulary_size")
        _optional_positive_int(self.level_count, field="columns.level_count")
        _optional_mapping(self.mapping, field="columns.mapping")
        lower_bound = _optional_float(self.lower_bound, field="columns.lower_bound")
        upper_bound = _optional_float(self.upper_bound, field="columns.upper_bound")
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError("columns.lower_bound must not exceed columns.upper_bound")

        _optional_member(
            self.time_value_kind,
            field="columns.time_value_kind",
            supported_values=SUPPORTED_TIME_VALUE_KINDS,
        )
        _optional_positive_float(
            self.time_value_scale_seconds,
            field="columns.time_value_scale_seconds",
        )
        _require_bool(
            self.time_value_use_local_normalized_time,
            field="columns.time_value_use_local_normalized_time",
        )
        _require_string(
            self.time_value_calendar_id,
            field="columns.time_value_calendar_id",
        )
        _optional_string(self.time_value_timezone, field="columns.time_value_timezone")
        _validate_time_value_options(self)
        if self.traits is not None:
            for trait_name, trait_value in _require_mapping(
                self.traits, field="columns.traits"
            ).items():
                _require_string(trait_name, field="columns.traits name")
                _require_string(trait_value, field=f"columns.traits.{trait_name}")

    def to_payload(self) -> dict[str, Any]:
        """Return this column descriptor as a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "name": self.name,
            "modality": self.modality,
        }
        if self.role != "feature":
            payload["role"] = self.role
        if self.nullable:
            payload["nullable"] = self.nullable
        if self.vocabulary_size is not None:
            payload["vocabulary_size"] = self.vocabulary_size
        if self.level_count is not None:
            payload["level_count"] = self.level_count
        if self.mapping is not None:
            payload["mapping"] = dict(self.mapping)
        if self.lower_bound is not None:
            payload["lower_bound"] = _optional_float(
                self.lower_bound,
                field="columns.lower_bound",
            )
        if self.upper_bound is not None:
            payload["upper_bound"] = _optional_float(
                self.upper_bound,
                field="columns.upper_bound",
            )
        if self.time_value_kind is not None:
            payload["time_value_kind"] = self.time_value_kind
        if self.time_value_scale_seconds is not None:
            payload["time_value_scale_seconds"] = _optional_float(
                self.time_value_scale_seconds,
                field="columns.time_value_scale_seconds",
            )
        if self.time_value_use_local_normalized_time:
            payload["time_value_use_local_normalized_time"] = True
        if self.time_value_calendar_id != DEFAULT_CALENDAR_ID:
            payload["time_value_calendar_id"] = self.time_value_calendar_id
        if self.time_value_timezone is not None:
            payload["time_value_timezone"] = self.time_value_timezone
        if self.traits is not None:
            payload["traits"] = dict(self.traits)
        return payload


@dataclass(frozen=True, slots=True)
class DataFrameSchema:
    """JSON-facing schema for the tabular history in one forecast request."""

    columns: Sequence[ColumnSpec]
    time_index_mode: TimeIndexMode
    time_column: str | None = None
    time_scale_seconds: float | int | None = None
    use_local_normalized_time: bool = False
    calendar_id: str = DEFAULT_CALENDAR_ID
    timezone: str | None = None

    def __post_init__(self) -> None:
        """Validate schema fields that span multiple columns."""
        columns = _require_sequence(self.columns, field="columns")
        for index, column in enumerate(columns):
            if not isinstance(column, ColumnSpec):
                raise ValueError(f"columns[{index}] must be a ColumnSpec")
        _require_member(
            self.time_index_mode,
            field="time_index_mode",
            supported_values=SUPPORTED_TIME_INDEX_MODES,
        )
        _optional_string(self.time_column, field="time_column")
        _optional_positive_float(self.time_scale_seconds, field="time_scale_seconds")
        _require_bool(
            self.use_local_normalized_time,
            field="use_local_normalized_time",
        )
        _require_string(self.calendar_id, field="calendar_id")
        _optional_string(self.timezone, field="timezone")

        column_names = [column.name for column in columns]
        if len(set(column_names)) != len(column_names):
            raise ValueError("columns must not contain duplicate names")
        if self.time_column is not None and self.time_column in set(column_names):
            raise ValueError("time_column must not also appear in columns")
        if self.time_index_mode == "absolute_datetime" and self.time_column is None:
            raise ValueError("absolute_datetime requests must provide time_column")

    def to_payload(self) -> dict[str, Any]:
        """Return this schema as the JSON fields expected by `/predict`."""
        payload: dict[str, Any] = {
            "time_index_mode": self.time_index_mode,
            "columns": [column.to_payload() for column in self.columns],
        }
        if self.time_column is not None:
            payload["time_column"] = self.time_column
        if self.time_scale_seconds is not None:
            payload["time_scale_seconds"] = _optional_float(
                self.time_scale_seconds,
                field="time_scale_seconds",
            )
        if self.use_local_normalized_time:
            payload["use_local_normalized_time"] = True
        if self.calendar_id != DEFAULT_CALENDAR_ID:
            payload["calendar_id"] = self.calendar_id
        if self.timezone is not None:
            payload["timezone"] = self.timezone
        return payload


@dataclass(frozen=True, slots=True)
class ForecastRequestMetadata:
    """Version and mode metadata for one forecast request."""

    model_version: str
    schema_version: str = SCHEMA_VERSION
    query_mode: QueryMode = "forecast"
    return_mode: ReturnMode = "mean"

    def __post_init__(self) -> None:
        """Validate the request metadata against the supported SDK contract."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        _require_string(self.model_version, field="model_version")
        _require_member(
            self.query_mode,
            field="query_mode",
            supported_values=SUPPORTED_QUERY_MODES,
        )
        _require_member(
            self.return_mode,
            field="return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return request metadata as JSON-compatible payload fields."""
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "query_mode": self.query_mode,
            "return_mode": self.return_mode,
        }


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    """Validated forecast request that can build a `/predict` payload."""

    metadata: ForecastRequestMetadata
    schema: DataFrameSchema
    history_rows: Sequence[Mapping[str, Any]]
    query_times: Sequence[Any]
    requested_columns: Sequence[str | int] | None = None
    n_samples: int | None = None
    quantiles: Sequence[float | int] | None = None
    seed: int | None = None
    query_row_ids: Sequence[int] | None = None
    condition: Condition | Sequence[Condition] | None = None
    query_rows: Sequence[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Validate payload controls and JSON-facing request arrays."""
        if not isinstance(self.metadata, ForecastRequestMetadata):
            raise ValueError("metadata must be a ForecastRequestMetadata")
        if not isinstance(self.schema, DataFrameSchema):
            raise ValueError("schema must be a DataFrameSchema")
        if self.query_row_ids is not None:
            raise ValueError("query_row_ids is not supported for forecast requests")

        history_rows = _require_sequence(self.history_rows, field="history_rows")
        for index, history_row in enumerate(history_rows):
            _require_mapping(history_row, field=f"history_rows[{index}]")
        _validate_history_declared_columns(history_rows, self.schema)
        _serialize_query_times(
            self.query_times, time_index_mode=self.schema.time_index_mode
        )
        _resolve_requested_columns(self.schema.columns, self.requested_columns)
        _optional_positive_int(self.n_samples, field="n_samples")
        _optional_int(self.seed, field="seed")

        if (self.metadata.query_mode == "condition") != (self.condition is not None):
            raise ValueError(
                "query_mode='condition' and a condition go together: one "
                "without the other means a different request than the caller wrote"
            )
        if self.condition is not None:
            self._validate_condition()

        if self.metadata.return_mode == "quantiles":
            _require_quantiles(self.quantiles)
        elif self.quantiles is not None:
            raise ValueError(
                "quantiles may be provided only when return_mode='quantiles'"
            )

        if (self.metadata.return_mode == "log_prob") != (self.query_rows is not None):
            raise ValueError(
                "return_mode='log_prob' and query_rows go together: scoring needs "
                "the observed values, and no other return mode reads them"
            )
        if self.query_rows is not None:
            self._validate_query_rows()

    def _validate_condition(self) -> None:
        """Check the conditions against this request's schema and positions.

        The deployment rejects all of this too, before executing anything; the
        point of repeating it here is that a caller learns which column name or
        position it got wrong without paying for a round trip.
        """
        assert self.condition is not None
        conditions = resolve_conditions(self.condition)
        declared = {column.name for column in self.schema.columns}
        unknown = [
            condition.column
            for condition in conditions
            if condition.column not in declared
        ]
        if unknown:
            raise ValueError(f"condition references undeclared columns {unknown}")
        horizon_count = len(_require_sequence(self.query_times, field="query_times"))
        for condition in conditions:
            outside = [
                index
                for index in condition.query_time_indices or ()
                if index >= horizon_count
            ]
            if outside:
                raise ValueError(
                    f"condition on {condition.column!r} names query_time_indices "
                    f"{outside} outside the {horizon_count} requested future positions"
                )
        for position in range(horizon_count):
            covering = [
                condition for condition in conditions if _covers(condition, position)
            ]
            if len(covering) < len(declared):
                continue
            if all(isinstance(condition, EqualityCondition) for condition in covering):
                raise ValueError(
                    f"pinning every column at query_time_indices position {position} "
                    "leaves nothing to read out; the joint density of those values "
                    "is what return_mode='log_prob' answers"
                )
            raise ValueError(
                "a request must leave at least one column unconditioned at "
                f"query_time_indices position {position}: what it reads out is "
                "the conditional distribution of the columns it does not condition"
            )

    def _validate_query_rows(self) -> None:
        """Check the observed rows a log-density request scores.

        The service scores the whole joint at each future position, so a row
        must carry every declared column and ``requested_columns`` must name
        every one of them in declared order: a narrower or reordered projection
        would score a different distribution than the caller believes it asked
        about. A ``condition`` changes nothing here — its columns are
        declared columns like any other — and omitting the field already
        resolves to exactly this projection.

        Only two per-value failures are decidable without the deployment's own
        encoding and are checked here — a missing value and a non-finite number.
        A categorical label passes, because the service maps it through the
        column's declared mapping before scoring it.
        """
        rows = _require_sequence(self.query_rows, field="query_rows")
        query_times = _require_sequence(self.query_times, field="query_times")
        if len(rows) != len(query_times):
            raise ValueError(
                f"query_rows must carry one row per query time: got {len(rows)} "
                f"rows for {len(query_times)} query_times"
            )
        declared = [column.name for column in self.schema.columns]
        required = (
            declared
            if self.schema.time_column is None
            else [
                *declared,
                self.schema.time_column,
            ]
        )
        for index, row in enumerate(rows):
            row_mapping = _require_mapping(row, field=f"query_rows[{index}]")
            missing = [name for name in required if name not in row_mapping]
            if missing:
                raise ValueError(
                    f"query_rows[{index}] is missing declared columns: {missing}"
                )
            for name in declared:
                _require_scored_value(
                    row_mapping[name], field=f"query_rows[{index}].{name}"
                )

        requested = _resolve_requested_columns(
            self.schema.columns, self.requested_columns
        )
        if requested is None:
            return
        if requested != declared:
            raise ValueError(
                "return_mode='log_prob' scores the whole joint, so "
                "requested_columns must list every declared column in declared "
                f"order: expected {declared}, got {requested}"
            )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible forecast request without mutating inputs."""
        payload = self.metadata.to_payload()
        payload.update(self.schema.to_payload())
        payload["history_rows"] = _serialize_rows(
            self.history_rows, field="history_rows"
        )
        payload["query_times"] = _serialize_query_times(
            self.query_times,
            time_index_mode=self.schema.time_index_mode,
        )
        requested_columns = _resolve_requested_columns(
            self.schema.columns,
            self.requested_columns,
        )
        if requested_columns is not None:
            payload["requested_columns"] = requested_columns
        if self.n_samples is not None:
            payload["n_samples"] = self.n_samples
        if self.quantiles is not None:
            payload["quantiles"] = _require_quantiles(self.quantiles)
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.condition is not None:
            payload["conditions"] = [
                condition.to_payload()
                for condition in resolve_conditions(self.condition)
            ]
        if self.query_rows is not None:
            payload["query_rows"] = _serialize_rows(self.query_rows, field="query_rows")
        return payload


def build_forecast_payload(
    *,
    model_version: str,
    schema: DataFrameSchema,
    history_rows: Sequence[Mapping[str, Any]],
    query_times: Sequence[Any],
    requested_columns: Sequence[str | int] | None = None,
    return_mode: ReturnMode = "mean",
    n_samples: int | None = None,
    quantiles: Sequence[float | int] | None = None,
    seed: int | None = None,
    schema_version: str = SCHEMA_VERSION,
    query_mode: QueryMode = "forecast",
    condition: Condition | Sequence[Condition] | None = None,
    query_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a validated JSON-compatible forecast request payload."""
    return ForecastRequest(
        metadata=ForecastRequestMetadata(
            model_version=model_version,
            schema_version=schema_version,
            query_mode=query_mode,
            return_mode=return_mode,
        ),
        schema=schema,
        history_rows=history_rows,
        query_times=query_times,
        requested_columns=requested_columns,
        n_samples=n_samples,
        quantiles=quantiles,
        seed=seed,
        condition=condition,
        query_rows=query_rows,
    ).to_payload()


def _validate_time_value_options(column_spec: ColumnSpec) -> None:
    uses_time_value_options = any(
        [
            column_spec.time_value_kind is not None,
            column_spec.time_value_scale_seconds is not None,
            column_spec.time_value_use_local_normalized_time,
            column_spec.time_value_calendar_id != DEFAULT_CALENDAR_ID,
            column_spec.time_value_timezone is not None,
        ]
    )
    if column_spec.modality != "time_value":
        if uses_time_value_options:
            raise ValueError("time_value options require modality='time_value'")
        return
    if column_spec.time_value_kind != "absolute_datetime" and any(
        [
            column_spec.time_value_scale_seconds is not None,
            column_spec.time_value_use_local_normalized_time,
            column_spec.time_value_calendar_id != DEFAULT_CALENDAR_ID,
            column_spec.time_value_timezone is not None,
        ]
    ):
        raise ValueError(
            "time_value absolute datetime options require time_value_kind='absolute_datetime'"
        )


def _validate_history_declared_columns(
    history_rows: Sequence[Mapping[str, Any]],
    schema: DataFrameSchema,
) -> None:
    present_columns: set[str] = set()
    for history_row in history_rows:
        present_columns.update(history_row.keys())
    expected_columns = {column.name for column in schema.columns}
    missing_columns = sorted(expected_columns.difference(present_columns))
    if missing_columns:
        raise ValueError(
            f"history_rows are missing declared columns: {missing_columns}"
        )
    if schema.time_column is not None and schema.time_column not in present_columns:
        raise ValueError(f"history_rows are missing time_column {schema.time_column!r}")


def _serialize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    return [
        {
            key: _to_json_compatible(value, field=f"{field}[{index}].{key}")
            for key, value in row.items()
        }
        for index, row in enumerate(rows)
    ]


def _require_scored_value(value: Any, *, field: str) -> None:
    if value is None:
        raise ValueError(f"{field} must carry an observed value to score")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


def _serialize_query_times(
    query_times: Sequence[Any],
    *,
    time_index_mode: TimeIndexMode,
) -> list[Any]:
    values = _require_sequence(query_times, field="query_times")
    if time_index_mode == "absolute_datetime":
        return [
            _serialize_absolute_datetime(value, field=f"query_times[{index}]")
            for index, value in enumerate(values)
        ]
    if time_index_mode == "ordinal":
        serialized: list[int] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"query_times[{index}] must be an integer for ordinal mode"
                )
            serialized.append(value)
        return serialized

    serialized_float_times: list[int | float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"query_times[{index}] must be numeric for continuous_float mode"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"query_times[{index}] must be finite")
        serialized_float_times.append(value)
    return serialized_float_times


def _serialize_absolute_datetime(value: Any, *, field: str) -> str:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        normalized_value = (
            value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
        )
        try:
            timestamp = datetime.fromisoformat(normalized_value)
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO 8601 datetime") from error
    else:
        raise ValueError(
            f"{field} must be a timezone-aware datetime or ISO 8601 string"
        )
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must include timezone information")
    return timestamp.astimezone(timezone.utc).isoformat()


def _resolve_requested_columns(
    column_specs: Sequence[ColumnSpec],
    requested_columns: Sequence[str | int] | None,
) -> list[str] | None:
    if requested_columns is None:
        return None
    requested_values = _require_sequence(requested_columns, field="requested_columns")
    column_names = [column.name for column in column_specs]
    resolved_names: list[str] = []
    for index, requested_column in enumerate(requested_values):
        if isinstance(requested_column, str):
            if requested_column not in column_names:
                raise ValueError(
                    f"requested_columns[{index}] references unknown column {requested_column!r}"
                )
            resolved_names.append(requested_column)
            continue
        if isinstance(requested_column, bool) or not isinstance(requested_column, int):
            raise ValueError(
                f"requested_columns[{index}] must be a string name or integer column index"
            )
        if requested_column < 0 or requested_column >= len(column_names):
            raise ValueError(
                f"requested_columns[{index}] index {requested_column} is out of bounds"
            )
        resolved_names.append(column_names[requested_column])
    if len(set(resolved_names)) != len(resolved_names):
        raise ValueError("requested_columns must not contain duplicates")
    return resolved_names


def _require_quantiles(value: Sequence[float | int] | None) -> list[float]:
    quantiles = _require_sequence(value, field="quantiles")
    parsed: list[float] = []
    for index, quantile in enumerate(quantiles):
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise ValueError(f"quantiles[{index}] must be numeric")
        parsed_quantile = float(quantile)
        if parsed_quantile <= 0.0 or parsed_quantile >= 1.0:
            raise ValueError(f"quantiles[{index}] must lie strictly between 0 and 1")
        parsed.append(parsed_quantile)
    return parsed


def _to_json_compatible(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    if isinstance(value, datetime):
        return _serialize_absolute_datetime(value, field=field)
    if isinstance(value, Mapping):
        return {
            _require_string(key, field=f"{field}.key"): _to_json_compatible(
                nested_value,
                field=f"{field}.{key}",
            )
            for key, nested_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _to_json_compatible(nested_value, field=f"{field}[]")
            for nested_value in value
        ]
    raise ValueError(f"{field} must be JSON-compatible")
