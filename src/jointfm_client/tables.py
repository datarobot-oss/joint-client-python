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

"""Table requests and answers: context rows, query rows, and requested cells.

A table has no time axis. Its request lists the table's context rows in
``history_rows`` and its query rows in ``query_rows``, every row holding every
declared column, and names the cells to predict in ``requested_cells``: per
row list, a mapping from a declared column name to the 0-based indices of the
rows whose cell in that column is requested, such as
``{"history_rows": {"price": [1]}, "query_rows": {"price": [0, 1]}}``. Any
cell can be requested, whatever its column's ``role``: a gap in a context row,
and any cells of a query row, a feature column's included. Every query row
requests at least one cell. A requested cell holds ``None``, except under
``return_mode="log_prob"``, which scores the value it holds.

The deployment reads each query row against the context rows and its own
cells alone, so a query row's answer does not depend on which other query
rows share the request. The answer covers the rows that carry a requested
cell, context rows first, and the columns that hold one, in declared order;
every cell the request did not ask for is ``null`` on the wire and NaN once
parsed.

A deployment answers tables when its health metadata lists
``TABLE_TIME_INDEX_MODE`` in ``supported_time_index_modes``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final, Self, TypeAlias

import numpy as np

from jointfm_client.adapters import (
    ColumnBounds,
    ColumnMapping,
    _column_value_to_json,
    _is_missing,
    _require_column_specs,
    _require_numpy,
    _require_pandas,
    infer_column_specs_from_dataframe,
)
from jointfm_client.configuration import DEFAULT_FORECAST_SCHEMA_VERSION
from jointfm_client.contract import (
    SCHEMA_VERSION,
    SUPPORTED_RETURN_MODES,
    TABLE_TIME_INDEX_MODE,
    ColumnRole,
    ColumnSpec,
    ForecastDiagnostics,
    ForecastRequestMetadata,
    HealthMetadata,
    LogProbScores,
    QuantileForecast,
    ReturnMode,
    StructuredError,
    _label_column,
    _optional_int,
    _optional_positive_int,
    _raise_for_response_errors,
    _require_float_array,
    _require_mapping,
    _require_member,
    _require_none,
    _require_pandas_module,
    _require_quantile_forecasts,
    _require_quantiles,
    _require_scored_value,
    _require_sequence,
    _require_string,
    _serialize_rows,
    _string_tuple,
    _structured_error_tuple,
)
from jointfm_client.exceptions import (
    UnsupportedModelVersionError,
    UnsupportedSchemaVersionError,
    UnsupportedServiceContractError,
)

# The request's two row lists, in the order the deployment stacks them.
TABLE_ROW_LISTS: Final[tuple[str, ...]] = ("history_rows", "query_rows")

RequestedCells: TypeAlias = Mapping[str, Mapping[str, Sequence[int]]]


def require_table_support(metadata: HealthMetadata) -> None:
    """Refuse a table request the mounted deployment has not advertised.

    Raises:
        UnsupportedServiceContractError: when ``supported_time_index_modes``
            lacks ``TABLE_TIME_INDEX_MODE``, as on every JointFM 1 deployment.
    """
    if TABLE_TIME_INDEX_MODE not in metadata.supported_time_index_modes:
        raise UnsupportedServiceContractError(
            f"The mounted {metadata.head!r} deployment does not answer table "
            f"requests (time_index_mode {TABLE_TIME_INDEX_MODE!r}); it advertises "
            f"time index modes {list(metadata.supported_time_index_modes)}"
        )


def _requested_cell_masks(
    requested_cells: RequestedCells,
    *,
    column_names: Sequence[str],
    row_counts: Mapping[str, int],
) -> dict[str, np.ndarray]:
    """Parse ``requested_cells`` into one boolean ``(row, column)`` mask per row list.

    Raises:
        ValueError: for a key that is not a row list, an undeclared column, an
            empty or duplicated index list, an index outside its row list, a
            request of no cell at all, or a query row that requests no cell.
    """
    parts = _require_mapping(requested_cells, field="requested_cells")
    unknown = sorted(set(map(str, parts)) - set(TABLE_ROW_LISTS))
    if unknown:
        raise ValueError(
            f"requested_cells holds {unknown}; its keys are row lists among "
            f"{list(TABLE_ROW_LISTS)}"
        )
    masks = {
        part: np.zeros((row_counts[part], len(column_names)), dtype=bool)
        for part in TABLE_ROW_LISTS
    }
    for part, raw_cells in parts.items():
        field = f"requested_cells.{part}"
        for column, raw_indices in _require_mapping(raw_cells, field=field).items():
            if column not in column_names:
                raise ValueError(f"{field} names undeclared column {column!r}")
            column_field = f"{field}.{column}"
            indices = _require_sequence(raw_indices, field=column_field)
            for index in indices:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError(f"{column_field} must hold integer row indices")
                if not 0 <= index < row_counts[part]:
                    raise ValueError(
                        f"{column_field} index {index} lies outside the "
                        f"{row_counts[part]} rows of {part}"
                    )
            if len(set(indices)) != len(indices):
                raise ValueError(f"{column_field} must not repeat a row index")
            masks[part][list(indices), column_names.index(column)] = True
    if not any(mask.any() for mask in masks.values()):
        raise ValueError("requested_cells requests no cell")
    silent_rows = np.flatnonzero(~masks["query_rows"].any(axis=1))
    if silent_rows.size:
        raise ValueError(
            f"query_rows[{int(silent_rows[0])}] requests no cell; every query row "
            "must request at least one"
        )
    return masks


def _require_table_columns(columns: Sequence[ColumnSpec]) -> tuple[ColumnSpec, ...]:
    """Return the declared columns, refusing an empty list or a repeated name."""
    column_specs = _require_column_specs(columns)
    names = [column.name for column in column_specs]
    if len(set(names)) != len(names):
        raise ValueError("columns must not contain duplicate names")
    return column_specs


def _require_table_rows(
    rows: Sequence[Mapping[str, Any]], *, field: str, column_names: Sequence[str]
) -> Sequence[Mapping[str, Any]]:
    """Require one non-empty row list whose rows hold exactly the declared columns."""
    sequence = _require_sequence(rows, field=field)
    declared = set(column_names)
    for index, row in enumerate(sequence):
        keys = set(map(str, _require_mapping(row, field=f"{field}[{index}]")))
        missing = [name for name in column_names if name not in keys]
        if missing:
            raise ValueError(f"{field}[{index}] is missing declared columns: {missing}")
        undeclared = sorted(keys - declared)
        if undeclared:
            raise ValueError(f"{field}[{index}] holds undeclared columns {undeclared}")
    return sequence


@dataclass(frozen=True, slots=True)
class TableRequest:
    """Validated table request that can build a `/predict` payload.

    ``history_rows`` holds the context rows and ``query_rows`` the query rows,
    each row a mapping from every declared column name to its value.
    ``requested_cells`` names the cells to predict, as the module docstring
    describes. A table answers ``query_mode="forecast"`` alone, since it takes
    no conditions, and every return mode.
    """

    metadata: ForecastRequestMetadata
    columns: Sequence[ColumnSpec]
    history_rows: Sequence[Mapping[str, Any]]
    query_rows: Sequence[Mapping[str, Any]]
    requested_cells: RequestedCells
    n_samples: int | None = None
    quantiles: Sequence[float | int] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the rows, the requested cells, and the sampling controls.

        The deployment refuses all of this too, before any model work; the
        point of repeating it is that a caller learns which row or cell is
        wrong without paying for a round trip.
        """
        if not isinstance(self.metadata, ForecastRequestMetadata):
            raise ValueError("metadata must be a ForecastRequestMetadata")
        if self.metadata.query_mode != "forecast":
            raise ValueError(
                "a table request takes no conditions, so its query_mode is "
                f"'forecast'; got {self.metadata.query_mode!r}"
            )
        names = [column.name for column in _require_table_columns(self.columns)]
        rows = {
            "history_rows": _require_table_rows(
                self.history_rows, field="history_rows", column_names=names
            ),
            "query_rows": _require_table_rows(
                self.query_rows, field="query_rows", column_names=names
            ),
        }
        masks = _requested_cell_masks(
            self.requested_cells,
            column_names=names,
            row_counts={part: len(part_rows) for part, part_rows in rows.items()},
        )
        scoring = self.metadata.return_mode == "log_prob"
        for part, mask in masks.items():
            for row_index, column_index in np.argwhere(mask):
                name = names[column_index]
                value = rows[part][row_index][name]
                field = f"{part}[{row_index}].{name}"
                if scoring:
                    _require_scored_value(value, field=field)
                elif value is not None:
                    raise ValueError(
                        f"{field} is a requested cell, so it holds None unless "
                        "return_mode='log_prob' scores its value"
                    )
        _optional_positive_int(self.n_samples, field="n_samples")
        _optional_int(self.seed, field="seed")
        if self.metadata.return_mode == "quantiles":
            _require_quantiles(self.quantiles)
        elif self.quantiles is not None:
            raise ValueError(
                "quantiles may be provided only when return_mode='quantiles'"
            )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible table request without mutating inputs.

        ``requested_cells`` is written per row list in declared column order
        with ascending indices, leaving out a row list that requests no cell.
        """
        names = [column.name for column in self.columns]
        masks = _requested_cell_masks(
            self.requested_cells,
            column_names=names,
            row_counts={
                "history_rows": len(self.history_rows),
                "query_rows": len(self.query_rows),
            },
        )
        payload = self.metadata.to_payload()
        payload["time_index_mode"] = TABLE_TIME_INDEX_MODE
        payload["columns"] = [column.to_payload() for column in self.columns]
        payload["history_rows"] = _serialize_rows(
            self.history_rows, field="history_rows"
        )
        payload["query_rows"] = _serialize_rows(self.query_rows, field="query_rows")
        payload["requested_cells"] = {
            part: {
                name: np.flatnonzero(mask[:, column_index]).tolist()
                for column_index, name in enumerate(names)
                if mask[:, column_index].any()
            }
            for part, mask in masks.items()
            if mask.any()
        }
        if self.n_samples is not None:
            payload["n_samples"] = self.n_samples
        if self.quantiles is not None:
            payload["quantiles"] = _require_quantiles(self.quantiles)
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


def build_table_payload(
    *,
    model_version: str,
    columns: Sequence[ColumnSpec],
    history_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    requested_cells: RequestedCells,
    return_mode: ReturnMode = "mean",
    n_samples: int | None = None,
    quantiles: Sequence[float | int] | None = None,
    seed: int | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a validated JSON-compatible table request payload."""
    return TableRequest(
        metadata=ForecastRequestMetadata(
            model_version=model_version,
            schema_version=schema_version,
            return_mode=return_mode,
        ),
        columns=columns,
        history_rows=history_rows,
        query_rows=query_rows,
        requested_cells=requested_cells,
        n_samples=n_samples,
        quantiles=quantiles,
        seed=seed,
    ).to_payload()


def build_table_payload_from_dataframe(
    history: Any,
    query_rows: Any,
    *,
    model_version: str,
    requested_cells: RequestedCells,
    columns: Sequence[ColumnSpec] | None = None,
    return_mode: ReturnMode = "mean",
    n_samples: int | None = None,
    quantiles: Sequence[float | int] | None = None,
    seed: int | None = None,
    schema_version: str = DEFAULT_FORECAST_SCHEMA_VERSION,
    target_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    column_roles: Mapping[str, ColumnRole] | None = None,
    categorical_columns: Sequence[str] | None = None,
    categorical_mappings: ColumnMapping | None = None,
    ordinal_columns: Sequence[str] | None = None,
    ordinal_mappings: ColumnMapping | None = None,
    count_columns: Sequence[str] | None = None,
    nullable_columns: Sequence[str] | None = None,
    bounds: ColumnBounds | None = None,
) -> dict[str, Any]:
    """Build a validated table payload from two pandas ``DataFrame`` objects.

    ``history`` holds the context rows and ``query_rows`` the query rows;
    ``requested_cells`` indexes each by row position, not by index label.
    Without ``columns`` the column descriptors are inferred from both frames
    together, so a categorical mapping covers every label either holds, and a
    column is nullable only when listed in ``nullable_columns`` or missing a
    value in a cell the request does not ask for. A requested cell is sent as
    ``None`` when the frame leaves it missing; outside
    ``return_mode="log_prob"`` a requested cell that holds a value is refused.
    """
    pandas_module = _require_pandas()
    for field, frame in (("history", history), ("query_rows", query_rows)):
        if not isinstance(frame, pandas_module.DataFrame):
            raise ValueError(f"{field} must be a pandas DataFrame")
    row_counts = {"history_rows": len(history), "query_rows": len(query_rows)}
    if columns is None:
        # A query frame whose requested column holds only None has object
        # dtype, which would otherwise read as a categorical column.
        combined = pandas_module.concat(
            [history, query_rows], ignore_index=True
        ).infer_objects()
        inferred = infer_column_specs_from_dataframe(
            combined,
            target_columns=target_columns,
            feature_columns=feature_columns,
            column_roles=column_roles,
            categorical_columns=categorical_columns,
            categorical_mappings=categorical_mappings,
            ordinal_columns=ordinal_columns,
            ordinal_mappings=ordinal_mappings,
            count_columns=count_columns,
            nullable_columns=nullable_columns,
            bounds=bounds,
        )
        names = [column.name for column in inferred]
        masks = _requested_cell_masks(
            requested_cells, column_names=names, row_counts=row_counts
        )
        # A requested cell is missing because the model predicts it, which
        # says nothing about whether the column can hold a missing value.
        unrequested_gaps = combined.loc[:, names].isna().to_numpy() & ~np.concatenate(
            [masks[part] for part in TABLE_ROW_LISTS]
        )
        listed = set(nullable_columns or ())
        column_specs = tuple(
            replace(
                column,
                nullable=column.name in listed
                or bool(unrequested_gaps[:, column_index].any()),
            )
            for column_index, column in enumerate(inferred)
        )
    else:
        column_specs = _require_table_columns(columns)
        masks = _requested_cell_masks(
            requested_cells,
            column_names=[column.name for column in column_specs],
            row_counts=row_counts,
        )
    return build_table_payload(
        model_version=model_version,
        columns=column_specs,
        history_rows=_table_rows_from_dataframe(
            history, column_specs, masks["history_rows"], field="history_rows"
        ),
        query_rows=_table_rows_from_dataframe(
            query_rows, column_specs, masks["query_rows"], field="query_rows"
        ),
        requested_cells=requested_cells,
        return_mode=return_mode,
        n_samples=n_samples,
        quantiles=quantiles,
        seed=seed,
        schema_version=schema_version,
    )


def _table_rows_from_dataframe(
    frame: Any,
    column_specs: Sequence[ColumnSpec],
    requested: np.ndarray,
    *,
    field: str,
) -> list[dict[str, Any]]:
    """Convert one frame into row payloads, sending a missing requested cell as ``None``.

    Every other cell goes through the forecast adapter's column encoding, so a
    categorical label or a timestamp is spelled as in a forecast request.
    """
    pandas_module = _require_pandas()
    numpy_module = _require_numpy()
    if len(frame) == 0:
        raise ValueError(f"{field} must contain at least one row")
    names = [column.name for column in column_specs]
    missing_columns = sorted(set(names).difference(map(str, frame.columns)))
    if missing_columns:
        raise ValueError(
            f"{field} frame is missing declared columns: {missing_columns}"
        )
    rows: list[dict[str, Any]] = []
    for row_index, values in enumerate(
        frame.loc[:, names].itertuples(index=False, name=None)
    ):
        row: dict[str, Any] = {}
        for column_index, (column, value) in enumerate(
            zip(column_specs, values, strict=True)
        ):
            if requested[row_index, column_index] and _is_missing(
                value, pandas_module=pandas_module
            ):
                row[column.name] = None
                continue
            row[column.name] = _column_value_to_json(
                value,
                column_spec=column,
                field=f"{field}[{row_index}].{column.name}",
                pandas_module=pandas_module,
                numpy_module=numpy_module,
            )
        rows.append(row)
    return rows


@dataclass(frozen=True, slots=True)
class TableRows:
    """The rows a table answer covers, as indices into the request's row lists.

    The answer stacks ``history_rows`` (context rows) before ``query_rows``,
    each in ascending order, and holds exactly the rows that carry a requested
    cell.
    """

    history_rows: tuple[int, ...]
    query_rows: tuple[int, ...]

    @property
    def labels(self) -> tuple[tuple[str, int], ...]:
        """Return ``(row list, row index)`` for each answer row, in answer order."""
        return (
            *(("history_rows", row) for row in self.history_rows),
            *(("query_rows", row) for row in self.query_rows),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse one ``outputs.rows`` object, which names both row lists."""
        keys = set(map(str, payload))
        if keys != set(TABLE_ROW_LISTS):
            raise ValueError(
                f"outputs.rows must name exactly {list(TABLE_ROW_LISTS)}; "
                f"got {sorted(keys)}"
            )
        parsed: dict[str, tuple[int, ...]] = {}
        for part in TABLE_ROW_LISTS:
            field = f"outputs.rows.{part}"
            indices = _require_sequence(payload[part], field=field, allow_empty=True)
            for index in indices:
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError(f"{field} must hold non-negative integers")
            parsed[part] = tuple(indices)
        return cls(history_rows=parsed["history_rows"], query_rows=parsed["query_rows"])

    @classmethod
    def from_masks(cls, masks: Mapping[str, np.ndarray]) -> Self:
        """Return the rows of per-row-list requested-cell masks that request a cell."""
        return cls(
            history_rows=tuple(
                int(row) for row in np.flatnonzero(masks["history_rows"].any(axis=1))
            ),
            query_rows=tuple(
                int(row) for row in np.flatnonzero(masks["query_rows"].any(axis=1))
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class _TableExpectations:
    """What a table request fixes about its answer."""

    schema_version: str
    model_version: str
    return_mode: ReturnMode
    rows: TableRows
    requested_columns: tuple[str, ...]
    requested_cells: np.ndarray
    nullable: tuple[bool, ...]
    history_row_count: int
    n_samples: int | None
    quantiles: tuple[float, ...] | None


def _table_expectations(request_payload: Mapping[str, Any]) -> _TableExpectations:
    """Read the answer layout a table request payload fixes."""
    columns = [
        _require_mapping(column, field=f"request_payload.columns[{index}]")
        for index, column in enumerate(
            _require_sequence(
                request_payload.get("columns"), field="request_payload.columns"
            )
        )
    ]
    names = [
        _require_string(column.get("name"), field="request_payload.columns.name")
        for column in columns
    ]
    row_counts = {
        part: len(
            _require_sequence(
                request_payload.get(part), field=f"request_payload.{part}"
            )
        )
        for part in TABLE_ROW_LISTS
    }
    masks = _requested_cell_masks(
        _require_mapping(
            request_payload.get("requested_cells"),
            field="request_payload.requested_cells",
        ),
        column_names=names,
        row_counts=row_counts,
    )
    stacked = np.concatenate([masks[part] for part in TABLE_ROW_LISTS])
    answer_columns = [int(index) for index in np.flatnonzero(stacked.any(axis=0))]
    cells = stacked[stacked.any(axis=1)][:, answer_columns]
    cells.flags.writeable = False
    quantiles = request_payload.get("quantiles")
    return _TableExpectations(
        schema_version=_require_string(
            request_payload.get("schema_version"),
            field="request_payload.schema_version",
        ),
        model_version=_require_string(
            request_payload.get("model_version"),
            field="request_payload.model_version",
        ),
        return_mode=_require_member(
            request_payload.get("return_mode"),
            field="request_payload.return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        ),
        rows=TableRows.from_masks(masks),
        requested_columns=tuple(names[index] for index in answer_columns),
        requested_cells=cells,
        nullable=tuple(
            columns[index].get("nullable") is True for index in answer_columns
        ),
        history_row_count=row_counts["history_rows"],
        n_samples=_optional_positive_int(
            request_payload.get("n_samples"), field="request_payload.n_samples"
        ),
        quantiles=None if quantiles is None else tuple(_require_quantiles(quantiles)),
    )


def _check_table_cells(
    array: np.ndarray,
    *,
    field: str,
    requested: np.ndarray,
    nullable: tuple[bool, ...],
) -> None:
    """Require values in the requested cells and NaN everywhere else.

    ``array`` ends in ``(row, column)`` axes; a requested cell may be NaN only
    in a column the request declares nullable, where ``null`` is a draw of the
    missing state.
    """
    answered = ~np.isnan(array)
    stray = answered & ~requested
    if stray.any():
        index = "".join(f"[{int(i)}]" for i in np.argwhere(stray)[0])
        raise ValueError(
            f"{field}{index} holds a value in a cell the request did not ask for"
        )
    unanswered = ~answered & requested & ~np.asarray(nullable, dtype=bool)
    if unanswered.any():
        index = "".join(f"[{int(i)}]" for i in np.argwhere(unanswered)[0])
        raise ValueError(
            f"{field}{index} is null in a requested cell; null is allowed only "
            "in columns the request declares nullable"
        )


def _require_table_array(
    value: Any,
    *,
    field: str,
    leading: tuple[int | None, ...],
    expect: _TableExpectations,
) -> np.ndarray:
    """Parse one answer array of shape ``leading + (row, column)``."""
    array = _require_float_array(
        value,
        field=field,
        expected_shape=(*leading, *expect.requested_cells.shape),
        missing_allowed=(True,) * len(expect.requested_columns),
    )
    _check_table_cells(
        array, field=field, requested=expect.requested_cells, nullable=expect.nullable
    )
    return array


@dataclass(frozen=True, slots=True, eq=False)
class TableResponse:
    """Shared metadata preserved on every parsed table answer.

    ``rows`` names the answer rows and ``requested_columns`` the answer
    columns; ``requested_cells`` is a read-only boolean array with axis order
    ``(row, column)`` that is true at each cell the request asked for. Every
    value array keeps that ``(row, column)`` layout behind its leading axis
    and holds NaN in every cell the request did not ask for. Results compare
    by identity, as forecast results do.
    """

    schema_version: str
    image_version: str
    model_version: str
    checkpoint_version: str
    head: str
    return_mode: ReturnMode
    rows: TableRows
    requested_columns: tuple[str, ...]
    requested_cells: np.ndarray
    diagnostics: ForecastDiagnostics
    errors: tuple[StructuredError, ...]

    def to_numpy(self) -> np.ndarray:
        """Return the parsed answer values as a NumPy array."""
        raise NotImplementedError

    def to_pandas_tidy(self) -> Any:
        """Return a tidy pandas DataFrame with one row per requested cell."""
        raise NotImplementedError

    def to_pandas_wide(self) -> Any:
        """Return a wide pandas DataFrame with one row per answer row."""
        raise NotImplementedError

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        request_payload: Mapping[str, Any],
    ) -> "TableResponse":
        """Parse one table answer into the concrete result class for its mode.

        ``request_payload`` is required because the request alone says which
        cells of the answer hold values; the answer's rows, columns, sizes,
        and versions must all match it.

        Raises:
            JointFMServiceError: when the response carries service errors.
            UnsupportedSchemaVersionError, UnsupportedModelVersionError: on a
                version the request or this SDK does not accept.
            ValueError: on any other departure from the request's layout.
        """
        _raise_for_response_errors(payload)
        expect = _table_expectations(request_payload)
        schema_version = _require_string(
            payload.get("schema_version"), field="schema_version"
        )
        if schema_version != SCHEMA_VERSION or schema_version != expect.schema_version:
            raise UnsupportedSchemaVersionError(
                "Unsupported JointFM schema_version: "
                f"expected {expect.schema_version!r}, got {schema_version!r}"
            )
        model_version = _require_string(
            payload.get("model_version"), field="model_version"
        )
        if model_version != expect.model_version:
            raise UnsupportedModelVersionError(
                "Unsupported JointFM model_version: "
                f"expected {expect.model_version!r}, got {model_version!r}"
            )
        query_mode = payload.get("query_mode")
        if query_mode != "forecast":
            raise ValueError(
                f"table response query_mode must be 'forecast'; got {query_mode!r}"
            )
        return_mode = _require_member(
            payload.get("return_mode"),
            field="return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        )
        if return_mode != expect.return_mode:
            raise ValueError(
                "table response return_mode mismatch: "
                f"expected {expect.return_mode!r}, got {return_mode!r}"
            )
        _require_none(payload.get("plausibility"), field="plausibility")

        outputs = _require_mapping(payload.get("outputs"), field="outputs")
        _require_none(outputs.get("query_times"), field="outputs.query_times")
        rows = TableRows.from_payload(
            _require_mapping(outputs.get("rows"), field="outputs.rows")
        )
        if rows != expect.rows:
            raise ValueError(
                f"table response rows mismatch: expected {expect.rows}, got {rows}"
            )
        requested_columns = _string_tuple(
            outputs.get("requested_columns"), field="outputs.requested_columns"
        )
        if requested_columns != expect.requested_columns:
            raise ValueError(
                "table response requested_columns mismatch: "
                f"expected {list(expect.requested_columns)!r}, "
                f"got {list(requested_columns)!r}"
            )
        diagnostics = ForecastDiagnostics.from_payload(
            _require_mapping(payload.get("diagnostics"), field="diagnostics")
        )
        if diagnostics.history_rows != expect.history_row_count:
            raise ValueError(
                "diagnostics.history_rows mismatch: "
                f"expected {expect.history_row_count}, got {diagnostics.history_rows}"
            )
        if diagnostics.horizon_count != len(rows.labels):
            raise ValueError(
                "diagnostics.horizon_count must count the answer rows: "
                f"expected {len(rows.labels)}, got {diagnostics.horizon_count}"
            )
        for other_mode in SUPPORTED_RETURN_MODES:
            if other_mode != return_mode:
                _require_none(outputs.get(other_mode), field=f"outputs.{other_mode}")

        shared_fields: dict[str, Any] = {
            "schema_version": schema_version,
            "image_version": _require_string(
                payload.get("image_version"), field="image_version"
            ),
            "model_version": model_version,
            "checkpoint_version": _require_string(
                payload.get("checkpoint_version"), field="checkpoint_version"
            ),
            "head": _require_string(payload.get("head"), field="head"),
            "return_mode": return_mode,
            "rows": rows,
            "requested_columns": requested_columns,
            "requested_cells": expect.requested_cells,
            "diagnostics": diagnostics,
            "errors": _structured_error_tuple(payload.get("errors")),
        }
        if return_mode == "mean":
            return TableMeanResult(
                mean=_require_table_array(
                    outputs.get("mean"), field="outputs.mean", leading=(), expect=expect
                ),
                **shared_fields,
            )
        if return_mode == "samples":
            return TableSampleResult(
                samples=_require_table_array(
                    outputs.get("samples"),
                    field="outputs.samples",
                    leading=(expect.n_samples,),
                    expect=expect,
                ),
                **shared_fields,
            )
        if return_mode == "quantiles":
            row_count, column_count = expect.requested_cells.shape
            quantiles = _require_quantile_forecasts(
                outputs.get("quantiles"),
                field="outputs.quantiles",
                expected_quantiles=expect.quantiles,
                expected_horizon_count=row_count,
                expected_column_count=column_count,
                missing_allowed=(True,) * column_count,
            )
            for index, entry in enumerate(quantiles):
                _check_table_cells(
                    entry.values,
                    field=f"outputs.quantiles[{index}].values",
                    requested=expect.requested_cells,
                    nullable=expect.nullable,
                )
            return TableQuantileResult(quantiles=quantiles, **shared_fields)
        return TableLogProbResult(
            log_prob=LogProbScores.from_payload(
                _require_mapping(outputs.get("log_prob"), field="outputs.log_prob"),
                expected_horizon_count=len(rows.labels),
            ),
            **shared_fields,
        )


@dataclass(frozen=True, slots=True, eq=False)
class TableMeanResult(TableResponse):
    """Parsed means of the requested cells.

    ``mean`` is a read-only float64 array with axis order ``(row, column)``.
    """

    mean: np.ndarray

    def to_numpy(self) -> np.ndarray:
        """Return the read-only values with axis order ``(row, column)``."""
        return self.mean

    def to_pandas_tidy(self) -> Any:
        """Return ``row_list``, ``row``, ``requested_column``, and ``value`` per requested cell."""
        return _table_tidy_frame(self.mean[np.newaxis], self)

    def to_pandas_wide(self) -> Any:
        """Return one row per answer row, NaN in each cell not requested."""
        return _table_wide_frame(self.mean[np.newaxis], self)


@dataclass(frozen=True, slots=True, eq=False)
class TableSampleResult(TableResponse):
    """Parsed joint draws of the requested cells.

    ``samples`` is a read-only float64 array with axis order
    ``(sample, row, column)``. The cells of one row in one draw belong
    together.
    """

    samples: np.ndarray

    def to_numpy(self) -> np.ndarray:
        """Return the read-only values with axis order ``(sample, row, column)``."""
        return self.samples

    def to_pandas_tidy(self) -> Any:
        """Return one row per draw and requested cell, led by ``sample``."""
        return _table_tidy_frame(
            self.samples,
            self,
            leading_name="sample",
            leading_labels=range(self.samples.shape[0]),
        )

    def to_pandas_wide(self) -> Any:
        """Return one row per draw and answer row, led by ``sample``."""
        return _table_wide_frame(
            self.samples,
            self,
            leading_name="sample",
            leading_labels=range(self.samples.shape[0]),
        )


@dataclass(frozen=True, slots=True, eq=False)
class TableQuantileResult(TableResponse):
    """Parsed quantiles of the requested cells, one ``(row, column)`` surface per level."""

    quantiles: tuple[QuantileForecast, ...]

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Return the ordered quantile levels preserved from the service payload."""
        return tuple(entry.quantile for entry in self.quantiles)

    def to_numpy(self) -> np.ndarray:
        """Return a new array with axis order ``(quantile, row, column)``."""
        return np.stack([entry.values for entry in self.quantiles])

    def to_pandas_tidy(self) -> Any:
        """Return one row per level and requested cell, led by ``quantile``."""
        return _table_tidy_frame(
            self.to_numpy(),
            self,
            leading_name="quantile",
            leading_labels=self.quantile_levels,
        )

    def to_pandas_wide(self) -> Any:
        """Return one row per level and answer row, led by ``quantile``."""
        return _table_wide_frame(
            self.to_numpy(),
            self,
            leading_name="quantile",
            leading_labels=self.quantile_levels,
        )


@dataclass(frozen=True, slots=True, eq=False)
class TableLogProbResult(TableResponse):
    """Parsed log densities of the values held in the requested cells.

    Each answer row is scored jointly over its requested cells, so the
    conversions have a row axis and no column axis.
    """

    log_prob: LogProbScores

    def to_numpy(self) -> np.ndarray:
        """Return the read-only log densities with axis order ``(row,)``."""
        return self.log_prob.values

    def to_pandas_tidy(self) -> Any:
        """Return ``row_list``, ``row``, ``log_prob``, and ``nll`` per answer row."""
        pandas_module = _require_pandas_module()
        return pandas_module.DataFrame.from_records(
            [
                {"row_list": part, "row": row, "log_prob": score, "nll": negated}
                for (part, row), score, negated in zip(
                    self.rows.labels,
                    self.log_prob.values,
                    self.log_prob.nll_values,
                    strict=True,
                )
            ]
        )

    def to_pandas_wide(self) -> Any:
        """Return the one-row summary over the scored rows."""
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


def _row_label_columns(
    pandas_module: Any, rows: TableRows, positions: np.ndarray
) -> dict[str, Any]:
    """Return the ``row_list`` and ``row`` label columns at answer-row ``positions``."""
    labels = rows.labels
    return {
        "row_list": _label_column(
            pandas_module, [part for part, _ in labels], positions
        ),
        "row": _label_column(pandas_module, [row for _, row in labels], positions),
    }


def _table_tidy_frame(
    values: np.ndarray,
    result: TableResponse,
    *,
    leading_name: str | None = None,
    leading_labels: Sequence[Any] = (),
) -> Any:
    """Return one row per ``(leading, requested cell)``, leaving out cells not asked for.

    ``values`` has axis order ``(leading, row, column)``; without a
    ``leading_name`` the leading axis has length one and emits no column.
    """
    pandas_module = _require_pandas_module()
    row_positions, column_positions = np.nonzero(result.requested_cells)
    leading_positions = np.repeat(np.arange(values.shape[0]), row_positions.size)
    rows = np.tile(row_positions, values.shape[0])
    columns_at = np.tile(column_positions, values.shape[0])
    columns: dict[str, Any] = {}
    if leading_name is not None:
        columns[leading_name] = _label_column(
            pandas_module, leading_labels, leading_positions
        )
    columns.update(_row_label_columns(pandas_module, result.rows, rows))
    columns["requested_column"] = _label_column(
        pandas_module, result.requested_columns, columns_at
    )
    columns["value"] = values[leading_positions, rows, columns_at]
    return pandas_module.DataFrame(columns)


def _table_wide_frame(
    values: np.ndarray,
    result: TableResponse,
    *,
    leading_name: str | None = None,
    leading_labels: Sequence[Any] = (),
) -> Any:
    """Return one row per ``(leading, answer row)`` with one column per requested column.

    Raises ``ValueError`` when a requested column is named like an index
    column, which would otherwise overwrite it.
    """
    pandas_module = _require_pandas_module()
    leading_count, row_count, column_count = values.shape
    flat_rows = np.arange(leading_count * row_count)
    columns: dict[str, Any] = {}
    if leading_name is not None:
        columns[leading_name] = _label_column(
            pandas_module, leading_labels, flat_rows // row_count
        )
    columns.update(
        _row_label_columns(pandas_module, result.rows, flat_rows % row_count)
    )
    clashing = sorted(set(columns) & set(result.requested_columns))
    if clashing:
        raise ValueError(
            f"requested columns {clashing!r} clash with the wide frame's index columns"
        )
    table = values.reshape(leading_count * row_count, column_count)
    for column_index, column_name in enumerate(result.requested_columns):
        columns[column_name] = table[:, column_index]
    return pandas_module.DataFrame(columns)
