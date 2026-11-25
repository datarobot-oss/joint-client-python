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

"""Value blocks of a forecast response: arrays, quantile surfaces, log densities, and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Final, Self

import numpy as np

from jointfm_client.contract.values import (
    _optional_int,
    _optional_positive_int,
    _require_mapping,
    _require_positive_int,
    _require_sequence,
    _string_tuple,
)
from jointfm_client.contract.vocabulary import (
    DERIVED_SCORE_TOLERANCE,
    ReturnMode,
)


@dataclass(frozen=True, slots=True)
class IntervalEstimator:
    """How a multi-column region's probability was estimated.

    Present only when some position bounds more than one column, which is the
    only case where a component's box probability has to be estimated; one
    interval column is exact, and a number that never varies would say nothing.
    When several positions are estimated, the accounting describes the one with
    the smallest effective sample size, because that position bounds how far
    the whole response can be trusted.
    """

    points: int
    effective_sample_size: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse the estimator accounting from a response payload."""
        return cls(
            points=_require_positive_int(
                payload.get("points"),
                field="diagnostics.interval_estimator.points",
            ),
            effective_sample_size=_require_output_float(
                payload.get("effective_sample_size"),
                field="diagnostics.interval_estimator.effective_sample_size",
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionPlausibility:
    """What the model thinks of the request it was asked to condition on.

    Without it a caller cannot separate "the model is confident" from "you
    conditioned on something the model finds absurd". Both numbers are in log
    space, and each is ``None`` when the request carried no condition of that
    kind. The service reports them and never refuses on them, so acting on them
    is the caller's decision.

    Each number is joint over every condition of its kind rather than one value
    per column: ``equality_log_density`` covers every pinned value at once, and
    ``region_log_probability`` covers the whole box at once. Do not rebuild
    either by multiplying per-column numbers, which would assume the columns are
    independent and so discard what a joint model is for. Across positions the
    numbers *are* sums: the deployment draws each future position from its own
    joint, independently of the others, so the value for a request covering
    several positions is the sum of the per-position values, exactly.

    ``region_log_probability`` is measured *after* the pinned columns are
    applied, so at a position carrying both kinds it is the log probability of
    the box **given the pins**, not the box's own. The two therefore chain, and
    their sum is the plausibility of the whole condition set::

        equality_log_density + region_log_probability
            == log( density(pins) * P(box | pins) )

    That sum is a density times a probability. Its ``exp`` is not a
    probability -- it carries the reciprocal units of the pinned columns and can
    exceed one -- so compare it, never threshold it: the difference between two
    scenarios pinning the same columns is a log likelihood ratio and is
    unit-free. For a request carrying only interval conditions there is nothing
    to combine and ``region_log_probability`` alone is the joint probability of
    the conditioned event.
    """

    equality_log_density: float | None = None
    region_log_probability: float | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse the plausibility block from a response payload."""
        equality = payload.get("equality_log_density")
        region = payload.get("region_log_probability")
        return cls(
            equality_log_density=(
                None
                if equality is None
                else _require_output_float(
                    equality, field="plausibility.equality_log_density"
                )
            ),
            region_log_probability=(
                None
                if region is None
                else _require_output_float(
                    region, field="plausibility.region_log_probability"
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ForecastDiagnostics:
    """Diagnostics block returned with one forecast response."""

    history_rows: int
    horizon_count: int
    seed: int | None = None
    condition_draws: int | None = None
    interval_estimator: IntervalEstimator | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse response diagnostics from a service payload."""
        raw_estimator = payload.get("interval_estimator")
        return cls(
            condition_draws=_optional_positive_int(
                payload.get("condition_draws"),
                field="diagnostics.condition_draws",
            ),
            interval_estimator=(
                None
                if raw_estimator is None
                else IntervalEstimator.from_payload(
                    _require_mapping(
                        raw_estimator, field="diagnostics.interval_estimator"
                    )
                )
            ),
            history_rows=_require_positive_int(
                payload.get("history_rows"),
                field="diagnostics.history_rows",
            ),
            horizon_count=_require_positive_int(
                payload.get("horizon_count"),
                field="diagnostics.horizon_count",
            ),
            seed=_optional_int(payload.get("seed"), field="diagnostics.seed"),
        )


@dataclass(frozen=True, slots=True, eq=False)
class QuantileForecast:
    """One quantile surface aligned to the response horizon and requested columns.

    ``values`` is a read-only float64 array with axis order ``(horizon, column)``.
    """

    quantile: float
    values: np.ndarray

    def to_numpy(self) -> np.ndarray:
        """Return the read-only values with axis order ``(horizon, column)``."""
        return self.values


@dataclass(frozen=True, slots=True, eq=False)
class LogProbScores:
    """Per-horizon log densities of one scored request, with its summaries.

    The service reports the same numbers several times over: ``values`` holds
    the log density of each scored future position, ``nll_values`` their
    negatives, and the four scalars the sum and the mean of each. Parsing
    recomputes every redundant field from ``values`` instead of trusting it, so
    a truncated or mismatched payload fails here rather than reading as a score.

    A log density is joint over the columns that were scored, so the block has
    no column axis however wide the request was: one number per horizon.
    ``values`` and ``nll_values`` are read-only float64 arrays of that length.
    """

    values: np.ndarray
    nll_values: np.ndarray
    total: float
    mean: float
    nll_total: float
    nll_mean: float

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_horizon_count: int | None = None,
    ) -> Self:
        """Parse one ``outputs.log_prob`` block and check its derived fields."""
        values = _require_float_array(
            payload.get("values"),
            field="outputs.log_prob.values",
            expected_shape=(expected_horizon_count,),
        )
        nll_values = _require_float_array(
            payload.get("nll_values"),
            field="outputs.log_prob.nll_values",
            expected_shape=(len(values),),
        )
        for index, (score, negated) in enumerate(
            zip(values.tolist(), nll_values.tolist(), strict=True)
        ):
            _require_derived_score(
                negated,
                expected=-score,
                field=f"outputs.log_prob.nll_values[{index}]",
            )
        summed = math.fsum(values.tolist())
        total = _require_derived_score(
            _require_output_float(payload.get("total"), field="outputs.log_prob.total"),
            expected=summed,
            field="outputs.log_prob.total",
        )
        mean = _require_derived_score(
            _require_output_float(payload.get("mean"), field="outputs.log_prob.mean"),
            expected=summed / len(values),
            field="outputs.log_prob.mean",
        )
        nll_total = _require_derived_score(
            _require_output_float(
                payload.get("nll_total"), field="outputs.log_prob.nll_total"
            ),
            expected=-total,
            field="outputs.log_prob.nll_total",
        )
        nll_mean = _require_derived_score(
            _require_output_float(
                payload.get("nll_mean"), field="outputs.log_prob.nll_mean"
            ),
            expected=-mean,
            field="outputs.log_prob.nll_mean",
        )
        return cls(
            values=values,
            nll_values=nll_values,
            total=total,
            mean=mean,
            nll_total=nll_total,
            nll_mean=nll_mean,
        )


@dataclass(frozen=True, slots=True, eq=False)
class ForecastOutputs:
    """Legacy-shaped forecast output arrays for one parsed forecast result."""

    query_times: tuple[Any, ...]
    requested_columns: tuple[str, ...]
    mean: np.ndarray | None = None
    samples: np.ndarray | None = None
    quantiles: tuple[QuantileForecast, ...] | None = None
    log_prob: LogProbScores | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        return_mode: ReturnMode,
        expected_horizon_count: int | None = None,
        expected_requested_column_count: int | None = None,
        expected_sample_count: int | None = None,
        expected_quantiles: Sequence[float] | None = None,
        nullable_columns: frozenset[str] = frozenset(),
    ) -> Self:
        """Parse and validate the `outputs` block from one forecast response.

        ``nullable_columns`` names the columns the request declared nullable:
        in those columns a ``null`` value is the observed missing state and
        parses to NaN; everywhere else ``null`` is refused. ``outputs.rows``
        names the rows of a table answer, so a forecast's is ``null`` or, from
        a service that answers no tables, absent.
        """
        _require_none(payload.get("rows"), field="outputs.rows")
        query_times = tuple(
            _require_sequence(payload.get("query_times"), field="outputs.query_times")
        )
        requested_columns = _string_tuple(
            payload.get("requested_columns"),
            field="outputs.requested_columns",
        )
        _validate_length(
            actual_length=len(query_times),
            expected_length=expected_horizon_count,
            field="outputs.query_times",
        )
        _validate_length(
            actual_length=len(requested_columns),
            expected_length=expected_requested_column_count,
            field="outputs.requested_columns",
        )

        horizon_count = len(query_times)
        column_count = len(requested_columns)
        missing_allowed = tuple(name in nullable_columns for name in requested_columns)
        mean: np.ndarray | None = None
        samples: np.ndarray | None = None
        quantiles: tuple[QuantileForecast, ...] | None = None
        log_prob: LogProbScores | None = None

        if return_mode == "mean":
            mean = _require_float_array(
                payload.get("mean"),
                field="outputs.mean",
                expected_shape=(horizon_count, column_count),
                missing_allowed=missing_allowed,
            )
            _require_none(payload.get("samples"), field="outputs.samples")
            _require_none(payload.get("quantiles"), field="outputs.quantiles")
            _require_none(payload.get("log_prob"), field="outputs.log_prob")
        elif return_mode == "samples":
            samples = _require_float_array(
                payload.get("samples"),
                field="outputs.samples",
                expected_shape=(expected_sample_count, horizon_count, column_count),
                missing_allowed=missing_allowed,
            )
            _require_none(payload.get("mean"), field="outputs.mean")
            _require_none(payload.get("quantiles"), field="outputs.quantiles")
            _require_none(payload.get("log_prob"), field="outputs.log_prob")
        elif return_mode == "quantiles":
            quantiles = _require_quantile_forecasts(
                payload.get("quantiles"),
                field="outputs.quantiles",
                expected_quantiles=expected_quantiles,
                expected_horizon_count=horizon_count,
                expected_column_count=column_count,
                missing_allowed=missing_allowed,
            )
            _require_none(payload.get("mean"), field="outputs.mean")
            _require_none(payload.get("samples"), field="outputs.samples")
            _require_none(payload.get("log_prob"), field="outputs.log_prob")
        elif return_mode == "log_prob":
            log_prob = LogProbScores.from_payload(
                _require_mapping(payload.get("log_prob"), field="outputs.log_prob"),
                expected_horizon_count=horizon_count,
            )
            _require_none(payload.get("mean"), field="outputs.mean")
            _require_none(payload.get("samples"), field="outputs.samples")
            _require_none(payload.get("quantiles"), field="outputs.quantiles")
        else:
            # A mode nobody parses must fail here rather than fall through to
            # another mode's branch and be read as that mode's shape.
            raise ValueError(
                f"outputs cannot be parsed for return_mode {return_mode!r}"
            )

        return cls(
            query_times=query_times,
            requested_columns=requested_columns,
            mean=mean,
            samples=samples,
            quantiles=quantiles,
            log_prob=log_prob,
        )


def _require_none(value: Any, *, field: str) -> None:
    if value is not None:
        raise ValueError(f"{field} must be null")


def _validate_length(
    *,
    actual_length: int,
    expected_length: int | None,
    field: str,
) -> None:
    if expected_length is None:
        return
    if actual_length != expected_length:
        raise ValueError(
            f"{field} length mismatch: expected {expected_length}, got {actual_length}"
        )


def _require_output_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _require_derived_score(value: float, *, expected: float, field: str) -> float:
    if not math.isclose(
        value,
        expected,
        rel_tol=DERIVED_SCORE_TOLERANCE,
        abs_tol=DERIVED_SCORE_TOLERANCE,
    ):
        raise ValueError(
            f"{field} disagrees with outputs.log_prob.values: "
            f"expected {expected}, got {value}"
        )
    return value


# NumPy dtype kinds a decoded JSON number array may infer: signed and unsigned
# integers and floats. Anything else (strings, None, nested objects) is refused.
_NUMERIC_ARRAY_KINDS: Final = frozenset({"i", "u", "f"})


def _require_float_array(
    value: Any,
    *,
    field: str,
    expected_shape: tuple[int | None, ...],
    missing_allowed: tuple[bool, ...] = (),
) -> np.ndarray:
    """Parse one nested JSON array of finite numbers into a read-only float64 array.

    ``expected_shape`` fixes the number of axes, and each entry that is not
    ``None`` fixes that axis's length. NumPy first infers the element type
    from the decoded values, so strings, nulls, and ragged nesting are refused
    rather than coerced; a boolean mixed into numbers is indistinguishable
    from ``0`` or ``1`` at that point and is read as one. Every value must be
    finite.

    ``missing_allowed`` holds one flag per entry of the last axis (the
    requested columns). Where it is true, ``null`` is the observed missing
    state of a nullable column and becomes NaN. An array holding a ``null`` is
    converted element by element, so in that case a numeric string would be
    read as its number; the server never sends one. Raises ``ValueError``
    naming ``field`` and, for a bad value, its index.
    """
    try:
        raw = np.asarray(value)
    except ValueError as error:
        raise ValueError(
            f"{field} must be a rectangular array of numbers: {error}"
        ) from error
    if raw.dtype.kind == "O" and any(missing_allowed):
        raw = _nulls_as_nan(raw, field=field)
    if raw.dtype.kind not in _NUMERIC_ARRAY_KINDS:
        raise ValueError(f"{field} must contain only numbers; got {raw.dtype} values")
    if raw.ndim != len(expected_shape):
        raise ValueError(
            f"{field} must have {len(expected_shape)} axes; got shape {raw.shape}"
        )
    for axis, (actual, expected) in enumerate(
        zip(raw.shape, expected_shape, strict=True)
    ):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{field} length mismatch on axis {axis}: "
                f"expected {expected}, got {actual}"
            )
    # A decoded JSON list yields a fresh array that is already float64, so no
    # copy is needed; a caller's own array is copied so that marking the result
    # read-only never freezes the caller's buffer.
    array = raw.astype(np.float64, copy=raw is value)
    refused = ~np.isfinite(array)
    if any(missing_allowed):
        refused &= ~(np.isnan(array) & np.asarray(missing_allowed, dtype=bool))
    if refused.any():
        index = "".join(f"[{int(i)}]" for i in np.argwhere(refused)[0])
        raise ValueError(
            f"{field}{index} must be finite; null is allowed only in columns the "
            "request declares nullable"
        )
    array.flags.writeable = False
    return array


def _nulls_as_nan(raw: np.ndarray, *, field: str) -> np.ndarray:
    """Return an object array of numbers and ``None`` as float64 with NaN for ``None``."""
    # On an object array ``==`` compares element by element.
    missing = raw == None  # noqa: E711
    try:
        return np.where(missing, np.nan, raw).astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain only numbers and nulls") from error


def _require_quantile_forecasts(
    value: Any,
    *,
    field: str,
    expected_quantiles: Sequence[float] | None = None,
    expected_horizon_count: int | None = None,
    expected_column_count: int | None = None,
    missing_allowed: tuple[bool, ...] = (),
) -> tuple[QuantileForecast, ...]:
    quantile_entries = _require_sequence(value, field=field)
    _validate_length(
        actual_length=len(quantile_entries),
        expected_length=None if expected_quantiles is None else len(expected_quantiles),
        field=field,
    )
    parsed_quantiles: list[QuantileForecast] = []
    for quantile_index, entry in enumerate(quantile_entries):
        mapping = _require_mapping(entry, field=f"{field}[{quantile_index}]")
        quantile = _require_output_float(
            mapping.get("quantile"),
            field=f"{field}[{quantile_index}].quantile",
        )
        if quantile <= 0.0 or quantile >= 1.0:
            raise ValueError(
                f"{field}[{quantile_index}].quantile must lie strictly between 0 and 1"
            )
        parsed_quantiles.append(
            QuantileForecast(
                quantile=quantile,
                values=_require_float_array(
                    mapping.get("values"),
                    field=f"{field}[{quantile_index}].values",
                    expected_shape=(expected_horizon_count, expected_column_count),
                    missing_allowed=missing_allowed,
                ),
            )
        )
    if expected_quantiles is not None and tuple(
        entry.quantile for entry in parsed_quantiles
    ) != tuple(expected_quantiles):
        raise ValueError(
            "forecast response quantiles mismatch: "
            f"expected {list(expected_quantiles)!r}, "
            f"got {[entry.quantile for entry in parsed_quantiles]!r}"
        )
    return tuple(parsed_quantiles)
