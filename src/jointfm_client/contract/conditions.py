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

"""Conditions of a ``condition`` query: equality pins and interval ranges at chosen query positions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, TypeAlias, cast

from jointfm_client.contract.values import (
    _require_sequence,
    _require_string,
)
from jointfm_client.contract.vocabulary import (
    ConditionKind,
)


def _require_query_time_indices(value: Any) -> tuple[int, ...] | None:
    """Validate the positions one condition covers, ``None`` meaning every one."""
    if value is None:
        return None
    indices = _require_sequence(value, field="condition.query_time_indices")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("condition.query_time_indices must hold integers")
        if index < 0:
            raise ValueError("condition.query_time_indices must not be negative")
    if len(set(indices)) != len(indices):
        raise ValueError("condition.query_time_indices names a position more than once")
    return tuple(indices)


@dataclass(frozen=True, slots=True)
class EqualityCondition:
    """One column pinned to a value at the future positions it covers.

    ``query_time_indices`` names those positions by index into the request's
    ``query_times``; ``None`` covers every one of them. An equality condition
    is an event of probability zero under a continuous head, so the deployment
    answers it analytically rather than by filtering draws. It fixes the
    column's value, so a projection that names the column reads that value back
    at every covered position rather than a model output — which is what lets
    a scenario answer line up column for column with an unconditioned one.
    """

    column: str
    value: float
    query_time_indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        """Reject a pin the service would refuse anyway."""
        _require_string(self.column, field="condition.column")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("condition value must be a number")
        if not math.isfinite(float(self.value)):
            raise ValueError("condition value must be finite")
        object.__setattr__(
            self,
            "query_time_indices",
            _require_query_time_indices(self.query_time_indices),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return this condition as JSON-compatible payload fields."""
        return {
            "column": self.column,
            "kind": "equality",
            "value": float(self.value),
            "query_time_indices": _query_time_indices_payload(self),
        }


@dataclass(frozen=True, slots=True)
class IntervalCondition:
    """One column confined to a range at the future positions it covers.

    ``None`` on either side leaves the range open, and ``query_time_indices``
    of ``None`` covers every requested position. The region carries positive
    mass, so the deployment answers it by composition on the reduced mixture,
    and the column stays readable: what it returns is its distribution inside
    the region.
    """

    column: str
    lower: float | None = None
    upper: float | None = None
    query_time_indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        """Reject a range that bounds nothing or bounds it backwards."""
        _require_string(self.column, field="condition.column")
        for bound, name in ((self.lower, "lower"), (self.upper, "upper")):
            if bound is None:
                continue
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(f"condition {name} bound must be a number or None")
            if not math.isfinite(float(bound)):
                raise ValueError(f"condition {name} bound must be finite or None")
        if self.lower is None and self.upper is None:
            raise ValueError(
                "an interval open on both sides conditions nothing; give at least one bound"
            )
        if (
            self.lower is not None
            and self.upper is not None
            and float(self.lower) >= float(self.upper)
        ):
            raise ValueError(
                f"condition needs lower < upper, got [{self.lower}, {self.upper}]"
            )
        object.__setattr__(
            self,
            "query_time_indices",
            _require_query_time_indices(self.query_time_indices),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return this condition as JSON-compatible payload fields."""
        return {
            "column": self.column,
            "kind": "interval",
            "lower": None if self.lower is None else float(self.lower),
            "upper": None if self.upper is None else float(self.upper),
            "query_time_indices": _query_time_indices_payload(self),
        }


Condition: TypeAlias = EqualityCondition | IntervalCondition


def _query_time_indices_payload(condition: Condition) -> list[int] | None:
    """Serialize a condition's positions, ``None`` staying the wire's ``null``."""
    if condition.query_time_indices is None:
        return None
    return list(condition.query_time_indices)


def _covers(condition: Condition, position: int) -> bool:
    """Return whether ``condition`` applies at future ``position``."""
    return (
        condition.query_time_indices is None or position in condition.query_time_indices
    )


def resolve_conditions(
    condition: Condition | Sequence[Condition],
) -> tuple[Condition, ...]:
    """Normalize one condition or a list of them into a validated tuple.

    Conditions on different columns may cover the same positions, which is how
    one position mixes a pin and a band. Two conditions on the *same* column
    must cover disjoint positions, because a column carries at most one
    condition per position; ``query_time_indices=None`` covers every position,
    so it overlaps any other condition on its column.

    Raises:
        ValueError: when the list is empty, holds something other than an
            ``EqualityCondition`` or ``IntervalCondition``, or puts two
            conditions on one column at a shared position.
    """
    conditions: tuple[Any, ...] = (
        (condition,)
        if isinstance(condition, (EqualityCondition, IntervalCondition))
        else tuple(_require_sequence(condition, field="condition"))
    )
    for index, entry in enumerate(conditions):
        if not isinstance(entry, (EqualityCondition, IntervalCondition)):
            raise ValueError(
                f"condition[{index}] must be an EqualityCondition or an "
                "IntervalCondition"
            )
    for later_index, later in enumerate(conditions):
        for earlier_index, earlier in enumerate(conditions[:later_index]):
            if earlier.column != later.column:
                continue
            if earlier.query_time_indices is None or later.query_time_indices is None:
                overlap = "every position"
            else:
                shared = sorted(
                    set(earlier.query_time_indices) & set(later.query_time_indices)
                )
                if not shared:
                    continue
                overlap = f"query_time_indices {shared}"
            raise ValueError(
                f"column {later.column!r} carries more than one condition at "
                f"{overlap}: condition[{later_index}] overlaps "
                f"condition[{earlier_index}]"
            )
    return cast(tuple[Condition, ...], conditions)


def condition_kinds(conditions: Sequence[Condition]) -> tuple[ConditionKind, ...]:
    """Return the condition kinds ``conditions`` use, which a deployment must advertise."""
    kinds: list[ConditionKind] = []
    for condition in conditions:
        kind: ConditionKind = (
            "equality" if isinstance(condition, EqualityCondition) else "interval"
        )
        if kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)
