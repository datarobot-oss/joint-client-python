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

"""Validators that read one JSON value of a payload and name its field when refusing it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, TypeVar, cast


def _string_tuple(
    value: Any, *, field: str, allow_empty: bool = False
) -> tuple[str, ...]:
    values = _require_sequence(value, field=field, allow_empty=allow_empty)
    return tuple(_require_string(item, field=f"{field}[]") for item in values)


def _optional_tuple(value: Any, *, field: str) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return tuple(_require_sequence(value, field=field, allow_empty=True))


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_sequence(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a JSON array")
    if not allow_empty and len(value) == 0:
        raise ValueError(f"{field} must not be empty")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field=field)


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _require_positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return parsed


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_positive_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field=field)


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _optional_positive_float(value: Any, *, field: str) -> float | None:
    parsed = _optional_float(value, field=field)
    if parsed is not None and parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _optional_mapping(value: Any, *, field: str) -> Mapping[str | int, int] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, field=field)
    parsed: dict[str | int, int] = {}
    for key, target in mapping.items():
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise ValueError(f"{field} keys must be strings or integers")
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError(f"{field} values must be integers")
        parsed[key] = target
    return parsed


_MemberT = TypeVar("_MemberT", bound=str)


def _require_member(
    value: Any, *, field: str, supported_values: Sequence[_MemberT]
) -> _MemberT:
    parsed = _require_string(value, field=field)
    if parsed not in supported_values:
        raise ValueError(
            f"Unsupported {field}: expected one of {tuple(supported_values)!r}, got {parsed!r}"
        )
    return cast(_MemberT, parsed)


def _optional_member(
    value: Any,
    *,
    field: str,
    supported_values: Sequence[_MemberT],
) -> _MemberT | None:
    if value is None:
        return None
    return _require_member(value, field=field, supported_values=supported_values)
