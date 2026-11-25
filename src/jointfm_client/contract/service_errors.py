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

"""Structured errors in a service response, and raising them as SDK exceptions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from jointfm_client.exceptions import (
    JointFMServiceError,
)
from jointfm_client.contract.values import (
    _optional_int,
    _optional_string,
    _require_mapping,
    _require_sequence,
    _require_string,
)


@dataclass(frozen=True, slots=True)
class StructuredError:
    """Structured error entry returned by the JointFM service.

    ``requested`` and ``allowed`` are the size a request asked for and the
    service's limit, sent on an ``INPUT_SIZE_EXCEEDED`` error; both are
    ``None`` on other errors and from a service that does not send them.
    """

    code: str
    message: str
    field: str | None = None
    requested: int | None = None
    allowed: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse one structured service error payload."""
        field = payload.get("field")
        return cls(
            code=_require_string(payload.get("code"), field="errors.code"),
            message=_require_string(payload.get("message"), field="errors.message"),
            field=_optional_string(field, field="errors.field"),
            requested=_optional_int(payload.get("requested"), field="errors.requested"),
            allowed=_optional_int(payload.get("allowed"), field="errors.allowed"),
        )


def _structured_error_tuple(value: Any) -> tuple[StructuredError, ...]:
    if value is None:
        return ()
    errors = _require_sequence(value, field="errors", allow_empty=True)
    parsed_errors: list[StructuredError] = []
    for index, error in enumerate(errors):
        parsed_errors.append(
            StructuredError.from_payload(
                _require_mapping(error, field=f"errors[{index}]")
            )
        )
    return tuple(parsed_errors)


def _raise_for_response_errors(payload: Mapping[str, Any]) -> None:
    errors = _structured_error_tuple(payload.get("errors"))
    if not errors:
        return
    error_payloads = tuple(_structured_error_payload(error) for error in errors)
    first_error = errors[0]
    message = f"JointFM response returned {first_error.code}: {first_error.message}"
    raise JointFMServiceError(message, jointfm_errors=error_payloads)


def _structured_error_payload(error: StructuredError) -> dict[str, str]:
    payload = {
        "code": error.code,
        "message": error.message,
    }
    if error.field is not None:
        payload["field"] = error.field
    return payload
