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

"""SDK exception hierarchy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class JointFMError(Exception):
    """Base class for JointFM SDK errors."""


class JointFMServiceError(JointFMError):
    """Raised when a successful HTTP response still carries JointFM service errors."""

    def __init__(
        self,
        message: str,
        *,
        jointfm_errors: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Store structured JointFM errors returned in a JSON response body."""
        super().__init__(message)
        self.jointfm_errors = tuple(jointfm_errors)


class JointFMConfigurationError(JointFMError):
    """Raised when local settings or deployment selection are invalid."""


class JointFMCapacityError(JointFMConfigurationError):
    """Raised when a planned forecast exceeds the deployed model's capacity envelope."""


class JointFMTransportError(JointFMError):
    """Base class for HTTP transport and response errors."""


class JointFMRequestEncodingError(JointFMTransportError):
    """Raised when an outbound JSON request cannot be encoded."""


class JointFMRequestError(JointFMTransportError):
    """Raised when a network request fails before a usable response is received."""


class JointFMResponseError(JointFMTransportError):
    """Raised when a service response is malformed or unsuccessful."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body_excerpt: str,
        datarobot_request_id: str | None = None,
        jointfm_errors: Sequence[Mapping[str, Any]] = (),
        retry_after_seconds: float | None = None,
    ) -> None:
        """Store HTTP metadata from one malformed or unsuccessful response."""
        super().__init__(message)
        self.status_code = status_code
        self.response_body_excerpt = response_body_excerpt
        self.datarobot_request_id = datarobot_request_id
        self.jointfm_errors = tuple(jointfm_errors)
        self.retry_after_seconds = retry_after_seconds


class JointFMResponseDecodeError(JointFMResponseError):
    """Raised when a service response is empty, non-JSON, or not a JSON object."""


class JointFMHTTPStatusError(JointFMResponseError):
    """Raised when the service returns a non-success HTTP status."""


class JointFMInputSizeExceededError(JointFMHTTPStatusError):
    """Raised when the service refuses a request with ``INPUT_SIZE_EXCEEDED``.

    The service gives this answer to a body above its ``max_request_bytes``
    (HTTP 413, refused before the body is read), to more samples than its
    ``max_sample_count``, and to more columns than its ``max_series``.
    ``field`` names the refused input, ``requested`` its size, and ``allowed``
    the service's limit; each is ``None`` when the service does not send it.
    The transport never retries the refusal, because the same request cannot
    succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body_excerpt: str,
        field: str | None,
        requested: int | None,
        allowed: int | None,
        datarobot_request_id: str | None = None,
        jointfm_errors: Sequence[Mapping[str, Any]] = (),
        retry_after_seconds: float | None = None,
    ) -> None:
        """Store the HTTP metadata and the refused input's size and limit."""
        super().__init__(
            message,
            status_code=status_code,
            response_body_excerpt=response_body_excerpt,
            datarobot_request_id=datarobot_request_id,
            jointfm_errors=jointfm_errors,
            retry_after_seconds=retry_after_seconds,
        )
        self.field = field
        self.requested = requested
        self.allowed = allowed


class JointFMCompatibilityError(JointFMError):
    """Base class for fail-fast service compatibility errors."""


class UnsupportedSchemaVersionError(JointFMCompatibilityError):
    """Raised when the service advertises a schema version unsupported by the SDK."""


class UnsupportedModelVersionError(JointFMCompatibilityError):
    """Raised when the service model version differs from the requested version."""


class UnsupportedServiceContractError(JointFMCompatibilityError):
    """Raised when advertised service capabilities do not match the SDK contract."""
