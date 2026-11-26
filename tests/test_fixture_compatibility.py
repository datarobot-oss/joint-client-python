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

"""Tests for the fixture compatibility surface of jointfm_client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jointfm_client import (
    ForecastResponse,
    HealthMetadata,
    SCHEMA_VERSION,
    JointFMServiceError,
    LogProbResult,
    MeanForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
    SUPPORTED_RETURN_MODES,
    validate_service_metadata,
)


def test_health_fixture_matches_current_service_contract(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The checked-in health fixture parses under the SDK's current contract."""
    payload = json_fixture_loader("jointfm_1/health_metadata")
    expected_model_version = payload["model_version"]
    assert isinstance(expected_model_version, str)

    validate_service_metadata(
        payload,
        expected_model_version=expected_model_version,
    )
    metadata = HealthMetadata.from_payload(payload)

    assert metadata.schema_version == SCHEMA_VERSION
    assert metadata.decoding_strategy == "parallel_dense"
    # The service lists its modes in an order of its own.
    assert set(metadata.supported_return_modes) == set(SUPPORTED_RETURN_MODES)


@pytest.mark.parametrize(
    ("request_fixture", "response_fixture", "result_type", "return_mode"),
    [
        (
            "jointfm_1/forecast_mean_request",
            "jointfm_1/forecast_mean_response",
            MeanForecastResult,
            "mean",
        ),
        (
            "jointfm_1/forecast_samples_request",
            "jointfm_1/forecast_samples_response",
            SampleForecastResult,
            "samples",
        ),
        (
            "jointfm_1/forecast_quantiles_request",
            "jointfm_1/forecast_quantiles_response",
            QuantileForecastResult,
            "quantiles",
        ),
        (
            "jointfm_1/forecast_log_prob_request",
            "jointfm_1/forecast_log_prob_response",
            LogProbResult,
            "log_prob",
        ),
        (
            "jointfm_1/condition_log_prob_request",
            "jointfm_1/condition_log_prob_response",
            LogProbResult,
            "log_prob",
        ),
    ],
)
def test_checked_in_fixture_payloads_parse_as_forecast_results(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    request_fixture: str,
    response_fixture: str,
    result_type: type[ForecastResponse],
    return_mode: str,
) -> None:
    """Checked in fixture payloads parse as forecast results."""
    request_payload = json_fixture_loader(request_fixture)
    response_payload = json_fixture_loader(response_fixture)

    result = ForecastResponse.from_payload(
        response_payload,
        request_payload=request_payload,
    )

    assert isinstance(result, result_type)
    assert result.return_mode == return_mode


@pytest.mark.parametrize(
    ("fixture_name", "error_code", "message_fragment"),
    [
        (
            "jointfm_1/validation_error_response",
            "VALIDATION_ERROR",
            "query_times must not be empty",
        ),
        (
            "jointfm_1/schema_version_mismatch_response",
            "SCHEMA_VERSION_MISMATCH",
            "Unsupported schema_version",
        ),
        (
            "jointfm_1/model_version_mismatch_response",
            "MODEL_VERSION_MISMATCH",
            "Unsupported model_version",
        ),
        (
            "jointfm_1/input_size_exceeded_response",
            "INPUT_SIZE_EXCEEDED",
            "n_samples exceeds the configured container cap",
        ),
        (
            "jointfm_1/request_body_too_large_response",
            "INPUT_SIZE_EXCEEDED",
            "exceeds the service's max_request_bytes",
        ),
    ],
)
def test_checked_in_error_fixtures_raise_typed_service_errors(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    fixture_name: str,
    error_code: str,
    message_fragment: str,
) -> None:
    """Checked in error fixtures raise typed service errors."""
    with pytest.raises(JointFMServiceError, match=message_fragment) as exc_info:
        ForecastResponse.raise_for_errors(json_fixture_loader(fixture_name))

    assert exc_info.value.jointfm_errors[0]["code"] == error_code
