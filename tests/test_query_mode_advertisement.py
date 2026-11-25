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

"""Tests for query modes a deployment advertises beyond the ones the client sends.

A JointFM 2 deployment that reads ordered rows also answers completions and
advertises ``complete`` in ``supported_query_modes``. The client reads that
advertisement, so its health check passes, while every request it builds
stays a forecast or a condition query. The health fixtures under
``jointfm_2/`` are the health payloads of two JointFM 2 deployments: one that
answers tables and one that reads ordered rows alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from jointfm_client import (
    COMPLETE_QUERY_MODE,
    EqualityCondition,
    ForecastRequestMetadata,
    HealthMetadata,
    UnsupportedServiceContractError,
    require_condition_support,
    validate_service_metadata,
)
from jointfm_client.contract import QueryMode

_HEALTH_FIXTURES = (
    "jointfm_2/health_metadata",
    "jointfm_2/ordered_health_metadata",
)


@pytest.mark.parametrize("fixture", _HEALTH_FIXTURES)
def test_a_deployment_that_answers_completions_passes_the_health_check(
    json_fixture_loader: Callable[[str], dict[str, Any]], fixture: str
) -> None:
    """Both deployments advertise ``complete``, and both pass and parse."""
    health = json_fixture_loader(fixture)
    assert COMPLETE_QUERY_MODE in health["supported_query_modes"]

    validate_service_metadata(health, expected_model_version=health["model_version"])
    parsed = HealthMetadata.from_payload(health)

    assert COMPLETE_QUERY_MODE in parsed.supported_query_modes


def test_a_query_mode_the_client_does_not_know_passes_and_the_call_gate_refuses(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A mode newer than this client passes health; a call needing an unadvertised mode is refused."""
    health = json_fixture_loader("jointfm_2/ordered_health_metadata")
    health["supported_query_modes"] = ["forecast", "explain"]

    validate_service_metadata(health)
    parsed = HealthMetadata.from_payload(health)

    assert parsed.supported_query_modes == ("forecast", "explain")
    with pytest.raises(
        UnsupportedServiceContractError, match="does not serve condition"
    ):
        require_condition_support(parsed, EqualityCondition(column="driver", value=1.0))


def test_the_client_never_sends_a_completion() -> None:
    """Reading the advertisement does not make ``complete`` a mode the client sends."""
    with pytest.raises(ValueError, match="query_mode"):
        ForecastRequestMetadata(
            model_version="any", query_mode=cast(QueryMode, COMPLETE_QUERY_MODE)
        )
