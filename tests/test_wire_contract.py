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

"""The runner's OpenAPI document this SDK ships, and the fixtures captured against it."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

import capture_fixtures
from jointfm_client import (
    SCHEMA_VERSION,
    HealthMetadata,
    StructuredError,
    openapi_document,
    unknown_contract_terms,
)

_DOCUMENT = openapi_document()
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jointfm_1"
_FIXTURE_FILES = sorted(_FIXTURES.glob("*.json"))
_CONTRACT = capture_fixtures.Contract(_DOCUMENT)
_REQUEST_SCHEMA = _DOCUMENT["paths"]["/predict"]["post"]["requestBody"]["content"][
    "application/json"
]["schema"]
_BAD_REQUEST = 400


def _schemas(document: dict[str, Any]) -> dict[str, Any]:
    """Return the component schemas of ``document``."""
    return document["components"]["schemas"]


def _read(path: Path) -> Any:
    """Read one fixture."""
    return json.loads(path.read_text(encoding="utf-8"))


def _refused(request_path: Path) -> bool:
    """Return whether the service refused the request stored at ``request_path``."""
    name = request_path.name.removesuffix(capture_fixtures.REQUEST_SUFFIX)
    return bool(
        _read(_FIXTURES / f"{name}{capture_fixtures.RESPONSE_SUFFIX}")["errors"]
    )


def _check_fixture(path: Path, payload: Any) -> None:
    """Raise unless ``payload``, stored at ``path``, is a body the document describes.

    A request the service refused is left unchecked, because breaking the
    request schema is what most refusal fixtures are for.
    """
    if path.name == capture_fixtures.HEALTH_FIXTURE:
        _CONTRACT.check_health(payload)
    elif path.name.endswith(capture_fixtures.REQUEST_SUFFIX):
        if not _refused(path):
            jsonschema.Draft202012Validator(
                {**_REQUEST_SCHEMA, "components": _DOCUMENT["components"]}
            ).validate(payload)
    elif payload["errors"]:
        _CONTRACT.check_predict(_BAD_REQUEST, payload)
    else:
        _CONTRACT.check_predict(200, payload)


def test_the_packaged_document_speaks_the_sdk_schema_version() -> None:
    """The document describes the contract version every request of this SDK names."""
    assert _DOCUMENT["info"]["version"] == SCHEMA_VERSION


def test_the_sdk_knows_every_term_of_the_runner_document() -> None:
    """A runner release that adds a field, mode, or code fails here until the SDK follows."""
    assert unknown_contract_terms(_DOCUMENT) == ()


@pytest.mark.parametrize(
    ("change", "term"),
    [
        (
            lambda schemas: schemas["HealthResponse"]["properties"].update(
                new_limit={"type": "integer"}
            ),
            "health field: new_limit",
        ),
        (
            lambda schemas: schemas["Error"]["properties"].update(
                hint={"type": "string"}
            ),
            "error field: hint",
        ),
        (
            lambda schemas: schemas["Error"]["properties"]["code"]["enum"].append(
                "RATE_LIMITED"
            ),
            "Error.code value: RATE_LIMITED",
        ),
        (
            lambda schemas: schemas["HealthResponse"]["properties"][
                "supported_return_modes"
            ]["items"]["enum"].append("median"),
            "supported_return_modes value: median",
        ),
    ],
)
def test_a_term_the_sdk_does_not_read_is_reported(
    change: Callable[[dict[str, Any]], None], term: str
) -> None:
    """Each kind of unknown term is named, so the failing test says what to follow."""
    document = copy.deepcopy(_DOCUMENT)
    change(_schemas(document))

    assert unknown_contract_terms(document) == (term,)


def test_an_enumerated_health_field_the_sdk_does_not_check_is_reported() -> None:
    """A new enumerated field needs a vocabulary or an explicit pass-through."""
    document = copy.deepcopy(_DOCUMENT)
    _schemas(document)["HealthResponse"]["properties"]["sampler"] = {
        "type": "string",
        "enum": ["exact"],
    }

    assert "unchecked enumerated health field: sampler" in unknown_contract_terms(
        document
    )


@pytest.mark.parametrize("path", _FIXTURE_FILES, ids=lambda path: path.name)
def test_every_captured_fixture_is_a_body_the_document_describes(path: Path) -> None:
    """A fixture missing a key the runner always sends, or carrying one it never sends, fails."""
    _check_fixture(path, _read(path))


@pytest.mark.parametrize(
    ("fixture", "change"),
    [
        ("forecast_mean_response.json", lambda payload: payload.pop("plausibility")),
        (
            "forecast_mean_response.json",
            lambda payload: payload["outputs"].pop("log_prob"),
        ),
        (
            "validation_error_response.json",
            lambda payload: payload["errors"][0].update(detail={}),
        ),
        ("health_metadata.json", lambda payload: payload.pop("tail_map")),
    ],
)
def test_a_fixture_that_drifts_from_the_document_fails(
    fixture: str, change: Callable[[dict[str, Any]], None]
) -> None:
    """The check refuses the drift that hand-written fixtures once carried."""
    payload = _read(_FIXTURES / fixture)
    change(payload)

    with pytest.raises(jsonschema.ValidationError):
        _check_fixture(_FIXTURES / fixture, payload)


def test_the_health_metadata_exposes_every_captured_field() -> None:
    """Precision, tail map, and request cap reach the caller as the service sent them."""
    payload = _read(_FIXTURES / capture_fixtures.HEALTH_FIXTURE)

    health = HealthMetadata.from_payload(payload)

    assert health.precision == payload["precision"]
    assert health.tail_map == payload["tail_map"]
    assert health.max_request_bytes == payload["max_request_bytes"]
    assert health.sample_row_draws is None


def test_a_health_payload_without_the_newer_fields_still_parses() -> None:
    """A deployment that predates the newer health fields reports none of them."""
    payload = _read(_FIXTURES / capture_fixtures.HEALTH_FIXTURE)
    for field in ("precision", "tail_map", "max_request_bytes"):
        del payload[field]

    health = HealthMetadata.from_payload(payload)

    assert (health.precision, health.tail_map, health.max_request_bytes) == (
        None,
        None,
        None,
    )


def test_a_size_refusal_exposes_both_sizes() -> None:
    """``requested`` and ``allowed`` travel beside the message of a size refusal."""
    payload = _read(_FIXTURES / "input_size_exceeded_response.json")

    error = StructuredError.from_payload(payload["errors"][0])

    assert (error.requested, error.allowed) == (
        payload["errors"][0]["requested"],
        payload["errors"][0]["allowed"],
    )
    assert error.requested is not None and error.requested > (error.allowed or 0)


def test_the_capture_needs_a_request_for_every_documented_mode(tmp_path: Path) -> None:
    """A mode the document lists but no request covers stops the capture."""
    problems = capture_fixtures._fixture_problems(tmp_path, ("forecast_mean",))

    assert problems == [
        "no request fixture covers the document's example 'forecast_mean'; "
        "add forecast_mean_request.json, built with this SDK"
    ]


def test_the_capture_refuses_a_fixture_it_cannot_capture(tmp_path: Path) -> None:
    """Only captured answers may sit beside the requests, never a hand-written one."""
    (tmp_path / "forecast_mean_request.json").write_text("{}", encoding="utf-8")
    (tmp_path / "forecast_mean_response.json").write_text("{}", encoding="utf-8")
    (tmp_path / "old_response.json").write_text("{}", encoding="utf-8")

    problems = capture_fixtures._fixture_problems(tmp_path, ("forecast_mean",))

    assert problems == [
        "old_response.json has no request to capture it from; delete it or add "
        "its request fixture"
    ]


def test_the_capture_replaces_served_values_by_placeholders() -> None:
    """Versions and paths of the capturing host never reach a fixture."""
    health: Mapping[str, Any] = {
        "model_version": "jointfm-inference:1.2.3+ckpt.abc",
        "checkpoint_path": "/tmp/x/tiny.pt",
        "checkpoint_version": "abc",
        "image_version": "1.2.3",
    }
    placeholders = capture_fixtures.Placeholders.from_health(health)

    replaced = placeholders.apply(
        {
            "model_version": health["model_version"],
            "errors": [{"message": "expected 'jointfm-inference:1.2.3+ckpt.abc'"}],
            "checkpoint_path": "/tmp/x/tiny.pt",
            "mean": [[1.2]],
        }
    )

    assert replaced == {
        "model_version": capture_fixtures.PLACEHOLDER_MODEL_VERSION,
        "errors": [
            {"message": f"expected '{capture_fixtures.PLACEHOLDER_MODEL_VERSION}'"}
        ],
        "checkpoint_path": capture_fixtures.PLACEHOLDER_CHECKPOINT_PATH,
        "mean": [[1.2]],
    }
