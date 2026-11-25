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

"""Capture a service's answers to the request fixtures, so the response fixtures stay real.

The test fixtures of one service live in one directory: each
``<name>_request.json`` is a request this SDK builds, and the matching
``<name>_response.json`` is what a running service answered to it, with
``health_metadata.json`` its ``GET /healthz``. This script sends every request
of the directory to the service at ``--service-url``, checks each answer
against the service's OpenAPI document, and rewrites the response fixtures
and the health fixture. Nothing in a response fixture is written by hand.

A fixture must not change with the version of the service or the place its
checkpoint lives, so the service's image version, model version, checkpoint
version, and checkpoint path are replaced wherever they appear by fixed
placeholders; a request fixture names the placeholder model version, which is
replaced by the served one before sending.

The capture fails before writing anything when the document lists a query and
return mode no request fixture covers, when a fixture has no request to
capture it from, when the service's image version differs from
``--image-version``, or when an answer does not match the document.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Final

import jsonschema
import requests

from jointfm_client import openapi_document

REQUEST_SUFFIX: Final = "_request.json"
RESPONSE_SUFFIX: Final = "_response.json"
HEALTH_FIXTURE: Final = "health_metadata.json"
HEALTH_ROUTE: Final = "/healthz"
PREDICT_ROUTE: Final = "/predict"
HEALTH_EXAMPLE: Final = "health"
TIMEOUT_SECONDS: Final = 60.0
JSON_INDENT: Final = 2

PLACEHOLDER_IMAGE_VERSION: Final = "0.3.0"
PLACEHOLDER_CHECKPOINT_VERSION: Final = "sdk-test"
PLACEHOLDER_MODEL_VERSION: Final = "jointfm-inference:0.3.0+ckpt.sdk-test"
PLACEHOLDER_CHECKPOINT_PATH: Final = "/models/jointfm.pt"


@dataclass(frozen=True)
class Placeholders:
    """The served values a capture replaces, each mapped to its fixed placeholder."""

    replacements: tuple[tuple[str, str], ...]

    @classmethod
    def from_health(cls, health: Mapping[str, Any]) -> Placeholders:
        """Read the served values from a health payload.

        The model version comes first because it contains the image and
        checkpoint versions, which would otherwise be replaced inside it.
        """
        return cls(
            replacements=(
                (health["model_version"], PLACEHOLDER_MODEL_VERSION),
                (health["checkpoint_path"], PLACEHOLDER_CHECKPOINT_PATH),
                (health["checkpoint_version"], PLACEHOLDER_CHECKPOINT_VERSION),
                (health["image_version"], PLACEHOLDER_IMAGE_VERSION),
            )
        )

    def apply(self, payload: Any) -> Any:
        """Return ``payload`` with every served value replaced in every string."""
        if isinstance(payload, str):
            for served, placeholder in self.replacements:
                payload = payload.replace(served, placeholder)
            return payload
        if isinstance(payload, list):
            return [self.apply(item) for item in payload]
        if isinstance(payload, dict):
            return {key: self.apply(value) for key, value in payload.items()}
        return payload


class Contract:
    """Validators for the bodies an OpenAPI document of a JointFM service describes."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        """Index the document's ``POST /predict`` and ``GET /healthz`` bodies."""
        self._document = document
        self._predict = document["paths"][PREDICT_ROUTE]["post"]
        self._health = document["paths"][HEALTH_ROUTE]["get"]

    def example_names(self) -> tuple[str, ...]:
        """Return the names of the example requests, the health probe excluded."""
        content = self._predict["requestBody"]["content"]["application/json"]
        return tuple(name for name in content["examples"] if name != HEALTH_EXAMPLE)

    def check_health(self, payload: Any) -> None:
        """Raise ``jsonschema.ValidationError`` unless ``payload`` is a documented health answer."""
        self._validate(self._health["responses"]["200"], payload)

    def check_predict(self, status: int, payload: Any) -> None:
        """Raise unless ``payload`` is what ``POST /predict`` documents for ``status``.

        Raises:
            ValueError: when the document lists no answer with ``status``.
            jsonschema.ValidationError: when the body does not match it.
        """
        responses = self._predict["responses"]
        if str(status) not in responses:
            raise ValueError(f"POST /predict documents no answer with status {status}")
        self._validate(responses[str(status)], payload)

    def _validate(self, response: Mapping[str, Any], payload: Any) -> None:
        """Validate ``payload`` against one documented response body."""
        schema = response["content"]["application/json"]["schema"]
        jsonschema.Draft202012Validator(
            {**schema, "components": self._document["components"]}
        ).validate(payload)


def _read_json(path: Path) -> Any:
    """Read one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """Write one fixture in the layout every fixture shares."""
    path.write_text(
        json.dumps(payload, indent=JSON_INDENT, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fixture_problems(fixtures: Path, examples: tuple[str, ...]) -> list[str]:
    """Return why the directory cannot be captured as it is, or nothing."""
    requests_present = {
        path.name.removesuffix(REQUEST_SUFFIX)
        for path in fixtures.glob(f"*{REQUEST_SUFFIX}")
    }
    problems = [
        f"no request fixture covers the document's example {name!r}; "
        f"add {name}{REQUEST_SUFFIX}, built with this SDK"
        for name in examples
        if name not in requests_present
    ]
    for path in sorted(fixtures.glob("*.json")):
        if path.name == HEALTH_FIXTURE or path.name.endswith(REQUEST_SUFFIX):
            continue
        if not (
            path.name.endswith(RESPONSE_SUFFIX)
            and path.name.removesuffix(RESPONSE_SUFFIX) in requests_present
        ):
            problems.append(
                f"{path.name} has no request to capture it from; delete it or add "
                "its request fixture"
            )
    return problems


def capture(
    *,
    service_url: str,
    fixtures: Path,
    document: Mapping[str, Any],
    image_version: str | None,
) -> list[str]:
    """Capture every response fixture of ``fixtures`` from ``service_url``.

    Returns the names of the files written. Every answer is fetched and
    checked before the first file is written, so a failed capture leaves the
    directory unchanged.

    Raises:
        SystemExit: when the directory cannot be captured as it is or the
            service is of another version than ``image_version``.
        jsonschema.ValidationError: when an answer does not match the document.
        requests.RequestException: when the service cannot be reached.
    """
    contract = Contract(document)
    problems = _fixture_problems(fixtures, contract.example_names())
    if problems:
        raise SystemExit(
            "cannot capture " + str(fixtures) + ":\n- " + "\n- ".join(problems)
        )

    session = requests.Session()
    health_response = session.get(service_url + HEALTH_ROUTE, timeout=TIMEOUT_SECONDS)
    health_response.raise_for_status()
    health = health_response.json()
    contract.check_health(health)
    if image_version is not None and health["image_version"] != image_version:
        raise SystemExit(
            f"the service runs version {health['image_version']}, not {image_version}; "
            "serve the release this SDK speaks"
        )
    placeholders = Placeholders.from_health(health)

    answers: dict[Path, Any] = {fixtures / HEALTH_FIXTURE: placeholders.apply(health)}
    for request_path in sorted(fixtures.glob(f"*{REQUEST_SUFFIX}")):
        request_payload = _read_json(request_path)
        if request_payload.get("model_version") == PLACEHOLDER_MODEL_VERSION:
            request_payload = {
                **request_payload,
                "model_version": health["model_version"],
            }
        response = session.post(
            service_url + PREDICT_ROUTE, json=request_payload, timeout=TIMEOUT_SECONDS
        )
        answer = response.json()
        contract.check_predict(response.status_code, answer)
        name = request_path.name.removesuffix(REQUEST_SUFFIX)
        answers[fixtures / f"{name}{RESPONSE_SUFFIX}"] = placeholders.apply(answer)

    for path, payload in answers.items():
        _write_json(path, payload)
    return [path.name for path in answers]


def main() -> None:
    """Parse the command line and capture one fixture directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--service-url", required=True, help="Base URL of the running service"
    )
    parser.add_argument(
        "--fixtures", required=True, type=Path, help="The directory to capture"
    )
    parser.add_argument(
        "--document",
        type=Path,
        help="The service's OpenAPI document (default: the runner's, as this package ships it)",
    )
    parser.add_argument(
        "--image-version",
        help="Refuse a service whose image_version differs (default: accept any)",
    )
    arguments = parser.parse_args()
    document = (
        openapi_document()
        if arguments.document is None
        else _read_json(arguments.document)
    )
    written = capture(
        service_url=arguments.service_url.rstrip("/"),
        fixtures=arguments.fixtures,
        document=document,
        image_version=arguments.image_version,
    )
    print(f"wrote {len(written)} fixtures to {arguments.fixtures}", file=sys.stderr)


if __name__ == "__main__":
    main()
