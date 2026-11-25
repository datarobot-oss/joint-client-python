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

"""The OpenAPI document of the inference runner, and what this SDK does not know of one.

The public inference runner (``jointfm-inference``) generates an OpenAPI 3.1
document of its HTTP contract from its own code: every request, response,
health, and error field of ``POST /predict`` and ``GET /healthz``, the values
each enumerated field may take, and one example request per query and return
mode. This package ships the document of the runner release it speaks, copied
by ``task contract:sync``, so tools and clients in other languages read the
same contract this SDK is tested against.

:func:`unknown_contract_terms` compares any JointFM service's document with
what this SDK reads, so a service that gains a health field, an error field,
an error code, or an advertised mode fails a test until the SDK follows it.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from importlib import resources
import json
from typing import Any, Final

from jointfm_client.contract import (
    KNOWN_QUERY_MODES,
    KNOWN_TIME_INDEX_MODES,
    STRUCTURED_ERROR_CODES,
    SUPPORTED_CONDITION_KINDS,
    SUPPORTED_RETURN_MODES,
    HealthMetadata,
    StructuredError,
)

OPENAPI_FILE_NAME: Final = "openapi.json"
HEALTH_COMPONENT: Final = "HealthResponse"
ERROR_COMPONENT: Final = "Error"
ERROR_CODE_FIELD: Final = "code"

# The advertised values this SDK knows; a deployment may advertise others,
# which the health check passes unread.
KNOWN_HEALTH_VOCABULARIES: Final[Mapping[str, tuple[str, ...]]] = {
    "supported_query_modes": KNOWN_QUERY_MODES,
    "supported_condition_kinds": SUPPORTED_CONDITION_KINDS,
    "supported_return_modes": SUPPORTED_RETURN_MODES,
    "supported_time_index_modes": KNOWN_TIME_INDEX_MODES,
}
# Enumerated health fields this SDK exposes as plain strings without a
# vocabulary of its own, because no request it builds depends on their value.
PASSED_THROUGH_HEALTH_FIELDS: Final[frozenset[str]] = frozenset(
    {"precision", "tail_map", "time_index_encoding"}
)


def openapi_document() -> dict[str, Any]:
    """Return the OpenAPI document of the runner release this SDK speaks.

    Each call parses the packaged copy anew, so a caller may change the
    returned mapping freely.
    """
    text = (
        resources.files("jointfm_client")
        .joinpath(OPENAPI_FILE_NAME)
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


def unknown_contract_terms(document: Mapping[str, Any]) -> tuple[str, ...]:
    """List what a JointFM service's OpenAPI document holds that this SDK does not read.

    ``document`` is the OpenAPI document of the public inference runner or of
    any service with the same ``HealthResponse`` and ``Error`` components. A
    term is unknown when it is a health field ``HealthMetadata`` lacks, an
    error field ``StructuredError`` lacks, an error code outside
    ``STRUCTURED_ERROR_CODES``, an advertised value outside the vocabulary this
    SDK knows for the field, or an enumerated health field the SDK has no
    vocabulary for and does not pass through. The result is sorted and empty
    when the SDK reads everything the document lists. A listed term does not
    stop the SDK from working with the service; it is what the SDK cannot show
    its user.

    Raises:
        KeyError: when the document lacks either component.
    """
    schemas = document["components"]["schemas"]
    health_fields = schemas[HEALTH_COMPONENT]["properties"]
    error_fields = schemas[ERROR_COMPONENT]["properties"]
    terms = [
        *_unknown_fields("health field", health_fields, HealthMetadata),
        *_unknown_fields("error field", error_fields, StructuredError),
        *_unknown_values(
            f"{ERROR_COMPONENT}.{ERROR_CODE_FIELD}",
            error_fields[ERROR_CODE_FIELD],
            STRUCTURED_ERROR_CODES,
        ),
    ]
    for name, field_schema in health_fields.items():
        vocabulary = KNOWN_HEALTH_VOCABULARIES.get(name)
        if vocabulary is not None:
            terms.extend(_unknown_values(name, field_schema, vocabulary))
        elif _enumerated_values(field_schema) and name not in (
            PASSED_THROUGH_HEALTH_FIELDS
        ):
            terms.append(f"unchecked enumerated health field: {name}")
    return tuple(sorted(terms))


def _unknown_fields(
    kind: str,
    fields: Mapping[str, Any],
    parsed_type: type[HealthMetadata] | type[StructuredError],
) -> list[str]:
    """Return the document's ``fields`` that the dataclass ``parsed_type`` lacks."""
    known = {parsed.name for parsed in dataclasses.fields(parsed_type)}
    return [f"{kind}: {name}" for name in fields if name not in known]


def _unknown_values(
    field: str, field_schema: Mapping[str, Any], vocabulary: tuple[str, ...]
) -> list[str]:
    """Return the values the document allows in ``field`` outside ``vocabulary``."""
    return [
        f"{field} value: {value}"
        for value in _enumerated_values(field_schema)
        if value not in vocabulary
    ]


def _enumerated_values(field_schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the enumeration of a field, or of an array field's items."""
    items = field_schema.get("items", {})
    return tuple(field_schema.get("enum", items.get("enum", ())))
