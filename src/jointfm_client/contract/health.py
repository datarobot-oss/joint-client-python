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

"""Service-health metadata and the checks a deployment's advertisement passes before a request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from jointfm_client.exceptions import (
    UnsupportedModelVersionError,
    UnsupportedSchemaVersionError,
    UnsupportedServiceContractError,
)
from jointfm_client.contract.conditions import (
    Condition,
    condition_kinds,
    resolve_conditions,
)
from jointfm_client.contract.values import (
    _optional_positive_int,
    _optional_string,
    _require_mapping,
    _require_positive_float,
    _require_positive_int,
    _require_string,
    _string_tuple,
)
from jointfm_client.contract.vocabulary import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class DataGenerationCapabilities:
    """Training-time data-generation envelope advertised by the deployment.

    Mirrors the inference service's ``data_generation`` health block.
    ``max_series`` bounds the *total* width of one request — feature and
    target columns counted together — because training draws that total and
    then splits it randomly, guaranteeing only that at least one column is a
    target. There is therefore no separate per-role cap: a request is
    admissible on width alone. ``min_series`` is the fewest columns a training
    sample carried; it describes the training data and limits no request,
    since one column is enough. ``n_input`` is the longest history and
    ``n_output`` the longest forecast horizon the deployment was trained to
    handle; both are upper bounds, and smaller requests are accepted.
    ``t_input`` and ``t_output`` are the simulated time spans of that history
    and horizon during training, which give the time scale the model expects.
    """

    min_series: int
    max_series: int
    t_input: float
    t_output: float
    n_input: int
    n_output: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse one ``data_generation`` block from a `/healthz` payload.

        Raises ``ValueError`` when a field is missing or malformed, or when the
        series bounds are inverted — an envelope no training sample could have
        met.
        """
        min_series = _require_positive_int(
            payload.get("min_series"),
            field="data_generation.min_series",
        )
        max_series = _require_positive_int(
            payload.get("max_series"),
            field="data_generation.max_series",
        )
        if min_series > max_series:
            raise ValueError(
                "data_generation.min_series must not exceed "
                f"data_generation.max_series; got {min_series} > {max_series}"
            )
        return cls(
            min_series=min_series,
            max_series=max_series,
            t_input=_require_positive_float(
                payload.get("t_input"),
                field="data_generation.t_input",
            ),
            t_output=_require_positive_float(
                payload.get("t_output"),
                field="data_generation.t_output",
            ),
            n_input=_require_positive_int(
                payload.get("n_input"),
                field="data_generation.n_input",
            ),
            n_output=_require_positive_int(
                payload.get("n_output"),
                field="data_generation.n_output",
            ),
        )


@dataclass(frozen=True, slots=True)
class HealthMetadata:
    """Typed representation of `/healthz` service metadata.

    ``generation`` names the JointFM generation of the mounted checkpoint
    (``jointfm-1`` or ``jointfm-2``). ``max_sample_count`` caps the samples
    one request may ask for, and ``max_concurrent_requests`` is how many
    requests the container serves at once; a further request waits for a
    free slot.

    ``precision`` is the numeric precision every model forward runs at
    (``fp32``, ``bf16``, or ``fp16``), and ``tail_map`` the per-column tail
    map the head's forecast applies, ``none`` when it applies none.
    ``max_request_bytes`` is the largest request body the service reads, and
    ``sample_row_draws`` how many answered rows times draws one sampling pass
    of a JointFM 2 service holds; a seed repeats an answer only under the same
    value. Each of these four is ``None`` from a deployment that does not send
    the field.

    ``decoding_strategy`` and the ``supported_*`` capability lists hold what
    the deployment advertises, values newer than this SDK's vocabularies
    included.
    """

    status: str
    schema_version: str
    image_version: str
    model_version: str
    checkpoint_version: str
    checkpoint_path: str
    device: str
    generation: str
    head: str
    decoding_strategy: str
    supported_query_modes: tuple[str, ...]
    supported_condition_kinds: tuple[str, ...]
    supported_return_modes: tuple[str, ...]
    supported_time_index_modes: tuple[str, ...]
    time_index_encoding: str
    max_sample_count: int
    max_concurrent_requests: int
    data_generation: DataGenerationCapabilities | None = None
    supported_metadata_traits: tuple[str, ...] = ()
    precision: str | None = None
    tail_map: str | None = None
    max_request_bytes: int | None = None
    sample_row_draws: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse and validate a JSON health payload from the service.

        ``supported_metadata_traits`` lists the expert-hint traits the mounted
        model reads. A deployment that does not send the field predates hints
        and drops them unread, so it parses as supporting none.
        """
        validate_service_metadata(payload)
        raw_data_generation = payload.get("data_generation")
        if raw_data_generation is None:
            parsed_data_generation: DataGenerationCapabilities | None = None
        else:
            parsed_data_generation = DataGenerationCapabilities.from_payload(
                _require_mapping(raw_data_generation, field="data_generation")
            )
        return cls(
            status=_require_string(payload.get("status"), field="status"),
            schema_version=_require_string(
                payload.get("schema_version"),
                field="schema_version",
            ),
            image_version=_require_string(
                payload.get("image_version"),
                field="image_version",
            ),
            model_version=_require_string(
                payload.get("model_version"),
                field="model_version",
            ),
            checkpoint_version=_require_string(
                payload.get("checkpoint_version"),
                field="checkpoint_version",
            ),
            checkpoint_path=_require_string(
                payload.get("checkpoint_path"),
                field="checkpoint_path",
            ),
            device=_require_string(payload.get("device"), field="device"),
            generation=_require_string(
                payload.get("generation"),
                field="generation",
            ),
            head=_require_string(payload.get("head"), field="head"),
            decoding_strategy=_require_decoding_strategy(
                payload.get("decoding_strategy")
            ),
            supported_query_modes=_string_tuple(
                payload.get("supported_query_modes"),
                field="supported_query_modes",
            ),
            # A deployment that cannot condition advertises an empty list,
            # which is an answer rather than a missing field.
            supported_condition_kinds=_string_tuple(
                payload.get("supported_condition_kinds"),
                field="supported_condition_kinds",
                allow_empty=True,
            ),
            supported_return_modes=_string_tuple(
                payload.get("supported_return_modes"),
                field="supported_return_modes",
            ),
            supported_time_index_modes=_string_tuple(
                payload.get("supported_time_index_modes"),
                field="supported_time_index_modes",
            ),
            time_index_encoding=_require_string(
                payload.get("time_index_encoding"),
                field="time_index_encoding",
            ),
            max_sample_count=_require_positive_int(
                payload.get("max_sample_count"),
                field="max_sample_count",
            ),
            max_concurrent_requests=_require_positive_int(
                payload.get("max_concurrent_requests"),
                field="max_concurrent_requests",
            ),
            data_generation=parsed_data_generation,
            supported_metadata_traits=(
                ()
                if payload.get("supported_metadata_traits") is None
                else _string_tuple(
                    payload.get("supported_metadata_traits"),
                    field="supported_metadata_traits",
                    allow_empty=True,
                )
            ),
            precision=_optional_string(payload.get("precision"), field="precision"),
            tail_map=_optional_string(payload.get("tail_map"), field="tail_map"),
            max_request_bytes=_optional_positive_int(
                payload.get("max_request_bytes"),
                field="max_request_bytes",
            ),
            sample_row_draws=_optional_positive_int(
                payload.get("sample_row_draws"),
                field="sample_row_draws",
            ),
        )


def require_metadata_trait_support(
    metadata: HealthMetadata,
    columns: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse expert hints the mounted deployment does not read.

    ``columns`` are request column descriptors as sent on the wire. A hint
    for a trait outside ``metadata.supported_metadata_traits`` would reach no
    model input, so the request is refused before it is sent.

    Raises:
        UnsupportedServiceContractError: when a column carries a trait the
            deployment does not advertise.
    """
    supported = set(metadata.supported_metadata_traits)
    for column in columns:
        unsupported = sorted(set(column.get("traits") or {}) - supported)
        if unsupported:
            raise UnsupportedServiceContractError(
                f"The mounted {metadata.head!r} deployment does not read metadata "
                f"traits {unsupported} on column {column.get('name')!r}; it "
                f"advertises {list(metadata.supported_metadata_traits)}"
            )


def require_condition_support(
    metadata: HealthMetadata,
    condition: Condition | Sequence[Condition],
) -> None:
    """Refuse a conditioning request the mounted deployment has not advertised.

    The gate exists so a caller learns a deployment cannot condition *before*
    paying for the request, rather than from the error on one it already sent.
    It reads the advertisement alone, which the deployment derives from its own
    head, so it can never claim a capability the checkpoint lacks.

    Raises:
        UnsupportedServiceContractError: when the deployment serves no
            ``condition`` mode, or not every condition kind ``condition`` uses.
    """
    if "condition" not in metadata.supported_query_modes:
        raise UnsupportedServiceContractError(
            f"The mounted {metadata.head!r} deployment does not serve condition "
            f"queries; it advertises {list(metadata.supported_query_modes)}"
        )
    missing = [
        kind
        for kind in condition_kinds(resolve_conditions(condition))
        if kind not in metadata.supported_condition_kinds
    ]
    if missing:
        raise UnsupportedServiceContractError(
            f"The mounted {metadata.head!r} deployment does not serve condition "
            f"kinds {missing}; it advertises "
            f"{list(metadata.supported_condition_kinds)}"
        )


def validate_service_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_model_version: str | None = None,
) -> None:
    """Validate `/healthz` metadata against the SDK's compatibility policy.

    Requires the schema version in ``SCHEMA_VERSION``, the expected model
    version when supplied, a non-empty ``decoding_strategy``, and each
    advertised capability list as a JSON array of non-empty strings, non-empty
    except ``supported_condition_kinds``. Advertised values this SDK does not
    know, such as a query or return mode added by a newer service, pass
    unread, so a client keeps working with every newer deployment of its
    schema version; a service that breaks older clients changes
    ``schema_version``. A call is refused when it needs a capability the
    deployment does not advertise: locally by the gates
    ``require_condition_support``, ``require_table_support``, and
    ``require_metadata_trait_support``, and by the service's own structured
    error for an unserved query or return mode.

    Raises:
        UnsupportedSchemaVersionError: when ``schema_version`` differs from
            ``SCHEMA_VERSION``.
        UnsupportedModelVersionError: when ``model_version`` differs from
            ``expected_model_version``.
        UnsupportedServiceContractError: when a required field is missing or
            malformed.
    """
    schema_version = _required_string(metadata, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            "Unsupported JointFM schema_version: "
            f"expected {SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    model_version = _required_string(metadata, "model_version")
    if expected_model_version is not None and model_version != expected_model_version:
        raise UnsupportedModelVersionError(
            "Unsupported JointFM model_version: "
            f"expected {expected_model_version!r}, got {model_version!r}"
        )

    _require_advertised_list(metadata, field="supported_query_modes", allow_empty=False)
    _require_advertised_list(
        metadata, field="supported_condition_kinds", allow_empty=True
    )
    _require_advertised_list(
        metadata, field="supported_return_modes", allow_empty=False
    )
    _require_advertised_list(
        metadata, field="supported_time_index_modes", allow_empty=False
    )
    _require_decoding_strategy(metadata.get("decoding_strategy"))


def _required_string(metadata: Mapping[str, Any], field: str) -> str:
    """Return a required non-empty string metadata field."""
    value = metadata.get(field)
    if not isinstance(value, str) or value == "":
        raise UnsupportedServiceContractError(
            f"JointFM health metadata field {field!r} must be a non-empty string"
        )
    return value


def _require_decoding_strategy(value: Any) -> str:
    """Return the horizon decoding strategy from health metadata.

    No request this SDK builds depends on the strategy, so a strategy newer
    than ``SUPPORTED_DECODING_STRATEGIES`` passes through as advertised.
    """
    if not isinstance(value, str) or value == "":
        raise UnsupportedServiceContractError(
            "JointFM health metadata field 'decoding_strategy' must be a "
            "non-empty string"
        )
    return value


def _require_advertised_list(
    metadata: Mapping[str, Any],
    *,
    field: str,
    allow_empty: bool,
) -> None:
    """Require one advertised capability list to be a JSON array of non-empty strings.

    The values are not checked against what this SDK knows: the service
    derives the list from its own checkpoint, so it may advertise fewer values
    than the SDK knows, and a newer service may advertise values the SDK never
    sends.
    """
    advertised_values = metadata.get(field)
    if not isinstance(advertised_values, Sequence) or isinstance(
        advertised_values, str | bytes | bytearray
    ):
        raise UnsupportedServiceContractError(
            f"JointFM health metadata field {field!r} must be a JSON array of strings"
        )

    parsed_values: list[str] = []
    for index, advertised_value in enumerate(advertised_values):
        if not isinstance(advertised_value, str) or advertised_value == "":
            raise UnsupportedServiceContractError(
                f"JointFM health metadata field {field!r}[{index}] must be a non-empty string"
            )
        parsed_values.append(advertised_value)

    if not parsed_values and not allow_empty:
        raise UnsupportedServiceContractError(
            f"JointFM health metadata field {field!r} advertises nothing"
        )
