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

"""Names, routes, versions, and value vocabularies of the JointFM wire contract."""

from __future__ import annotations

import importlib.metadata
from typing import Final, Literal, TypeAlias


DISTRIBUTION_NAME: Final = "jointfm-client"


IMPORT_NAMESPACE: Final = "jointfm_client"


FIRST_SUPPORTED_PYTHON_VERSION: Final = "3.11"


# Read from installed distribution metadata so the version cannot drift from
# pyproject.toml the way a hand-maintained literal here did.
PACKAGE_VERSION: Final = importlib.metadata.version(DISTRIBUTION_NAME)


SCHEMA_VERSION: Final = "v6"


# The service sums and averages its log densities in the model's own precision,
# so its summaries differ from a recomputation in the last few digits.
DERIVED_SCORE_TOLERANCE: Final = 1e-6


DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE: Final = (
    "deployments/{deployment_id}/predictionsUnstructured"
)


LOCAL_HEALTH_ROUTE: Final = "/healthz"


LOCAL_PREDICT_ROUTE: Final = "/predict"


DEFAULT_CALENDAR_ID: Final = "pandas-default"


PREDICT_REQUEST_TYPE: Final = "predict"


HEALTH_REQUEST_TYPE: Final = "health"


SUPPORTED_REQUEST_TYPES: Final[tuple[str, ...]] = (
    PREDICT_REQUEST_TYPE,
    HEALTH_REQUEST_TYPE,
)


QueryMode: TypeAlias = Literal["forecast", "condition"]


ConditionKind: TypeAlias = Literal["equality", "interval"]


ReturnMode: TypeAlias = Literal["mean", "samples", "quantiles", "log_prob"]


TimeIndexMode: TypeAlias = Literal[
    "ordinal",
    "continuous_float",
    "absolute_datetime",
]


DecodingStrategy: TypeAlias = Literal[
    "parallel_dense",
    "parallel_scalable",
    "autoregressive",
]


ColumnModality: TypeAlias = Literal[
    "numeric",
    "categorical",
    "ordinal",
    "binary",
    "count",
    "time_value",
]


ColumnRole: TypeAlias = Literal[
    "target",
    "known_dynamic",
    "past_dynamic",
    "static",
    "feature",
]


TimeValueKind: TypeAlias = Literal["continuous_float", "absolute_datetime"]


StructuredErrorCode: TypeAlias = Literal[
    "VALIDATION_ERROR",
    "UNSUPPORTED_HEAD_QUERY_COMBINATION",
    "UNSUPPORTED_RETURN_MODE",
    "SCHEMA_VERSION_MISMATCH",
    "MODEL_VERSION_MISMATCH",
    "INPUT_SIZE_EXCEEDED",
    "INTERNAL_ERROR",
]


SUPPORTED_QUERY_MODES: Final[tuple[QueryMode, ...]] = ("forecast", "condition")


# A completion asks an ordered request for cells of its own history rows; a
# deployment that answers it advertises this mode beside the ones this SDK
# sends, and the SDK reads the advertisement without sending the mode.
COMPLETE_QUERY_MODE: Final = "complete"


KNOWN_QUERY_MODES: Final[tuple[str, ...]] = (
    *SUPPORTED_QUERY_MODES,
    COMPLETE_QUERY_MODE,
)


SUPPORTED_CONDITION_KINDS: Final[tuple[ConditionKind, ...]] = ("equality", "interval")


SUPPORTED_RETURN_MODES: Final[tuple[ReturnMode, ...]] = (
    "mean",
    "samples",
    "quantiles",
    "log_prob",
)


SUPPORTED_TIME_INDEX_MODES: Final[tuple[TimeIndexMode, ...]] = (
    "ordinal",
    "continuous_float",
    "absolute_datetime",
)


# A table has no time axis; a deployment that answers table requests
# advertises this mode beside the forecast modes.
TABLE_TIME_INDEX_MODE: Final = "unordered"


KNOWN_TIME_INDEX_MODES: Final[tuple[str, ...]] = (
    *SUPPORTED_TIME_INDEX_MODES,
    TABLE_TIME_INDEX_MODE,
)


SUPPORTED_DECODING_STRATEGIES: Final[tuple[DecodingStrategy, ...]] = (
    "parallel_dense",
    "parallel_scalable",
    "autoregressive",
)


SUPPORTED_COLUMN_MODALITIES: Final[tuple[ColumnModality, ...]] = (
    "numeric",
    "categorical",
    "ordinal",
    "binary",
    "count",
    "time_value",
)


SUPPORTED_COLUMN_ROLES: Final[tuple[ColumnRole, ...]] = (
    "target",
    "known_dynamic",
    "past_dynamic",
    "static",
    "feature",
)


SUPPORTED_TIME_VALUE_KINDS: Final[tuple[TimeValueKind, ...]] = (
    "continuous_float",
    "absolute_datetime",
)


STRUCTURED_ERROR_CODES: Final[tuple[StructuredErrorCode, ...]] = (
    "VALIDATION_ERROR",
    "UNSUPPORTED_HEAD_QUERY_COMBINATION",
    "UNSUPPORTED_RETURN_MODE",
    "SCHEMA_VERSION_MISMATCH",
    "MODEL_VERSION_MISMATCH",
    "INPUT_SIZE_EXCEEDED",
    "INTERNAL_ERROR",
)
