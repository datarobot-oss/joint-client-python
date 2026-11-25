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

"""JointFM service contract constants, models, and compatibility checks.

The submodules group the contract by concern; every public name is
re-exported here, so ``jointfm_client.contract`` stays the import path.
"""

from jointfm_client.contract.vocabulary import (
    COMPLETE_QUERY_MODE,
    ColumnModality,
    ColumnRole,
    ConditionKind,
    DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE,
    DEFAULT_CALENDAR_ID,
    DERIVED_SCORE_TOLERANCE,
    DISTRIBUTION_NAME,
    DecodingStrategy,
    FIRST_SUPPORTED_PYTHON_VERSION,
    HEALTH_REQUEST_TYPE,
    IMPORT_NAMESPACE,
    KNOWN_QUERY_MODES,
    KNOWN_TIME_INDEX_MODES,
    LOCAL_HEALTH_ROUTE,
    LOCAL_PREDICT_ROUTE,
    PACKAGE_VERSION,
    PREDICT_REQUEST_TYPE,
    QueryMode,
    ReturnMode,
    SCHEMA_VERSION,
    STRUCTURED_ERROR_CODES,
    SUPPORTED_COLUMN_MODALITIES,
    SUPPORTED_COLUMN_ROLES,
    SUPPORTED_CONDITION_KINDS,
    SUPPORTED_DECODING_STRATEGIES,
    SUPPORTED_QUERY_MODES,
    SUPPORTED_REQUEST_TYPES,
    SUPPORTED_RETURN_MODES,
    SUPPORTED_TIME_INDEX_MODES,
    SUPPORTED_TIME_VALUE_KINDS,
    StructuredErrorCode,
    TABLE_TIME_INDEX_MODE,
    TimeIndexMode,
    TimeValueKind,
)
from jointfm_client.contract.conditions import (
    Condition,
    EqualityCondition,
    IntervalCondition,
    condition_kinds,
    resolve_conditions,
)
from jointfm_client.contract.request import (
    ColumnSpec,
    DataFrameSchema,
    ForecastRequest,
    ForecastRequestMetadata,
    build_forecast_payload,
)
from jointfm_client.contract.health import (
    DataGenerationCapabilities,
    HealthMetadata,
    require_condition_support,
    require_metadata_trait_support,
    validate_service_metadata,
)
from jointfm_client.contract.service_errors import (
    StructuredError,
)
from jointfm_client.contract.outputs import (
    ConditionPlausibility,
    ForecastDiagnostics,
    ForecastOutputs,
    IntervalEstimator,
    LogProbScores,
    QuantileForecast,
)
from jointfm_client.contract.response import (
    ForecastResponse,
    LogProbResult,
    MeanForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)

__all__ = [
    "COMPLETE_QUERY_MODE",
    "ColumnModality",
    "ColumnRole",
    "ColumnSpec",
    "Condition",
    "ConditionKind",
    "ConditionPlausibility",
    "DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE",
    "DEFAULT_CALENDAR_ID",
    "DERIVED_SCORE_TOLERANCE",
    "DISTRIBUTION_NAME",
    "DataFrameSchema",
    "DataGenerationCapabilities",
    "DecodingStrategy",
    "EqualityCondition",
    "FIRST_SUPPORTED_PYTHON_VERSION",
    "ForecastDiagnostics",
    "ForecastOutputs",
    "ForecastRequest",
    "ForecastRequestMetadata",
    "ForecastResponse",
    "HEALTH_REQUEST_TYPE",
    "HealthMetadata",
    "IMPORT_NAMESPACE",
    "IntervalCondition",
    "IntervalEstimator",
    "KNOWN_QUERY_MODES",
    "KNOWN_TIME_INDEX_MODES",
    "LOCAL_HEALTH_ROUTE",
    "LOCAL_PREDICT_ROUTE",
    "LogProbResult",
    "LogProbScores",
    "MeanForecastResult",
    "PACKAGE_VERSION",
    "PREDICT_REQUEST_TYPE",
    "QuantileForecast",
    "QuantileForecastResult",
    "QueryMode",
    "ReturnMode",
    "SCHEMA_VERSION",
    "STRUCTURED_ERROR_CODES",
    "SUPPORTED_COLUMN_MODALITIES",
    "SUPPORTED_COLUMN_ROLES",
    "SUPPORTED_CONDITION_KINDS",
    "SUPPORTED_DECODING_STRATEGIES",
    "SUPPORTED_QUERY_MODES",
    "SUPPORTED_REQUEST_TYPES",
    "SUPPORTED_RETURN_MODES",
    "SUPPORTED_TIME_INDEX_MODES",
    "SUPPORTED_TIME_VALUE_KINDS",
    "SampleForecastResult",
    "StructuredError",
    "StructuredErrorCode",
    "TABLE_TIME_INDEX_MODE",
    "TimeIndexMode",
    "TimeValueKind",
    "build_forecast_payload",
    "condition_kinds",
    "require_condition_support",
    "require_metadata_trait_support",
    "resolve_conditions",
    "validate_service_metadata",
]
