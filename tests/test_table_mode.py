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

"""Tests for table requests: context rows, query rows, and requested cells.

The ``table_*`` and ``jointfm_2_*`` fixtures are a JointFM 2 deployment's own
output for requests this client built: each request fixture is what the
client sent, and each response fixture what the deployment answered.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from typing import Any

import numpy as np
import pytest

from jointfm_client import (
    ColumnSpec,
    ForecastResponse,
    JointFMClient,
    JointFMServiceError,
    MeanForecastResult,
    TableLogProbResult,
    TableMeanResult,
    TableQuantileResult,
    TableResponse,
    TableRows,
    TableSampleResult,
    UnsupportedServiceContractError,
    build_table_payload,
    build_table_payload_from_dataframe,
    validate_service_metadata,
)
from jointfm_client.contract import ReturnMode

_COLUMNS = (
    ColumnSpec(name="size", modality="numeric"),
    ColumnSpec(name="rooms", modality="numeric"),
    ColumnSpec(name="price", modality="numeric", role="target"),
)
_CONTEXT_ROWS = (
    {"size": 50.0, "rooms": 2.0, "price": 210.0},
    {"size": 72.0, "rooms": 3.0, "price": None},
    {"size": 95.0, "rooms": 4.0, "price": 380.0},
    {"size": 61.0, "rooms": 2.0, "price": 250.0},
    {"size": 120.0, "rooms": 5.0, "price": 470.0},
)
_QUERY_ROWS = (
    {"size": 80.0, "rooms": 3.0, "price": None},
    {"size": 105.0, "rooms": None, "price": None},
)
# Context row 1's price is a gap; query row 0 asks for its price, and query
# row 1 for its price and its rooms, a feature column.
_REQUESTED_CELLS = {
    "history_rows": {"price": [1]},
    "query_rows": {"price": [0, 1], "rooms": [1]},
}
# Values a log_prob request scores at each requested cell.
_SCORED_VALUES = {
    ("history_rows", 1, "price"): 300.0,
    ("query_rows", 0, "price"): 330.0,
    ("query_rows", 1, "price"): 420.0,
    ("query_rows", 1, "rooms"): 4.0,
}
# Answer rows: context row 1, then query rows 0 and 1; answer columns rooms
# and price. True where a cell was requested.
_ANSWER_CELLS = np.array([[False, True], [False, True], [True, True]])
_CONTROLS: dict[str, dict[str, Any]] = {
    "mean": {},
    "samples": {"n_samples": 3},
    "quantiles": {"quantiles": [0.1, 0.9]},
    "log_prob": {},
}
_RESULT_TYPES: dict[str, type[TableResponse]] = {
    "mean": TableMeanResult,
    "samples": TableSampleResult,
    "quantiles": TableQuantileResult,
    "log_prob": TableLogProbResult,
}


def _rows(return_mode: str) -> dict[str, list[dict[str, Any]]]:
    """Return both row lists, holding the scored values under ``log_prob``."""
    rows = {
        "history_rows": [dict(row) for row in _CONTEXT_ROWS],
        "query_rows": [dict(row) for row in _QUERY_ROWS],
    }
    if return_mode == "log_prob":
        for (part, row, column), value in _SCORED_VALUES.items():
            rows[part][row][column] = value
    return rows


def _payload(return_mode: ReturnMode = "mean", **overrides: Any) -> dict[str, Any]:
    """Build one table payload from row mappings."""
    rows = _rows(return_mode)
    fields: dict[str, Any] = {
        "model_version": "jointfm-serving:0.2.0+ckpt.table-1",
        "columns": _COLUMNS,
        "history_rows": rows["history_rows"],
        "query_rows": rows["query_rows"],
        "requested_cells": _REQUESTED_CELLS,
        "return_mode": return_mode,
        "seed": 3,
        **_CONTROLS[return_mode],
        **overrides,
    }
    return build_table_payload(**fields)


class _TableTransport:
    """Fake JSON transport for a deployment that serves one canned answer."""

    def __init__(
        self,
        *,
        health_payload: Mapping[str, Any],
        predict_payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Remember the advertisement and the canned answer to serve."""
        self.health_payload = health_payload
        self.predict_payload = predict_payload
        self.payloads: list[dict[str, Any]] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Serve the health advertisement on the local health route."""
        assert url == "http://127.0.0.1:8080/healthz"
        return self.health_payload

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Record the request and answer it."""
        assert url == "http://127.0.0.1:8080/predict"
        assert self.predict_payload is not None
        self.payloads.append(dict(payload))
        return self.predict_payload


def _client(transport: _TableTransport) -> JointFMClient:
    """Build a local-service client over the fake transport."""
    return JointFMClient(
        health_url="http://127.0.0.1:8080/healthz",
        predict_url="http://127.0.0.1:8080/predict",
        transport=transport,
    )


def _frames(return_mode: str) -> tuple[Any, Any]:
    """Return the context and query rows as pandas frames."""
    pandas = pytest.importorskip("pandas")
    rows = _rows(return_mode)
    return pandas.DataFrame(rows["history_rows"]), pandas.DataFrame(rows["query_rows"])


def test_a_deployment_that_answers_tables_passes_the_health_check(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Advertising the table mode beside the forecast modes is a valid contract."""
    health = json_fixture_loader("jointfm_2_health_metadata")
    assert "unordered" in health["supported_time_index_modes"]

    validate_service_metadata(health, expected_model_version=health["model_version"])


def test_a_time_index_mode_the_client_does_not_know_is_refused(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A subset check still refuses a mode newer than this client."""
    health = json_fixture_loader("jointfm_2_health_metadata")
    health["supported_time_index_modes"] = ["ordinal", "spiral"]

    with pytest.raises(UnsupportedServiceContractError, match="spiral"):
        validate_service_metadata(health)


def test_a_jointfm_1_deployment_refuses_a_table_before_any_round_trip(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A deployment that does not advertise tables never receives one."""
    transport = _TableTransport(health_payload=json_fixture_loader("health_metadata"))

    with pytest.raises(UnsupportedServiceContractError, match="does not answer table"):
        _client(transport).predict_table(
            list(_CONTEXT_ROWS),
            query_rows=list(_QUERY_ROWS),
            requested_cells=_REQUESTED_CELLS,
            columns=_COLUMNS,
        )
    assert transport.payloads == []


@pytest.mark.parametrize("return_mode", list(_CONTROLS))
def test_the_dataframe_adapter_builds_the_request_the_deployment_accepted(
    return_mode: ReturnMode,
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Frames whose requested cells are missing infer no nullable column."""
    history, query = _frames(return_mode)

    payload = build_table_payload_from_dataframe(
        history,
        query,
        model_version="jointfm-serving:0.2.0+ckpt.table-1",
        requested_cells=_REQUESTED_CELLS,
        target_columns=["price"],
        return_mode=return_mode,
        seed=3,
        **_CONTROLS[return_mode],
    )

    assert payload == json_fixture_loader(f"table_{return_mode}_request")


@pytest.mark.parametrize("return_mode", list(_CONTROLS))
def test_row_mappings_build_the_same_request(
    return_mode: ReturnMode,
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Row mappings with explicit columns spell the same payload as frames."""
    assert _payload(return_mode) == json_fixture_loader(f"table_{return_mode}_request")


@pytest.mark.parametrize("return_mode", list(_CONTROLS))
def test_each_answer_parses_into_its_table_result(
    return_mode: str,
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Values sit in the requested cells and NaN in every other answer cell."""
    result = TableResponse.from_payload(
        json_fixture_loader(f"table_{return_mode}_response"),
        request_payload=json_fixture_loader(f"table_{return_mode}_request"),
    )

    assert isinstance(result, _RESULT_TYPES[return_mode])
    assert result.rows == TableRows(history_rows=(1,), query_rows=(0, 1))
    assert result.rows.labels == (
        ("history_rows", 1),
        ("query_rows", 0),
        ("query_rows", 1),
    )
    assert result.requested_columns == ("rooms", "price")
    np.testing.assert_array_equal(result.requested_cells, _ANSWER_CELLS)
    values = result.to_numpy()
    if return_mode == "log_prob":
        assert values.shape == (3,)
        assert np.isfinite(values).all()
        return
    leading = {"mean": (), "samples": (3,), "quantiles": (2,)}[return_mode]
    assert values.shape == (*leading, 3, 2)
    np.testing.assert_array_equal(
        ~np.isnan(values), np.broadcast_to(_ANSWER_CELLS, values.shape)
    )


def test_the_client_sends_a_table_and_types_the_answer(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The client checks the advertisement, sends the frames, and parses the answer."""
    transport = _TableTransport(
        health_payload=json_fixture_loader("jointfm_2_health_metadata"),
        predict_payload=json_fixture_loader("table_quantiles_response"),
    )
    history, query = _frames("quantiles")

    result = _client(transport).predict_table(
        history,
        query_rows=query,
        requested_cells=_REQUESTED_CELLS,
        target_columns=["price"],
        return_mode="quantiles",
        quantiles=[0.1, 0.9],
        seed=3,
    )

    assert isinstance(result, TableQuantileResult)
    assert result.quantile_levels == (0.1, 0.9)
    assert transport.payloads == [json_fixture_loader("table_quantiles_request")]


def test_the_tidy_view_lists_only_the_requested_cells(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Four cells were requested, so the tidy mean has four rows."""
    pytest.importorskip("pandas")
    result = TableResponse.from_payload(
        json_fixture_loader("table_mean_response"),
        request_payload=json_fixture_loader("table_mean_request"),
    )

    tidy = result.to_pandas_tidy()
    wide = result.to_pandas_wide()

    assert list(tidy.columns) == ["row_list", "row", "requested_column", "value"]
    assert list(tidy.itertuples(index=False, name=None))[:2] == [
        ("history_rows", 1, "price", result.to_numpy()[0, 1]),
        ("query_rows", 0, "price", result.to_numpy()[1, 1]),
    ]
    assert len(tidy) == 4
    assert list(wide.columns) == ["row_list", "row", "rooms", "price"]
    assert wide["rooms"].isna().tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"requested_cells": {"history_rows": {"price": [1]}}},
            r"query_rows\[0\] requests no cell",
        ),
        (
            {"requested_cells": {"future_rows": {"price": [0]}}},
            r"holds \['future_rows'\]",
        ),
        (
            {"requested_cells": {"query_rows": {"area": [0, 1]}}},
            "undeclared column 'area'",
        ),
        (
            {"requested_cells": {"query_rows": {"price": [0, 2]}}},
            "index 2 lies outside",
        ),
        (
            {"requested_cells": {"query_rows": {"price": [0, 0, 1]}}},
            "must not repeat",
        ),
        ({"quantiles": [0.5]}, "only when return_mode='quantiles'"),
    ],
)
def test_a_request_the_deployment_would_refuse_is_refused_locally(
    overrides: dict[str, Any], message: str
) -> None:
    """Each rule the deployment enforces on requested cells holds here first."""
    with pytest.raises(ValueError, match=message):
        _payload(**overrides)


def test_a_requested_cell_holds_none_unless_it_is_scored() -> None:
    """Outside log_prob a value in a requested cell is refused, under it a gap."""
    rows = _rows("mean")
    rows["query_rows"][0]["price"] = 300.0
    with pytest.raises(ValueError, match=r"query_rows\[0\]\.price is a requested"):
        _payload(query_rows=rows["query_rows"])

    scored = _rows("log_prob")
    scored["history_rows"][1]["price"] = None
    with pytest.raises(ValueError, match="must carry an observed value"):
        _payload("log_prob", history_rows=scored["history_rows"])


def test_a_gap_outside_the_requested_cells_makes_a_column_nullable() -> None:
    """A missing value the request does not ask for is the column's missing state."""
    history, query = _frames("mean")
    history.loc[3, "size"] = None

    payload = build_table_payload_from_dataframe(
        history,
        query,
        model_version="m",
        requested_cells=_REQUESTED_CELLS,
        target_columns=["price"],
    )

    nullable = {
        column["name"]: column.get("nullable", False) for column in payload["columns"]
    }
    assert nullable == {"size": True, "rooms": False, "price": False}
    assert payload["history_rows"][3]["size"] is None


def _fill_an_unrequested_cell(outputs: dict[str, Any]) -> None:
    """Answer the rooms cell of context row 1, which the request did not ask for."""
    outputs["mean"][0][0] = 1.5


def _blank_a_requested_cell(outputs: dict[str, Any]) -> None:
    """Leave the requested price of query row 0 unanswered."""
    outputs["mean"][1][1] = None


def _drop_an_answer_row(outputs: dict[str, Any]) -> None:
    """Name fewer query rows than the request asked about."""
    outputs["rows"]["query_rows"] = [0]


def _add_query_times(outputs: dict[str, Any]) -> None:
    """Give the table answer the time axis a table does not have."""
    outputs["query_times"] = [0]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            _fill_an_unrequested_cell,
            r"outputs\.mean\[0\]\[0\] holds a value in a cell the request did not ask for",
        ),
        (
            _blank_a_requested_cell,
            r"outputs\.mean\[1\]\[1\] is null in a requested cell",
        ),
        (_drop_an_answer_row, "rows mismatch"),
        (_add_query_times, r"outputs\.query_times must be null"),
    ],
)
def test_an_answer_that_departs_from_the_request_is_refused(
    edit: Callable[[dict[str, Any]], None],
    message: str,
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Every answer cell, row, and field must match what the request asked for."""
    response = copy.deepcopy(json_fixture_loader("table_mean_response"))
    edit(response["outputs"])

    with pytest.raises(ValueError, match=message):
        TableResponse.from_payload(
            response, request_payload=json_fixture_loader("table_mean_request")
        )


def test_a_forecast_from_a_jointfm_2_deployment_carries_null_rows(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A forecast answer names no table rows, and one that does is refused."""
    request = json_fixture_loader("jointfm_2_forecast_mean_request")
    response = json_fixture_loader("jointfm_2_forecast_mean_response")
    assert response["outputs"]["rows"] is None

    assert isinstance(
        ForecastResponse.from_payload(response, request_payload=request),
        MeanForecastResult,
    )
    response["outputs"]["rows"] = {"history_rows": [], "query_rows": [0]}
    with pytest.raises(ValueError, match=r"outputs\.rows must be null"):
        ForecastResponse.from_payload(response, request_payload=request)


def test_a_malformed_table_answer_surfaces_as_a_service_error(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The client reports a contract break as a service error, not a ValueError."""
    response = copy.deepcopy(json_fixture_loader("table_mean_response"))
    response["outputs"]["requested_columns"] = ["price"]
    transport = _TableTransport(
        health_payload=json_fixture_loader("jointfm_2_health_metadata"),
        predict_payload=response,
    )

    with pytest.raises(JointFMServiceError, match="requested_columns mismatch"):
        _client(transport).predict_table(
            list(_CONTEXT_ROWS),
            query_rows=list(_QUERY_ROWS),
            requested_cells=_REQUESTED_CELLS,
            columns=_COLUMNS,
            seed=3,
        )
