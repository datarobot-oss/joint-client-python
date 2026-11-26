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

"""Tests for the request body: its encoding, the advertised size cap, and the service's size refusal."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

import jointfm_client.transport as transport_module
from jointfm_client import (
    ColumnSpec,
    DataFrameSchema,
    JointFMCapacityError,
    JointFMClient,
    JointFMHTTPStatusError,
    JointFMHTTPTransport,
    JointFMInputSizeExceededError,
    JointFMRequestEncodingError,
    JointFMRetryConfig,
    JOINTFM_LOCAL_BASE_URLS_ENV,
    JOINTFM_SCHEMA_VERSION_ENV,
    load_settings,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jointfm_1"
_RETRYING = JointFMRetryConfig(max_attempts=3, backoff_seconds=0.0)
_POOL_URLS = ("http://127.0.0.1:8080", "http://127.0.0.1:8081")


def _read(name: str) -> dict[str, Any]:
    """Read one captured fixture."""
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


@dataclass
class _Service:
    """A local HTTP service answering health and predictions from fixed bodies."""

    health: Mapping[str, Any]
    answers: list[tuple[HTTPStatus, Mapping[str, Any]]]
    posted_bodies: list[bytes] = field(default_factory=list)
    url: str = ""


@pytest.fixture
def service() -> Iterator[_Service]:
    """Serve ``/healthz`` and ``/predict`` on loopback from a mutable ``_Service``."""
    state = _Service(health=_read("health_metadata.json"), answers=[])

    class Handler(BaseHTTPRequestHandler):
        """Answer from ``state``; record every POST body."""

        def do_GET(self) -> None:
            """Answer the health route."""
            self._answer(HTTPStatus.OK, state.health)

        def do_POST(self) -> None:
            """Record the body and give the next queued answer."""
            length = int(self.headers["Content-Length"])
            state.posted_bodies.append(self.rfile.read(length))
            status, payload = state.answers[len(state.posted_bodies) - 1]
            self._answer(status, payload)

        def _answer(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            """Write one JSON answer."""
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test output quiet."""
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def _client(service: _Service) -> JointFMClient:
    """Return a client of ``service`` whose transport retries."""
    return JointFMClient(
        health_url=f"{service.url}/healthz",
        predict_url=f"{service.url}/predict",
        retry_config=_RETRYING,
    )


def test_a_non_finite_value_is_refused_unsent_and_unretried(service: _Service) -> None:
    """NaN has no JSON spelling, so the request fails to encode once instead of retrying."""
    transport = JointFMHTTPTransport(retry_config=_RETRYING)

    with pytest.raises(JointFMRequestEncodingError, match="JSON"):
        transport.post_json(f"{service.url}/predict", {"value": math.nan})

    assert service.posted_bodies == []


def test_a_retried_request_is_encoded_once(
    service: _Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every attempt sends the bytes the first encoding produced."""
    encoded: list[bytes] = []
    encode = transport_module.encode_json_body

    def _recording_encode(payload: Mapping[str, Any]) -> bytes:
        """Record each encoding."""
        encoded.append(encode(payload))
        return encoded[-1]

    monkeypatch.setattr(transport_module, "encode_json_body", _recording_encode)
    service.answers = [(HTTPStatus.SERVICE_UNAVAILABLE, {}), (HTTPStatus.OK, {"ok": 1})]
    transport = JointFMHTTPTransport(retry_config=_RETRYING)

    assert transport.post_json(f"{service.url}/predict", {"value": 1.5}) == {"ok": 1}
    assert len(encoded) == 1
    assert service.posted_bodies == [encoded[0], encoded[0]]


def test_the_services_size_refusal_surfaces_once_as_a_typed_error(
    service: _Service,
) -> None:
    """The runner's 413 is not retried and carries both sizes and the refused field."""
    refusal = _read("request_body_too_large_response.json")
    service.health = {
        key: value
        for key, value in service.health.items()
        if key != "max_request_bytes"
    }
    service.answers = [(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, refusal)] * 3

    with pytest.raises(JointFMInputSizeExceededError) as exc_info:
        _client(service).predict(_read("request_body_too_large_request.json"))

    error = exc_info.value
    assert isinstance(error, JointFMHTTPStatusError)
    assert error.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert (error.field, error.requested, error.allowed) == (
        refusal["errors"][0]["field"],
        refusal["errors"][0]["requested"],
        refusal["errors"][0]["allowed"],
    )
    assert len(service.posted_bodies) == 1


def test_a_request_above_the_advertised_cap_is_refused_before_the_upload(
    service: _Service,
) -> None:
    """The client measures the body it would send and names both sizes and the request's shape."""
    request = _read("request_body_too_large_request.json")
    cap = service.health["max_request_bytes"]
    body_bytes = len(transport_module.encode_json_body(request))

    with pytest.raises(JointFMCapacityError) as exc_info:
        _client(service).predict(request)

    message = str(exc_info.value)
    assert f"{body_bytes} bytes" in message
    assert f"max_request_bytes {cap}" in message
    assert f"{len(request['history_rows'])} history rows" in message
    assert "1 columns" in message and "1 query times" in message
    assert service.posted_bodies == []


def test_a_request_within_the_advertised_cap_goes_out_unchanged(
    service: _Service,
) -> None:
    """A body at most the cap is sent as the client encoded it."""
    request = _read("forecast_mean_request.json")
    service.answers = [(HTTPStatus.OK, _read("forecast_mean_response.json"))]

    _client(service).predict(request)

    assert service.posted_bodies == [transport_module.encode_json_body(request)]


class _PoolTransport:
    """Pool peers advertising a cap below every request, recording each prediction."""

    def __init__(self, max_request_bytes: int) -> None:
        """Advertise ``max_request_bytes`` from every peer."""
        self.health = {
            **_read("health_metadata.json"),
            "max_request_bytes": max_request_bytes,
        }
        self.predictions: list[Mapping[str, Any]] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Refuse: pool peers answer health on the predict route."""
        raise AssertionError(f"unexpected GET {url}")

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Answer health; record anything else."""
        if payload.get("request_type") == "health":
            return self.health
        self.predictions.append(payload)
        raise AssertionError("a prediction above the cap was uploaded")


def test_a_pool_refuses_oversized_sample_batches_before_any_upload() -> None:
    """Parallel sample batches are measured before the first one goes out."""
    transport = _PoolTransport(max_request_bytes=64)
    client = JointFMClient(
        settings=load_settings(
            env={
                JOINTFM_LOCAL_BASE_URLS_ENV: ",".join(_POOL_URLS),
                JOINTFM_SCHEMA_VERSION_ENV: "v6",
            },
            dotenv_path=None,
        ),
        transport=transport,
    )

    with pytest.raises(JointFMCapacityError, match="max_request_bytes 64"):
        client.forecast_samples(
            [{"target": 10.0}, {"target": 11.0}],
            schema=DataFrameSchema(
                columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
                time_index_mode="ordinal",
            ),
            query_times=[2],
            requested_columns=["target"],
            model_version="jointfm-inference:0.3.0+ckpt.sdk-test",
            n_samples=4,
            seed=7,
        )

    assert transport.predictions == []
