"""Small, dependency-free client for the Jev System One API."""

from __future__ import annotations

import http.client
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit


@dataclass(frozen=True)
class RawResponse:
    """An HTTP response with wire-level observations."""

    status: int
    headers: dict[str, str]
    body: bytes
    ttfb_s: float
    total_s: float

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class Transport(Protocol):
    """Transport interface used by the real client and unit-test fakes."""

    def post(self, url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> RawResponse:
        """Send one POST request."""


class StdlibTransport:
    """HTTP transport based only on :mod:`http.client`."""

    def post(self, url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> RawResponse:
        parts = urlsplit(url)
        connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(parts.hostname, parts.port, timeout=timeout)
        path = parts.path or "/"
        if parts.query:
            path += f"?{parts.query}"
        started = time.perf_counter()
        try:
            connection.request("POST", path, body=body, headers=dict(headers))
            response = connection.getresponse()
            ttfb = time.perf_counter() - started
            raw_body = response.read()
            total = time.perf_counter() - started
            response_headers = {name.lower(): value for name, value in response.getheaders()}
            return RawResponse(response.status, response_headers, raw_body, ttfb, total)
        finally:
            connection.close()


class JevClient:
    """Jev API client supporting parsed and raw responses."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: Transport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ["TYPESAFE_API_KEY"]
        self.transport = transport or StdlibTransport()
        self.timeout = timeout

    @staticmethod
    def encode_payload(payload: Mapping[str, Any]) -> bytes:
        """Encode JSON without ASCII-escaping Unicode or reordering keys."""
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def ask_raw(self, state: str, questions: dict[str, Any], model: str = "jev-latest") -> RawResponse:
        """Submit a request and retain status, headers, bytes, and timings."""
        payload = {"state": state, "model": model, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return self.transport.post(
            f"{self.base_url}/v1/systemone",
            headers,
            self.encode_payload(payload),
            self.timeout,
        )

    def ask(self, state: str, questions: dict[str, Any], model: str = "jev-latest") -> dict[str, Any]:
        response = self.ask_raw(state, questions, model)
        if response.status != 200:
            message = response.body.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Jev API error {response.status}: {message}")
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise RuntimeError("Jev API returned a non-object JSON response")
        return parsed

    def noul(self, state: str, question_id: str, instructions: str, **kwargs: Any) -> dict[str, Any]:
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if "criteria" in kwargs:
            question["criteria"] = kwargs["criteria"]
        return self.ask(state, {question_id: question}, **{key: value for key, value in kwargs.items() if key == "model"})

    def choice(
        self,
        state: str,
        question_id: str,
        instructions: str,
        criteria: dict[str, str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        question = {"type": "choice", "instructions": instructions, "criteria": criteria}
        return self.ask(state, {question_id: question}, **{key: value for key, value in kwargs.items() if key == "model"})

    def score(
        self,
        state: str,
        question_id: str,
        instructions: str,
        criteria: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        question = {"type": "score", "instructions": instructions, "criteria": criteria}
        return self.ask(state, {question_id: question}, **{key: value for key, value in kwargs.items() if key == "model"})

    def close(self) -> None:
        """Retained for compatibility; connections are request-scoped."""

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
