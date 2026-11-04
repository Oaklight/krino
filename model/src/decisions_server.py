#!/usr/bin/env python3
"""Serve a Krino decision model over llm-rosetta decision formats.

Two wire formats share one backend:

    POST /v1/decisions   OpenAI Decisions  (questions: list, answers: list)
    POST /v1/systemone   TypeSafe System One (questions: map, answers: map)

The backend is any object with ``predict(state, question) -> dict``.
``DecisionModel`` already implements that interface, so a trained checkpoint
needs no adapter.

Run against a checkpoint:

    python -m model.src.decisions_server \
        --model Qwen/Qwen3-Reranker-0.6B --heads checkpoints/best_heads.pt \
        --rank 64 --port 8000

Run without a checkpoint, to test the wire format:

    python -m model.src.decisions_server --stub --port 8000

Smoke test:

    curl -s localhost:8000/v1/decisions -H 'content-type: application/json' -d '{
      "model": "krino",
      "input": "I was charged twice.",
      "questions": [
        {"type": "predicate", "name": "urgent", "instructions": "Is this urgent?"},
        {"type": "choice", "name": "dept", "instructions": "Which team?",
         "choices": [{"value": "billing", "description": "Payments"},
                     {"value": "technical", "description": "Bugs"}]},
        {"type": "score", "name": "severity", "instructions": "How severe?",
         "levels": [{"label": "Cosmetic"}, {"label": "Blocked"}]}
      ]}'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_BODY_BYTES = 32 * 1024 * 1024


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class Backend(Protocol):
    model_name: str

    def predict(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        ...

    def count_tokens(self, text: str) -> int:
        ...


class KrinoBackend:
    """Adapter around ``DecisionModel.predict`` with a serialized GPU lock."""

    def __init__(self, model: Any, tokenizer: Any = None, model_name: str = "krino") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._lock = threading.Lock()

    def predict(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self.model.predict(state, question)

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return 0


class StubBackend:
    """Deterministic backend for wire-format testing. No torch, no weights.

    Probabilities are derived from a hash of the state and question, so the
    same request always returns the same answer.
    """

    model_name = "krino-stub"

    def __init__(self, model_name: str = "krino-stub") -> None:
        self.model_name = model_name

    def predict(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        seed = int.from_bytes(
            hashlib.sha256(
                (state + json.dumps(question, sort_keys=True, ensure_ascii=False)).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        rng = random.Random(seed)
        q_type = question.get("type", "noul")

        if q_type == "noul":
            return {"type": "noul", "noul": round(rng.uniform(0.05, 0.95), 4)}

        if q_type == "choice":
            criteria = question["criteria"]
            keys = list(criteria.keys())
            weights = [rng.random() + 1e-6 for _ in keys]
            total = sum(weights)
            probs = {k: w / total for k, w in zip(keys, weights)}
            return {
                "type": "choice",
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
            }

        if q_type == "score":
            criteria = question["criteria"]
            weights = [rng.random() + 1e-6 for _ in criteria]
            total = sum(weights)
            probs = {str(i): w / total for i, w in enumerate(weights)}
            score = sum(i * p for i, p in enumerate(probs.values()))
            return {
                "type": "score",
                "score": score,
                "probabilities": probs,
                "legend": {str(i): desc for i, desc in enumerate(criteria)},
            }

        raise ValueError(f"unknown question type: {q_type}")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def confidence(probabilities: dict[str, float]) -> float:
    """Normalized top-1 confidence, matching the existing Krino server."""
    k = len(probabilities)
    if k <= 1:
        return 1.0
    p_max = max(probabilities.values())
    return max(0.0, (p_max - 1.0 / k) / (1.0 - 1.0 / k))


def flatten_input(value: Any) -> str:
    """Flatten OpenAI ``input`` (string or message array) into one state string."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return "" if value is None else str(value)

    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind in ("input_text", "text", "output_text"):
                    parts.append(part.get("text", ""))
                elif kind == "input_json":
                    parts.append(json.dumps(part.get("data"), ensure_ascii=False))
                elif kind in ("input_image", "image_url"):
                    parts.append("[image omitted]")
        if item.get("type") in ("function_call", "custom_tool_call"):
            parts.append(f"{item.get('name', 'tool')}({item.get('arguments', '')})")
        elif item.get("type") in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output")
            parts.append(output if isinstance(output, str) else json.dumps(output, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def openai_question_to_internal(q: dict[str, Any]) -> dict[str, Any]:
    q_type = q.get("type")
    instructions = q.get("instructions", "")
    if isinstance(instructions, list):
        instructions = flatten_input(instructions)

    if q_type == "predicate":
        return {"type": "noul", "instructions": instructions}

    if q_type == "choice":
        criteria: dict[str, str] = {}
        for choice in q.get("choices", []):
            value = choice.get("value")
            if value is None:
                value = choice.get("label")
            value = str(value)
            criteria[value] = choice.get("description") or value
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    if q_type == "score":
        levels = []
        for level in q.get("levels", []):
            label = level.get("label")
            levels.append(level.get("description") or ("" if label is None else str(label)))
        return {"type": "score", "instructions": instructions, "criteria": levels}

    raise ValueError(f"unsupported OpenAI question type: {q_type!r}")


def systemone_question_to_internal(q: dict[str, Any]) -> dict[str, Any]:
    q_type = q.get("type")
    instructions = q.get("instructions", "")
    if q_type == "noul":
        # TypeSafe noul carries a criteria map for the true/false labels; the
        # model only uses the instruction, so the map is dropped.
        return {"type": "noul", "instructions": instructions}
    if q_type in ("choice", "score"):
        return {"type": q_type, "instructions": instructions, "criteria": q["criteria"]}
    raise ValueError(f"unsupported System One question type: {q_type!r}")


def internal_to_openai_answer(
    name: str,
    answer: dict[str, Any],
    refusal_threshold: float = 0.0,
) -> dict[str, Any]:
    answer_type = answer.get("type")

    if answer_type == "noul":
        probability = float(answer["noul"])
        return {"type": "predicate", "name": name, "probability": probability}

    if answer_type == "choice":
        probabilities = answer.get("probabilities", {})
        if refusal_threshold > 0.0 and probabilities and max(probabilities.values()) < refusal_threshold:
            return {"type": "refusal", "name": name}
        return {
            "type": "choice",
            "name": name,
            "choice": answer.get("choice", ""),
            "probabilities": [
                {"value": key, "probability": float(value)}
                for key, value in probabilities.items()
            ],
            "confidence": confidence(probabilities),
        }

    if answer_type == "score":
        probabilities = answer.get("probabilities", {})
        if refusal_threshold > 0.0 and probabilities and max(probabilities.values()) < refusal_threshold:
            return {"type": "refusal", "name": name}
        legend = answer.get("legend", {})
        return {
            "type": "score",
            "name": name,
            "score": float(answer.get("score", 0.0)),
            "probabilities": [
                {
                    "value": int(index),
                    "label": legend.get(str(index), str(index)),
                    "probability": float(value),
                }
                for index, value in probabilities.items()
            ],
            "confidence": confidence(probabilities),
        }

    raise ValueError(f"unsupported internal answer type: {answer_type!r}")


def internal_to_systemone_answer(name: str, answer: dict[str, Any]) -> dict[str, Any]:
    answer_type = answer.get("type")
    if answer_type == "noul":
        return {"type": "noul", "noul": float(answer["noul"])}
    if answer_type == "choice":
        probabilities = answer.get("probabilities", {})
        return {
            "type": "choice",
            "choice": answer.get("choice", ""),
            "probabilities": {k: float(v) for k, v in probabilities.items()},
            "confidence": confidence(probabilities),
        }
    if answer_type == "score":
        probabilities = answer.get("probabilities", {})
        return {
            "type": "score",
            "score": float(answer.get("score", 0.0)),
            "legend": answer.get("legend", {}),
            "probabilities": {k: float(v) for k, v in probabilities.items()},
            "confidence": confidence(probabilities),
        }
    raise ValueError(f"unsupported internal answer type: {answer_type!r}")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class DecisionHandler(BaseHTTPRequestHandler):
    backend: Backend
    refusal_threshold: float = 0.0

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": error_type}})

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._error(400, "empty body")
            return None
        if length > MAX_BODY_BYTES:
            self._error(413, "request body too large")
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._error(400, f"invalid JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            self._error(400, "request body must be a JSON object")
            return None
        return payload

    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok", "model": self.backend.model_name})
            return
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.backend.model_name,
                            "object": "model",
                            "owned_by": "krino",
                        }
                    ],
                },
            )
            return
        self._error(404, f"unknown path: {self.path}")

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path == "/v1/decisions":
            self._handle_openai()
            return
        if path == "/v1/systemone":
            self._handle_systemone()
            return
        self._error(404, f"unknown path: {self.path}")

    def _handle_openai(self) -> None:
        payload = self._read_json()
        if payload is None:
            return

        state = flatten_input(payload.get("input", ""))
        questions = payload.get("questions")
        if not isinstance(questions, list) or not questions:
            self._error(400, "questions must be a non-empty list")
            return

        answers = []
        used = 0
        try:
            for question in questions:
                name = question.get("name")
                if not name:
                    self._error(400, "every question needs a name")
                    return
                internal = openai_question_to_internal(question)
                used += self.backend.count_tokens(state + internal.get("instructions", ""))
                answer = self.backend.predict(state, internal)
                answers.append(
                    internal_to_openai_answer(name, answer, self.refusal_threshold)
                )
        except ValueError as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:  # model failure is a server error
            self._error(500, f"inference failed: {exc}", "server_error")
            return

        self._send_json(
            200,
            {
                "id": f"dec_{hashlib.sha256(state.encode('utf-8')).hexdigest()[:24]}",
                "object": "decision",
                "model": payload.get("model") or self.backend.model_name,
                "answers": answers,
                "usage": {
                    "input_tokens": used,
                    "output_tokens": 0,
                    "total_tokens": used,
                },
            },
        )

    def _handle_systemone(self) -> None:
        payload = self._read_json()
        if payload is None:
            return

        state = payload.get("state")
        if not isinstance(state, str) or not state:
            self._error(400, "state must be a non-empty string")
            return
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            self._error(400, "questions must be a non-empty object")
            return

        answers: dict[str, Any] = {}
        used = 0
        try:
            for name, question in questions.items():
                internal = systemone_question_to_internal(question)
                used += self.backend.count_tokens(state + internal.get("instructions", ""))
                answer = self.backend.predict(state, internal)
                answers[name] = internal_to_systemone_answer(name, answer)
        except ValueError as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:
            self._error(500, f"inference failed: {exc}", "server_error")
            return

        self._send_json(
            200,
            {
                "model": payload.get("model") or self.backend.model_name,
                "answers": answers,
                "usage": {"input_tokens": used, "output_tokens": 0},
            },
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_backend(args: argparse.Namespace) -> Backend:
    if args.stub:
        return StubBackend(args.model_name or "krino-stub")

    if not args.model or not args.heads:
        raise SystemExit("--model and --heads are required unless --stub is set")

    from model.src.backbone import load_causal_lm, load_encoder
    from model.src.decision_model import DecisionModel

    loader = load_encoder if args.encoder else load_causal_lm
    backbone, tokenizer = loader(args.model, device=args.device, freeze=True)
    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=args.rank,
        rival_aware=args.rival_aware,
        readout=args.readout,
        max_length=args.max_length,
    )
    model.load_heads(args.heads)
    model.eval()

    model_name = args.model_name or Path(args.model).name or "krino"
    print(
        f"loaded {model_name}: {model.trainable_parameters():,} trainable, "
        f"{model.frozen_parameters():,} frozen",
        flush=True,
    )
    return KrinoBackend(model, tokenizer, model_name=model_name)


def serve(backend: Backend, host: str, port: int, refusal_threshold: float = 0.0) -> None:
    DecisionHandler.backend = backend
    DecisionHandler.refusal_threshold = refusal_threshold
    server = ThreadingHTTPServer((host, port), DecisionHandler)
    server.daemon_threads = True
    print(f"serving {backend.model_name} on http://{host}:{port}", flush=True)
    print("  POST /v1/decisions   (OpenAI Decisions)", flush=True)
    print("  POST /v1/systemone   (TypeSafe System One)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="backbone name or path")
    parser.add_argument("--encoder", action="store_true", help="load an encoder backbone")
    parser.add_argument("--heads", type=Path, help="path to a saved heads checkpoint")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--rival-aware", action="store_true")
    parser.add_argument("--readout", default="last")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stub", action="store_true", help="serve deterministic fake answers")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=0.0,
        help="return a refusal when top probability is below this value (0 disables)",
    )
    args = parser.parse_args()
    serve(load_backend(args), args.host, args.port, args.refusal_threshold)


if __name__ == "__main__":
    main()
