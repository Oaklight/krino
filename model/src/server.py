"""Minimal /v1/systemone API server matching TypeSafe's contract.

Pluggable backend: any object with a .evaluate(state, questions) method
that returns {question_id: answer_dict}.

Usage:
    from model.src.server import create_app
    app = create_app(backend)
    # Run with: uvicorn model.src.server:app
"""

from __future__ import annotations

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Protocol


class DecisionBackend(Protocol):
    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Evaluate typed questions against a state.

        Returns: {question_id: answer_dict} where answer_dict has:
            noul: {"type": "noul", "noul": float}
            choice: {"type": "choice", "choice": str, "probabilities": dict, "confidence": float}
            score: {"type": "score", "score": float, "probabilities": dict, "legend": dict, "confidence": float}
        """
        ...


def _confidence(probabilities: dict[str, float]) -> float:
    k = len(probabilities)
    if k <= 1:
        return 1.0
    p_max = max(probabilities.values())
    return (p_max - 1.0 / k) / (1.0 - 1.0 / k)


def _format_response(answers: dict[str, Any], model_name: str, input_tokens: int = 0, output_tokens: int = 0) -> dict[str, Any]:
    formatted = {}
    for qid, answer in answers.items():
        q_type = answer.get("type", "noul")
        if q_type == "noul":
            formatted[qid] = {"type": "noul", "noul": round(answer.get("noul", 0.5), 2)}
        elif q_type == "choice":
            probs = {k: round(v, 2) for k, v in answer.get("probabilities", {}).items()}
            formatted[qid] = {
                "type": "choice",
                "choice": answer.get("choice", max(probs, key=probs.get) if probs else ""),
                "probabilities": probs,
                "confidence": round(_confidence(probs), 2),
            }
        elif q_type == "score":
            probs = {str(k): round(v, 2) for k, v in answer.get("probabilities", {}).items()}
            legend = answer.get("legend", {})
            score_val = answer.get("score", 0.0)
            formatted[qid] = {
                "type": "score",
                "score": round(score_val, 2),
                "probabilities": probs,
                "legend": legend,
                "confidence": round(_confidence(probs), 2),
            }
    return {
        "model": model_name,
        "answers": formatted,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


class SystemOneHandler(BaseHTTPRequestHandler):
    backend: DecisionBackend
    model_name: str = "decision-model"

    def do_POST(self) -> None:
        if self.path not in ("/v1/systemone", "/v1/systemone/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        state = payload.get("state", "")
        questions = payload.get("questions", {})
        if not questions:
            self.send_error(400, "No questions provided")
            return
        try:
            answers = self.backend.evaluate(state, questions)
        except Exception as e:
            self.send_error(500, str(e)[:200])
            return
        response = _format_response(answers, self.model_name)
        response_bytes = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            response = {"models": [{"name": self.model_name, "description": "Open decision model"}]}
            response_bytes = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def serve(backend: DecisionBackend, host: str = "0.0.0.0", port: int = 8000, model_name: str = "decision-model") -> None:
    SystemOneHandler.backend = backend
    SystemOneHandler.model_name = model_name
    server = HTTPServer((host, port), SystemOneHandler)
    print(f"Serving {model_name} on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
