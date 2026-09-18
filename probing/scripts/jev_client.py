"""Thin Jev API client using zerodep httpclient. No external dependencies."""

import json
import os
import sys

# Vendor httpclient.py alongside this file, or add zerodep to sys.path
sys.path.insert(0, os.path.dirname(__file__))
from httpclient import Client


class JevClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).rstrip("/")
        self.api_key = api_key or os.environ["TYPESAFE_API_KEY"]
        self._client = Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    def ask(self, state, questions: dict, model: str = "jev-latest") -> dict:
        payload = {"state": state, "model": model, "questions": questions}
        resp = self._client.post(f"{self.base_url}/v1/systemone", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Jev API error {resp.status_code}: {resp.text}")
        return resp.json()

    def noul(self, state, question_id: str, instructions: str, **kwargs) -> dict:
        q = {"type": "noul", "instructions": instructions}
        if "criteria" in kwargs:
            q["criteria"] = kwargs["criteria"]
        return self.ask(state, {question_id: q}, **{k: v for k, v in kwargs.items() if k == "model"})

    def choice(self, state, question_id: str, instructions: str, criteria: dict, **kwargs) -> dict:
        q = {"type": "choice", "instructions": instructions, "criteria": criteria}
        return self.ask(state, {question_id: q}, **{k: v for k, v in kwargs.items() if k == "model"})

    def score(self, state, question_id: str, instructions: str, criteria: list, **kwargs) -> dict:
        q = {"type": "score", "instructions": instructions, "criteria": criteria}
        return self.ask(state, {question_id: q}, **{k: v for k, v in kwargs.items() if k == "model"})

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
