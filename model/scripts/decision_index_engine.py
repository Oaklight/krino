"""Engine adapter from Krino's per-option scoring to the Decision Index kit.

The Decision Index reproduction kit (``apolinario/decision-index``, MIT) calls an
engine once per request::

    response, raw = engine(state, questions)

and validates that every question is answered, that every choice option carries a
finite probability, and that the probabilities sum to one. Our scorer already
produces that shape through ``predict_row``; this module only translates.

Use it in-process, not through ``model/src/server.py``: that server rounds every
probability to two decimals, which destroys calibration and can break the kit's
sum-to-one check. The in-process engine keeps the full float distribution.

Example, scoring the v4 control on the rebuilt public suite::

    python -m decision_index pipeline \
        --engine model.scripts.decision_index_engine:KrinoRawEngine \
        --model Qwen/Qwen3.5-4B \
        --option shared_prefix=true \
        --option strict_length=true \
        --option device=cuda \
        --out runs/v4-control

The kit forbids truncation: a request that does not fit must be refused, not cut.
``strict_length=True`` mirrors the token accounting of ``RawReranker`` and raises
``Unsupported`` before a call that would truncate. Leave it on for any run whose
score is reported.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

try:  # the kit is not a dependency of the Krino test environment
    from decision_index.engines.base import Engine as _BaseEngine
    from decision_index.engines.base import Unsupported
except ImportError:  # pragma: no cover - exercised by the unit tests

    class _BaseEngine:  # type: ignore[no-redef]
        """Stand-in for the kit's Engine when it is not installed."""

        name = "engine"
        provenance: dict[str, Any] = {}
        latency = "in-process request wall time; excludes model loading"

        def __init__(self, **options: Any) -> None:
            self.options = options

        def warmup(self) -> None:
            warm = {
                "warmup": {
                    "type": "choice",
                    "instructions": "Which color is named?",
                    "criteria": {"red": "red", "blue": "blue"},
                }
            }
            self("The color is red.", warm)

        def synchronize(self) -> None:
            pass

        def runtime(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            pass

    class Unsupported(ValueError):  # type: ignore[no-redef]
        """Stand-in for the kit's declared-capacity exception."""


_SCORE_REFUSAL = (
    "the Decision Index interface has choice and noul questions only; "
    "a score question cannot be represented"
)


def _raw_options(question: dict[str, Any]) -> list[tuple[str, str]]:
    """Options in the order and spelling ``eval_gate._raw_options`` uses.

    Kept in sync with ``model/scripts/eval_gate.py`` on purpose: the noul
    default descriptions and the stringified keys are part of the prompt, so a
    different spelling here would score a different prompt from the gate.
    """
    q_type = question.get("type")
    criteria = question.get("criteria") or {}
    if q_type == "noul":
        return [
            ("no", str(criteria.get("false", criteria.get("no", "No")))),
            ("yes", str(criteria.get("true", criteria.get("yes", "Yes")))),
        ]
    if q_type == "choice":
        return [(str(key), str(value)) for key, value in criteria.items()]
    raise Unsupported(f"{_SCORE_REFUSAL} (got {q_type!r})")


def _document(q_type: str, label: str, description: str, *, shared: bool) -> str:
    """The option document exactly as ``RawReranker`` renders it."""
    if q_type == "score":
        body = f"Rubric level {label}: {description}"
    else:
        body = f"Answer option {label}: {description}"
    return " " + body if shared else body


def capacity_violation(
    scorer: Any,
    state: Any,
    instructions: str,
    q_type: str,
    options: Iterable[tuple[str, str]],
    instruction: str,
) -> str | None:
    """Why a request would be truncated, or ``None`` when it fits.

    Mirrors the token accounting of ``RawReranker.score_many`` when
    ``shared_prefix`` is set and of ``RawReranker._score_option`` otherwise.
    That code cuts the document with ``[:budget]``; the kit records a cut row as
    a silent wrong answer, so the adapter turns the cut into a declared refusal.

    Returns ``None`` when the scorer does not expose the fields needed to check.
    A caller that sets ``strict_length`` on such a scorer is trusting it not to
    truncate.
    """
    tokenizer = getattr(scorer, "tokenizer", None)
    max_length = getattr(scorer, "max_length", None)
    prefix_tokens = getattr(scorer, "prefix_tokens", None)
    suffix_tokens = getattr(scorer, "suffix_tokens", None)
    if (
        tokenizer is None
        or max_length is None
        or prefix_tokens is None
        or suffix_tokens is None
    ):
        return None

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    query = f"{state}\n\nQuestion: {instructions}"
    shared = bool(getattr(scorer, "shared_prefix", False))
    suffix_len = len(suffix_tokens)

    if shared:
        prefix_text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>:"
        prefix_len = len(prefix_tokens) + len(encode(prefix_text))
        budget = max_length - prefix_len - suffix_len
        if budget <= 0:
            return (
                f"the shared prefix is {prefix_len} tokens and the suffix is "
                f"{suffix_len}; no budget remains under max_length={max_length}"
            )
        for label, description in options:
            needed = len(encode(_document(q_type, label, description, shared=True)))
            if needed > budget:
                return (
                    f"option {label!r} needs {needed} tokens after a "
                    f"{prefix_len}-token prefix; budget is {budget} "
                    f"(max_length={max_length})"
                )
        return None

    budget = max_length - len(prefix_tokens) - suffix_len
    for label, description in options:
        text = (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {_document(q_type, label, description, shared=False)}"
        )
        needed = len(encode(text))
        if needed > budget:
            return (
                f"option {label!r} needs {needed} tokens; budget is {budget} "
                f"(max_length={max_length})"
            )
    return None


class KrinoEngine(_BaseEngine):
    """Wrap any scorer with ``predict_row(item)`` as a Decision Index engine.

    ``scorer`` must expose ``predict_row`` returning a
    ``model.evaluation.gate.Prediction`` whose ``keys`` and ``probs`` align. Both
    ``RawReranker`` and ``ResidualScorer`` in ``model/scripts/eval_gate.py`` do.
    """

    name = "krino"

    def __init__(
        self,
        scorer: Any,
        model_name: str = "krino",
        *,
        strict_length: bool = True,
        instruction: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.scorer = scorer
        self.model_name = model_name
        self.strict_length = strict_length
        self.instruction = instruction
        if self.strict_length and self.instruction is None:
            self.instruction = self._load_instruction()
        self.provenance = getattr(scorer, "provenance", None) or {
            "kind": "krino-engine",
            "model": model_name,
            "scorer": type(scorer).__name__,
        }
        self.latency = (
            "in-process request wall time including prompt construction and "
            "model inference; excludes loading"
        )

    @staticmethod
    def _load_instruction() -> str:
        from model.scripts.eval_gate import RAW_RERANKER_INSTRUCTION

        return RAW_RERANKER_INSTRUCTION

    def __call__(self, state: Any, questions: dict[str, Any]):
        answers: dict[str, Any] = {}
        for question_id, question in questions.items():
            q_type = question.get("type")
            options = _raw_options(question)
            if self.strict_length:
                reason = capacity_violation(
                    self.scorer,
                    state,
                    str(question.get("instructions", "")),
                    q_type,
                    options,
                    self.instruction,
                )
                if reason is not None:
                    raise Unsupported(reason)

            item = SimpleNamespace(
                id=question_id,
                state=state,
                question=question,
                label=None,
                source="decision-index",
                split="test",
                group=None,
            )
            prediction = self.scorer.predict_row(item)
            probabilities = {
                str(key): float(prob)
                for key, prob in zip(prediction.keys, prediction.probs)
            }

            if q_type == "choice":
                missing = set(question["criteria"]) - set(probabilities)
                if missing:
                    raise ValueError(
                        f"scorer omitted options for {question_id!r}: {sorted(missing)}"
                    )
                if not probabilities:
                    raise ValueError(f"scorer returned no options for {question_id!r}")
                answers[question_id] = {
                    "type": "choice",
                    "choice": max(probabilities, key=probabilities.get),
                    "probabilities": probabilities,
                }
            else:
                p_yes = probabilities.get("yes")
                if p_yes is None:
                    raise ValueError(
                        f"noul answer for {question_id!r} has no 'yes' probability"
                    )
                answers[question_id] = {"type": "noul", "noul": p_yes}

        return {"model": self.model_name, "answers": answers}, None


class KrinoRawEngine(KrinoEngine):
    """The v4 control path: a raw backbone scored by yes/no logits.

    Builds ``eval_gate.RawReranker`` with the requested loader. Qwen3.5 4B needs
    the architecture-aware loader; ``RawReranker`` falls back to it when
    ``AutoModelForCausalLM`` rejects the config, which is why this class exists
    instead of the kit's stock ``transformers`` engine.
    """

    name = "krino-raw"

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        shared_prefix: bool = True,
        loader_type: str | None = None,
        strict_length: bool = True,
        **options: Any,
    ) -> None:
        from model.scripts.eval_gate import RAW_RERANKER_INSTRUCTION, RawReranker

        scorer = RawReranker(model, device, loader_type=loader_type)
        scorer.shared_prefix = bool(shared_prefix)
        super().__init__(
            scorer,
            model_name=model,
            strict_length=bool(strict_length),
            instruction=RAW_RERANKER_INSTRUCTION,
            **options,
        )
        self.provenance = {
            "kind": "krino-raw-reranker",
            "model": model,
            "device": device,
            "shared_prefix": bool(shared_prefix),
            "loader_type": loader_type,
            "scoring_path": scorer.scoring_path,
        }
