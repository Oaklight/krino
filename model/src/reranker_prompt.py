"""Score a typed question's options the way the base reranker does.

Qwen3-Reranker is a causal LM that judges relevance with a yes/no logit. Scoring
an option means wrapping ``<Instruct>/<Query>/<Document>`` in the model's chat
template, running the model once per option, and reading P(yes) at the last
position. That is a different scorer from the trained choice head, which pools
option hidden states through an attention head, and the two disagree by 14-21
points on JevBench (#180). Keeping the prompt construction here lets the JevBench
harness and the general calibration harness score options identically, so a
comparison between them compares scorers rather than prompt formats.
"""

from __future__ import annotations

from typing import Any, Sequence

INSTRUCTION = (
    "Classify: which option correctly answers the question for the given situation? "
    "Rate how well each option fits."
)

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_query_doc(
    state: str, instructions: str, q_type: str, label: str, description: str
) -> str:
    """Build one query-document pair in Qwen3-Reranker's expected format.

    Args:
        state: The situation text.
        instructions: The question being asked of the situation.
        q_type: ``noul``, ``choice`` or ``score``; score rubrics are named
            differently from answer options.
        label: Option key as presented.
        description: Option text.
    """
    query_text = f"{state}\n\nQuestion: {instructions}"
    if q_type == "score":
        doc_text = f"Rubric level {label}: {description}"
    else:
        doc_text = f"Answer option {label}: {description}"
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {query_text}\n<Document>: {doc_text}"


def get_options(item: Any) -> list[tuple[str, str]]:
    """Return ``(label, description)`` pairs for each answer option of an item."""
    q = item.question
    criteria = q.get("criteria", {})
    if q["type"] == "noul":
        return [
            ("no", str(criteria.get("false", criteria.get("no", "No")))),
            ("yes", str(criteria.get("true", criteria.get("yes", "Yes")))),
        ]
    if q["type"] == "choice":
        return [(k, str(v)) for k, v in criteria.items()]
    if q["type"] == "score":
        if isinstance(criteria, list):
            return [(str(index), str(desc)) for index, desc in enumerate(criteria)]
        return [(k, str(v)) for k, v in criteria.items()]
    return []


def yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve the token ids the reranker answers with.

    The words are looked up in the vocabulary rather than encoded, matching the
    JevBench harness, and an unknown token raises instead of silently becoming
    whatever id the tokenizer returns for ``<unk>``.
    """
    true_id = tokenizer.convert_tokens_to_ids("yes")
    false_id = tokenizer.convert_tokens_to_ids("no")
    unk = getattr(tokenizer, "unk_token_id", None)
    if true_id is None or false_id is None or unk in (true_id, false_id):
        raise ValueError(f"vocabulary has no yes/no tokens ({true_id}, {false_id})")
    return int(true_id), int(false_id)


def score_options_native(
    backbone: torch.nn.Module,
    tokenizer: Any,
    state: str,
    question: dict[str, Any],
    options: Sequence[tuple[str, str]],
    max_length: int = 8192,
    device: str | None = None,
) -> list[float]:
    """Score each option independently and return P(yes) per option.

    Args:
        backbone: The causal LM, in eval mode.
        tokenizer: Its tokenizer.
        state: Situation text.
        question: The question dict, for ``instructions`` and ``type``.
        options: ``(label, description)`` pairs.
        max_length: Truncation length for one prompt.
        device: Device holding ``backbone``; inferred when omitted.

    Returns:
        One probability per option, in the order given. They are independent
        per-option scores and do not sum to one.
    """
    import torch

    device = device or str(next(backbone.parameters()).device)
    true_id, false_id = yes_no_token_ids(tokenizer)
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    scores: list[float] = []
    for label, description in options:
        text = build_query_doc(
            state, str(question.get("instructions", "")), question["type"], label, description
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokens[: max_length - len(prefix_tokens) - len(suffix_tokens)]
        input_ids = prefix_tokens + tokens + suffix_tokens
        inputs = {
            "input_ids": torch.tensor([input_ids], device=device),
            "attention_mask": torch.ones(1, len(input_ids), device=device),
        }
        with torch.no_grad():
            logits = backbone(**inputs).logits[0, -1, :].float()
        pair = torch.softmax(torch.stack([logits[false_id], logits[true_id]]), dim=0)
        scores.append(pair[1].item())
    return scores
