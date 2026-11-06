"""The ability taxonomy for the canonical corpus (#247).

The ten names and the one-line definitions are the decision shapes from the
TypeSafe use-case map, which is also the list #207 is organised around.  Every
report and label uses these exact names; do not introduce a second vocabulary.

A record's ``ability`` is the decision the question asks for, not the dataset
it came from.  ``label_method`` records how the label was assigned and is one
of ``source_rule``, ``heuristic`` or ``model``.  Rules and heuristics cover the
whole corpus today, so no record carries a model label; the constant exists so
a later residue stage cannot silently mix methods.
"""

from __future__ import annotations

import re
from typing import Any

CLASSIFICATION = "Classification"
DETECTION = "Detection"
SCORING = "Scoring"
ROUTING = "Routing"
SEARCH = "Search"
RETRIEVAL = "Retrieval"
RANKING = "Ranking"
VERIFICATION = "Verification"
ML_FEATURE_EXTRACTION = "ML Feature Extraction"
STRUCTURED_DATA_EXTRACTION = "Structured Data Extraction"

ABILITIES: tuple[str, ...] = (
    CLASSIFICATION,
    DETECTION,
    SCORING,
    ROUTING,
    SEARCH,
    RETRIEVAL,
    RANKING,
    VERIFICATION,
    ML_FEATURE_EXTRACTION,
    STRUCTURED_DATA_EXTRACTION,
)

ABILITY_DEFINITIONS: dict[str, str] = {
    CLASSIFICATION: "One known category should win.",
    DETECTION: "A probability that one property is present.",
    SCORING: "The answer belongs on an ordered rubric.",
    ROUTING: "A category selects the next code path.",
    SEARCH: "Find items that match a natural-language query.",
    RETRIEVAL: "Fetch the most relevant context or records for a workflow.",
    RANKING: "Order items by semantic relevance or quality.",
    VERIFICATION: "Check an artifact for specific failure modes.",
    ML_FEATURE_EXTRACTION: (
        "Extract probabilistic signals for a downstream classical model."
    ),
    STRUCTURED_DATA_EXTRACTION: ("Recover known fields from unstructured input."),
}

# Whether the public cookbooks demonstrate the ability as the model's job.
# This decides how to read a zero row: a demonstrated ability is a real gap,
# an ability the cookbooks leave to code is not.  Checked against
# https://docs.typesafe.ai/cookbooks on 2026-11-06.
ABILITY_COOKBOOK: dict[str, str] = {
    CLASSIFICATION: "demonstrated (Hierarchical classification)",
    DETECTION: "demonstrated (Twitch chat filter; Guardrails for LLMs)",
    SCORING: "demonstrated (Re-ranking; Intelliprompter; entity alignment)",
    ROUTING: "demonstrated (Function calling; Skill suggestion)",
    SEARCH: "demonstrated (Line-by-line search)",
    RETRIEVAL: (
        "not demonstrated as the model's job; cookbooks retrieve in code "
        "(BM25 shortlists) and ask Jev to score or rank the candidates"
    ),
    RANKING: "demonstrated (Re-ranking; Skill suggestion top-3 rerank)",
    VERIFICATION: (
        "demonstrated (Double-checking citations; SDE cascade field checks)"
    ),
    ML_FEATURE_EXTRACTION: (
        "demonstrated (Autoresearch feature discovery: free text to numeric "
        "features for a CatBoost regressor)"
    ),
    STRUCTURED_DATA_EXTRACTION: (
        "demonstrated (SDE cascade; Date extraction; Pre-parsed value "
        "extraction; Shopping cart)"
    ),
}

SOURCE_RULE = "source_rule"
HEURISTIC = "heuristic"
MODEL = "model"
LABEL_METHODS: tuple[str, ...] = (SOURCE_RULE, HEURISTIC, MODEL)

# Upstream dataset name -> ability.  These are the names in each record's
# ``source`` field, which the canonical builder preserves.  The split between
# classification and verification follows the decision the question asks:
# NLI-style sources check whether an artifact is supported by evidence, while
# sentiment, topic, intent and multiple-choice knowledge sources pick a
# category.
SOURCE_ABILITY: dict[str, str] = {
    # Category selection.
    "agnews": CLASSIFICATION,
    "arc": CLASSIFICATION,
    "banking77": CLASSIFICATION,
    "boolq": CLASSIFICATION,
    "commonsenseqa": CLASSIFICATION,
    "hellaswag": CLASSIFICATION,
    "logiqa": CLASSIFICATION,
    "mmlu": CLASSIFICATION,
    "piqa": CLASSIFICATION,
    "race": CLASSIFICATION,
    "swag": CLASSIFICATION,
    "winogrande": CLASSIFICATION,
    "sst2": CLASSIFICATION,
    "sst5": CLASSIFICATION,
    "yelp": CLASSIFICATION,
    # Evidence and answer checking.
    "anli": VERIFICATION,
    "contractnli": VERIFICATION,
    "fever": VERIFICATION,
    "mednli": VERIFICATION,
    "mnli": VERIFICATION,
    "multirc": VERIFICATION,
    "tabfact": VERIFICATION,
    "openjev_authored144": VERIFICATION,
    # An ordered rubric.
    "stsb": SCORING,
    # Code search.
    "codesearchnet": SEARCH,
    # Canonical files whose question form is uniform.
    "mmlu_pro_short_reasoning_choice": CLASSIFICATION,
    "hotpotqa_long_choice": CLASSIFICATION,
    "musique_long_choice": CLASSIFICATION,
    "longbench_long_choice": CLASSIFICATION,
    "twowiki_long_choice": CLASSIFICATION,
    "2wiki_long_choice": CLASSIFICATION,
    "race_short_passage_choice": CLASSIFICATION,
}

# The apigen files ask five different decisions.  The file, not the question
# text, is the rule.
APIGEN_ABILITY: dict[str, str] = {
    "apigen_policy_compliance": VERIFICATION,
    "apigen_tool_action": ROUTING,
    "apigen_tool_argument": STRUCTURED_DATA_EXTRACTION,
    "apigen_tool_choice": ROUTING,
    "apigen_tool_next_action": ROUTING,
}

# typed-decisions has twenty instruction families across four domains.  The
# instruction text is exact and unique per family, so it is a source rule.
TYPED_INSTRUCTION_ABILITY: dict[str, str] = {
    "What should the observability system do with this trace?": ROUTING,
    "The evidence indicates a credential or account has been compromised.": DETECTION,
    "This trace requires human review.": ROUTING,
    "How did this agent run turn out?": CLASSIFICATION,
    "How risky was the agent's behaviour in this trace?": SCORING,
    "How quickly does this trace need attention?": SCORING,
    "How should this security alert be dispositioned?": ROUTING,
    "This alert reflects genuinely malicious or unauthorised activity.": DETECTION,
    "How quickly must a responder act on this alert?": SCORING,
    "How severe is the potential impact if this alert is real?": SCORING,
    "How should this vendor invoice be dispositioned?": ROUTING,
    "The invoice reconciles with the purchase order and the recorded delivery.": VERIFICATION,
    "This invoice appears to duplicate an invoice already submitted.": DETECTION,
    "How material is any discrepancy between the invoice, the order and the delivery?": SCORING,
    "How time-sensitive is processing this invoice?": SCORING,
    "What is this customer conversation primarily about?": CLASSIFICATION,
    "This conversation requires a human agent rather than automated handling.": ROUTING,
    "What should the assistant do next with this conversation?": ROUTING,
    "How likely is this customer to stop doing business with us because of this interaction?": SCORING,
    "How time-sensitive is this conversation?": SCORING,
}

# The dominant generated template in each synthetic domain.  Free-form
# variants fall through to the ordered heuristic patterns below.
SYNTHETIC_TEMPLATE_ABILITY: dict[str, str] = {
    "What type of reasoning error, if any, is present in this question's common wrong answers?": DETECTION,
    "What is the relationship between premise and hypothesis?": CLASSIFICATION,
    "What type of issue, if any, is present in this code?": DETECTION,
    "What type of commonsense reasoning best explains this scenario?": CLASSIFICATION,
    "What is the primary rhetorical purpose of this content?": CLASSIFICATION,
    "What is the primary issue with this student's response?": DETECTION,
    "What investment action is most appropriate given this analysis?": ROUTING,
    "Which action is the strongest strategic move?": ROUTING,
    "Which legal determination best applies to this situation?": CLASSIFICATION,
    "What comprehension strategy is most needed to answer questions about this document?": CLASSIFICATION,
    "What is the appropriate triage level for this patient?": ROUTING,
    "What coordination strategy should the deciding agent adopt?": ROUTING,
    "What type of multi-hop reasoning is required?": CLASSIFICATION,
    "Which navigation strategy should be used next?": ROUTING,
    "What numerical operation is needed to answer the question?": CLASSIFICATION,
    "What is the best resource allocation strategy?": ROUTING,
    "What moderation action should be taken on this content?": ROUTING,
    "What conclusion is best supported by this experimental evidence?": VERIFICATION,
    "What should the next action be?": ROUTING,
    "Which direction should you travel to reach the target from the starting point?": ROUTING,
}

# Ordered patterns for free-form synthetic instructions and unrecognised
# sources.  The first match wins.  They are deliberately narrow: a question
# that matches none falls back to the question type, and the coverage report
# counts the fallbacks.
_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "structured_value",
        re.compile(
            r"\b(?:what value|which value|extract the|what (?:is|was) the "
            r"(?:value|amount|date|number|score) of)\b",
            re.IGNORECASE,
        ),
        STRUCTURED_DATA_EXTRACTION,
    ),
    (
        "routing",
        re.compile(
            r"\b(?:what should .{0,60} do (?:next|now|upon|first)|"
            r"which (?:action|tool|strategy|route|direction)|next action|"
            r"disposition|triage level|moderation action|"
            r"which .{0,30} should (?:be used|the agent|the assistant|you))\b",
            re.IGNORECASE,
        ),
        ROUTING,
    ),
    (
        "evidence_sufficiency",
        re.compile(
            r"\b(?:"
            r"(?:sufficient|insufficient|enough (?:information|detail))\b.{0,80}"
            r"\b(?:determine|establish|conclude|answer|decide|confirm)"
            r"|is there (?:evidence|enough detail)"
            r")",
            re.IGNORECASE,
        ),
        VERIFICATION,
    ),
    (
        "verification",
        re.compile(
            r"\b(?:does .{0,60}(?:support|entail|reconcile|comply|match|verify)|"
            r"is (?:it|this) true|assess the claim|is this answer correct|"
            r"supports? the claim|is entailed|correctly implements)\b",
            re.IGNORECASE,
        ),
        VERIFICATION,
    ),
    (
        "detection",
        re.compile(
            r"\b(?:is .{0,40} present|malicious|violation|risk factor|"
            r"(?:pose|poses|posing) .{0,30}threat|explicit threat|"
            r"safety concern|bottleneck|information gap|suspicious)\b",
            re.IGNORECASE,
        ),
        DETECTION,
    ),
)

TYPE_DEFAULT: dict[str, str] = {
    # A generic yes/no question is a two-way category choice, not a detection
    # of a named property.  Detection is reserved for the explicit patterns.
    "noul": CLASSIFICATION,
    "choice": CLASSIFICATION,
    "score": SCORING,
}


def _instruction(question: dict[str, Any]) -> str:
    return str(question.get("instructions") or "").strip()


def _synthetic_domain(item_id: str) -> str:
    parts = item_id.split("-")
    return parts[1] if len(parts) > 1 else ""


def classify(
    source: str,
    item_id: str,
    question: dict[str, Any],
) -> tuple[str, str, str]:
    """Return ``(ability, label_method, rule_id)`` for one record.

    Args:
        source: The record's ``source`` field.
        item_id: The record id; the synthetic domain is read from it.
        question: The record's question object.

    Returns:
        The ability name, the label method, and a stable rule id naming the
        rule that fired.

    Raises:
        ValueError: The source is not in the taxonomy's rule tables.  A new
            source must be mapped explicitly rather than defaulted.
    """
    q_type = str(question.get("type") or "")
    instruction = _instruction(question)

    if q_type == "score":
        return SCORING, SOURCE_RULE, "question_type:score"

    if source == "typed_decisions":
        ability = TYPED_INSTRUCTION_ABILITY.get(instruction)
        if ability is not None:
            return ability, SOURCE_RULE, "typed_instruction"
        return _heuristic(instruction, q_type, "typed_decisions")

    if source in APIGEN_ABILITY:
        return APIGEN_ABILITY[source], SOURCE_RULE, f"apigen_file:{source}"

    if source in SOURCE_ABILITY:
        return SOURCE_ABILITY[source], SOURCE_RULE, f"upstream_source:{source}"

    if source == "synthetic":
        ability = SYNTHETIC_TEMPLATE_ABILITY.get(instruction)
        if ability is not None:
            return ability, SOURCE_RULE, "synthetic_template"
        return _heuristic(instruction, q_type, _synthetic_domain(item_id))

    raise ValueError(f"no ability rule for source {source!r}")


def _heuristic(instruction: str, q_type: str, source: str) -> tuple[str, str, str]:
    for name, pattern, ability in _PATTERNS:
        if pattern.search(instruction):
            return ability, HEURISTIC, f"pattern:{name}:{source}"
    default = TYPE_DEFAULT.get(q_type, CLASSIFICATION)
    return default, HEURISTIC, f"type_default:{q_type}:{source}"


def is_known_source(source: str) -> bool:
    """Return whether :func:`classify` has an explicit rule for ``source``."""
    return (
        source in SOURCE_ABILITY
        or source in APIGEN_ABILITY
        or source in ("typed_decisions", "synthetic")
    )
