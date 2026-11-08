"""Generate ordinal (score) questions over existing states.

The generator in ``scripts/generate_score_questions.py`` uses these pieces:

* candidate selection: distinct normalised states that do not already carry a
  score question, stratified across sources;
* rubric assignment: one ordinal rubric per source family, resolved from
  ``data/score_rubrics.json`` or ``data.synthetic_templates.py``;
* response validation: a Jev score answer must cover every level, sum to one,
  and not collapse onto a single level;
* row construction: ``label`` is the argmax level, the full distribution is
  kept in ``teacher_probs["jev"]``, and the provenance is ``teacher``.

Nothing here calls the network.  The script owns the API loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Candidate",
    "allocate_quotas",
    "build_item",
    "candidate_id",
    "evidence_problem",
    "expected_level",
    "iter_candidates",
    "label_level",
    "load_rubric_config",
    "normalise_state",
    "order_candidates",
    "select_candidates",
    "source_slug",
    "validate_distribution",
]

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_SLUG = re.compile(r"[^a-z0-9]+")


def normalise_state(text: Any) -> str:
    """Return the comparison form of a state: NFC, casefold, one space."""
    value = unicodedata.normalize("NFC", str(text)).casefold()
    return _NON_WORD.sub(" ", value).strip()


def source_slug(source: str) -> str:
    return _SLUG.sub("-", source.lower()).strip("-") or "unknown"


def candidate_id(source: str, state: str) -> str:
    """Stable id for a generated score question over ``state``."""
    digest = hashlib.sha1(normalise_state(state).encode("utf-8")).hexdigest()[:10]
    return f"scoregen-{source_slug(source)}-{digest}"


@dataclass(frozen=True)
class Candidate:
    """One state selected for a generated score question."""

    source: str
    state: str
    split: str | None
    group: str | None
    rubric: Mapping[str, Any]
    original_id: str = ""

    @property
    def id(self) -> str:
        return candidate_id(self.source, self.state)


def evidence_problem(state: str, evidence: Any) -> str | None:
    """Return why ``state`` does not carry the rubric's evidence, or ``None``.

    ``evidence`` names where in the state the rubric's evidence lives.  The
    literal ``"state text"`` means the state itself; a JSON object state may
    carry the named key; otherwise the name is a labelled section such as
    ``"Contract clause"`` or ``"System"`` and must be followed by content.
    """
    text = state.strip()
    if not text:
        return "empty state"
    if not isinstance(evidence, str) or not evidence.strip():
        return "no evidence field declared"
    if evidence == "state text":
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and evidence in parsed:
        value = parsed[evidence]
        return None if value not in (None, "", [], {}) else f"{evidence!r} is empty"
    marker = f"{evidence}:"
    start = text.find(marker)
    if start < 0:
        return f"{evidence!r} not found in the state"
    body = text[start + len(marker):].split("\n\n", 1)[0]
    return None if body.strip() else f"{evidence!r} is empty in the state"


def iter_candidates(
    items: Iterable[Mapping[str, Any]],
    score_states: set[str],
    rubrics: Mapping[str, Any],
    skipped: Counter[str] | None = None,
) -> Iterable[Candidate]:
    """Yield the first item for each distinct state with no score question.

    ``score_states`` holds the normalised states of every existing score item.
    ``rubrics`` maps a source name to its resolved rubric, evidence included.  A
    source with no entry is an error: there is no default rubric, because a
    rubric may only be assigned to a source whose state is declared to carry the
    evidence for it.  A state that does not carry the declared evidence is
    skipped and counted in ``skipped["evidence:<source>"]``.
    """
    seen: set[str] = set()
    for item in items:
        question = item.get("question") or {}
        state = item.get("state")
        if not isinstance(state, str) or not state.strip():
            continue
        normalised = normalise_state(state)
        if not normalised or normalised in score_states or normalised in seen:
            continue
        source = str(item.get("source") or "missing")
        rubric = rubrics.get(source)
        if rubric is None:
            raise ValueError(
                f"source {source!r} has no rubric in data/score_rubrics.json; "
                "declare the field that carries the rubric's evidence, or drop "
                "the source from the generation run"
            )
        problem = evidence_problem(state, rubric.get("evidence"))
        if problem is not None:
            if skipped is not None:
                skipped[f"evidence:{source}"] += 1
            continue
        seen.add(normalised)
        yield Candidate(
            source=source,
            state=state,
            split=item.get("split"),
            group=item.get("group"),
            rubric=rubric,
            original_id=str(item.get("id") or ""),
        )


def allocate_quotas(
    pool_sizes: Mapping[str, int], target: int, floor: int = 25, cap: int = 350
) -> dict[str, int]:
    """Return a deterministic per-source quota summing to ``target``.

    Every source first gets ``min(pool, floor)``; the remainder is water-filled
    towards the largest remaining pool, so an abundant source gains seats
    without a rare source losing coverage.  Sources are capped at ``cap`` or
    their pool size.
    """
    sources = sorted(pool_sizes)
    if not sources or target <= 0:
        return {source: 0 for source in sources}
    quota = {source: min(pool_sizes[source], floor) for source in sources}
    capacity = {source: min(pool_sizes[source], cap) for source in sources}
    total_capacity = sum(capacity.values())
    target = min(target, total_capacity)
    while sum(quota.values()) > target:
        # Give back from the largest quota above zero, deterministically.
        source = max(sources, key=lambda name: (quota[name], name))
        if quota[source] == 0:
            break
        quota[source] -= 1
    while sum(quota.values()) < target:
        room = [source for source in sources if quota[source] < capacity[source]]
        if not room:
            break
        source = max(
            room,
            key=lambda name: (pool_sizes[name] / (quota[name] + 1), name),
        )
        quota[source] += 1
    return quota


def select_candidates(
    candidates: Sequence[Candidate], target: int, floor: int, cap: int, seed: int
) -> list[Candidate]:
    """Stratified, seeded selection of ``target`` candidates across sources."""
    by_source: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate.source, []).append(candidate)
    quotas = allocate_quotas(
        {source: len(values) for source, values in by_source.items()},
        target,
        floor=floor,
        cap=cap,
    )
    selected: list[Candidate] = []
    for source in sorted(by_source):
        values = sorted(
            by_source[source],
            key=lambda candidate: hashlib.sha256(
                f"{seed}:{source}:{normalise_state(candidate.state)}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(values[: quotas[source]])
    return selected


def order_candidates(candidates: Sequence[Candidate], source_order: Sequence[str]) -> list[Candidate]:
    """Interleave sources round-robin so early stopping does not bias the mix."""
    by_source: dict[str, list[Candidate]] = {source: [] for source in source_order}
    for candidate in candidates:
        by_source.setdefault(candidate.source, []).append(candidate)
    ordered: list[Candidate] = []
    index = 0
    while True:
        added = False
        for source in source_order:
            values = by_source.get(source, [])
            if index < len(values):
                ordered.append(values[index])
                added = True
        if not added:
            return ordered
        index += 1


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def validate_distribution(
    probabilities: Mapping[str, Any], n_levels: int
) -> str | None:
    """Return a rejection reason, or ``None`` when the distribution is usable.

    Reasons: ``keys`` (missing/extra levels), ``nonnumeric``, ``range``
    (outside [0, 1]), ``sum`` (not within 0.02 of 1), ``degenerate`` (one level
    at or above 0.99, or fewer than two levels above 0.01).
    """
    expected = [str(index) for index in range(n_levels)]
    if sorted(probabilities) != expected:
        return "keys"
    values = []
    for key in expected:
        number = _numeric(probabilities[key])
        if number is None:
            return "nonnumeric"
        values.append(number)
    if any(value < 0.0 or value > 1.0 for value in values):
        return "range"
    total = sum(values)
    if abs(total - 1.0) > 0.02:
        return "sum"
    if max(values) >= 0.99 or sum(1 for value in values if value > 0.01) < 2:
        return "degenerate"
    return None


def expected_level(probabilities: Mapping[str, Any]) -> float:
    values = [_numeric(probabilities[str(index)]) or 0.0 for index in range(len(probabilities))]
    return sum(index * value for index, value in enumerate(values))


def label_level(probabilities: Mapping[str, Any]) -> float:
    """Return the argmax level as a float, the row's ``label``."""
    key = max(probabilities, key=lambda name: (_numeric(probabilities[name]) or 0.0, name))
    return float(int(key))


def build_item(candidate: Candidate, probabilities: Mapping[str, Any]) -> dict[str, Any]:
    """Build one typed score row from a validated distribution.

    ``label`` is the argmax level and is for evaluation only.  The training
    target is the distribution in ``teacher_probs["jev"]``; ``teacher``
    provenance is not eligible for a hard-label loss.
    """
    clean = {
        str(index): round(_numeric(probabilities[str(index)]) or 0.0, 6)
        for index in range(len(probabilities))
    }
    item: dict[str, Any] = {
        "id": candidate.id,
        "state": candidate.state,
        "question": {
            "type": "score",
            "instructions": candidate.rubric["instructions"],
            "criteria": list(candidate.rubric["criteria"]),
        },
        "label": label_level(clean),
        "source": candidate.source,
        "split": candidate.split or "train",
        "provenance": "teacher",
        "teacher_probs": {"jev": clean},
    }
    if candidate.group is not None:
        item["group"] = candidate.group
    return item


def load_rubric_config(path: str | Path, templates: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve ``data/score_rubrics.json`` against the domain templates.

    A source entry is either ``{"template": name, "evidence": ...}`` (resolved
    from ``templates``) or an inline ``{"instructions", "criteria", "evidence"}``
    object.  ``evidence`` is required on every entry: it names where in the
    state the rubric's evidence lives.  A ``default`` rubric is rejected, because
    it would assign a rubric to a source whose evidence nobody declared.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if "default" in config:
        raise ValueError(
            "data/score_rubrics.json may not carry a `default` rubric: every "
            "source must declare the field that carries its evidence"
        )
    resolved: dict[str, Any] = {}
    for source, entry in config.get("sources", {}).items():
        evidence = entry.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(
                f"source {source!r} has no `evidence` field naming where the "
                "rubric's evidence lives in the state"
            )
        if "template" in entry:
            name = entry["template"]
            rubric = dict(templates[name]["score_template"])
        else:
            rubric = {
                "instructions": entry["instructions"],
                "criteria": list(entry["criteria"]),
            }
        rubric["evidence"] = evidence.strip()
        resolved[source] = rubric
    return resolved
