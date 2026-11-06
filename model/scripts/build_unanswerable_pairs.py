"""Build paired unanswerable variants of existing typed questions.

The abstention head needs negatives that differ from an answerable original in
exactly one factor.  This builder derives them by construction: each transform
edits only the evidence part of ``state`` and leaves the question untouched, so
the resulting label needs no teacher call.

Three transforms are produced:

``support_removed``
    Delete the evidence sentence that best supports the question.  The item is
    unanswerable because its support is gone.  Requires an answerable noul item
    (or a ``supports`` fever item) whose evidence has at least two sentences.
``out_of_scope``
    Replace the evidence with evidence from an item in another group of the
    same source.  The question stays, the evidence no longer belongs to it.
``entity_corrupted``
    Replace every mention of an entity shared by the question and the evidence
    with an entity from another group of the same source that has the same
    token shape and the same coarse type (proper name, common term, acronym,
    organisation, or digit string).  Type matching keeps the substitution
    wrong in meaning but plausible in kind, so a surface classifier cannot
    learn "a person name in a contract means unanswerable".  The
    ``--entity-donor-match shape`` option drops the type match to measure what
    it costs in rows.

Every variant keeps the original ``label`` as provenance, sets
``answerable=False`` and a closed-set ``reason``, and strips ``teacher_probs``
so a soft target cannot leak the old label.  The paired answerable original is
written with ``answerable=True``.  A pair is verified before it is emitted: the
question, source and group are identical and the state differs only inside the
evidence span.

Because the label is kept for provenance, a consumer that trains the task head
must mask reason-bearing rows out of the task loss; only the answerable head
should see them.  The design note for that integration is
``research/09-answerable-head-integration.md``.

Output:

- ``--out``: pairs, each answerable original followed by its variants.
- ``--negatives-out``: variants only, for ``train_residual --answerable-items``
  (the originals already live in the main corpus and must not be duplicated).
- ``--stats``: counts per transform, per source, per type, and the length delta.

Usage:
    python model/scripts/build_unanswerable_pairs.py \
        --items data/benchmarks/label_target_30k_jev.jsonl \
                data/benchmarks/residual_train_20000_jev.jsonl \
        --out data/benchmarks/unanswerable_30k/pairs.jsonl \
        --negatives-out data/benchmarks/unanswerable_30k/negatives.jsonl \
        --stats results/abstention/unanswerable_30k.stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.format import TypedQuestion  # noqa: E402
from data.pipeline import load_jsonl, save_jsonl  # noqa: E402

TRANSFORMS = ("support_removed", "out_of_scope", "entity_corrupted")

#: Transform name -> closed-set reason written on the variant.
REASON_BY_TRANSFORM = {
    "support_removed": "support_removed",
    "out_of_scope": "out_of_scope",
    "entity_corrupted": "entity_corrupted",
}

# ---------------------------------------------------------------------------
# State layouts
# ---------------------------------------------------------------------------

#: Sources whose state separates an evidence span from the question span.
#: The named group is the evidence; the remaining text is the question and is
#: never edited.
STATE_FORMATS: dict[str, re.Pattern[str]] = {
    "boolq": re.compile(
        r"\APassage:\s*(?P<support>.*?)\n\nQuestion:\s*(?P<query>.*)\Z", re.S
    ),
    "multirc": re.compile(
        r"\APassage:\s*(?P<support>.*?)\n\nQuestion:\s*(?P<query>.*?)"
        r"\n\nAnswer:\s*(?P<answer>.*)\Z",
        re.S,
    ),
    "fever": re.compile(
        r"\AClaim:\s*(?P<query>.*?)\n\nEvidence:\s*\n?(?P<support>.*)\Z", re.S
    ),
    "nli": re.compile(
        r"\APremise:\s*(?P<support>.*?)\nHypothesis:\s*(?P<query>.*)\Z", re.S
    ),
    "contractnli": re.compile(
        r"\AContract clause:\s*(?P<support>.*?)\n+Proposition:\s*(?P<query>.*)\Z",
        re.S,
    ),
    "tabfact": re.compile(
        r"\ATable:\s*(?P<support>.*?)\n+Statement:\s*(?P<query>.*)\Z", re.S
    ),
}

SOURCE_FORMAT = {
    "boolq": "boolq",
    "multirc": "multirc",
    "fever": "fever",
    "mnli": "nli",
    "anli": "nli",
    "mednli": "nli",
    "contractnli": "contractnli",
    "tabfact": "tabfact",
}

#: A table is not a sentence sequence, so sentence-level transforms are
#: undefined there.  State swapping still applies.
NO_SENTENCE_TRANSFORMS = {"tabfact"}

WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")
#: Tokens that are already lowercase in the original text.
LOWERCASE_WORD = re.compile(r"[a-z][a-z'\-]*")
STOPWORDS = set(
    "the a an and or but if then of to in on at for with by from as is are was "
    "were be been being this that these those it its their his her our your my "
    "we they he she you i not no do does did have has had can could would should "
    "may might must will shall about into over under between among during without "
    "which what who whom whose when where why how than more most less least such "
    "other same".split()
)

#: Abbreviations that must not end a sentence.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "corp", "st", "vs",
    "etc", "eg", "ie", "no", "fig", "al", "jr", "sr", "mt", "ft", "ave",
    "blvd", "rd", "hwy", "gen", "col", "lt", "sgt", "capt", "rev", "hon",
    "gov", "sen", "rep", "pres", "dept", "univ", "assoc", "bros", "mfg",
    "misc", "approx", "est", "min", "max", "vol", "pp", "ed", "eds", "trans",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
}

#: Capitalised words that are grammar, not an entity.
NON_ENTITIES = set(
    "The A An This That These Those It Its In On At For From As Is Are Was Were "
    "Be What Which Who When Where Why How Does Do Did Based Question Answer "
    "Passage Claim Evidence Premise Hypothesis Contract Table Statement "
    "Proposition If While After Before Since Although Because However But And Or "
    "Not No Yes I We They He She You There Here To By With During Of Into Over "
    "Under Between Among Than More Most Less Least Such Other Same".split()
)

ENTITY = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+(?:[A-Z][\w'’\-]*|[A-Z]{2,}))*\b")
ENTITY_TOKEN = re.compile(r"[A-Za-z0-9][\w'’\-]*")

#: Corporate suffixes that mark an entity as an organisation.
ORG_SUFFIXES = {
    "inc", "ltd", "llc", "plc", "corp", "corporation", "company", "co", "group",
    "gmbh", "ag", "sa", "bv", "llp", "lp", "holdings", "partners",
}

SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+")
TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z'’\-]*)$")


def entity_spans(text: str) -> list[str]:
    """Capitalised entity spans of ``text``, without leading grammar words.

    A greedy capitalised-sequence match can start at a sentence-initial
    grammar word (``Did Acme ...``).  This keeps only the runs of tokens that
    are capitalised and are not grammar, so the result is ``Acme`` and not
    ``Did Acme``.
    """
    spans: list[str] = []
    for match in ENTITY.finditer(text):
        span = match.group(0)
        tokens = ENTITY_TOKEN.findall(span)
        run_start: int | None = None
        run_end = 0
        for token in ENTITY_TOKEN.finditer(span):
            word = token.group(0)
            grammar = word in NON_ENTITIES or (
                len(tokens) == 1 and word.lower() in STOPWORDS
            )
            if word[0].isupper() and not grammar:
                if run_start is None:
                    run_start = token.start()
                run_end = token.end()
            elif run_start is not None:
                spans.append(span[run_start:run_end])
                run_start = None
        if run_start is not None:
            spans.append(span[run_start:run_end])
    return spans


@dataclass(frozen=True)
class SplitState:
    """A state split into a contiguous evidence span and its surroundings."""

    fmt: str
    prefix: str
    support: str
    suffix: str
    query: str

    def with_support(self, state: str, new_support: str) -> str:
        """Replace the evidence span, leaving every other character in place."""
        end = len(state) - len(self.suffix)
        return self.prefix + new_support + state[end:]


def split_state(item: TypedQuestion) -> SplitState | None:
    """Split a known state layout into evidence and question.

    Returns ``None`` when the source has no supported layout or the state does
    not match it.
    """
    fmt = SOURCE_FORMAT.get(item.source)
    if fmt is None:
        return None
    match = STATE_FORMATS[fmt].match(item.state)
    if match is None:
        return None
    start, end = match.span("support")
    groups = match.groupdict()
    query = " ".join(
        part for part in (groups.get("query", ""), groups.get("answer", "")) if part
    )
    return SplitState(
        fmt=fmt,
        prefix=item.state[:start],
        support=item.state[start:end],
        suffix=item.state[end:],
        query=query,
    )


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return non-empty ``(start, end, sentence)`` spans of ``text``.

    The splitter skips a boundary after a single capital letter or a known
    abbreviation, so ``D.C. is ...`` and ``Inc. ...`` stay in one sentence.
    """
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        before = text[: match.start() + 1]
        tail = before[-2:-1]
        if tail.isupper() and tail.isalpha():
            continue
        word = TRAILING_WORD.search(before[:-1])
        if word and word.group(1).lower() in ABBREVIATIONS:
            continue
        end = match.start() + 1
        if text[start:end].strip():
            spans.append((start, end, text[start:end]))
        start = match.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text), text[start:]))
    return spans


def content_tokens(text: str) -> list[str]:
    """Lowercase content tokens, dropping stopwords and one-letter tokens."""
    return [
        token
        for token in WORD.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def common_word_vocabulary(items: Iterable[TypedQuestion]) -> set[str]:
    """Words that occur in lowercase anywhere in the corpus.

    A capitalised span whose tokens all occur lowercase elsewhere is a
    common-noun term (``Confidential Information``); a span whose tokens never
    do is a proper name (``Jackie Evancho``).  This is a cheap stand-in for a
    named-entity type, derived from the corpus itself.
    """
    vocabulary: set[str] = set()
    for item in items:
        vocabulary.update(
            token
            for token in LOWERCASE_WORD.findall(item.state)
            if len(token) >= 3 and token not in STOPWORDS
        )
    return vocabulary


def _is_true_label(label: object) -> bool:
    return label is True or (isinstance(label, str) and label.lower() == "true")


def is_answerable_original(item: TypedQuestion) -> bool:
    """Whether the gold label is supported by the current state.

    Only a supported noul item, or a fever item labelled ``supports``, can lose
    its support.  Other labels are still answerable in principle, but removing
    evidence would not flip them, so sentence-level transforms skip them.
    """
    if item.question.get("type") == "noul":
        return _is_true_label(item.label)
    return item.source == "fever" and item.label == "supports"


# ---------------------------------------------------------------------------
# Transform primitives
# ---------------------------------------------------------------------------


def _cut_span(text: str, start: int, end: int) -> str:
    """Remove ``text[start:end]`` plus one adjacent run of whitespace."""
    if end < len(text) and text[end].isspace():
        while end < len(text) and text[end].isspace():
            end += 1
    elif start > 0 and text[start - 1].isspace():
        while start > 0 and text[start - 1].isspace():
            start -= 1
    return text[:start] + text[end:]


def remove_supporting_sentences(
    split: SplitState,
    token_idf: dict[str, float],
    limit: int,
) -> list[tuple[str, str]]:
    """Return up to ``limit`` ``(new_support, removed_sentence)`` pairs.

    The supporting sentence is the evidence sentence with the highest
    IDF-weighted overlap with the question, normalised by sentence length.
    Only sentences that share at least one content token with the question
    qualify.
    """
    query_tokens = set(content_tokens(split.query))
    units = split_sentences(split.support)
    if len(units) < 2 or not query_tokens:
        return []
    scored: list[tuple[float, int, int, str]] = []
    for start, end, sentence in units:
        sentence_tokens = content_tokens(sentence)
        overlap = query_tokens & set(sentence_tokens)
        if not overlap:
            continue
        weight = sum(token_idf.get(token, 1.0) for token in overlap) / math.sqrt(
            max(1, len(sentence_tokens))
        )
        scored.append((weight, start, end, sentence))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [
        (_cut_span(split.support, start, end), sentence)
        for _weight, start, end, sentence in scored[:limit]
    ]


def key_entities(split: SplitState) -> list[str]:
    """Capitalised spans shared by the question and the evidence, longest first."""
    lowered = split.support.lower()
    entities: list[str] = []
    seen: set[str] = set()
    for entity in entity_spans(split.query):
        if entity.lower() not in lowered or entity.lower() in seen:
            continue
        if len(entity.split()) > 4:
            continue
        seen.add(entity.lower())
        entities.append(entity)
    entities.sort(key=lambda entity: (-len(entity.split()), split.query.find(entity)))
    return entities


def entity_type(entity: str, common_words: set[str]) -> str:
    """Coarse entity type from surface form and a corpus common-word set.

    The types are deliberately coarse: digit strings, acronyms, organisations,
    common-noun terms (``Confidential Information``), proper names (``Jackie
    Evancho``), and mixtures.  A type-matched donor keeps the substitution
    wrong in meaning but plausible in kind, so a surface classifier cannot
    learn "a person name in a contract means unanswerable".
    """
    tokens = ENTITY_TOKEN.findall(entity)
    if any(character.isdigit() for character in entity):
        return "digit"
    if any(token.isupper() and len(token) >= 2 for token in tokens):
        return "acronym"
    if any(token.lower() in ORG_SUFFIXES for token in tokens):
        return "org"
    if not tokens:
        return "other"
    common = sum(1 for token in tokens if token.lower() in common_words)
    if common == len(tokens):
        return "common"
    if common == 0:
        return "proper"
    return "mixed"


def entity_shape(
    entity: str, common_words: set[str], match: str = "type"
) -> tuple[int, bool, str]:
    """Shape used to pick a plausible replacement.

    ``match="type"`` requires the same token count, digit presence, and coarse
    entity type.  ``match="shape"`` drops the type, the pre-type-match
    behaviour, and is kept to measure what the type match costs in rows.
    """
    type_label = entity_type(entity, common_words) if match == "type" else "any"
    return (
        len(entity.split()),
        any(character.isdigit() for character in entity),
        type_label,
    )


def corrupt_entities(
    split: SplitState,
    own_group: str,
    donors: dict[tuple[int, bool, str], dict[str, list[str]]],
    rng: random.Random,
    limit: int,
    common_words: set[str],
    match: str = "type",
) -> list[tuple[str, str, str]]:
    """Return up to ``limit`` ``(new_support, original_entity, donor_entity)``.

    Every mention of a key entity in the evidence is replaced with an entity
    from another group of the same source that has the same shape (and, by
    default, the same coarse type).
    """
    results: list[tuple[str, str, str]] = []
    for entity in key_entities(split):
        by_group = donors.get(entity_shape(entity, common_words, match)) or {}
        groups = [group for group in by_group if group and group != own_group]
        rng.shuffle(groups)
        donor = None
        for group in groups:
            bucket = by_group[group]
            for _attempt in range(min(8, len(bucket))):
                candidate = rng.choice(bucket)
                if (
                    candidate.lower() != entity.lower()
                    and candidate.lower() not in entity.lower()
                    and entity.lower() not in candidate.lower()
                ):
                    donor = candidate
                    break
            if donor is not None:
                break
        if donor is None:
            continue
        pattern = re.compile(
            r"(?<!\w)" + re.escape(entity) + r"(?!\w)", re.IGNORECASE
        )
        new_support, replaced = pattern.subn(donor, split.support)
        if replaced:
            results.append((new_support, entity, donor))
        if len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------------------
# Pair construction and verification
# ---------------------------------------------------------------------------


def _clone_variant(item: TypedQuestion, **changes: object) -> TypedQuestion:
    return replace(
        item,
        teacher_probs=None,
        teacher_name=None,
        answerable=False,
        **changes,
    )


def pair_error(
    original: TypedQuestion,
    variant: TypedQuestion,
    transform: str,
    expected_support: str,
) -> str | None:
    """Return the first construction violation, or ``None``.

    The check is by construction, not by a teacher: the variant must keep the
    question, source and group, and its state must equal the original state
    with only the evidence span replaced by ``expected_support``.
    """
    if variant.question != original.question:
        return "question changed"
    if variant.source != original.source or variant.group != original.group:
        return "source or group changed"
    if variant.answerable is not False:
        return "variant is not marked answerable=False"
    if variant.reason != REASON_BY_TRANSFORM[transform]:
        return "reason does not name the transform"
    if variant.teacher_probs is not None:
        return "teacher_probs leaked into the variant"
    original_split = split_state(original)
    variant_split = split_state(variant)
    if original_split is None or variant_split is None:
        return "state no longer parses"
    if variant_split.prefix != original_split.prefix:
        return "text before the evidence changed"
    if variant_split.suffix != original_split.suffix:
        return "text after the evidence changed"
    if variant_split.support != expected_support:
        return "evidence does not equal the constructed evidence"
    if not variant.id.startswith(f"{original.id}-unans-"):
        return "variant id is not paired to the original"
    return None


def build_pairs(
    items: list[TypedQuestion],
    transforms: Iterable[str] = TRANSFORMS,
    per_item_limit: int = 2,
    seed: int = 42,
    entity_match: str = "type",
) -> tuple[list[TypedQuestion], Counter, Counter, dict]:
    """Build verified pairs, each original immediately followed by its variants.

    Args:
        items: Source corpus, already de-duplicated by id.
        transforms: Subset of :data:`TRANSFORMS` to apply.
        per_item_limit: Maximum variants per transform per item.
        seed: Seed for donor selection, so a run is reproducible.
        entity_match: ``"type"`` requires the same entity type and token shape;
            ``"shape"`` keeps only the token shape.

    Returns:
        ``(rows, variant_counts, original_counts, diagnostics)``.  The counters
        are keyed by transform.  ``diagnostics`` holds per-transform metadata
        for the stats file and a ``pair_failures`` list.
    """
    if entity_match not in ("type", "shape"):
        raise ValueError(f"entity_match={entity_match!r} is not 'type' or 'shape'")
    enabled = [transform for transform in TRANSFORMS if transform in set(transforms)]
    rng = random.Random(seed)

    parsed: dict[str, SplitState] = {}
    for item in items:
        split = split_state(item)
        if split is not None:
            parsed[item.id] = split

    # Document frequency over evidence sentences, for the support ranking.
    document_frequency: Counter = Counter()
    total_units = 0
    for split in parsed.values():
        for _start, _end, sentence in split_sentences(split.support):
            total_units += 1
            for token in set(content_tokens(sentence)):
                document_frequency[token] += 1
    token_idf = {
        token: math.log((total_units + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }

    # Donor indexes, keyed by source so the replacement stays in-distribution.
    common_words = common_word_vocabulary(items)
    evidence_by_group: dict[str, dict[str, list[tuple[TypedQuestion, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    entity_donors: dict[
        str, dict[tuple[int, bool, str], dict[str, list[str]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in items:
        split = parsed.get(item.id)
        if split is None or not split.support:
            continue
        group = item.group or ""
        evidence_by_group[item.source][group].append((item, split.support))
        for entity in entity_spans(split.query):
            if len(entity.split()) > 4:
                continue
            shape = entity_shape(entity, common_words, entity_match)
            bucket = entity_donors[item.source][shape][group]
            if entity not in bucket:
                bucket.append(entity)

    # Sorted evidence lengths let a swap find an in-band donor exactly instead
    # of sampling blindly and falling back to a wildly different length.
    evidence_lengths: dict[tuple[str, str], list[int]] = {}
    for source, groups in evidence_by_group.items():
        for group, pool in groups.items():
            pool.sort(key=lambda entry: len(entry[1]))
            evidence_lengths[(source, group)] = [len(entry[1]) for entry in pool]

    rows: list[TypedQuestion] = []
    variant_counts: Counter = Counter()
    original_counts: Counter = Counter()
    diagnostics: dict[str, dict] = {
        transform: {
            "length_delta": [],
            "sources": Counter(),
            "types": Counter(),
        }
        for transform in enabled
    }
    if "entity_corrupted" in diagnostics:
        diagnostics["entity_corrupted"]["entity_types"] = Counter()
    failures: list[str] = []

    def make_variant(
        item: TypedQuestion,
        transform: str,
        index: int,
        new_support: str,
        split: SplitState,
    ) -> TypedQuestion | None:
        variant = _clone_variant(
            item,
            id=f"{item.id}-unans-{REASON_BY_TRANSFORM[transform]}-{index}",
            state=split.with_support(item.state, new_support),
            reason=REASON_BY_TRANSFORM[transform],
        )
        error = pair_error(item, variant, transform, new_support)
        if error is not None:
            failures.append(f"{variant.id}: {error}")
            return None
        variant_counts[transform] += 1
        diagnostics[transform]["length_delta"].append(
            len(variant.state) - len(item.state)
        )
        diagnostics[transform]["sources"][item.source] += 1
        diagnostics[transform]["types"][item.question["type"]] += 1
        return variant

    for item in items:
        split = parsed.get(item.id)
        item_variants: list[TypedQuestion] = []

        if split is not None and "support_removed" in enabled:
            if is_answerable_original(item) and split.fmt not in NO_SENTENCE_TRANSFORMS:
                candidates = remove_supporting_sentences(
                    split, token_idf, per_item_limit
                )
                for index, (new_support, _removed) in enumerate(candidates, 1):
                    variant = make_variant(
                        item, "support_removed", index, new_support, split
                    )
                    if variant is not None:
                        item_variants.append(variant)

        if split is not None and "out_of_scope" in enabled and split.support:
            by_group = evidence_by_group.get(item.source) or {}
            groups = [group for group in by_group if group != (item.group or "")]
            rng.shuffle(groups)
            donor: tuple[TypedQuestion, str] | None = None
            for group in groups:
                pool = by_group[group]
                lengths = evidence_lengths[(item.source, group)]
                low = bisect_left(lengths, 0.5 * len(split.support))
                high = bisect_right(lengths, 2.0 * len(split.support))
                for _attempt in range(min(8, high - low)):
                    entry = pool[rng.randrange(low, high)]
                    if entry[1] != split.support:
                        donor = entry
                        break
                if donor is not None:
                    break
            if donor is not None:
                variant = make_variant(item, "out_of_scope", 1, donor[1], split)
                if variant is not None:
                    item_variants.append(variant)

        if split is not None and "entity_corrupted" in enabled:
            if is_answerable_original(item) and split.fmt not in NO_SENTENCE_TRANSFORMS:
                candidates = corrupt_entities(
                    split,
                    item.group or "",
                    entity_donors.get(item.source, {}),
                    rng,
                    per_item_limit,
                    common_words,
                    entity_match,
                )
                for index, (new_support, entity, _donor) in enumerate(candidates, 1):
                    variant = make_variant(
                        item, "entity_corrupted", index, new_support, split
                    )
                    if variant is not None:
                        item_variants.append(variant)
                        diagnostics["entity_corrupted"]["entity_types"][
                            entity_type(entity, common_words)
                        ] += 1

        if item_variants:
            rows.append(replace(item, answerable=True))
            original_counts["answerable_originals"] += 1
            original_counts[item.question["type"]] += 1
            rows.extend(item_variants)

    diagnostics["pair_failures"] = failures
    return rows, variant_counts, original_counts, diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_delta": sum(values) / len(values),
        "min_delta": min(values),
        "median_delta": statistics.median(values),
        "max_delta": max(values),
    }


def _load_inputs(paths: list[Path]) -> tuple[list[TypedQuestion], int]:
    items: list[TypedQuestion] = []
    seen: set[str] = set()
    duplicates = 0
    for path in paths:
        for item in load_jsonl(path):
            if item.id in seen:
                duplicates += 1
                continue
            seen.add(item.id)
            items.append(item)
    return items, duplicates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="pair file")
    parser.add_argument(
        "--negatives-out",
        type=Path,
        default=None,
        help="variants-only file (default: negatives.jsonl beside --out)",
    )
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument(
        "--transforms",
        default=",".join(TRANSFORMS),
        help=f"Comma-separated subset of {TRANSFORMS}",
    )
    parser.add_argument(
        "--per-item-limit",
        type=int,
        default=2,
        help="Maximum variants per transform per item",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap inputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--entity-donor-match",
        choices=("type", "shape"),
        default="type",
        help="Entity donor match: same type and token shape, or token shape only",
    )
    args = parser.parse_args(argv)

    transforms = [part.strip() for part in args.transforms.split(",") if part.strip()]
    unknown = [transform for transform in transforms if transform not in TRANSFORMS]
    if unknown:
        raise SystemExit(f"unknown transform(s): {unknown}")
    if args.per_item_limit < 1:
        raise SystemExit("--per-item-limit must be positive")

    items, duplicates = _load_inputs(args.items)
    if args.limit:
        items = items[: args.limit]
    rows, variant_counts, original_counts, diagnostics = build_pairs(
        items,
        transforms=transforms,
        per_item_limit=args.per_item_limit,
        seed=args.seed,
        entity_match=args.entity_donor_match,
    )
    failures = diagnostics.pop("pair_failures")
    if failures:
        for failure in failures[:20]:
            print(f"pair check failed: {failure}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} pair check failure(s)")

    save_jsonl(rows, args.out)
    negatives_path = args.negatives_out or args.out.with_name("negatives.jsonl")
    negatives = [row for row in rows if getattr(row, "answerable", None) is False]
    save_jsonl(negatives, negatives_path)

    by_transform: dict[str, dict] = {}
    for transform in transforms:
        detail = diagnostics[transform]
        block = {
            "variants": variant_counts[transform],
            "reason": REASON_BY_TRANSFORM[transform],
            "by_source": dict(detail["sources"].most_common()),
            "by_type": dict(detail["types"]),
            "length_delta": _length_summary(detail["length_delta"]),
        }
        if "entity_types" in detail:
            block["entity_types"] = dict(detail["entity_types"].most_common())
        by_transform[transform] = block
    stats = {
        "inputs": [str(path) for path in args.items],
        "out": str(args.out),
        "negatives_out": str(negatives_path),
        "seed": args.seed,
        "per_item_limit": args.per_item_limit,
        "entity_donor_match": args.entity_donor_match,
        "transforms": transforms,
        "input_items": len(items),
        "duplicate_ids_dropped": duplicates,
        "answerable_originals": original_counts["answerable_originals"],
        "originals_by_type": {
            key: value
            for key, value in original_counts.items()
            if key != "answerable_originals"
        },
        "variants": sum(variant_counts.values()),
        "by_transform": by_transform,
        "pair_checks": {"passed": sum(variant_counts.values()), "failed": 0},
    }
    payload = json.dumps(stats, indent=1, sort_keys=True)
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(payload + "\n")
    print(payload, flush=True)
    print(
        f"wrote {len(rows)} pair rows ({sum(variant_counts.values())} variants) "
        f"to {args.out}; negatives to {negatives_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
