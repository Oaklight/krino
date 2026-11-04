"""Derive long-context multi-hop choice items from open QA datasets.

JevBench hard is long-context, multi-hop choice with K in 6-20, and the
current corpus barely covers it.  This module converts extraction QA rows
into ``TypedQuestion`` choice items instead of generating them synthetically:

- state: the supporting paragraphs first, then enough distractor paragraphs to
  reach ``--target-chars`` (capped by ``--max-chars``);
- instructions: the dataset question;
- criteria: the gold answer plus ``K-1`` answers mined from other rows of the
  same dataset, same answer type, not a substring of the gold answer and not
  present in the state;
- label: the gold answer's option key.

Adapters: HotpotQA's distractor setting (10 paragraphs, ~5.7K median chars)
and MuSiQue (2-4 hop, 20 paragraphs, ~9.8K median chars; answer aliases are
excluded from the distractor pool).  Adapters return ``(id, question,
paragraphs, supporting_titles, answer, group)`` plus optional ``aliases``.

Usage:
    python -m data.long_context_choice \
        --source hotpotqa --split validation --limit 500 \
        --k 8 --min-chars 4000 --target-chars 5000 --max-chars 8000 \
        --out data/benchmarks/hotpotqa_long_choice.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.format import TypedQuestion  # noqa: E402
from data.pipeline import save_jsonl  # noqa: E402

YES_NO = {"yes", "no", "true", "false"}
PROPER = re.compile(r"^[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)+$")
PLACE = re.compile(
    r"\b(?:County|City|State|River|Lake|Bay|Strait|Straits|Mount|Mountain|"
    r"Island|Mosque|Province|Region|District|Park|Valley|Desert|Sea|Ocean|"
    r"Gulf|Peninsula|Canyon|Forest|Village|Town|Port|Airport|Gmina)\b",
    re.IGNORECASE,
)
ORG = re.compile(
    r"\b(?:Inc|Corp|Corporation|Company|Media|Group|Holdings|Ltd|LLC|University|"
    r"College|Institute|Foundation|Association|Agency|Bank|Records|Studios|"
    r"Pictures|Entertainment|Publishing|Press|Times|Journal|Network|Airways|"
    r"Airlines|Motors|Foods|Pharmaceuticals|Hospital|School|Academy|Committee|"
    r"Council|Department|Ministry|Bureau|Office|Commission)\b",
    re.IGNORECASE,
)
TITLE = re.compile(
    r"\b(?:Chief|Director|Minister|Secretary|President|Governor|Mayor|Senator|"
    r"Ambassador|Commissioner|Officer|Chair(?:man|woman|person)?|Leader|Head|"
    r"King|Queen|Prince|Princess|Pope|Dean|Professor|Coach|Captain|General|"
    r"Admiral|Judge|Justice|Sheriff|Treasurer|Chancellor|Provost|Speaker|Whip|"
    r"Consul|Envoy|Delegate|Representative|Congressman|Congresswoman|"
    r"Assemblyman|Assemblywoman|Premier|Monarch)\b",
    re.IGNORECASE,
)
YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _title_position_like(text: str) -> bool:
    """True when a title word behaves like a position, not part of a name.

    "Chief of Protocol", "Israeli Ambassador" and "a governor" are positions;
    "King Diamond" and "Captain Marvel" are names that happen to start with a
    title word.
    """
    match = TITLE.search(text)
    if match is None:
        return False
    head = text[: match.start()].strip(" \"'(\u201c").lower()
    tail = text[match.end() :].strip(" \"').,").lower()
    if not tail:
        return True
    if tail.startswith(("of ", "for ", "at ")):
        return True
    return head in ("the", "a", "an")


def answer_type(answer: str) -> str:
    """Coarse answer type used to keep distractors plausible."""
    text = answer.strip()
    if not text:
        return "empty"
    lowered = text.lower()
    if lowered in YES_NO:
        return "yes_no"
    if NUMBER.fullmatch(text):
        return "date" if YEAR.search(text) else "number"
    if YEAR.search(text) and len(text) <= 40:
        return "date"
    if _title_position_like(text):
        return "title"
    if PLACE.search(text):
        return "place"
    if ORG.search(text):
        return "organization"
    if PROPER.fullmatch(text):
        return "proper"
    return "entity"


def expected_type(question: str, gold: str) -> str:
    """Answer type implied by the question, falling back to the gold surface."""
    text = question.lower()
    # An explicit position cue wins over a relative "who" clause: "the woman
    # who portrayed X" asks for a position, not for a person.
    if re.search(r"\b(position|title|role|office|job)\b", text):
        return "title"
    if re.match(r"^\s*(where|in which)\b", text) or re.search(
        r"\bwhich (city|country|state|province|region)\b", text
    ):
        return "place"
    if re.match(r"^\s*(who|whom)\b", text):
        return "proper"
    if re.search(r"\b(when|what year|which year|what date)\b", text):
        return "date"
    if re.search(r"\bhow many\b|\bwhat number\b", text):
        return "number"
    return answer_type(gold)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def acceptable_distractor(
    candidate: str,
    gold: str,
    state: str,
    kind: str,
) -> bool:
    """A distractor is usable when it is same-type, distinct and not in the state."""
    cand = _normalise(candidate)
    gold_norm = _normalise(gold)
    if not cand or cand == gold_norm:
        return False
    if cand in gold_norm or gold_norm in cand:
        return False
    if answer_type(candidate) != kind:
        return False
    # Length ratio keeps one-word entities from mixing with long phrases.
    if not 0.33 <= len(cand) / max(1, len(gold_norm)) <= 3.0:
        return False
    if cand in _normalise(state):
        return False
    return True


def pick_distractors(
    gold: str,
    pool: Iterable[str],
    state: str,
    k: int,
    rng: random.Random,
    question: str = "",
    forbidden: Iterable[str] = (),
) -> list[str] | None:
    """Return ``k - 1`` distractors, or None when the pool is too small.

    ``forbidden`` holds the gold answer's aliases: they are correct answers in
    another spelling and must not become distractors.
    """
    kind = expected_type(question, gold)
    if kind in ("yes_no", "empty"):
        return None
    forbidden_norm = {_normalise(value) for value in forbidden if value}
    unique: dict[str, str] = {}
    for candidate in pool:
        candidate = candidate.strip()
        normalised = _normalise(candidate)
        if normalised and normalised not in unique:
            unique[normalised] = candidate
    candidates = sorted(
        candidate
        for normalised, candidate in unique.items()
        if normalised not in forbidden_norm
        and acceptable_distractor(candidate, gold, state, kind)
    )
    if len(candidates) < k - 1:
        return None
    return rng.sample(candidates, k - 1)


def select_paragraphs(
    paragraphs: list[tuple[str, str]],
    supporting_titles: set[str],
    target_chars: int,
    max_chars: int,
) -> str:
    """Supporting paragraphs first, then fill with distractors up to the target."""
    ordered = [pair for pair in paragraphs if pair[0] in supporting_titles]
    ordered += [pair for pair in paragraphs if pair[0] not in supporting_titles]
    blocks: list[str] = []
    total = 0
    for title, text in ordered:
        block = f"{title}: {text}".strip()
        if not block:
            continue
        if blocks and total + len(block) + 2 > max_chars:
            break
        blocks.append(block)
        total += len(block) + 2
        if total >= target_chars:
            break
    return "\n\n".join(blocks)


def hotpot_row(row: dict, split: str) -> dict | None:
    """Normalise a HotpotQA distractor row to the adapter shape."""
    context = row.get("context") or {}
    titles = context.get("title") or []
    sentence_lists = context.get("sentences") or []
    paragraphs = [
        (str(title), " ".join(str(sentence) for sentence in sentences))
        for title, sentences in zip(titles, sentence_lists)
    ]
    supporting = row.get("supporting_facts") or {}
    supporting_titles = {str(title) for title in (supporting.get("title") or [])}
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer or not paragraphs:
        return None
    return {
        "id": f"hotpotqa-long-{row.get('id', '')}",
        "question": question,
        "paragraphs": paragraphs,
        "supporting_titles": supporting_titles,
        "answer": answer,
        "group": str(row.get("type") or "unknown"),
        "split": split,
    }


def musique_row(row: dict, split: str) -> dict | None:
    """Normalise a MuSiQue row (2-4 hop, 20 paragraphs)."""
    if not row.get("answerable", True):
        return None
    paragraphs = []
    supporting: set[str] = set()
    for paragraph in row.get("paragraphs") or []:
        title = str(paragraph.get("title") or "")
        text = str(paragraph.get("paragraph_text") or "").strip()
        if not text:
            continue
        paragraphs.append((title, text))
        if paragraph.get("is_supporting"):
            supporting.add(title)
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer or not paragraphs:
        return None
    hops = len(row.get("question_decomposition") or [])
    aliases = [str(alias).strip() for alias in row.get("answer_aliases") or []]
    return {
        "id": f"musique-long-{row.get('id', '')}",
        "question": question,
        "paragraphs": paragraphs,
        "supporting_titles": supporting,
        "answer": answer,
        "aliases": aliases,
        "group": f"{hops}hop",
        "split": split,
    }


def build_item(
    example: dict,
    pool: list[str],
    k: int,
    min_chars: int,
    target_chars: int,
    max_chars: int,
    rng: random.Random,
) -> TypedQuestion | None:
    state = select_paragraphs(
        example["paragraphs"], example["supporting_titles"], target_chars, max_chars
    )
    if len(state) < min_chars:
        return None
    distractors = pick_distractors(
        example["answer"], pool, state, k, rng, example["question"],
        forbidden=example.get("aliases") or (),
    )
    if distractors is None:
        return None
    options = [example["answer"]] + distractors
    rng.shuffle(options)
    criteria = {f"option_{index}": option for index, option in enumerate(options)}
    gold_key = next(key for key, value in criteria.items() if value == example["answer"])
    return TypedQuestion(
        id=example["id"],
        state=state,
        question={"type": "choice", "instructions": example["question"], "criteria": criteria},
        label=gold_key,
        source="hotpotqa_long_choice",
        split=example["split"],
        group=example["group"],
    )


def load_source(source: str, split: str, limit: int) -> tuple[list[dict], list[str]]:
    """Return ``(examples, answer_pool)`` for the split.

    The pool uses every answer in the split, so same-type distractors are not
    limited to the few rows being converted.  Only the converted rows have
    their paragraphs materialised.
    """
    from datasets import load_dataset

    if source == "hotpotqa":
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        normalise = hotpot_row
    elif source == "musique":
        dataset = load_dataset("dgslibisey/MuSiQue", split=split)
        normalise = musique_row
    else:
        raise SystemExit(
            f"unknown --source {source!r}; expected 'hotpotqa' or 'musique'"
        )

    pool: set[str] = set()
    for row in dataset:
        answer = str(row.get("answer") or "").strip()
        if answer:
            pool.add(answer)
        for alias in row.get("answer_aliases") or []:
            alias = str(alias).strip()
            if alias:
                pool.add(alias)
    rows = []
    for row in dataset:
        example = normalise(row, split)
        if example is not None:
            rows.append(example)
        if limit and len(rows) >= limit:
            break
    return rows, sorted(pool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="hotpotqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=500,
                        help="number of source rows to inspect (0 = all)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=4000)
    parser.add_argument("--target-chars", type=int, default=5000)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()

    if not 6 <= args.k <= 20:
        raise SystemExit("--k must be in [6, 20] for this dataset line")
    if not args.min_chars <= args.target_chars <= args.max_chars:
        raise SystemExit("need min_chars <= target_chars <= max_chars")

    examples, pool = load_source(args.source, args.split, args.limit)
    rng = random.Random(args.seed)
    items: list[TypedQuestion] = []
    dropped: Counter = Counter()
    length_buckets: Counter = Counter()
    for example in examples:
        if answer_type(example["answer"]) == "yes_no":
            dropped["yes_no_answer"] += 1
            continue
        item = build_item(
            example, pool, args.k, args.min_chars, args.target_chars,
            args.max_chars, rng,
        )
        if item is None:
            dropped["short_or_no_pool"] += 1
            continue
        items.append(item)
        length = len(item.state)
        bucket = min(12000, (length // 2000) * 2000)
        length_buckets[bucket] += 1

    save_jsonl(items, args.out)
    stats = {
        "source": args.source,
        "split": args.split,
        "inspected": len(examples),
        "selected": len(items),
        "k": args.k,
        "min_chars": args.min_chars,
        "target_chars": args.target_chars,
        "max_chars": args.max_chars,
        "dropped": dict(dropped),
        "by_group": dict(Counter(item.group for item in items)),
        "by_answer_type": dict(
            Counter(
                answer_type(item.question["criteria"][item.label])
                for item in items
            )
        ),
        "length_buckets_2k_chars": dict(sorted(length_buckets.items())),
        "out": str(args.out),
    }
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True), flush=True)
    print(f"wrote {len(items)} items to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
