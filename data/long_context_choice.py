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

Adapters return ``(id, question, paragraphs, supporting_titles, answer, group,
split)`` plus optional ``aliases``.  Adapters may add adapter-only fields such
as ``options`` and ``answer_index``; the core shape is unchanged.

Sources:

- ``hotpotqa``: distractor setting, 10 paragraphs, ~5.7K median chars.
- ``musique``: 2-4 hop, 20 paragraphs, ~9.8K median chars; answer aliases are
  excluded from the distractor pool.
- ``mmlu_pro``: 10-option multiple choice.  This is a short-context *reasoning*
  source, not long-context: the state is the question text and the 10 options
  are the criteria.  Rows without exactly 10 options are dropped.  K is fixed
  at 10 and no distractors are mined.
- ``race``: reading-comprehension passages with 4 options.  The 4 original
  options are preserved; same-type option texts mined from other rows fill to
  ``--k``.  Measured article length is ~1.6K median chars, so this is labelled
  a short-passage source, not long-context.
- ``longbench``: raw JSONL subsets from the canonical ``zai-org/LongBench``
  mirror (``data.zip``), bypassing the removed loading script.  The state is
  the long context (capped at ``--max-chars``); criteria are the gold answer
  plus mined same-type answers from the same subset.
- ``twowiki``: 2WikiMultiHopQA raw JSON from ``voidful/2WikiMultihopQA``,
  which keeps the original context and supporting facts.  This bypasses the
  column-cast failure of the script-based mirrors.

State-length filters do not apply to ``mmlu_pro``: its state is the question
text, so ``--min-chars``/``--target-chars``/``--max-chars`` are ignored there.

Blocked sources:

- ``allenai/qasper``: the HF repo ships a loading script and dummy data only.
  datasets 4+ refuses script datasets, and the AI2 S3 mirror returns 403.
  No separate Qasper adapter is added; LongBench's ``qasper`` subset is used
  instead.
- ``thinkall/2WikiMultihopQA``: the raw files drop supporting facts, so the
  ``voidful`` raw JSON (original full format) is the working path.

Usage:
    python -m data.long_context_choice \
        --source hotpotqa --split validation --limit 500 \
        --k 8 --min-chars 4000 --target-chars 5000 --max-chars 8000 \
        --out data/benchmarks/hotpotqa_long_choice.jsonl

    python -m data.long_context_choice \
        --source longbench --subsets narrativeqa,qasper,multifieldqa_en \
        --k 8 --min-chars 4000 --target-chars 8000 --max-chars 32000 \
        --out data/benchmarks/longbench_long_choice.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

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

SOURCES = ("hotpotqa", "musique", "mmlu_pro", "race", "longbench", "twowiki")
LONGBENCH_SUBSETS = ("narrativeqa", "qasper", "multifieldqa_en")
TWOWIKI_SPLITS = {
    "validation": "dev",
    "dev": "dev",
    "train": "train",
    "test": "test",
}


def _title_position_like(text: str) -> bool:
    """True when a title word behaves like a position, not part of a name.

    "Chief of Protocol", "Israeli Ambassador" and "a governor" are positions;
    "King Diamond" and "Captain Marvel" are names that happen to start with a
    title word.
    """
    match = TITLE.search(text)
    if match is None:
        return False
    head = text[: match.start()].strip(" \"'(“").lower()
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


def _acceptable_normalised(
    candidate: str,
    gold_norm: str,
    state_norm: str,
    kind: str,
) -> bool:
    """Type, length, substring and state checks on normalised inputs."""
    cand = _normalise(candidate)
    if not cand or cand == gold_norm:
        return False
    if cand in gold_norm or gold_norm in cand:
        return False
    if answer_type(candidate) != kind:
        return False
    # Length ratio keeps one-word entities from mixing with long phrases.
    if not 0.33 <= len(cand) / max(1, len(gold_norm)) <= 3.0:
        return False
    if cand in state_norm:
        return False
    return True


def acceptable_distractor(
    candidate: str,
    gold: str,
    state: str,
    kind: str,
) -> bool:
    """A distractor is usable when it is same-type, distinct and not in the state."""
    return _acceptable_normalised(
        candidate, _normalise(gold), _normalise(state), kind
    )


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


def pick_distractors_scan(
    gold: str,
    pool: Iterable[str],
    state: str,
    k: int,
    rng: random.Random,
    question: str = "",
    forbidden: Iterable[str] = (),
    max_scan: int = 20000,
) -> list[str] | None:
    """Return ``k - 1`` distractors by scanning a randomised pool slice.

    Large option pools (RACE has ~300K option texts) make sorting every
    candidate per item expensive.  This variant starts at a random offset and
    stops once enough valid candidates are found.  It applies the same filters
    as :func:`pick_distractors`.
    """
    need = k - 1
    if need <= 0:
        return []
    kind = expected_type(question, gold)
    if kind in ("yes_no", "empty"):
        return None
    gold_norm = _normalise(gold)
    state_norm = _normalise(state)
    forbidden_norm = {_normalise(value) for value in forbidden if value}
    forbidden_norm.add(gold_norm)

    pool_list = list(pool)
    if not pool_list:
        return None
    start = rng.randrange(len(pool_list))
    picked: list[str] = []
    picked_norm: set[str] = set()
    scanned = 0
    for offset in range(len(pool_list)):
        if scanned >= max_scan:
            break
        candidate = pool_list[(start + offset) % len(pool_list)]
        candidate = str(candidate).strip()
        normalised = _normalise(candidate)
        scanned += 1
        if not normalised or normalised in forbidden_norm:
            continue
        if normalised in picked_norm:
            continue
        if _acceptable_normalised(candidate, gold_norm, state_norm, kind):
            picked.append(candidate)
            picked_norm.add(normalised)
            if len(picked) == need:
                return picked
    return None


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
        # Single-document adapters use an empty title to mean "state is the
        # text itself", without a title prefix.
        block = f"{title}: {text}".strip() if title else str(text).strip()
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


def mmlu_pro_row(row: dict, split: str) -> dict | None:
    """Normalise an MMLU-Pro row.  The state is the question text.

    The 10 options are criteria, not state.  ``options`` and ``answer_index``
    are adapter-only fields consumed by :func:`build_mmlu_item`.
    """
    options = [str(option).strip() for option in row.get("options") or []]
    answer_index = row.get("answer_index")
    question = str(row.get("question") or "").strip()
    if not question or not options:
        return None
    if not isinstance(answer_index, int) or not 0 <= answer_index < len(options):
        return None
    if not options[answer_index]:
        return None
    category = str(row.get("category") or "unknown")
    return {
        "id": f"mmlu-pro-{row.get('question_id', '')}",
        "question": question,
        "paragraphs": [("", question)],
        "supporting_titles": {""},
        "answer": options[answer_index],
        "options": options,
        "answer_index": answer_index,
        "group": f"short_reasoning:{category}",
        "split": split,
    }


def race_row(row: dict, split: str) -> dict | None:
    """Normalise a RACE row.  The 4 original options are preserved."""
    options = [str(option).strip() for option in row.get("options") or []]
    answer = str(row.get("answer") or "").strip().upper()
    article = str(row.get("article") or "").strip()
    question = str(row.get("question") or "").strip()
    if len(options) != 4 or any(not option for option in options):
        return None
    if answer not in ("A", "B", "C", "D"):
        return None
    if not article or not question:
        return None
    answer_index = ord(answer) - ord("A")
    return {
        "id": f"race-{row.get('example_id', '')}",
        "question": question,
        "paragraphs": [("", article)],
        "supporting_titles": {""},
        "answer": options[answer_index],
        "options": options,
        "answer_index": answer_index,
        "group": "short_passage_rc",
        "split": split,
    }


def longbench_row(row: dict, subset: str, split: str) -> dict | None:
    """Normalise a LongBench JSONL row with a short gold answer."""
    answers = row.get("answers")
    if isinstance(answers, str):
        answers = [answers]
    answers = [str(answer).strip() for answer in answers or [] if str(answer).strip()]
    context = str(row.get("context") or "").strip()
    question = str(row.get("input") or "").strip()
    if not context or not question or not answers:
        return None
    return {
        "id": f"longbench-{subset}-{row.get('_id', '')}",
        "question": question,
        "paragraphs": [("", context)],
        "supporting_titles": {""},
        "answer": answers[0],
        "aliases": answers[1:],
        "group": subset,
        "split": split,
    }


def twowiki_row(row: dict, split: str) -> dict | None:
    """Normalise a 2WikiMultiHopQA raw JSON row (original full format)."""
    paragraphs = []
    for entry in row.get("context") or []:
        if isinstance(entry, dict):
            title = str(entry.get("title") or "")
            sentences = entry.get("sentences") or entry.get("text") or []
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            title = str(entry[0])
            sentences = entry[1] or []
        else:
            continue
        if isinstance(sentences, str):
            sentences = [sentences]
        text = " ".join(str(sentence) for sentence in sentences).strip()
        if text:
            paragraphs.append((title, text))
    supporting_titles: set[str] = set()
    for fact in row.get("supporting_facts") or []:
        if isinstance(fact, (list, tuple)) and fact:
            supporting_titles.add(str(fact[0]))
        elif isinstance(fact, dict) and fact.get("title"):
            supporting_titles.add(str(fact["title"]))
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer or not paragraphs:
        return None
    return {
        "id": f"2wiki-long-{row.get('_id', '')}",
        "question": question,
        "paragraphs": paragraphs,
        "supporting_titles": supporting_titles,
        "answer": answer,
        "group": str(row.get("type") or "unknown"),
        "split": split,
    }


def _assemble_choice(
    example: dict,
    state: str,
    options: list[str],
    gold_index: int,
    source: str,
    rng: random.Random,
    shuffle: bool = True,
) -> TypedQuestion:
    """Build a choice item, tracking the gold by position, not by text."""
    entries = [(text, index == gold_index) for index, text in enumerate(options)]
    if shuffle:
        rng.shuffle(entries)
    criteria = {
        f"option_{index}": text for index, (text, _) in enumerate(entries)
    }
    gold_key = next(
        f"option_{index}"
        for index, (_, is_gold) in enumerate(entries)
        if is_gold
    )
    return TypedQuestion(
        id=example["id"],
        state=state,
        question={
            "type": "choice",
            "instructions": example["question"],
            "criteria": criteria,
        },
        label=gold_key,
        source=source,
        split=example["split"],
        group=example["group"],
    )


def _has_duplicate_options(options: list[str]) -> bool:
    normalised = [_normalise(option) for option in options]
    return len(normalised) != len(set(normalised))


def build_item_reason(
    example: dict,
    pool: list[str],
    k: int,
    min_chars: int,
    target_chars: int,
    max_chars: int,
    rng: random.Random,
    source: str = "hotpotqa_long_choice",
) -> tuple[TypedQuestion | None, str | None]:
    """Build a paragraph-backed choice item and return a drop reason on failure."""
    state = select_paragraphs(
        example["paragraphs"], example["supporting_titles"], target_chars, max_chars
    )
    if len(state) < min_chars:
        return None, "short_state"
    if answer_type(example["answer"]) == "yes_no":
        return None, "yes_no_answer"
    distractors = pick_distractors(
        example["answer"], pool, state, k, rng, example["question"],
        forbidden=example.get("aliases") or (),
    )
    if distractors is None:
        return None, "insufficient_distractors"
    options = [example["answer"]] + distractors
    if _has_duplicate_options(options):
        return None, "duplicate_options"
    item = _assemble_choice(
        example, state, options, gold_index=0, source=source,
        rng=rng,
    )
    return item, None


def build_item(
    example: dict,
    pool: list[str],
    k: int,
    min_chars: int,
    target_chars: int,
    max_chars: int,
    rng: random.Random,
    source: str = "hotpotqa_long_choice",
) -> TypedQuestion | None:
    """Backward-compatible wrapper around :func:`build_item_reason`."""
    item, _ = build_item_reason(
        example, pool, k, min_chars, target_chars, max_chars, rng, source=source
    )
    return item


def build_mmlu_item(
    example: dict,
    k: int = 10,
) -> tuple[TypedQuestion | None, str | None]:
    """Build a fixed K=10 MMLU-Pro item; no distractors are mined."""
    options = example.get("options") or []
    if len(options) != 10 or k != 10:
        return None, "not_10_options"
    if _has_duplicate_options(options):
        return None, "duplicate_options"
    state = example["question"]
    if not state.strip():
        return None, "empty_state"
    return (
        _assemble_choice(
            example,
            state,
            options,
            gold_index=example["answer_index"],
            source="mmlu_pro_short_reasoning_choice",
            rng=random.Random(0),
            shuffle=False,
        ),
        None,
    )


def build_race_item(
    example: dict,
    pool: list[str],
    k: int,
    min_chars: int,
    max_chars: int,
    rng: random.Random,
) -> tuple[TypedQuestion | None, str | None]:
    """Build a RACE item: keep the 4 original options, mine to reach K."""
    state = example["paragraphs"][0][1]
    if len(state) < min_chars:
        return None, "short_state"
    state = state[:max_chars]
    originals = example["options"]
    gold_index = example["answer_index"]
    gold = originals[gold_index]
    needed = k - len(originals)
    if needed < 0:
        return None, "k_below_option_count"
    distractors: list[str] = []
    if needed:
        distractors = pick_distractors_scan(
            gold,
            pool,
            state,
            needed + 1,
            rng,
            example["question"],
            forbidden=originals,
        ) or []
        if len(distractors) < needed:
            return None, "insufficient_distractors"
    options = originals + distractors
    if _has_duplicate_options(options):
        return None, "duplicate_options"
    return (
        _assemble_choice(
            example,
            state,
            options,
            gold_index=gold_index,
            source="race_short_passage_choice",
            rng=rng,
        ),
        None,
    )


def build_long_context_item(
    example: dict,
    pool: list[str],
    k: int,
    min_chars: int,
    max_chars: int,
    rng: random.Random,
) -> tuple[TypedQuestion | None, str | None]:
    """Build a single-document long-context item.

    ``target_chars`` does not apply here: the document is kept whole up to
    ``max_chars``.  ``min_chars`` is a hard filter.
    """
    context = example["paragraphs"][0][1]
    if len(context) < min_chars:
        return None, "short_state"
    if answer_type(example["answer"]) == "yes_no":
        return None, "yes_no_answer"
    state = context[:max_chars]
    distractors = pick_distractors_scan(
        example["answer"],
        pool,
        state,
        k,
        rng,
        example["question"],
        forbidden=example.get("aliases") or (),
    )
    if distractors is None:
        return None, "insufficient_distractors"
    options = [example["answer"]] + distractors
    if _has_duplicate_options(options):
        return None, "duplicate_options"
    return (
        _assemble_choice(
            example,
            state,
            options,
            gold_index=0,
            source="longbench_long_choice",
            rng=rng,
        ),
        None,
    )


def _load_longbench(
    subsets: list[str], limit: int
) -> tuple[list[dict], dict[str, list[str]]]:
    """Read LongBench JSONL subsets straight from the canonical data.zip.

    The dataset card no longer supports the loading script under datasets 4+,
    so the raw archive is the stable path.
    """
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        "zai-org/LongBench", "data.zip", repo_type="dataset"
    )
    examples: list[dict] = []
    pools: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        for subset in subsets:
            member = f"data/{subset}.jsonl"
            if member not in members:
                raise SystemExit(
                    f"LongBench has no subset {subset!r} ({member} missing)"
                )
            rows = [
                json.loads(line)
                for line in archive.read(member).decode("utf-8").splitlines()
                if line.strip()
            ]
            pool: set[str] = set()
            for row in rows:
                answers = row.get("answers")
                if isinstance(answers, str):
                    answers = [answers]
                for answer in answers or []:
                    answer = str(answer).strip()
                    if answer:
                        pool.add(answer)
            pools[subset] = sorted(pool)
            inspected = 0
            for row in rows:
                example = longbench_row(row, subset, "test")
                if example is None:
                    continue
                examples.append(example)
                inspected += 1
                if limit and inspected >= limit:
                    break
    return examples, pools


def _load_twowiki(split: str, limit: int) -> tuple[list[dict], list[str]]:
    """Read 2WikiMultiHopQA raw JSON, bypassing the failing script loader."""
    from huggingface_hub import hf_hub_download

    split_name = TWOWIKI_SPLITS.get(split)
    if split_name is None:
        raise SystemExit(
            f"unknown 2Wiki split {split!r}; expected validation/dev/train/test"
        )
    path = hf_hub_download(
        "voidful/2WikiMultihopQA", f"{split_name}.json", repo_type="dataset"
    )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = sorted(
        {
            str(row.get("answer") or "").strip()
            for row in data
            if str(row.get("answer") or "").strip()
        }
    )
    examples: list[dict] = []
    for row in data:
        example = twowiki_row(row, split)
        if example is None:
            continue
        examples.append(example)
        if limit and len(examples) >= limit:
            break
    return examples, pool


def load_source(
    source: str,
    split: str,
    limit: int,
    subsets: list[str] | None = None,
) -> tuple[list[dict], list[str] | dict[str, list[str]]]:
    """Return ``(examples, answer_pool)`` for the split.

    The pool uses every answer (or option text) in the split, so same-type
    distractors are not limited to the few rows being converted.  LongBench
    returns a per-subset pool dict because its subsets are different domains.
    """
    from datasets import load_dataset

    if source == "hotpotqa":
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        normalise = hotpot_row
    elif source == "musique":
        dataset = load_dataset("dgslibisey/MuSiQue", split=split)
        normalise = musique_row
    elif source == "mmlu_pro":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        rows = []
        for row in dataset:
            example = mmlu_pro_row(row, split)
            if example is not None:
                rows.append(example)
            if limit and len(rows) >= limit:
                break
        return rows, []
    elif source == "race":
        dataset = load_dataset("ehovy/race", "all", split=split)
        pool = sorted(
            {
                str(option).strip()
                for row in dataset
                for option in (row.get("options") or [])
                if str(option).strip()
            }
        )
        rows = []
        for row in dataset:
            example = race_row(row, split)
            if example is not None:
                rows.append(example)
            if limit and len(rows) >= limit:
                break
        return rows, pool
    elif source == "longbench":
        return _load_longbench(subsets or list(LONGBENCH_SUBSETS), limit)
    elif source == "twowiki":
        return _load_twowiki(split, limit)
    else:
        raise SystemExit(
            f"unknown --source {source!r}; expected one of {', '.join(SOURCES)}"
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


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="hotpotqa", choices=SOURCES)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=500,
                        help="number of source rows to inspect (0 = all); "
                             "for longbench the limit is per subset")
    parser.add_argument("--subsets", default=",".join(LONGBENCH_SUBSETS),
                        help="comma-separated LongBench subsets")
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
    if args.source == "mmlu_pro" and args.k != 10:
        print("note: mmlu_pro is fixed at K=10; overriding --k", flush=True)
        args.k = 10

    subsets = [value.strip() for value in args.subsets.split(",") if value.strip()]
    examples, pool = load_source(
        args.source, args.split, args.limit, subsets=subsets
    )
    rng = random.Random(args.seed)
    items: list[TypedQuestion] = []
    dropped: Counter = Counter()
    length_buckets: Counter = Counter()
    length_buckets_8k: Counter = Counter()
    truncated_state = 0

    for example in examples:
        if args.source == "mmlu_pro":
            item, reason = build_mmlu_item(example, k=args.k)
        elif args.source == "race":
            item, reason = build_race_item(
                example, pool, args.k, args.min_chars, args.max_chars, rng
            )
        elif args.source == "longbench":
            assert isinstance(pool, dict)
            item, reason = build_long_context_item(
                example, pool[example["group"]], args.k,
                args.min_chars, args.max_chars, rng,
            )
        else:
            assert not isinstance(pool, dict)
            item, reason = build_item_reason(
                example, pool, args.k, args.min_chars,
                args.target_chars, args.max_chars, rng,
                source="2wiki_long_choice" if args.source == "twowiki"
                else "hotpotqa_long_choice",
            )
        if item is None:
            dropped[reason or "unknown"] += 1
            continue
        if args.source in ("race", "longbench"):
            context = example["paragraphs"][0][1]
            if len(context) > args.max_chars:
                truncated_state += 1
        items.append(item)
        length = len(item.state)
        length_buckets[min(12000, (length // 2000) * 2000)] += 1
        length_buckets_8k[min(64000, (length // 8000) * 8000)] += 1

    save_jsonl(items, args.out)
    state_lengths = sorted(len(item.state) for item in items)
    stats = {
        "source": args.source,
        "split": args.split,
        "subsets": subsets if args.source == "longbench" else None,
        "inspected": len(examples),
        "selected": len(items),
        "k": args.k,
        "min_chars": args.min_chars,
        "target_chars": args.target_chars,
        "max_chars": args.max_chars,
        "dropped": dict(sorted(dropped.items())),
        "by_source": dict(
            sorted(Counter(item.source for item in items).items())
        ),
        "by_group": dict(sorted(Counter(item.group for item in items).items())),
        "k_distribution": dict(
            sorted(
                (str(k), count)
                for k, count in Counter(
                    len(item.question["criteria"]) for item in items
                ).items()
            )
        ),
        "by_answer_type": dict(
            sorted(
                Counter(
                    answer_type(item.question["criteria"][item.label])
                    for item in items
                ).items()
            )
        ),
        "state_chars": {
            "min": state_lengths[0] if state_lengths else 0,
            "p50": _percentile(state_lengths, 0.50),
            "p90": _percentile(state_lengths, 0.90),
            "max": state_lengths[-1] if state_lengths else 0,
        },
        "length_buckets_2k_chars": dict(sorted(length_buckets.items())),
        "length_buckets_8k_chars": dict(sorted(length_buckets_8k.items())),
        "truncated_state": truncated_state,
        "out": str(args.out),
    }
    stats = {key: value for key, value in stats.items() if value is not None}
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True), flush=True)
    print(f"wrote {len(items)} items to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
