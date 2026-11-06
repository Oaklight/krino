"""Audit the canonical training corpus for duplicates and JevBench contamination.

The rule is fixed in issue #235 before the run and is not tuned here.

Contamination: a record is contaminated when any of these holds.

1. Its ``id`` equals a public JevBench ``id``.
2. Its normalized ``state`` equals a JevBench normalized ``state``.
3. Its normalized ``state`` shares at least three 8-word shingles with one
   JevBench normalized ``state``.

Normalization is Unicode NFC, casefold, every run of non-word characters
replaced by one space, then strip. JevBench structured states are rendered with
``data.pipeline.render_jevbench_state``, the same text the model reads.

Near-duplicates: MinHash over character 5-grams of the normalized state, 128
permutations, 32 bands, confirmed when the MinHash Jaccard estimate is at least
0.7. That threshold is the one ``data.synthetic_dedup`` already uses for state
text; it was chosen before the counts were seen.

The canonical source list and id-dedup order are the ones fixed by #234:
target-30K, residual-20K, typed-decisions, assembled synthetic, then the
remaining corpora. Exact dedup is reported, not applied: a repeated id with a
different question is a distinct record, not a duplicate. RACE is the case that
matters, with 498 questions over 144 passage ids.

Usage:
    python -m data.corpus_audit --artifact research/235-corpus-audit.json
    python -m data.corpus_audit --check --skip-near-dup
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import multiprocessing
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .lsh import LSHIndex, MinHash, char_ngrams

try:  # optional fast path; the pure-Python comparison is the fallback
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on a minimal install
    _np = None

WORD_SHINGLE = 8
MIN_SHARED_SHINGLES = 3

NEAR_DUP_NGRAM = 5
NEAR_DUP_THRESHOLD = 0.7
NEAR_DUP_NUM_PERM = 128
NEAR_DUP_BANDS = 32
NEAR_DUP_SEED = 42

JEVBENCH_TIERS = ("easy", "hard", "original")
JEVBENCH_EXPECTED = {"easy": 48, "hard": 111, "original": 72}

# Canonical priority order from #234. Synthetic is assembled between the
# typed-decisions corpus and the remaining files.
HIGH_PRIORITY_FILES: tuple[tuple[str, str], ...] = (
    ("label_target_30k_jev", "label_target_30k_jev.jsonl"),
    ("residual_train_20000_jev", "residual_train_20000_jev.jsonl"),
)
TYPED_DECISIONS = ("typed_decisions", "typed_decisions.jsonl")
LOW_PRIORITY_FILES: tuple[tuple[str, str], ...] = (
    ("apigen_policy_compliance", "apigen_policy_compliance.jsonl"),
    ("apigen_tool_action", "apigen_tool_action.jsonl"),
    ("apigen_tool_argument", "apigen_tool_argument.jsonl"),
    ("apigen_tool_choice", "apigen_tool_choice.jsonl"),
    ("apigen_tool_next_action", "apigen_tool_next_action.jsonl"),
    ("abstention_train_20k", "abstention_train_20k.jsonl"),
    ("abstention_train_30k", "abstention_train_30k.jsonl"),
    ("mmlu_pro_short_reasoning_choice", "mmlu_pro_short_reasoning_choice.jsonl"),
    ("hotpotqa_long_choice", "hotpotqa_long_choice.jsonl"),
    ("musique_long_choice", "musique_long_choice.jsonl"),
    ("longbench_long_choice", "longbench_long_choice.jsonl"),
    ("twowiki_long_choice", "twowiki_long_choice.jsonl"),
    ("race_short_passage_choice", "race_short_passage_choice.jsonl"),
)
CANONICAL_FILES = HIGH_PRIORITY_FILES + (TYPED_DECISIONS,) + LOW_PRIORITY_FILES

CANONICAL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "abstention_train_20k": (
        "abstention_train_20k.jsonl",
        "abstention_20k/train.jsonl",
    ),
    "abstention_train_30k": (
        "abstention_train_30k.jsonl",
        "abstention_30k/train.jsonl",
    ),
}

SYNTHETIC_RE = re.compile(r"^synthetic-(?P<domain>.+?)-\d{4}-")

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize_state(text: str) -> str:
    """Return the comparison form of a state string."""
    value = unicodedata.normalize("NFC", str(text)).casefold()
    return _NON_WORD.sub(" ", value).strip()


def word_shingles(text: str, n: int = WORD_SHINGLE) -> set[str]:
    """Return the set of n-word shingles of a state."""
    words = normalize_state(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _state_digest(normalized: str) -> bytes:
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def _question_digest(question: dict[str, Any]) -> bytes:
    return hashlib.blake2b(
        json.dumps(
            question, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"),
        digest_size=16,
    ).digest()


def _record_digest(
    record_id: str,
    normalized_state: str,
    question: dict[str, Any],
    label: Any,
) -> bytes:
    payload = json.dumps(
        [record_id, normalized_state, question, label],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


@dataclass(frozen=True)
class JevItem:
    id: str
    tier: str
    state: str
    normalized_state: str
    shingles: frozenset[str]


@dataclass(slots=True)
class CorpusRecord:
    source: str
    id: str
    state: str
    question: dict[str, Any]
    label: Any
    group: str | None


@dataclass(slots=True)
class StateInfo:
    text: str
    sources: set[str] = field(default_factory=set)
    rows: int = 0
    example_id: str = ""


@dataclass(slots=True)
class SourceAccum:
    rows: int = 0
    ids: collections.Counter = field(default_factory=collections.Counter)
    states: collections.Counter = field(default_factory=collections.Counter)
    records: set[bytes] = field(default_factory=set)
    id_first_question: dict[str, bytes] = field(default_factory=dict)
    ids_with_multiple_questions: set[str] = field(default_factory=set)
    state_first_question: dict[bytes, bytes] = field(default_factory=dict)
    states_with_multiple_questions: set[bytes] = field(default_factory=set)

    def add(self, record: CorpusRecord) -> None:
        normalized = normalize_state(record.state)
        state_key = _state_digest(normalized)
        question_key = _question_digest(record.question)

        self.rows += 1
        self.ids[record.id] += 1
        self.states[state_key] += 1
        self.records.add(_record_digest(record.id, normalized, record.question, record.label))

        first_id_question = self.id_first_question.get(record.id)
        if first_id_question is None:
            self.id_first_question[record.id] = question_key
        elif first_id_question != question_key:
            self.ids_with_multiple_questions.add(record.id)

        first_state_question = self.state_first_question.get(state_key)
        if first_state_question is None:
            self.state_first_question[state_key] = question_key
        elif first_state_question != question_key:
            self.states_with_multiple_questions.add(state_key)

    def summary(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "distinct_ids": len(self.ids),
            "distinct_normalized_states": len(self.states),
            "distinct_records": len(self.records),
            "exact_duplicate_rows": self.rows - len(self.records),
            "rows_sharing_id": self.rows - len(self.ids),
            "rows_sharing_state": self.rows - len(self.states),
            "ids_with_multiple_questions": len(self.ids_with_multiple_questions),
            "states_with_multiple_questions": len(self.states_with_multiple_questions),
            "max_rows_per_id": max(self.ids.values()) if self.ids else 0,
            "max_rows_per_state": max(self.states.values()) if self.states else 0,
        }


def load_jevbench(directory: Path) -> list[JevItem]:
    """Load the 231 public JevBench items, rendered as the model reads them."""
    from .pipeline import render_jevbench_state

    items: list[JevItem] = []
    for tier in JEVBENCH_TIERS:
        path = directory / f"{tier}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing JevBench tier: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                state = render_jevbench_state(raw["state"])
                items.append(
                    JevItem(
                        id=str(raw["id"]).strip(),
                        tier=tier,
                        state=state,
                        normalized_state=normalize_state(state),
                        shingles=frozenset(word_shingles(state)),
                    )
                )
    expected = sum(JEVBENCH_EXPECTED.values())
    if len(items) != expected:
        raise ValueError(f"expected {expected} JevBench items, found {len(items)}")
    counts = collections.Counter(item.tier for item in items)
    if dict(counts) != JEVBENCH_EXPECTED:
        raise ValueError(f"JevBench tier counts {dict(counts)} != {JEVBENCH_EXPECTED}")
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("JevBench ids are not unique")
    return items


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _resolve_source(root: Path, name: str, relative: str) -> Path | None:
    for candidate in CANONICAL_FALLBACKS.get(name, (relative,)):
        path = root / candidate
        if path.exists():
            return path
    return None


def _synthetic_source(record_id: str) -> str:
    match = SYNTHETIC_RE.match(record_id)
    return f"synthetic/{match.group('domain')}" if match else "synthetic/unknown"


def _record_from_item(item: Any) -> CorpusRecord:
    return CorpusRecord(
        source=_synthetic_source(item.id) if item.source == "synthetic" else item.source,
        id=item.id,
        state=item.state,
        question=item.question,
        label=item.label,
        group=item.group,
    )


def iter_file_records(path: Path, source: str) -> Iterator[CorpusRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            yield CorpusRecord(
                source=source,
                id=str(raw["id"]).strip(),
                state=str(raw.get("state", "")),
                question=raw.get("question") or {},
                label=raw.get("label"),
                group=raw.get("group"),
            )


def iter_synthetic_records() -> Iterator[CorpusRecord]:
    from .pipeline import load_synthetic

    logging.getLogger("data.synthetic").setLevel(logging.ERROR)
    for item in load_synthetic():
        yield _record_from_item(item)


def iter_corpus(
    root: Path,
    include_synthetic: bool = True,
    include_typed_decisions: bool = True,
) -> tuple[Iterator[CorpusRecord], list[dict[str, Any]], list[str]]:
    """Return a record iterator plus coverage and missing-source lists.

    Sources are yielded in the canonical id-dedup priority order from #234.
    """
    files: list[dict[str, Any]] = []
    missing: list[str] = []

    def register(name: str, relative: str) -> Path | None:
        path = _resolve_source(root, name, relative)
        if path is None:
            missing.append(relative)
            return None
        files.append(
            {
                "source": name,
                "path": str(path),
                "relpath": _relative(path, root),
                "file": path.name,
                "bytes": path.stat().st_size,
            }
        )
        return path

    def generator() -> Iterator[CorpusRecord]:
        for name, relative in HIGH_PRIORITY_FILES:
            path = register(name, relative)
            if path is not None:
                yield from iter_file_records(path, name)

        if include_typed_decisions:
            typed_path = _resolve_source(root, *TYPED_DECISIONS)
            if typed_path is not None:
                files.append(
                    {
                        "source": "typed_decisions",
                        "path": str(typed_path),
                        "relpath": _relative(typed_path, root),
                        "file": typed_path.name,
                        "bytes": typed_path.stat().st_size,
                    }
                )
                yield from iter_file_records(typed_path, "typed_decisions")
            else:
                try:
                    from .pipeline import load_typed_decisions

                    for item in load_typed_decisions():
                        yield _record_from_item(item)
                    files.append(
                        {
                            "source": "typed_decisions",
                            "path": "",
                            "relpath": "typed_decisions (HF parquet)",
                            "file": "typed_decisions (HF parquet)",
                            "bytes": None,
                        }
                    )
                except Exception as exc:  # network or pyarrow unavailable
                    missing.append(TYPED_DECISIONS[1])
                    print(f"typed_decisions unavailable: {exc}", file=sys.stderr)

        if include_synthetic:
            synthetic_dir = root / "synthetic"
            if synthetic_dir.exists():
                for path in sorted(synthetic_dir.glob("*_families.jsonl")):
                    domain = path.name[: -len("_families.jsonl")]
                    files.append(
                        {
                            "source": f"synthetic/{domain}",
                            "path": str(path),
                            "relpath": _relative(path, root),
                            "file": path.name,
                            "bytes": path.stat().st_size,
                        }
                    )
                yield from iter_synthetic_records()
            else:
                missing.append("synthetic/")

        for name, relative in LOW_PRIORITY_FILES:
            path = register(name, relative)
            if path is not None:
                yield from iter_file_records(path, name)

    return generator(), files, missing


class ContaminationIndex:
    """Inverted index over the 231 public JevBench states."""

    def __init__(self, items: Sequence[JevItem]):
        self.items = list(items)
        self.by_id = {item.id: item for item in items}
        self.by_state = {item.normalized_state: item for item in items}
        self.postings: dict[str, list[int]] = collections.defaultdict(list)
        for index, item in enumerate(self.items):
            for shingle in item.shingles:
                self.postings[shingle].append(index)

    def check_state(
        self, normalized: str, shingles: set[str]
    ) -> list[tuple[JevItem, str, int]]:
        hits: list[tuple[JevItem, str, int]] = []
        exact = self.by_state.get(normalized)
        if exact is not None:
            return [(exact, "state", 0)]

        counts: collections.Counter[int] = collections.Counter()
        for shingle in shingles:
            for index in self.postings.get(shingle, ()):
                counts[index] += 1
        for index, shared in counts.items():
            if shared >= MIN_SHARED_SHINGLES:
                hits.append((self.items[index], "shingle", shared))
        return hits


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _signature_chunk(payload: tuple[int, list[str]]) -> list[tuple[int, tuple[int, ...]]]:
    start, texts = payload
    out: list[tuple[int, tuple[int, ...]]] = []
    for offset, text in enumerate(texts):
        minhash = MinHash(num_perm=NEAR_DUP_NUM_PERM, seed=NEAR_DUP_SEED)
        minhash.update(char_ngrams(text, NEAR_DUP_NGRAM))
        out.append((start + offset, minhash.signature))
    return out


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != root:
            self.parent[value], value = root, self.parent[value]
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _compute_signatures(texts: list[str], workers: int) -> list[tuple[int, ...]]:
    total = len(texts)
    if total == 0:
        return []
    chunk_size = max(1, min(500, total // max(1, workers * 4)))
    chunks = [
        (start, texts[start:start + chunk_size])
        for start in range(0, total, chunk_size)
    ]
    signatures: list[tuple[int, ...] | None] = [None] * total
    if workers <= 1:
        for chunk in chunks:
            for index, signature in _signature_chunk(chunk):
                signatures[index] = signature
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_signature_chunk, chunks):
                for index, signature in result:
                    signatures[index] = signature
    missing = sum(1 for signature in signatures if signature is None)
    if missing:
        raise RuntimeError(f"{missing} states produced no MinHash signature")
    return [signature for signature in signatures if signature is not None]


def near_duplicate_report(
    state_info: dict[bytes, StateInfo],
    workers: int,
    max_clusters: int,
    progress: bool = False,
) -> dict[str, Any]:
    keys = list(state_info)
    texts = [state_info[key].text for key in keys]
    started = time.time()
    if progress:
        print(
            f"near-duplicate: computing {len(texts)} signatures with {workers} worker(s)",
            file=sys.stderr,
            flush=True,
        )
    signatures = _compute_signatures(texts, workers)
    signature_matrix = (
        _np.asarray(signatures, dtype=_np.uint64) if _np is not None else None
    )
    if progress:
        print(
            f"near-duplicate: signatures done in {time.time() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    if progress:
        print("near-duplicate: building the LSH index", file=sys.stderr, flush=True)
    index = LSHIndex(
        num_perm=NEAR_DUP_NUM_PERM,
        bands=NEAR_DUP_BANDS,
        seed=NEAR_DUP_SEED,
    )
    union = _UnionFind(len(keys))
    pair_sources: collections.Counter[tuple[str, str]] = collections.Counter()
    candidate_pairs = 0
    merging_pairs = 0
    for position, signature in enumerate(signatures):
        position_root = union.find(position)
        for candidate in index.query(signature):
            candidate_pairs += 1
            if union.find(candidate) == position_root:
                continue
            if signature_matrix is not None:
                similarity = float(
                    (signature_matrix[position] == signature_matrix[candidate]).sum()
                ) / NEAR_DUP_NUM_PERM
            else:
                similarity = _jaccard(signature, signatures[candidate])
            if similarity < NEAR_DUP_THRESHOLD:
                continue
            union.union(position, candidate)
            position_root = union.find(position)
            merging_pairs += 1
            left = state_info[keys[position]].sources
            right = state_info[keys[candidate]].sources
            for left_source in left:
                for right_source in right:
                    if left_source != right_source:
                        pair_sources[tuple(sorted((left_source, right_source)))] += 1
        index.insert(position, signature)

    if progress:
        print(
            f"near-duplicate: {candidate_pairs} candidate pairs, "
            f"{merging_pairs} merging pairs in {time.time() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for position in range(len(keys)):
        groups[union.find(position)].append(position)

    clusters = [members for members in groups.values() if len(members) >= 2]
    clusters.sort(key=len, reverse=True)

    by_source: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"states_in_clusters": 0, "rows_in_clusters": 0, "clusters_touching": 0}
    )
    cross_source_clusters = 0
    same_family_clusters = 0
    total_rows_in_clusters = 0
    cluster_entries: list[dict[str, Any]] = []
    state_to_cluster: dict[str, str] = {}
    for cluster_index, members in enumerate(clusters):
        cluster_id = f"nd-{cluster_index:05d}"
        member_keys = [keys[index] for index in members]
        for key in member_keys:
            state_to_cluster[key.hex()] = cluster_id
        infos = [state_info[key] for key in member_keys]
        sources = sorted({source for info in infos for source in info.sources})
        rows = sum(info.rows for info in infos)
        total_rows_in_clusters += rows
        if len(sources) > 1:
            cross_source_clusters += 1
        example_ids = [info.example_id for info in infos[:5]]
        families = {
            match.group(0)
            for example in example_ids
            if (match := re.match(r"^synthetic-(.+?)-\d{4}-", example))
        }
        if (
            sources
            and all(source.startswith("synthetic/") for source in sources)
            and len(families) == 1
        ):
            same_family_clusters += 1
        for source in sources:
            by_source[source]["clusters_touching"] += 1
        for info in infos:
            for source in info.sources:
                by_source[source]["states_in_clusters"] += 1
                by_source[source]["rows_in_clusters"] += info.rows
        if len(cluster_entries) < max_clusters:
            entry: dict[str, Any] = {
                "cluster_id": cluster_id,
                "size": len(members),
                "sources": sources,
                "rows": rows,
                "example_ids": example_ids,
                "state_ids": [key.hex() for key in member_keys],
            }
            cluster_entries.append(entry)

    clustered_states = sum(len(members) for members in clusters)
    return {
        "threshold": NEAR_DUP_THRESHOLD,
        "num_perm": NEAR_DUP_NUM_PERM,
        "bands": NEAR_DUP_BANDS,
        "ngram": NEAR_DUP_NGRAM,
        "unique_states": len(keys),
        "candidate_pairs": candidate_pairs,
        "merging_pairs": merging_pairs,
        "clusters": len(clusters),
        "cross_source_clusters": cross_source_clusters,
        "same_synthetic_family_clusters": same_family_clusters,
        "states_in_clusters": clustered_states,
        "rows_in_clusters": total_rows_in_clusters,
        "largest_cluster": len(clusters[0]) if clusters else 0,
        "by_source": dict(sorted(by_source.items())),
        "cross_source_pairs": [
            {"sources": list(pair), "pairs": count}
            for pair, count in pair_sources.most_common(200)
        ],
        "cluster_details": cluster_entries,
        "state_to_cluster": state_to_cluster,
        "unclustered_states": len(keys) - clustered_states,
        "truncated_clusters": max(0, len(clusters) - len(cluster_entries)),
        "elapsed_seconds": round(time.time() - started, 1),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_file_hashes(
    corpus_root: Path, files: list[dict[str, Any]], include_synthetic: bool
) -> dict[str, str]:
    """Hash every file the audit reads, including synthetic variants and labels."""
    hashes: dict[str, str] = {}
    for entry in files:
        if entry["path"]:
            hashes[entry["source"]] = _sha256(Path(entry["path"]))
    if include_synthetic:
        synthetic_dir = corpus_root / "synthetic"
        if synthetic_dir.exists():
            for path in sorted(synthetic_dir.glob("*.jsonl")):
                hashes[f"synthetic/{path.name}"] = _sha256(path)
    return hashes


def audit(
    corpus_root: Path,
    jevbench_root: Path,
    include_synthetic: bool = True,
    include_typed_decisions: bool = True,
    skip_near_dup: bool = False,
    workers: int = 1,
    max_clusters: int = 5000,
    max_hits: int = 20000,
    canonical_dedup: bool = True,
    keep_same_source_id_reuse: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    started = time.time()
    jevbench = load_jevbench(jevbench_root)
    contamination = ContaminationIndex(jevbench)

    records, files, missing = iter_corpus(
        corpus_root, include_synthetic, include_typed_decisions
    )
    per_source: dict[str, SourceAccum] = collections.defaultdict(SourceAccum)
    raw_rows_by_source: collections.Counter[str] = collections.Counter()
    raw_id_groups: dict[str, set[str]] = collections.defaultdict(set)
    raw_id_rows: collections.Counter[str] = collections.Counter()
    state_source_pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    state_info: dict[bytes, StateInfo] = {}

    seen_id_group: dict[str, str] = {}
    seen_record_digests: set[bytes] = set()
    dropped_by_source: collections.Counter[str] = collections.Counter()
    dropped_pairs: collections.Counter[tuple[str, str]] = collections.Counter()

    raw_rows = 0
    checked_rows = 0
    dropped_rows = 0
    same_source_extra_rows = 0

    contamination_cache: dict[bytes, list[tuple[str, str, str, int]]] = {}
    hit_rows = 0
    hit_ids: set[str] = set()
    hit_entries: list[dict[str, Any]] = []
    hits_by_source: collections.Counter[str] = collections.Counter()
    hits_by_reason: collections.Counter[str] = collections.Counter()
    hits_by_jevbench: collections.Counter[str] = collections.Counter()
    truncated_hits = 0

    for record in records:
        raw_rows += 1
        if progress and raw_rows % 25000 == 0:
            print(
                f"progress: {raw_rows} rows loaded, {checked_rows} audited, "
                f"{hit_rows} hit rows, {time.time() - started:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        raw_rows_by_source[record.source] += 1
        group = (
            "synthetic" if record.source.startswith("synthetic/") else record.source
        )
        raw_id_groups[record.id].add(group)
        raw_id_rows[record.id] += 1

        normalized = normalize_state(record.state)
        state_key = _state_digest(normalized)
        record_digest = _record_digest(
            record.id, normalized, record.question, record.label
        )

        if canonical_dedup:
            if record_digest in seen_record_digests:
                dropped_rows += 1
                dropped_by_source[record.source] += 1
                first_group = seen_id_group.get(record.id)
                if first_group is not None and first_group != group:
                    dropped_pairs[(first_group, group)] += 1
                continue
            first_group = seen_id_group.get(record.id)
            if first_group is None:
                seen_id_group[record.id] = group
            elif first_group != group:
                dropped_rows += 1
                dropped_by_source[record.source] += 1
                dropped_pairs[(first_group, group)] += 1
                continue
            elif not keep_same_source_id_reuse:
                dropped_rows += 1
                dropped_by_source[record.source] += 1
                continue
            else:
                same_source_extra_rows += 1

        seen_record_digests.add(record_digest)
        checked_rows += 1
        per_source[record.source].add(record)
        info = state_info.get(state_key)
        if info is None:
            state_info[state_key] = StateInfo(
                text=normalized,
                sources={record.source},
                rows=1,
                example_id=record.id,
            )
        else:
            for existing in info.sources:
                if existing != record.source:
                    state_source_pairs[tuple(sorted((existing, record.source)))] += 1
            info.sources.add(record.source)
            info.rows += 1

        state_hits = contamination_cache.get(state_key)
        if state_hits is None:
            state_hits = [
                (item.id, item.tier, reason, shared)
                for item, reason, shared in contamination.check_state(
                    normalized, word_shingles(normalized)
                )
            ]
            contamination_cache[state_key] = state_hits

        record_hits: list[tuple[str, str, str, int]] = []
        exact = contamination.by_id.get(record.id)
        if exact is not None:
            record_hits.append((exact.id, exact.tier, "id", 0))
        for hit in state_hits:
            if hit[0] != record.id:
                record_hits.append(hit)

        if record_hits:
            hit_rows += 1
            hit_ids.add(record.id)
            hits_by_source[record.source] += 1
            for jev_id, tier, reason, shared in record_hits:
                hits_by_reason[reason] += 1
                hits_by_jevbench[jev_id] += 1
                if len(hit_entries) < max_hits:
                    hit_entries.append(
                        {
                            "record_id": record.id,
                            "source": record.source,
                            "jevbench_id": jev_id,
                            "jevbench_tier": tier,
                            "reason": reason,
                            "shared_shingles": shared,
                        }
                    )
                else:
                    truncated_hits += 1

    source_summary: dict[str, dict[str, Any]] = {}
    for source, accum in sorted(per_source.items()):
        summary = accum.summary()
        summary["raw_rows"] = raw_rows_by_source[source]
        source_summary[source] = summary

    cross_source_ids = [
        {"id": record_id, "sources": sorted(groups), "rows": raw_id_rows[record_id]}
        for record_id, groups in raw_id_groups.items()
        if len(groups) > 1
    ]
    cross_source_states = [
        {
            "state_id": key.hex(),
            "sources": sorted(info.sources),
            "rows": info.rows,
            "example_id": info.example_id,
        }
        for key, info in state_info.items()
        if len(info.sources) > 1
    ]
    cross_source_states.sort(key=lambda entry: (-entry["rows"], entry["state_id"]))

    canonical_ids = len(seen_id_group) if canonical_dedup else len(raw_id_groups)
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": {
            "normalization": "Unicode NFC, casefold, non-word runs -> space, strip",
            "word_shingle": WORD_SHINGLE,
            "min_shared_shingles": MIN_SHARED_SHINGLES,
            "near_dup": {
                "ngram": NEAR_DUP_NGRAM,
                "threshold": NEAR_DUP_THRESHOLD,
                "num_perm": NEAR_DUP_NUM_PERM,
                "bands": NEAR_DUP_BANDS,
                "confirmation": "MinHash Jaccard estimate",
            },
        },
        "coverage": {
            "canonical_dedup": canonical_dedup,
            "keep_same_source_id_reuse": keep_same_source_id_reuse,
            "files": [
                {
                    "source": entry["source"],
                    "file": entry["file"],
                    "bytes": entry["bytes"],
                }
                for entry in files
            ],
            "missing_sources": missing,
            "jevbench_items": len(jevbench),
            "jevbench_tiers": dict(
                collections.Counter(item.tier for item in jevbench)
            ),
            "raw_rows_loaded": raw_rows,
            "canonical_records": checked_rows,
            "passage_ids": canonical_ids,
            "dropped_duplicate_rows": dropped_rows,
            "same_source_id_extra_rows": same_source_extra_rows,
            "dropped_by_source": dict(sorted(dropped_by_source.items())),
            "dropped_cross_source_pairs": [
                {"first": first, "dropped": dropped, "rows": rows}
                for (first, dropped), rows in sorted(
                    dropped_pairs.items(), key=lambda item: -item[1]
                )
            ],
            "distinct_normalized_states": len(state_info),
        },
        "dedup": {
            "per_source": source_summary,
            "global": {
                "records": checked_rows,
                "passage_ids": canonical_ids,
                "distinct_all_ids": len(raw_id_groups),
                "distinct_normalized_states": len(state_info),
                "exact_duplicate_rows": checked_rows - len(seen_record_digests),
                "ids_in_multiple_sources": len(cross_source_ids),
                "states_in_multiple_sources": len(cross_source_states),
            },
            "cross_source_ids": cross_source_ids[:500],
            "cross_source_ids_truncated": max(0, len(cross_source_ids) - 500),
            "cross_source_state_pairs": [
                {"sources": list(pair), "states": count}
                for pair, count in state_source_pairs.most_common(200)
            ],
            "cross_source_states": cross_source_states[:500],
            "cross_source_states_truncated": max(0, len(cross_source_states) - 500),
        },
        "contamination": {
            "checked_records": checked_rows,
            "checked_passage_ids": canonical_ids,
            "hit_rows": hit_rows,
            "hit_ids": len(hit_ids),
            "hits": hit_entries,
            "hits_truncated": truncated_hits,
            "by_source": dict(sorted(hits_by_source.items())),
            "by_reason": dict(sorted(hits_by_reason.items())),
            "by_jevbench_id": dict(sorted(hits_by_jevbench.items())),
        },
        "file_hashes": _input_file_hashes(corpus_root, files, include_synthetic),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if skip_near_dup:
        report["near_duplicates"] = {"skipped": True}
    else:
        report["near_duplicates"] = near_duplicate_report(
            state_info,
            workers=workers,
            max_clusters=max_clusters,
            progress=progress,
        )
        report["elapsed_seconds"] = round(time.time() - started, 1)

    return report


def write_cluster_artifact(report: dict[str, Any], path: Path) -> None:
    """Write the near-duplicate clusters as a standalone artifact for consumers."""
    near = report.get("near_duplicates") or {}
    if near.get("skipped"):
        raise ValueError("near-duplicate pass was skipped")
    payload = {
        "generated_at": report["generated_at"],
        "parameters": {
            "ngram": near["ngram"],
            "threshold": near["threshold"],
            "num_perm": near["num_perm"],
            "bands": near["bands"],
            "confirmation": report["rule"]["near_dup"]["confirmation"],
        },
        "normalization": report["rule"]["normalization"],
        "state_key": "blake2b-128 of the normalized state text, hex",
        "cluster_id_rule": (
            "clusters sorted by size descending; nd-00000 is the largest"
        ),
        "corpus": {
            "canonical_records": report["coverage"]["canonical_records"],
            "passage_ids": report["coverage"]["passage_ids"],
            "distinct_normalized_states": report["coverage"][
                "distinct_normalized_states"
            ],
        },
        "cluster_count": near["clusters"],
        "distinct_states_in_clusters": near["states_in_clusters"],
        "rows_in_clusters": near["rows_in_clusters"],
        "clusters": near["cluster_details"],
        "state_to_cluster": near["state_to_cluster"],
        "unclustered_states": near["unclustered_states"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    coverage = report["coverage"]
    dedup = report["dedup"]
    contamination = report["contamination"]
    print(f"raw rows loaded: {coverage['raw_rows_loaded']}")
    print(
        f"canonical records: {coverage['canonical_records']} "
        f"(passage ids: {coverage['passage_ids']})"
    )
    print(
        "dropped duplicate rows: "
        f"{coverage['dropped_duplicate_rows']} "
        f"(same-source id reuse kept: {coverage['same_source_id_extra_rows']})"
    )
    print(f"distinct normalized states: {coverage['distinct_normalized_states']}")
    print(f"exact duplicate rows: {dedup['global']['exact_duplicate_rows']}")
    print(f"ids in multiple sources: {dedup['global']['ids_in_multiple_sources']}")
    print(f"states in multiple sources: {dedup['global']['states_in_multiple_sources']}")
    print(
        "contamination: "
        f"{contamination['hit_rows']} of {coverage['canonical_records']} "
        "canonical records hit, "
        f"{len(contamination['hits'])} hit entries "
        f"(truncated {contamination['hits_truncated']})"
    )
    if contamination["by_source"]:
        for source, count in contamination["by_source"].items():
            print(f"  {source}: {count} rows")
    if coverage["missing_sources"]:
        print(f"missing sources: {', '.join(coverage['missing_sources'])}")
    near = report.get("near_duplicates") or {}
    if not near.get("skipped"):
        print(
            "near-duplicates: "
            f"{near.get('clusters', 0)} clusters, "
            f"{near.get('states_in_clusters', 0)} states, "
            f"{near.get('cross_source_clusters', 0)} cross-source clusters"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--jevbench-root", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--clusters-artifact", type=Path, default=None)
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--no-typed-decisions", action="store_true")
    parser.add_argument("--skip-near-dup", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-clusters", type=int, default=5000)
    parser.add_argument("--max-hits", type=int, default=20000)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--no-canonical-dedup",
        action="store_true",
        help="audit every raw row instead of the canonical id-deduped set",
    )
    parser.add_argument(
        "--strict-id-dedup",
        action="store_true",
        help="drop a repeated id even when it carries a different question",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the contamination rule fires on any row",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    jevbench_root = args.jevbench_root or args.corpus_root / "jevbench"
    report = audit(
        corpus_root=args.corpus_root,
        jevbench_root=jevbench_root,
        include_synthetic=not args.no_synthetic,
        include_typed_decisions=not args.no_typed_decisions,
        skip_near_dup=args.skip_near_dup,
        workers=args.workers,
        max_clusters=args.max_clusters,
        max_hits=args.max_hits,
        canonical_dedup=not args.no_canonical_dedup,
        keep_same_source_id_reuse=not args.strict_id_dedup,
        progress=args.progress,
    )
    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps(report, indent=1, sort_keys=False))
        print(f"wrote {args.artifact}")
    if args.clusters_artifact:
        write_cluster_artifact(report, args.clusters_artifact)
        print(f"wrote {args.clusters_artifact}")
    _print_summary(report)
    if args.check and report["contamination"]["hit_rows"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
