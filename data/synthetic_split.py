"""Cluster-aware train/eval splitting for synthetic data families.

The split is computed per *near-duplicate cluster* rather than per family
(issue #120). Family-level splitting (#117) keeps every variant of a family
on one side, but it is not state-disjoint: a small number of states recur in
more than one family, and when those families land on opposite sides the
same state is evaluated on text it was trained on.

Generation-time dedup cannot be relied on to have removed those families.
It LSH-bands at 16 bands x 8 rows, which gives only ~0.61 recall at jaccard
0.7, so roughly a third of true near-duplicate pairs survive it by
construction. This module therefore clusters the corpus at load time, with a
banding that is effectively exhaustive at the same threshold (32 bands of 2
rows: a pair at jaccard 0.7 is missed with probability ~3e-10), and gives one
split per cluster. Low row counts raise the candidate rate at lower
similarities, but a candidate check is 64 integer comparisons, so the cost
stays in the signatures rather than the banding.

Two deliberate choices keep results comparable with earlier runs:

* A family in a singleton cluster keeps exactly the split the family-level
  scheme gave it, so only the clustered families move.
* A cluster takes the split of its lowest-indexed member, which is the
  family-level split of that member. Eval-set membership therefore changes
  only for families that share a state with another family.

Clustering is per domain: generation and dedup are per domain too, and
states do not repeat across domains in this corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .lsh import LSHIndex, char_ngrams

logger = logging.getLogger(__name__)

CACHE_NAME = ".split_clusters.json"
# Bump whenever the clustering algorithm changes: the cache stores computed
# splits, so a code change without a version bump would silently serve splits
# produced by the previous algorithm.
CACHE_VERSION = 2
DEFAULT_THRESHOLD = 0.7
DEFAULT_NUM_PERM = 64
DEFAULT_BANDS = 32


def split_from_key(key: str, eval_ratio: float = 0.1) -> str:
    """Map a stable string key to a train/test split.

    Uses md5 rather than ``hash()`` because Python's string hash is
    randomised per process (PYTHONHASHSEED), which would make the split
    unstable across runs and machines.

    Args:
        key: Stable identifier of the split unit.
        eval_ratio: Fraction of units held out for evaluation.

    Returns:
        ``"test"`` or ``"train"``.
    """
    digest = hashlib.md5(key.encode()).hexdigest()
    return "test" if int(digest[:8], 16) / 0xFFFFFFFF < eval_ratio else "train"


def family_state_texts(
    family: dict[str, Any],
    variants: dict[str, Any] | None = None,
) -> list[str]:
    """Collect every state text an item of this family can be built from.

    ``family_to_typed_questions`` attaches the base state to most items, the
    counterfactual state to the counterfactual items, and the paraphrased
    state to the ``para-state`` items. All three must take part in the
    clustering, because a state can be shared through any of them.

    Args:
        family: A family record from ``<domain>_families.jsonl``.
        variants: The matching record from ``<domain>_variants.jsonl``.

    Returns:
        Normalised, de-duplicated, non-empty state texts.
    """
    variants = variants or {}
    counterfactual = variants.get("counterfactual") or {}
    paraphrase = variants.get("paraphrase") or {}
    raw = [
        family.get("state") or "",
        counterfactual.get("state") or "",
        paraphrase.get("state_paraphrase") or "",
    ]

    texts: list[str] = []
    seen: set[str] = set()
    for text in raw:
        normalised = " ".join(str(text).split())
        if normalised and normalised not in seen:
            seen.add(normalised)
            texts.append(normalised)
    return texts


class _UnionFind:
    """Disjoint-set with path halving."""

    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        parent = self._parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            # Smaller index wins, so roots are the lowest-indexed members.
            if root_left < root_right:
                self._parent[root_right] = root_left
            else:
                self._parent[root_left] = root_right


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Approximate Jaccard from two MinHash signatures."""
    if len(left) != len(right):
        raise ValueError("signature length mismatch")
    if not left:
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def _exact_jaccard(left: str, right: str) -> float:
    """Exact character 5-gram Jaccard between two state texts."""
    a, b = char_ngrams(left), char_ngrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_families(
    family_states: Sequence[Sequence[str]],
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    seed: int = 42,
    force_group: Sequence[Any] | None = None,
) -> list[int]:
    """Group families whose states are near-duplicates of each other.

    Similarity is character 5-gram Jaccard, the same measure ``dedup_families``
    uses, so a cluster is exactly a set of families that would have been
    deduplicated had the banding found them. Clusters are transitive: if A is
    near-duplicate of B and B of C, A and C land in the same cluster even when
    A and C themselves fall below the threshold.

    Banding proposes candidate pairs and MinHash estimates rank them, but the
    decision is made on the exact Jaccard. A 64-permutation estimate near 0.7
    has a standard error of ~0.06, which is wide enough to leave pairs at 0.71
    on opposite sides of the threshold about half the time.

    Args:
        family_states: State texts per family, as returned by
            :func:`family_state_texts`.
        threshold: Jaccard at or above which two states are duplicates.
        num_perm: MinHash permutations. Must be divisible by ``bands``.
        bands: LSH bands. More bands raise recall; with ``num_perm=64`` this
            defaults to 32 bands of 2 rows, which misses a pair at jaccard 0.7
            with probability ~3e-10.
        seed: MinHash permutation seed.
        force_group: Optional key per family. Families sharing a key are put in
            one cluster even when their states are unrelated. ``_family_idx``
            is not unique in the corpus (two domains reuse indices), and two
            families sharing an index also share their item ids and group
            name, so they must share a split.

    Returns:
        Cluster label per family: the lowest family index in its cluster, so a
        family with no near-duplicate is labelled with its own index.
    """
    lsh = LSHIndex(num_perm=num_perm, bands=bands, seed=seed)
    union_find = _UnionFind(len(family_states))

    if force_group is not None:
        first_position: dict[Any, int] = {}
        for position, key in enumerate(force_group):
            if key in first_position:
                union_find.union(first_position[key], position)
            else:
                first_position[key] = position
    signatures: list[tuple[int, ...]] = []
    state_of: list[str] = []
    owners: list[int] = []
    # Cache of *text* pairs already compared. It must not be keyed on the family
    # pair: a family contributes up to three state texts, so a family-level key
    # would make the first combination the only one ever tested and a duplicate
    # reachable through a counterfactual or paraphrase would stay in another
    # split. Keying on the pair of inserted signatures is exact and still visits
    # each text pair once.
    checked: set[tuple[int, int]] = set()
    # The signature only estimates Jaccard: at 64 permutations its standard
    # error near the threshold is ~0.057, so a pair at exactly 0.7 is called a
    # duplicate only ~half the time. Candidates are banded at the threshold and
    # confirmed with the exact measure; the prefilter only bounds wasted work,
    # and its statistical justification lives in ``_prefilter_for``.
    prefilter = _prefilter_for(threshold, num_perm)

    for family_idx, texts in enumerate(family_states):
        for text in texts:
            signature = lsh.make_minhash(text).signature
            current = len(signatures)
            for candidate in lsh.query(signature):
                other_family = owners[candidate]
                if other_family == family_idx:
                    continue
                pair = (
                    min(other_family, family_idx),
                    max(other_family, family_idx),
                )
                text_pair = (min(candidate, current), max(candidate, current))
                if text_pair in checked:
                    continue
                checked.add(text_pair)
                if _jaccard(signature, signatures[candidate]) < prefilter:
                    continue
                if _exact_jaccard(text, state_of[candidate]) >= threshold:
                    union_find.union(pair[0], pair[1])
            lsh.insert(current, signature)
            signatures.append(signature)
            state_of.append(text)
            owners.append(family_idx)

    return [union_find.find(idx) for idx in range(len(family_states))]


def _prefilter_for(threshold: float, num_perm: int) -> float:
    """MinHash estimate below which a candidate pair is not worth exact comparison.

    The exact Jaccard decides cluster membership, so this only bounds wasted work
    — but a too-tight bound silently becomes the dominant false-negative source,
    which is what a fixed ``threshold - 0.15`` was: at 64 permutations the
    estimate's standard error near 0.7 is ~0.057, so that cutoff sat 2.6 sigma
    below the true value and dropped roughly 0.5% of genuine threshold pairs,
    seven orders of magnitude more than banding misses.

    Four standard errors below the threshold puts the per-pair false-negative
    rate near 3e-5.
    """
    sigma = math.sqrt(max(threshold * (1.0 - threshold), 0.0) / num_perm)
    return max(0.0, threshold - 4.0 * sigma)


def assign_splits(
    domain: str,
    clusters: Sequence[int],
    eval_ratio: float = 0.1,
    family_keys: Sequence[int] | None = None,
) -> list[str]:
    """Assign one split per family, shared by every member of a cluster.

    The split key of a cluster is the lowest ``_family_idx`` among its members,
    not the lowest position and not the cluster label. Two reasons: a family
    alone in its cluster must keep exactly the split the family-level scheme
    gives it, and this function must agree with the loader. Keying on position
    disagreed with :func:`domain_splits` on 258 of 4,476 families, which is how a
    verification table came to describe a run that never happens.

    Args:
        domain: Synthetic domain name.
        clusters: Cluster label per family, from :func:`cluster_families`.
        eval_ratio: Fraction of clusters held out for evaluation.
        family_keys: ``_family_idx`` per family, in the same order as
            ``clusters``. Defaults to the position, which is only correct when
            the indices are positional — they are not in most domains.

    Returns:
        Split per family, aligned with ``clusters``.
    """
    if family_keys is None:
        family_keys = list(range(len(clusters)))

    cluster_keys: dict[int, int] = {}
    for position, label in enumerate(clusters):
        key = family_keys[position]
        if label not in cluster_keys or key < cluster_keys[label]:
            cluster_keys[label] = key

    return [
        split_from_key(f"{domain}-{cluster_keys[label]}", eval_ratio)
        for label in clusters
    ]


def _cache_identity(
    threshold: float,
    num_perm: int,
    bands: int,
    eval_ratio: float,
    content_key: str,
) -> dict[str, Any]:
    return {
        "content_key": content_key,
        "threshold": threshold,
        "num_perm": num_perm,
        "bands": bands,
        "eval_ratio": eval_ratio,
    }


def _read_cache(
    cache_path: Path, domain: str, identity: dict[str, Any]
) -> list[str] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    entry = (payload.get("domains") or {}).get(domain)
    if not isinstance(entry, dict):
        return None
    if entry.get("identity") != identity:
        return None
    splits = entry.get("splits")
    if not isinstance(splits, list):
        return None
    return splits


def _write_cache(
    cache_path: Path, domain: str, identity: dict[str, Any], splits: list[str]
) -> None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("version") != CACHE_VERSION:
            raise ValueError
    except (OSError, ValueError):
        payload = {"version": CACHE_VERSION, "domains": {}}
    payload.setdefault("domains", {})[domain] = {
        "identity": identity,
        "splits": splits,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle)
            temp_path = Path(handle.name)
        os.replace(temp_path, cache_path)
    except OSError as exc:
        logger.warning("Could not write split cache %s: %s", cache_path, exc)


def domain_splits(
    domain: str,
    families: Sequence[dict[str, Any]],
    variants: Sequence[dict[str, Any]] | None = None,
    family_idxs: Sequence[int] | None = None,
    eval_ratio: float = 0.1,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    content_key: str = "",
    cache_path: Path | None = None,
) -> list[str]:
    """Split one domain's families so that near-duplicate states cannot leak.

    Args:
        domain: Synthetic domain name.
        families: Family records, in file order.
        variants: Variant records aligned with ``families`` by position.
        family_idxs: ``_family_idx`` per family, defaulting to the position.
            Used for the split key so that a family keeps its family-level
            split whenever it has no near-duplicate.
        eval_ratio: Fraction of clusters held out for evaluation.
        threshold: Jaccard at or above which two states are duplicates.
        num_perm: MinHash permutations.
        bands: LSH bands.
        content_key: Digest of the corpus files backing this domain, used to
            invalidate the cache. Empty disables caching.
        cache_path: Cache file to read and update.

    Returns:
        Split per family, aligned with ``families``.
    """
    if not families:
        return []

    if family_idxs is None:
        family_idxs = list(range(len(families)))
    variants = variants or []

    identity = _cache_identity(threshold, num_perm, bands, eval_ratio, content_key)
    cacheable = cache_path is not None and bool(content_key)
    if cacheable:
        cached = _read_cache(cache_path, domain, identity)
        if cached is not None and len(cached) == len(families):
            return cached

    states = [
        family_state_texts(
            family, variants[idx] if idx < len(variants) else None
        )
        for idx, family in enumerate(families)
    ]
    clusters = cluster_families(
        states,
        threshold=threshold,
        num_perm=num_perm,
        bands=bands,
        force_group=family_idxs,
    )

    # The split key is the lowest ``_family_idx`` in the cluster, not the
    # position: a family alone in its cluster then keeps exactly the split the
    # family-level scheme gives it. ``assign_splits`` owns this rule so the
    # leakage checker and the loader cannot drift apart again.
    splits = assign_splits(domain, clusters, eval_ratio, family_keys=family_idxs)

    clustered = len({label for label in clusters})
    if clustered != len(families):
        logger.info(
            "Split: %s has %d families in %d near-duplicate clusters "
            "(threshold=%.2f)",
            domain,
            len(families),
            clustered,
            threshold,
        )

    if cacheable:
        _write_cache(cache_path, domain, identity, splits)
    return splits

def corpus_fingerprint(paths: Sequence[Path]) -> str:
    """Digest the corpus files that a cached domain split depends on.

    Uses file contents rather than mtimes so the cache stays valid across
    machines and copies on a shared filesystem.

    Args:
        paths: Files whose contents determine the split.

    Returns:
        Hex digest, or ``""`` when a file cannot be read.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        try:
            digest.update(Path(path).name.encode())
            digest.update(Path(path).read_bytes())
        except OSError:
            return ""
    return digest.hexdigest()
