"""Group-aware train/val splitting.

A row-level split lets near-duplicate rows land on both sides: two rows that
differ only by whitespace, a paraphrase of the same passage, or membership of
one near-duplicate cluster teach the val set.  Every trainer therefore splits
whole *groups*, where an item's group is the union of three relations:

* the near-duplicate cluster from the cluster/id manifest,
* the exact normalized state (Unicode NFKC, whitespace collapsed),
* the item's existing ``group`` key (passage, family, ...).

Held-out ids are removed by the caller before this function sees them; the
splitter never re-splits them.

``KRINO_DISABLE_GROUP_SPLIT`` makes every item its own group, i.e. the old
row-level behaviour.  ``tests/conftest.py`` re-runs the group test with it set
and requires the test to fail; it is not a production option.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from model.src.feature_flags import feature_disabled

GROUP_SPLIT_SWITCH = "KRINO_DISABLE_GROUP_SPLIT"

_NON_WORD = re.compile(r"\W+")


def normalise_state(state: str) -> str:
    """The #235 exact-state key: NFC, casefold, non-word runs to a space, strip."""
    text = unicodedata.normalize("NFC", state).casefold()
    return _NON_WORD.sub(" ", text).strip()


def state_key(state: str) -> str:
    """``blake2b-128`` hex of :func:`normalise_state`, as the #235 clusters use."""
    return hashlib.blake2b(normalise_state(state).encode(), digest_size=16).hexdigest()


def load_cluster_manifest(path: Path | str) -> dict[str, str]:
    """Load a cluster manifest into a prefixed ``key -> cluster`` mapping.

    Accepted shapes:

    * a JSON object mapping row id to cluster key;
    * ``{"clusters": [[id, ...], ...]}`` (or a bare list of lists);
    * the #235 ``research/235-near-duplicate-clusters.json`` shape, whose
      ``state_to_cluster`` maps a :func:`state_key` to a cluster.

    Keys are prefixed ``id:`` or ``state:`` so one lookup function can serve
    both artifacts.  Ids and states absent from the manifest fall back to the
    other two split relations, so a partial manifest is usable.
    """
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "state_to_cluster" in payload:
        return {
            f"state:{key}": str(cluster)
            for key, cluster in payload["state_to_cluster"].items()
        }
    if isinstance(payload, dict) and "clusters" in payload:
        clusters = payload["clusters"]
    elif isinstance(payload, list):
        clusters = payload
    elif isinstance(payload, dict):
        return {f"id:{key}": str(value) for key, value in payload.items()}
    else:
        raise ValueError(f"{path}: unrecognised cluster manifest shape")

    mapping: dict[str, str] = {}
    for index, members in enumerate(clusters):
        cluster_key = f"cluster:{index}"
        for member in members:
            mapping[f"id:{member}"] = cluster_key
    return mapping


def cluster_for(
    manifest: dict[str, str] | None,
    item_id: str,
    state: str,
) -> str | None:
    """Cluster of one item, by id first and then by its state key."""
    if not manifest:
        return None
    found = manifest.get(f"id:{item_id}")
    if found is not None:
        return found
    return manifest.get(f"state:{state_key(state)}")


class _UnionFind:
    """Disjoint-set over item positions."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_group_keys(
    items: Sequence[Any],
    *,
    id_of: Callable[[Any], str],
    state_of: Callable[[Any], str],
    group_of: Callable[[Any], str | None],
    cluster_of: Callable[[Any], str | None] | None = None,
    namespace_of: Callable[[Any], str | None] | None = None,
    use_group: bool = True,
) -> tuple[list[str], int]:
    """Return ``(group key per item, number of groups)``.

    Items are unioned when they share a non-empty cluster key, normalized state
    or existing group key.  A group key is unique per group even when two items
    share an id (RACE carries several questions over one passage id), so the
    result is parallel to ``items`` and never keyed by id.

    ``namespace_of`` scopes the state and group relations (for example by
    source, or by source+family for synthetic rows) so two unrelated rows that
    happen to share a short state do not chain across sources.  Cluster ids
    stay global: a near-duplicate cluster is one unit even when the manifest
    spans sources.  ``use_group=False`` drops the row's own group key from the
    union; a coarse group value (``race-high``) otherwise chains tens of
    thousands of unrelated rows into one key.
    """
    if not items:
        return [], 0
    union = _UnionFind(len(items))
    seen: dict[tuple[str, str], int] = {}
    relations: list[tuple[str, Callable[[Any], str | None]]] = [
        ("state", lambda item: normalise_state(state_of(item))),
    ]
    if use_group:
        relations.append(("group", group_of))
    if cluster_of is not None:
        relations.append(("cluster", cluster_of))
    for index, item in enumerate(items):
        if feature_disabled(GROUP_SPLIT_SWITCH):
            continue
        namespace = ""
        if namespace_of is not None:
            namespace = str(namespace_of(item) or "")
        for name, accessor in relations:
            value = accessor(item)
            if not value:
                continue
            relation = (name, str(value))
            if namespace and name != "cluster":
                relation = (name, f"{namespace}\x1f{value}")
            first = seen.setdefault(relation, index)
            union.union(first, index)

    roots: dict[int, str] = {}
    for index, item in enumerate(items):
        root = union.find(index)
        roots.setdefault(root, f"{root}:{id_of(item)}")
    return [roots[union.find(index)] for index in range(len(items))], len(roots)


def split_items_by_group(
    items: Sequence[Any],
    val_fraction: float,
    seed: int,
    *,
    id_of: Callable[[Any], str],
    state_of: Callable[[Any], str],
    group_of: Callable[[Any], str | None],
    cluster_of: Callable[[Any], str | None] | None = None,
    stratum_of: Callable[[Any], Any] | None = None,
    namespace_of: Callable[[Any], str | None] | None = None,
    use_group: bool = True,
    force_train_clusters: Sequence[str] = (),
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Split items into train and val by whole groups.

    ``stratum_of`` (for example ``(type, K bucket)``) keeps the val mix close to
    the corpus by assigning whole groups per stratum.  A group's stratum is the
    stratum of its first item in input order; a group that spans strata is not
    broken apart.

    ``force_train_clusters`` names cluster ids whose component must stay in
    train; those groups are skipped when val candidates are drawn.  It exists
    for clusters the manifest over-merges and that cannot be split without
    leaking: a component that lands wholly in val would otherwise remove a
    source from training.

    Returns:
        ``(train, val, report)``.  Both lists keep the input order.
    """
    keys, n_groups = build_group_keys(
        items,
        id_of=id_of,
        state_of=state_of,
        group_of=group_of,
        cluster_of=cluster_of,
        namespace_of=namespace_of,
        use_group=use_group,
    )
    if not items:
        return [], [], {"items": 0, "groups": 0, "clusters": 0, "train": 0, "val": 0}

    forced_keys: set[str] = set()
    forced = {str(value) for value in force_train_clusters}
    if forced and cluster_of is not None:
        for index, item in enumerate(items):
            value = cluster_of(item)
            if value is not None and str(value) in forced:
                forced_keys.add(keys[index])

    members: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        members[key].append(index)

    stratum_of = stratum_of or (lambda item: "all")
    by_stratum: dict[Any, list[str]] = defaultdict(list)
    for group_key, indices in members.items():
        by_stratum[stratum_of(items[indices[0]])].append(group_key)

    rng = random.Random(seed)
    val_groups: set[str] = set()
    for stratum in sorted(by_stratum, key=str):
        group_keys = by_stratum[stratum]
        rng.shuffle(group_keys)
        total = sum(len(members[key]) for key in group_keys)
        if val_fraction <= 0.0 or total <= 1:
            target = 0
        else:
            target = min(total, max(1, int(round(total * val_fraction))))
        taken = 0
        for group_key in group_keys:
            if taken >= target:
                break
            if group_key in forced_keys:
                continue
            val_groups.add(group_key)
            taken += len(members[group_key])

    train = [item for index, item in enumerate(items) if keys[index] not in val_groups]
    val = [item for index, item in enumerate(items) if keys[index] in val_groups]
    clusters = set()
    if cluster_of is not None:
        clusters = {value for item in items if (value := cluster_of(item))}
    report = {
        "items": len(items),
        "groups": n_groups,
        "clusters": len(clusters),
        "train": len(train),
        "val": len(val),
        "val_fraction": val_fraction,
        "groups_with_over_one_item": sum(
            1 for group in members.values() if len(group) > 1
        ),
    }
    return train, val, report


def unique_ids(items: Iterable[Any], id_of: Callable[[Any], str]) -> int:
    """Count distinct ids; a diagnostic, not a split input."""
    return len({id_of(item) for item in items})
