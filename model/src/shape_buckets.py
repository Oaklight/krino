"""Fixed-shape buckets for inference latency.

CUDA loads a kernel variant the first time a new tensor shape reaches an
operator.  At 4B parameters that first call costs 0.4-1.3 s on H100, and an
evaluation that walks 231 items with 231 distinct sequence lengths pays it
231 times.  Padding every request into a handful of bucket shapes pays it
once per bucket.

Padding is always masked out (attention mask 0, position ids held at 0), so
the real tokens see exactly the same computation as an unpadded sequence.
Bucketing is therefore a latency change, not a modelling change.
"""

from __future__ import annotations

#: Prefix lengths.  Coarse on purpose: each bucket is one cold kernel load.
#: 1536/1792/1984 exist so a 2048-token head window has buckets below its
#: 64-token option reserve; without them every long context fell back to an
#: exact shape and paid the cold load again.
PREFIX_BUCKETS = (
    128, 256, 512, 1024, 1536, 1792, 1984, 2048, 3072, 4096, 6144, 8192
)

#: Option/suffix widths.  64 covers a normal option description plus the
#: reranker suffix; 96 covers an option plus the marker token.
SUFFIX_BUCKETS = (16, 32, 64, 96, 128, 256, 512)

#: Batch widths.  A group is padded to its bucket so the GEMM shape does not
#: vary with K.
ROW_BUCKETS = (1, 2, 4, 8, 16)


def bucket_length(
    length: int,
    buckets: tuple[int, ...] = PREFIX_BUCKETS,
    limit: int | None = None,
) -> int:
    """Smallest bucket at least ``length``, optionally capped by ``limit``.

    Returns ``length`` unchanged when no bucket fits, so a caller never has
    to special-case a sequence longer than the coarsest bucket.
    """
    for bucket in buckets:
        if bucket >= length and (limit is None or bucket <= limit):
            return bucket
    return length


def bucket_rows(rows: int) -> int:
    """Smallest batch bucket at least ``rows``."""
    return bucket_length(rows, ROW_BUCKETS)


def left_pad(
    ids: list[int],
    target: int,
    pad_id: int,
) -> tuple[list[int], int]:
    """Left-pad ``ids`` to ``target``; returns the padded ids and pad count.

    Left padding keeps the real tokens contiguous at the end, so the last
    position is the sequence's last real token whether or not padding is
    present.
    """
    pad = max(0, target - len(ids))
    return [pad_id] * pad + ids, pad
