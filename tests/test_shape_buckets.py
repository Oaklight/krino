"""Shape-bucket helpers used by the inference latency path."""

from __future__ import annotations

from model.src.shape_buckets import (
    PREFIX_BUCKETS,
    ROW_BUCKETS,
    SUFFIX_BUCKETS,
    bucket_length,
    bucket_rows,
    left_pad,
)


def test_bucket_length_picks_smallest_fitting_bucket():
    assert bucket_length(1) == 128
    assert bucket_length(128) == 128
    assert bucket_length(129) == 256
    assert bucket_length(1000) == 1024
    assert bucket_length(1025) == 1536


def test_bucket_length_respects_limit():
    # A 1984-token head window cannot use the 2048 bucket: the option block
    # still needs its 64-token reserve.
    assert bucket_length(1025, limit=1984) == 1536
    assert bucket_length(1984, limit=1984) == 1984
    assert bucket_length(1985, limit=1984) == 1985


def test_bucket_length_returns_exact_when_nothing_fits():
    assert bucket_length(9000) == 9000
    assert bucket_length(9000, limit=8192) == 9000


def test_bucket_rows():
    assert bucket_rows(1) == 1
    assert bucket_rows(3) == 4
    assert bucket_rows(5) == 8
    assert bucket_rows(8) == 8
    assert bucket_rows(16) == 16


def test_left_pad_keeps_real_tokens_at_the_end():
    padded, pad = left_pad([4, 5, 6], 6, 0)
    assert pad == 3
    assert padded == [0, 0, 0, 4, 5, 6]


def test_left_pad_noop_when_target_matches():
    padded, pad = left_pad([4, 5], 2, 0)
    assert pad == 0
    assert padded == [4, 5]


def test_buckets_are_sorted_and_unique():
    for buckets in (PREFIX_BUCKETS, SUFFIX_BUCKETS, ROW_BUCKETS):
        assert list(buckets) == sorted(set(buckets))
