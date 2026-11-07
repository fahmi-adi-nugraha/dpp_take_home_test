"""Tests for Yelp text preprocessing."""

from __future__ import annotations

import polars as pl
import pytest
from yelp_sentiment.preprocess import (
    TextDecodeError,
    clean_text,
    clean_text_series,
    collapse_whitespace,
    decode_text,
    normalize_for_matching,
)


def test_decode_text_handles_json_escapes() -> None:
    raw = r"Line one\nLine two \"quoted\" \u00e9"

    assert decode_text(raw) == 'Line one\nLine two "quoted" é'


def test_clean_text_collapses_whitespace_and_preserves_tokens() -> None:
    raw = r"Not   GREAT!!\n\nWe\tloved it.  "

    assert clean_text(raw) == "Not GREAT!! We loved it."


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("  a \n b\t c  ") == "a b c"


def test_normalize_for_matching() -> None:
    assert normalize_for_matching("Hello,   WORLD!!") == "hello world"


def test_clean_text_series() -> None:
    series = pl.Series("text", [r"one\ntwo", r"three   four"])

    cleaned = clean_text_series(series)

    assert cleaned.to_list() == ["one two", "three four"]


def test_malformed_escape_raises() -> None:
    with pytest.raises(TextDecodeError):
        decode_text(r"bad \q escape")


def test_non_string_raises() -> None:
    with pytest.raises(TextDecodeError):
        decode_text(None)  # type: ignore[arg-type]
