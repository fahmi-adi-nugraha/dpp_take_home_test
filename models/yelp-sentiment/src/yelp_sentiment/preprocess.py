"""Text preprocessing helpers for Yelp reviews."""

from __future__ import annotations

import json
import re

import polars as pl

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


class TextDecodeError(ValueError):
    """Raised when a raw review is not a valid escaped JSON string."""


def decode_text(value: str) -> str:
    """Decode literal JSON-style escapes in a raw review."""
    if not isinstance(value, str):
        raise TextDecodeError(f"Expected a string, received {type(value).__name__}.")
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError as exc:
        raise TextDecodeError(f"Could not decode review text: {value[:80]!r}") from exc


def collapse_whitespace(value: str) -> str:
    """Collapse all whitespace runs to one space and strip the result."""
    return _WHITESPACE_RE.sub(" ", value).strip()


def clean_text(value: str) -> str:
    """Decode escapes and collapse whitespace while preserving the review."""
    return collapse_whitespace(decode_text(value))


def clean_text_series(series: pl.Series) -> pl.Series:
    """Apply :func:`clean_text` to a Polars string series."""
    return series.map_elements(clean_text, return_dtype=pl.String)


def normalize_for_matching(value: str) -> str:
    """Normalize text for near-duplicate diagnostics without changing model input."""
    lowered = value.lower()
    normalized = _NON_ALNUM_RE.sub(" ", lowered)
    return collapse_whitespace(normalized)
