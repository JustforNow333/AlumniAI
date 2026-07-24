"""Bounded, deterministic value parsing for typed row predicates."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta, timezone
from numbers import Real
from typing import Any

import pandas as pd


MAX_TYPED_VALUE_LENGTH = 120
_NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_WHITESPACE = re.compile(r"\s+")


def parse_numeric_value(value: Any) -> int | float | None:
    """Parse a finite scalar number without interpreting arbitrary text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() else number

    text = str(value).strip()
    if not text or len(text) > MAX_TYPED_VALUE_LENGTH:
        return None
    if text.casefold() in {"n/a", "na", "nan", "none", "null", "not available"}:
        return None

    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()
    text = text.replace("$", "").replace(",", "").strip()
    if not _NUMERIC_PATTERN.fullmatch(text):
        return None
    number = float(text)
    if negative_parentheses:
        number = -number
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def parse_date_value(value: Any) -> date | None:
    """Parse a date and normalize it to a timezone-independent calendar date."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return _timestamp_date(value)
    if isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
        return _timestamp_date(timestamp)
    if isinstance(value, date):
        return value

    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number.is_integer() and 1900 <= number <= 2200:
            return date(int(number), 1, 1)
        # Excel's default 1900 date system uses 1899-12-30 as the practical
        # origin after accounting for its historic leap-year bug.
        if 1 <= number <= 2_958_465:
            try:
                return date(1899, 12, 30) + timedelta(days=number)
            except (OverflowError, ValueError):
                return None
        return None

    text = str(value).strip()
    if not text or len(text) > MAX_TYPED_VALUE_LENGTH:
        return None
    if text.casefold() in {"n/a", "na", "nan", "none", "null", "nat", "not available"}:
        return None

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%B %Y",
        "%b %Y",
        "%Y",
    )
    for format_string in formats:
        try:
            return datetime.strptime(text, format_string).date()
        except ValueError:
            continue

    try:
        parsed = pd.to_datetime(text, errors="raise")
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        return None
    timestamp = pd.Timestamp(parsed)
    if pd.isna(timestamp):
        return None
    return _timestamp_date(timestamp)


def normalize_membership_value(value: Any) -> str | None:
    if value is None:
        return None
    text = _WHITESPACE.sub(" ", str(value).strip())
    if not text or text.casefold() in {"nan", "none", "null", "nat", "n/a", "na"}:
        return None
    return text.casefold()


def normalize_string_value(value: Any) -> str | None:
    if value is None:
        return None
    text = _WHITESPACE.sub(" ", str(value).strip())
    return text if text else None


def relative_date_boundary(days: int, *, now: Any = None) -> date:
    if not isinstance(days, int) or isinstance(days, bool) or days < 0 or days > 36_600:
        raise ValueError("Relative date days must be an integer between 0 and 36600.")
    current = _coerce_current_date(now)
    return current - timedelta(days=days)


def _coerce_current_date(now: Any) -> date:
    if now is None:
        return datetime.now(timezone.utc).date()
    parsed = parse_date_value(now)
    if parsed is None:
        raise ValueError("The injected current time must be a valid date or datetime.")
    return parsed


def _timestamp_date(value: pd.Timestamp) -> date:
    if value.tzinfo is not None:
        value = value.tz_convert("UTC")
    return value.date()
