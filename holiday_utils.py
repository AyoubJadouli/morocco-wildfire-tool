# -*- coding: utf-8 -*-
"""holiday_utils.py

Light‑weight wrapper around *python‑holidays* (and optionally *workalendar*) to
annotate dataframes with public‑holiday flags, weekend flags, and basic
calendar columns that are reused across the Morocco wildfire pipeline.

Only one public helper is exported: ``mark_holidays``.

Example
~~~~~~~
>>> import pandas as pd, holiday_utils as hu
>>> df = pd.DataFrame({"acq_date": pd.date_range("2023-01-01", periods=10)})
>>> df = hu.mark_holidays(df, country="MA")
>>> df.head(3)[["acq_date", "is_holiday", "is_weekend"]]
    acq_date  is_holiday  is_weekend
0 2023-01-01           1           1
1 2023-01-02           0           0
2 2023-01-03           0           0
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from functools import lru_cache
from typing import Collection, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import holidays as _pyholidays
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "holiday_utils requires the 'holidays' package – install with\n"
        "    pip install holidays[arabic]"
    ) from exc

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_holiday_set(country: str | None = "MA", years: Iterable[int] | None = None) -> set[date]:
    """Return a *set* of `date` objects that are public holidays for *country*.

    If *country* is ``None`` or an empty string, the returned set is empty.
    If the ISO code is not supported by *python‑holidays*, we fall back to
    *workalendar* when available; otherwise we log a warning and return an
    empty set.
    """

    if not country:
        return set()

    years = sorted(set(years)) if years else None

    # Prefer python‑holidays – it supports Morocco (MA) natively.
    try:
        hol_cal = _pyholidays.country_holidays(country.upper(), years=years, language="en")
        return {d for d in hol_cal}
    except (KeyError, NotImplementedError):  # country not found
        logger.warning("⚠️  Holidays for ISO '%s' not found in python‑holidays.", country)

    # Fallback to workalendar if installed.
    try:
        from workalendar.registry import registry as _wc_registry

        cal_cls = _wc_registry.get(country.upper())
        if cal_cls is None:
            raise LookupError
        cal = cal_cls()  # type: ignore[call-arg]
        years = years or range(date.today().year - 5, date.today().year + 1)
        dates: set[date] = set()
        for yr in years:
            dates.update({d for d, _ in cal.holidays(yr)})
        return dates
    except Exception:  # pragma: no cover – workalendar optional
        logger.warning(
            "⚠️  workalendar not available or country '%s' unsupported; 'is_holiday' will be 0.",
            country,
        )
        return set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mark_holidays(
    df: pd.DataFrame,
    *,
    country: str | None = "MA",
    date_col: str = "acq_date",
    weekend_days: Collection[int] = (5, 6),  # Friday & Saturday considered weekend in MA
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add calendar columns – *is_holiday*, *is_weekend*, *day_of_week*, *day_of_year*.

    Parameters
    ----------
    df
        DataFrame containing a date column.
    country
        Two‑letter ISO‑3166 code understood by *python‑holidays* (e.g. ``"MA"``).
        If ``None`` or unknown, *is_holiday* will be 0.
    date_col
        Name of the column with tz‑naive timestamps or Python ``date`` objects.
    weekend_days
        Iterable of integers 0–6 representing weekend days (``datetime.weekday``).
    prefix
        If supplied, col names become ``f"{prefix}_is_holiday"`` … (useful when
        merging multiple date domains).

    Returns
    -------
    pandas.DataFrame
        Same as *df* but with new/overwritten columns.
    """

    if date_col not in df.columns:
        raise KeyError(f"DataFrame has no column '{date_col}'.")

    # Ensure datetime64[ns]
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    years = df[date_col].dt.year.unique()
    holidays_set = _get_holiday_set(country, years)

    # Build new columns
    _pref = (prefix + "_") if prefix else ""

    df[f"{_pref}day_of_week"] = df[date_col].dt.dayofweek + 1  # 1–7 to match ISO
    df[f"{_pref}day_of_year"] = df[date_col].dt.dayofyear

    # Weekend & holiday as int32 0/1
    wkd_set = set(weekend_days)
    df[f"{_pref}is_weekend"] = df[date_col].dt.dayofweek.isin(wkd_set).astype("int8")
    if holidays_set:
        df[f"{_pref}is_holiday"] = df[date_col].dt.date.isin(holidays_set).astype("int8")
    else:
        df[f"{_pref}is_holiday"] = 0

    return df
