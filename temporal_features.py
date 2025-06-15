# -*- coding: utf-8 -*-
"""temporal_features.py

Helpers that turn tidy *daily* time‑series (one row per day) into a rich set of
lagged and rolling‑window statistical features suitable for machine‑learning
models. The module is intentionally *pandas‑only* so it works in both local and
Dask/Polars downstream replacements.

Typical usage
-------------
>>> from temporal_features import add_lag_features, add_rollups
>>> df = add_lag_features(df, cols=["tavg", "tmax"], lags=range(1, 16), group_cols=["station_id"])
>>> df = add_rollups(df, cols=["tavg", "tmax"], windows={"weekly":7, "monthly":30}, group_cols=["station_id"])

All functions are side‑effect free: they *copy* the incoming dataframe when
necessary and always return a reference to the modified frame so chaining is
possible.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "add_lag_features",
    "add_rollups",
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _validate_inputs(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Raise if any column is missing."""

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")


def _shift_grp(grp: pd.DataFrame, lag: int, col: str) -> pd.Series:
    """Internal helper for groupby shift – keeps index aligned."""

    return grp[col].shift(lag)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    *,
    cols: Iterable[str],
    lags: Iterable[int] | range,
    group_cols: Iterable[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Create *n* lagged versions of each `cols` column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a monotonic `date` column in **daily** frequency. If not
        monotonic, it will be sorted.
    cols : list[str]
        Numeric columns to shift (e.g. ``["tavg", "tmax"]``).
    lags : Iterable[int]
        Positive integers – number of *days* to shift by.
    group_cols : list[str] | None, default None
        If provided, lagging is performed within each group separately
        (e.g. per‑station). Ordering inside each group is determined by the
        **`date`** column.
    inplace : bool, default False
        Whether to mutate `df` directly or work on a copy.

    Returns
    -------
    pd.DataFrame
        Dataframe with extra columns like ``tavg_lag_1, tavg_lag_2, …``
    """

    if "date" not in df.columns:
        raise KeyError("Dataframe must include a 'date' column before lagging.")

    _validate_inputs(df, cols)

    if not inplace:
        df = df.copy()

    # Ensure deterministic order first
    sort_cols = list(group_cols) + ["date"] if group_cols else ["date"]
    df.sort_values(sort_cols, inplace=True)

    lags = sorted(set(int(l) for l in lags if l > 0))

    for col in cols:
        for lag in lags:
            new_col = f"{col}_lag_{lag}"
            if group_cols:
                df[new_col] = (
                    df.groupby(list(group_cols), sort=False)
                    .apply(lambda g: _shift_grp(g, lag, col))
                    .droplevel(level=list(range(len(group_cols))))
                )
            else:
                df[new_col] = df[col].shift(lag)

            logger.debug("Created lag column %s", new_col)

    return df


def add_rollups(
    df: pd.DataFrame,
    *,
    cols: Iterable[str],
    windows: Mapping[str, int],
    stats: Iterable[str] = ("mean", "max", "min", "std"),
    group_cols: Iterable[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add rolling‑window statistics.

    Each window entry is `{name: size}` where *size* is **days**. For example
    ``{"weekly": 7, "monthly": 30}`` will produce columns like
    ``tavg_weekly_mean, tavg_monthly_std``, etc.

    Parameters
    ----------
    df, cols, group_cols, inplace : see :func:`add_lag_features`.
    windows : Mapping[str,int]
        Names and window sizes (in days).
    stats : Iterable[str]
        Any combination of pandas‐supported aggregation funcs ("mean", "max",
        "min", "std").

    Notes
    -----
    • Windows are *trailing* (i.e. use observations up to but **not including**
      the current day): same semantics as ``df.rolling(window, closed="left")``.
    • NaNs are preserved when the window is not fully available.
    """

    if "date" not in df.columns:
        raise KeyError("Dataframe must include a 'date' column before rollups.")

    _validate_inputs(df, cols)

    if not inplace:
        df = df.copy()

    # Make sure date is index for rolling
    if not df.index.is_monotonic_increasing or df.index.name != "date":
        df = df.set_index("date", drop=False).sort_index()

    windows = {k: int(v) for k, v in windows.items() if v >= 2}

    # Build groupby object once
    gb = df.groupby(list(group_cols)) if group_cols else [((), df)]

    for col in cols:
        for win_name, win_size in windows.items():
            rolled = []
            for key, sub in gb:
                r = (
                    sub[col]
                    .rolling(window=win_size, closed="left", min_periods=win_size)
                    .agg(stats)
                )
                rolled.append(r)
            combined = pd.concat(rolled).sort_index()

            # If multiple stats, combined is a frame; make columns
            if isinstance(combined, pd.DataFrame):
                for stat in stats:
                    df[f"{col}_{win_name}_{stat}"] = combined[stat]
            else:  # single stat
                stat = stats[0]
                df[f"{col}_{win_name}_{stat}"] = combined

            logger.debug(
                "Added rollup for %s window=%s stats=%s", col, win_name, stats
            )

    # Restore original order if mutated
    df.sort_values(list(group_cols) + ["date"] if group_cols else ["date"], inplace=True)

    return df
