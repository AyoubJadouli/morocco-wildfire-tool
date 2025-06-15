# -*- coding: utf-8 -*-
"""augment_data.py

Spatial augmentation utilities that *densify* a point wildfire dataset by
scattering synthetic observations in a small radius around each fire pixel.
This helps mitigate extreme class imbalance—particularly when pairing each
fire point with many *no‑fire* controls—while preserving local spatial context.

Public API
~~~~~~~~~~
>>> import pandas as pd
>>> from augment_data import jitter_fire_points
>>> df_fire = pd.read_parquet("all_firms_fire_points.parquet")
>>> df_aug = jitter_fire_points(df_fire, n_per_fire=100, radius_m=300)
>>> combined = pd.concat([df_fire, df_aug])

Design notes
------------
* Uses a *uniform‑area* sampling: radius ∝ sqrt(U) so points are equally likely
  to fall anywhere inside the circle.
* Great‑circle adjustment done via local metres‑per‑degree factors; accurate
  enough for ≤2 km radii.
* Adds metadata columns: ``is_augmented`` (bool), ``parent_fire_id`` (index of
  source row), and ``jitter_m`` (actual random displacement in metres).
* Returns *only* the jittered rows—caller decides whether to concatenate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Sequence

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean radius
_METERS_PER_DEG_LAT = 111_132.954  # approx at 30°–40°N


def _meters_per_degree_lon(lat_rad: np.ndarray) -> np.ndarray:
    """Vectorised metres‑per‑degree‑longitude given latitude in **radians**."""
    return (
        np.cos(lat_rad) * 2 * np.pi * EARTH_RADIUS_M / 360.0
    )  # circumference at that latitude / 360°


def jitter_fire_points(
    fire_df: pd.DataFrame,
    *,
    n_per_fire: int = 100,
    radius_m: Union[int, float] = 300.0,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    copy_metadata: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scatter *n_per_fire* synthetic points in a circle of ``radius_m`` metres.

    Parameters
    ----------
    fire_df : pd.DataFrame
        Source fire points with at least *latitude* and *longitude* columns.
    n_per_fire : int, default 100
        Number of jittered points to create for **each** original fire pixel.
    radius_m : float, default 300.0
        Maximum radial distance in metres for jittering.
    lat_col, lon_col : str
        Column names for latitude / longitude (° WGS‑84).
    copy_metadata : Sequence[str] | None
        Additional columns to copy verbatim onto the augmented points. If
        None, copies **all** columns *except* geometry and the coordinate pair.

    Returns
    -------
    pd.DataFrame
        New dataframe *only* with the synthetic points. It contains *all*
        requested metadata columns + ``latitude``, ``longitude``, ``acq_date``
        (if present), ``parent_fire_id``, ``jitter_m``, and ``is_augmented``.
    """
    if n_per_fire <= 0:
        raise ValueError("n_per_fire must be positive")

    fire_df = fire_df.reset_index(drop=False).rename(columns={"index": "_src_idx"})
    n_src = len(fire_df)
    if n_src == 0:
        raise ValueError("fire_df is empty – nothing to augment")

    # Decide which columns to copy.
    if copy_metadata is None:
        copy_metadata = [c for c in fire_df.columns if c not in {lat_col, lon_col}]

    # Prepare broadcast arrays.
    repeat_factor = n_per_fire
    base_lat = np.repeat(fire_df[lat_col].to_numpy(), repeat_factor)
    base_lon = np.repeat(fire_df[lon_col].to_numpy(), repeat_factor)
    base_idx = np.repeat(fire_df["_src_idx"].to_numpy(), repeat_factor)

    # Random radii (uniform‑area) and angles.
    rng = np.random.default_rng()
    theta = rng.uniform(0, 2 * np.pi, size=base_lat.size)
    # r^2 ~ U -> r = sqrt(U) * radius_m
    r = np.sqrt(rng.uniform(0, 1, size=base_lat.size)) * radius_m

    # Convert metres to degrees.
    lat_rad = np.deg2rad(base_lat)
    dlat_deg = (r * np.cos(theta)) / _METERS_PER_DEG_LAT
    meters_per_deg_lon = _meters_per_degree_lon(lat_rad)
    dlon_deg = (r * np.sin(theta)) / meters_per_deg_lon

    jitter_lat = base_lat + dlat_deg
    jitter_lon = base_lon + dlon_deg

    out = pd.DataFrame({
        lat_col: jitter_lat.astype(np.float32),
        lon_col: jitter_lon.astype(np.float32),
        "parent_fire_id": base_idx.astype(np.int32),
        "jitter_m": r.astype(np.float32),
        "is_augmented": True,
    })

    # Copy requested metadata columns.
    for col in copy_metadata:
        if col in fire_df.columns:
            out[col] = np.repeat(fire_df[col].to_numpy(), repeat_factor)

    return out


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description="Scatter synthetic points around each fire pixel (augmentation)."
    )
    parser.add_argument("input", type=pathlib.Path, help="Parquet/CSV of fire points")
    parser.add_argument("output", type=pathlib.Path, help="Where to save augmented Parquet")
    parser.add_argument("--n", type=int, default=100, help="Points per fire pixel")
    parser.add_argument("--radius", type=float, default=300.0, help="Radius in metres")
    args = parser.parse_args()

    if args.input.suffix == ".parquet":
        df_fire = pd.read_parquet(args.input)
    else:
        df_fire = pd.read_csv(args.input)

    df_aug = jitter_fire_points(df_fire, n_per_fire=args.n, radius_m=args.radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_parquet(args.output, index=False)
    print(f"Augmented data written to {args.output} (rows: {len(df_aug)})")