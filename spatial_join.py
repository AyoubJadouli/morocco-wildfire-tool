# -*- coding: utf-8 -*-
"""spatial_join.py

Fast, vectorised nearest‑neighbour joins between point observations
(fire/control pixels, weather stations…) and environmental grids stored as
`lat / lon / value` dataframes.

The implementation converts geographic coordinates into 3‑D Cartesian
coordinates on the unit sphere so that an ordinary Euclidean KD‑tree yields
the great‑circle (*haversine*) nearest neighbour. This is ~2× faster and
*orders of magnitude* lighter than shapely/GeoPandas spatial joins when you
only need the closest grid cell.

Public API
==========

>>> from spatial_join import nearest_grid_value
>>> df_points = pd.DataFrame({
...     "latitude": [34.02, 30.11],
...     "longitude": [-6.83, -9.21],
... })
>>> grid = pd.DataFrame({
...     "lat": [34.0, 30.0],
...     "lon": [-7.0, -9.0],
...     "NDVI": [0.52, 0.47],
... })
>>> out = nearest_grid_value(df_points, grid, value_col="NDVI")
>>> out[["latitude", "longitude", "NDVI"]]
   latitude  longitude  NDVI
0     34.02      -6.83  0.52
1     30.11      -9.21  0.47

A thin OO‑wrapper `SpatialJoiner` is also provided for repeated queries
against the same grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "to_unit_xyz",
    "build_kdtree",
    "nearest_grid_value",
    "SpatialJoiner",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert *degrees* lat / lon to 3‑D unit‑sphere Cartesian coordinates.

    Parameters
    ----------
    lat, lon : np.ndarray
        Latitude and longitude in *degrees*.

    Returns
    -------
    xyz : np.ndarray of shape (n, 3)
        X, Y, Z coordinates on the unit sphere.
    """
    rad_lat = np.deg2rad(lat)
    rad_lon = np.deg2rad(lon)
    cos_lat = np.cos(rad_lat)
    x = cos_lat * np.cos(rad_lon)
    y = cos_lat * np.sin(rad_lon)
    z = np.sin(rad_lat)
    return np.column_stack((x, y, z))


def build_kdtree(lat: np.ndarray, lon: np.ndarray, leafsize: int = 40) -> cKDTree:
    """Build a cKDTree on unit‑sphere Cartesian coords."""
    xyz = to_unit_xyz(lat, lon)
    return cKDTree(xyz, leafsize=leafsize)


# ---------------------------------------------------------------------------
# Main functional interface
# ---------------------------------------------------------------------------

def nearest_grid_value(
    points_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    value_col: str,
    new_name: Optional[str] = None,
    *,
    point_lat: str = "latitude",
    point_lon: str = "longitude",
    grid_lat: str = "lat",
    grid_lon: str = "lon",
    overwrite: bool = True,
) -> pd.DataFrame:
    """Attach *value_col* from the closest grid cell to each point row.

    Parameters
    ----------
    points_df : pd.DataFrame
        Table with point coordinates (degrees) in columns *point_lat* and
        *point_lon*.
    grid_df : pd.DataFrame
        Environmental grid with columns *grid_lat*, *grid_lon*, and
        *value_col*.
    value_col : str
        Name of the column in *grid_df* that holds the value to be joined.
    new_name : str, optional
        Column name to write in *points_df*. Defaults to *value_col*.
    point_lat, point_lon : str
        Column names for point latitude / longitude.
    grid_lat, grid_lon : str
        Column names for grid latitude / longitude.
    overwrite : bool, default True
        If *new_name* already exists, whether to overwrite it.

    Returns
    -------
    points_out : pd.DataFrame
        Copy of *points_df* with an extra column.
    """
    if value_col not in grid_df.columns:
        raise KeyError(f"{value_col!r} not in grid_df columns")

    new_name = new_name or value_col
    if not overwrite and new_name in points_df.columns:
        logger.debug("Column %s already exists and overwrite=False; returning unchanged df", new_name)
        return points_df

    # Build KDTree on grid
    tree = build_kdtree(grid_df[grid_lat].values, grid_df[grid_lon].values)

    # Query nearest neighbour for all points
    xyz_points = to_unit_xyz(points_df[point_lat].values, points_df[point_lon].values)
    dist, idx = tree.query(xyz_points, k=1, workers=-1)

    # Map values
    values = grid_df.iloc[idx][value_col].values
    out = points_df.copy()
    out[new_name] = values
    return out


# ---------------------------------------------------------------------------
# Object‑oriented wrapper for repeated queries
# ---------------------------------------------------------------------------

class SpatialJoiner:
    """Reusable KD‑tree wrapper for a single environmental grid."""

    def __init__(
        self,
        grid_df: pd.DataFrame,
        value_col: str,
        *,
        lat_col: str = "lat",
        lon_col: str = "lon",
        leafsize: int = 40,
    ) -> None:
        if value_col not in grid_df.columns:
            raise KeyError(f"{value_col!r} not found in grid_df")
        self.grid = grid_df.reset_index(drop=True)
        self.value_col = value_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self._tree = build_kdtree(grid_df[lat_col].values, grid_df[lon_col].values, leafsize)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(
        self,
        points_df: pd.DataFrame,
        *,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        new_name: Optional[str] = None,
        overwrite: bool = True,
    ) -> pd.DataFrame:
        """Return *points_df* with nearest grid value attached."""
        new_name = new_name or self.value_col
        if not overwrite and new_name in points_df.columns:
            logger.debug("Column %s already exists and overwrite=False; returning unchanged df", new_name)
            return points_df

        xyz_points = to_unit_xyz(points_df[lat_col].values, points_df[lon_col].values)
        _, idx = self._tree.query(xyz_points, k=1, workers=-1)
        values = self.grid.iloc[idx][self.value_col].values
        out = points_df.copy()
        out[new_name] = values
        return out

    # ------------------------------------------------------------------
    # Convenience dunder
    # ------------------------------------------------------------------
    def __call__(
        self,
        points_df: pd.DataFrame,
        *,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        new_name: Optional[str] = None,
        overwrite: bool = True,
    ) -> pd.DataFrame:
        """Alias for :meth:`query`."""
        return self.query(points_df, lat_col=lat_col, lon_col=lon_col, new_name=new_name, overwrite=overwrite)


# ---------------------------------------------------------------------------
# CLI for ad‑hoc sanity checks
# ---------------------------------------------------------------------------

def _example_usage() -> None:
    """Run a tiny demo when executed as a script."""
    pts = pd.DataFrame({"latitude": [33.5, 35.0], "longitude": [-7.6, -5.0]})
    grid = pd.DataFrame({
        "lat": [33.0, 35.0],
        "lon": [-8.0, -5.0],
        "NDVI": [0.41, 0.55],
    })
    print(nearest_grid_value(pts, grid, "NDVI"))


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Ad‑hoc nearest‑grid join test")
    parser.add_argument("--demo", action="store_true", help="Run a built‑in demo and exit")
    args = parser.parse_args()
    if args.demo:
        _example_usage()
        sys.exit(0)

    parser.print_help()
