# -*- coding: utf-8 -*-
"""geoutils.py

Light‑weight geographic utility helpers used across the wildfire pipeline.
All functions are *dependency‑lean* (only `numpy`, `pandas`, `geopandas`,
`shapely`, and `scipy` if available) so they can run on a minimal install.

Public API
~~~~~~~~~~
* ``nearest_station(fire_df, station_df, …)`` – vectorised nearest‑neighbour
  search that appends *station_name*, *station_lat*, *station_lon*, and the
  great‑circle distance (km) to each fire/control record.
* ``build_coast_kdtree(coast_gdf)`` – helper that converts a *coastline* GDF
  to a KD‑tree of unit‑sphere coordinates for fast look‑ups.
* ``distance_to_coast(lat, lon, tree, coast_xyz)`` – haversine distance (km)
  from a single point to the nearest coastline vertex.
* ``haversine(lat1, lon1, lat2, lon2)`` – scalar / vectorised great‑circle
  distance (km) using `numpy` broadcasting.

If *scipy* is not installed, the module gracefully falls back to a pure‑Python
*n²* search but logs a warning so you know to `pip install scipy` for speed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union, Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points

try:
    from scipy.spatial import cKDTree  # type: ignore
except ImportError:  # pragma: no cover – optional dep
    cKDTree = None  # type: ignore


EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _to_unit_sphere(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/lon degrees to *unit‑sphere* Cartesian XYZ."""

    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))


# -----------------------------------------------------------------------------
# Great‑circle distance (vectorised) – can be used standalone
# -----------------------------------------------------------------------------

def haversine(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    radius: float = EARTH_RADIUS_KM,
) -> Union[float, np.ndarray]:
    """Great‑circle distance between two points or arrays of points (km)."""

    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c


# -----------------------------------------------------------------------------
# Coastline KD‑tree builder & distance query
# -----------------------------------------------------------------------------

def build_coast_kdtree(coast_gdf: gpd.GeoDataFrame) -> Tuple["cKDTree", np.ndarray]:
    """Build a KD‑tree on coastline vertices projected to unit‑sphere XYZ.

    Returns the *tree* and the *XYZ array* so distance queries can be batched.
    """

    if cKDTree is None:
        raise ImportError("scipy is required for KD‑tree operations – pip install scipy")

    # Explode multilines and collect vertices
    coords: list[tuple[float, float]] = []
    for geom in coast_gdf.geometry:
        if isinstance(geom, (LineString, MultiLineString)):
            for line in geom.geoms if isinstance(geom, MultiLineString) else [geom]:
                coords.extend(line.coords)

    coords_arr = np.array(coords)  # shape (N, 2) – lon, lat order!
    xyz = _to_unit_sphere(coords_arr[:, 1], coords_arr[:, 0])
    tree = cKDTree(xyz)
    return tree, xyz


def distance_to_coast(
    lat: float,
    lon: float,
    tree: "cKDTree",
    coast_xyz: np.ndarray,
) -> float:
    """Distance (km) from a point to the nearest coastline vertex using the KD‑tree."""

    # Convert query point to XYZ
    query_xyz = _to_unit_sphere(np.array([lat]), np.array([lon]))
    dist_chord, idx = tree.query(query_xyz, k=1, n_jobs=-1)
    # Convert *chord length on unit sphere* to central angle, then km
    central_angle = 2 * np.arcsin(np.clip(dist_chord / 2, 0, 1))
    return float(central_angle * EARTH_RADIUS_KM)


# -----------------------------------------------------------------------------
# Nearest station lookup – main public helper
# -----------------------------------------------------------------------------

def nearest_station(
    fire_df: pd.DataFrame,
    station_df: pd.DataFrame,
    fire_lat_col: str = "latitude",
    fire_lon_col: str = "longitude",
    station_lat_col: str = "lat",
    station_lon_col: str = "lon",
    station_name_col: str = "station_name",
    return_distance: bool = True,
) -> pd.DataFrame:
    """Attach closest station to every row in *fire_df*.

    Parameters
    ----------
    fire_df : DataFrame with columns *fire_lat_col* & *fire_lon_col*.
    station_df : DataFrame with columns *station_lat_col*, *station_lon_col*, and
        *station_name_col*.
    return_distance : Whether to append a *station_distance_km* column.

    Returns
    -------
    The original *fire_df* with new columns *station_name*, *station_lat*,
    *station_lon*, and optionally *station_distance_km*.
    """

    if cKDTree is None:
        logger.warning("scipy not available → falling back to O(N×M) brute force for nearest_station(); install scipy for speed.")
        return _nearest_station_bruteforce(
            fire_df,
            station_df,
            fire_lat_col,
            fire_lon_col,
            station_lat_col,
            station_lon_col,
            station_name_col,
            return_distance,
        )

    # Build KD‑tree on station unit‑sphere coordinates
    station_xyz = _to_unit_sphere(station_df[station_lat_col].values, station_df[station_lon_col].values)
    tree = cKDTree(station_xyz)

    # Query all fire points at once
    fire_xyz = _to_unit_sphere(fire_df[fire_lat_col].values, fire_df[fire_lon_col].values)
    dist_chord, idx = tree.query(fire_xyz, k=1, n_jobs=-1)

    # Map indices back to station attributes
    fire_df = fire_df.copy()
    fire_df["station_name"] = station_df.iloc[idx][station_name_col].values
    fire_df["station_lat"] = station_df.iloc[idx][station_lat_col].values
    fire_df["station_lon"] = station_df.iloc[idx][station_lon_col].values

    if return_distance:
        central_angle = 2 * np.arcsin(np.clip(dist_chord / 2, 0, 1))
        fire_df["station_distance_km"] = central_angle * EARTH_RADIUS_KM

    return fire_df


# -----------------------------------------------------------------------------
# Brute‑force fallback (no scipy)
# -----------------------------------------------------------------------------

def _nearest_station_bruteforce(
    fire_df: pd.DataFrame,
    station_df: pd.DataFrame,
    fire_lat_col: str,
    fire_lon_col: str,
    station_lat_col: str,
    station_lon_col: str,
    station_name_col: str,
    return_distance: bool,
) -> pd.DataFrame:
    """Slow O(N×M) nearest station lookup – only used if *scipy* missing."""

    fire_df = fire_df.copy()
    station_lats = station_df[station_lat_col].values
    station_lons = station_df[station_lon_col].values

    # Vectorised distance to *all* stations for each fire point → min
    fire_coords = fire_df[[fire_lat_col, fire_lon_col]].values
    dists = np.empty((fire_coords.shape[0], station_lats.size), dtype="float32")
    for j, (s_lat, s_lon) in enumerate(zip(station_lats, station_lons)):
        dists[:, j] = haversine(fire_coords[:, 0], fire_coords[:, 1], s_lat, s_lon)

    idx = dists.argmin(axis=1)
    fire_df["station_name"] = station_df.iloc[idx][station_name_col].values
    fire_df["station_lat"] = station_lats[idx]
    fire_df["station_lon"] = station_lons[idx]

    if return_distance:
        fire_df["station_distance_km"] = dists[np.arange(dists.shape[0]), idx]

    return fire_df
