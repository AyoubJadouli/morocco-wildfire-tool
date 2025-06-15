# -*- coding: utf-8 -*-
"""control_sampler.py

Generate *no‑fire* control points uniformly at random inside Morocco’s land
polygon for a given date interval.

These points balance the wildfire dataset to roughly 50 / 50 fire vs no‑fire
and provide negative samples for classification.

Public helpers
~~~~~~~~~~~~~~
* ``sample_no_fire(n, date_min, date_max, polygon, seed=None)`` → pd.DataFrame
* ``sample_no_fire_country(n, date_min, date_max, country_iso='MAR', ...)``
  (loads Natural‑Earth shapefiles automatically)

A thin CLI wrapper allows running from the shell:

```bash
python control_sampler.py 100000 2010-01-01 2024-12-31 \
       --output Data/WildFireHist/FIRMS/no_fire_2010_2024.parquet
```

Dependencies
------------
* numpy
* pandas
* geopandas
* shapely
* click (optional – only for CLI if installed)

Note
----
The algorithm uses *rejection sampling* inside the bounding box of the target
polygon. For Morocco (~710 000 km²) the acceptance rate is ~40 %, so generating
100 000 points typically takes <2 seconds on a laptop.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def _uniform_dates(n: int, start: datetime, end: datetime) -> np.ndarray:
    """Return *n* random datetime64[ns] uniformly between *start* and *end*."""
    delta = (end - start).total_seconds()
    seconds = np.random.rand(n) * delta
    return np.array([start + timedelta(seconds=float(s)) for s in seconds], dtype="datetime64[ns]")


def _random_points_in_bbox(n: int, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Return *n* random lon/lat pairs inside bounding *bounds* (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds
    xs = np.random.rand(n) * (maxx - minx) + minx
    ys = np.random.rand(n) * (maxy - miny) + miny
    return np.column_stack([xs, ys])


def _ensure_polygon(poly) -> Polygon | MultiPolygon:
    if isinstance(poly, (Polygon, MultiPolygon)):
        return poly
    raise TypeError("polygon must be shapely Polygon or MultiPolygon")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def sample_no_fire(
    n: int,
    date_min: datetime | str,
    date_max: datetime | str,
    polygon: Polygon | MultiPolygon,
    seed: Optional[int] = None,
    chunk: int = 20000,
) -> pd.DataFrame:
    """Return *n* uniformly distributed no‑fire control points inside *polygon*.

    Parameters
    ----------
    n : int
        Number of points to return.
    date_min, date_max : datetime or str (YYYY‑MM‑DD)
        Inclusive date range from which *acq_date* will be sampled.
    polygon : shapely Polygon or MultiPolygon
        Target geographic mask (in WGS‑84 CRS).
    seed : int, optional
        RNG seed for reproducibility.
    chunk : int, default 20 000
        Internal batch size for rejection sampling (speed‑vs‑memory trade‑off).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    poly = _ensure_polygon(polygon)
    bounds = poly.bounds

    points: list[Point] = []
    while len(points) < n:
        # Draw a batch of candidate points
        coords = _random_points_in_bbox(chunk, bounds)
        cand_points = [Point(lon, lat) for lon, lat in coords]
        # Filter those inside polygon
        inside = [pt for pt in cand_points if poly.contains(pt)]
        points.extend(inside)
        logger.debug("generated %d / %d", len(points), n)

    points = points[:n]
    lats = np.fromiter((p.y for p in points), dtype=float, count=n)
    lons = np.fromiter((p.x for p in points), dtype=float, count=n)

    # Dates
    if isinstance(date_min, str):
        date_min = datetime.fromisoformat(date_min)
    if isinstance(date_max, str):
        date_max = datetime.fromisoformat(date_max)
    dates = _uniform_dates(n, date_min, date_max)

    df = pd.DataFrame(
        {
            "acq_date": dates,
            "latitude": lats.astype(np.float32),
            "longitude": lons.astype(np.float32),
            "is_fire": np.zeros(n, dtype=np.uint8),
        }
    )
    return df


def sample_no_fire_country(
    n: int,
    date_min: datetime | str,
    date_max: datetime | str,
    country_iso: str = "MAR",
    shapefile_path: Optional[Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper that loads a Natural‑Earth country polygon and delegates to
    *sample_no_fire()*.

    Parameters
    ----------
    shapefile_path : Path, optional
        If given, overrides auto‑discovery of the NE shapefile.
    """

    if shapefile_path is None:
        from config import DIRS  # late import to avoid circulars

        shapefile_path = DIRS["shapefiles"] / "ne_10m_admin_0_countries_mar.shp"

    gdf = gpd.read_file(shapefile_path)
    poly = gdf.iloc[0].geometry

    return sample_no_fire(n, date_min, date_max, poly, **kwargs)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cli():  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser("Generate no‑fire control points for Morocco")
    parser.add_argument("n", type=int, help="Number of points to generate")
    parser.add_argument("date_min", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("date_max", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output Parquet path")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")

    args = parser.parse_args()

    df = sample_no_fire_country(
        args.n,
        args.date_min,
        args.date_max,
        seed=args.seed,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("✅ Saved %d points to %s", len(df), out_path)
    else:
        print(df.head())


if __name__ == "__main__":  # pragma: no cover
    _cli()
