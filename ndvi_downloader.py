# -*- coding: utf-8 -*-
"""ndvi_downloader.py

Download & cache MODIS MOD13A2 16‑day NDVI composites for Morocco (2000‑present).

The module offers two levels of API:

* **fetch_mod13a2(years)** – high‑level helper that iterates over a list of
  years, ensures all HDF tiles are present locally, extracts the *NDVI* layer,
  and returns a tidy `pandas.DataFrame` with columns
  `[date, lat, lon, NDVI]` (scaled to −1…1).
* **to_parquet(df, path)** – convenience saver that writes the tidy frame to
  Parquet (Snappy, float32) so it can be spatially joined later by
  `spatial_join.py`.

Implementation notes
--------------------
* **Download source** – we use NASA LP DAAC’s LAADS backend. Tiles are fetched
  anonymously (no Earthdata login) via HTTPS. For regions where anonymous
  access is disabled you can export the environment variables
  `EARTHDATA_USER` and `EARTHDATA_PASS`; the code will then switch to
  authenticated requests.
* **Tile selection** – Morocco falls entirely inside MODIS sinusoidal tiles
  *h17v05*, *h18v05* and *h18v06*. If you later need Western Sahara or the
  Canary Islands, add *h17v04*.
* **HDF parsing** – we rely on *rasterio* (which links to GDAL) to read the
  `1 km 16 days NDVI` sub‑dataset (index 0). The scaling factor 0.0001 and the
  fill value −3000 are handled transparently.

Dependencies
------------
```
pandas
numpy
rasterio
requests
python‑wget (optional)
```

Example
-------
>>> from ndvi_downloader import fetch_mod13a2, to_parquet
>>> df = fetch_mod13a2(years=[2023])
>>> to_parquet(df, Path("Data/GeoData/NDVI/morocco_ndvi_2023.parquet"))
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio import windows
from rasterio.vrt import WarpedVRT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRODUCT = "MOD13A2.061"  # Collection 6.1
TILES: List[Tuple[int, int]] = [(17, 5), (18, 5), (18, 6)]
CACHE_DIR = Path("Data/GeoData/NDVI")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Target CRS for downstream spatial joins (WGS84)
TARGET_CRS = "EPSG:4326"

LAADS_BASE = (
    "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/{product}/{year}/{doy:03d}/"
)

# MOD13A2 gives composites every **16 days**; day‑of‑year list helper
_DOYS = list(range(1, 367, 16))  # 1,17,33,…  (23 composites per year)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates_for_year(year: int) -> List[Tuple[int, date]]:
    """Return (DOY, datetime.date) tuples that exist for the MOD13A2 product."""
    days: List[Tuple[int, date]] = []
    for doy in _DOYS:
        try:
            days.append((doy, date.fromordinal(date(year, 1, 1).toordinal() + doy - 1)))
        except ValueError:
            # skip 366 in non‑leap years
            continue
    return days


def _filename(tile_h: int, tile_v: int, d: date) -> str:
    """MODIS naming convention: MOD13A2.AYYYYDDD.hHHvVV.061.NNNNNNNNNNNN.hdf"""
    return (
        f"MOD13A2.A{d.year}{d.timetuple().tm_yday:03d}."
        f"h{tile_h:02d}v{tile_v:02d}.{PRODUCT.split('.')[-1]}."
        "*.hdf"  # wildcard, version/timecode resolved via LAADS HTML listing
    )


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* (with optional Earthdata authentication)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return  # cached

    session = requests.Session()
    user = os.getenv("EARTHDATA_USER")
    password = os.getenv("EARTHDATA_PASS")
    if user and password:
        session.auth = (user, password)

    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
    shutil.move(tmp_path, dest)


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _extract_ndvi(hdf_path: Path) -> pd.DataFrame:
    """Return tidy `[lat, lon, NDVI]` frame for a single HDF tile."""

    # Sub‑dataset 0 == NDVI; add WarpedVRT to reproject on the fly.
    with rasterio.open(f"HDF4_EOS:EOS_GRID:{hdf_path}:MOD_Grid_16DAY_1km_VI:1 km 16 days NDVI") as src:
        with WarpedVRT(src, crs=TARGET_CRS, resampling=rasterio.enums.Resampling.nearest) as vrt:
            arr = vrt.read(1).astype(np.int16)
            arr = arr.astype("float32") * 1e-4  # scale factor
            mask = arr == -0.3  # fill value → NaN after scaling (*‑3000*1e‑4)
            arr[mask] = np.nan
            # Build coordinate grids
            rows, cols = np.indices(arr.shape)
            xs, ys = rasterio.transform.xy(vrt.transform, rows, cols)
            df = pd.DataFrame({"lon": xs.ravel(), "lat": ys.ravel(), "NDVI": arr.ravel()})
            return df.dropna(subset=["NDVI"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_mod13a2(years: Iterable[int]) -> pd.DataFrame:
    """Ensure all tiles for *years* exist locally, then return tidy dataframe."""

    frames: list[pd.DataFrame] = []

    for year in years:
        for doy, dt in _dates_for_year(year):
            for h, v in TILES:
                # Resolve real filename via LAADS listing (one HTML GET)
                listing_url = LAADS_BASE.format(product=PRODUCT, year=year, doy=doy)
                html = requests.get(listing_url, timeout=30).text
                pattern = _filename(h, v, dt).replace("*", "(\\d{13})")
                m = re.search(pattern, html)
                if not m:
                    continue  # tile not yet processed – skip
                fname = m.group(0)
                tile_url = f"{listing_url}{fname}"
                tile_path = CACHE_DIR / str(year) / f"h{h:02d}v{v:02d}" / fname
                _download(tile_url, tile_path)

                # Parse NDVI and append
                df_tile = _extract_ndvi(tile_path)
                df_tile["date"] = dt
                frames.append(df_tile)

    if not frames:
        raise RuntimeError("No MOD13A2 tiles could be downloaded – check connectivity.")

    df_all = pd.concat(frames, ignore_index=True)
    # Keep float32 to reduce memory
    df_all["NDVI"] = df_all["NDVI"].astype("float32")
    return df_all


def to_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:  # noqa: D401 – imperative
    """Write *df* to *path* using Snappy compression, float32 dtypes preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False, **kwargs)


# ---------------------------------------------------------------------------
# CLI helper (python ndvi_downloader.py 2023 2024 …)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch MOD13A2 NDVI for Morocco.")
    parser.add_argument("years", nargs="+", type=int, help="Years, e.g. 2022 2023")
    parser.add_argument("--out", type=Path, default=CACHE_DIR / "morocco_ndvi.parquet")
    args = parser.parse_args()

    df_ndvi = fetch_mod13a2(args.years)
    to_parquet(df_ndvi, args.out)
    print(f"✅ Saved {len(df_ndvi):,} rows to {args.out}")
