# # -*- coding: utf-8 -*-
# """soil_moisture_downloader.py

# Download and process LPRM AMSR‑2 soil‑moisture grids for Morocco.

# * Product: **LPRM_AMSR2_DS_A_SOILM3** (V6) – daily, 0.25° × 0.25° global grids
#   released by the Department of Hydrology and Geo‑Environmental Sciences, VU
#   University Amsterdam, and hosted at NASA Goddard (GES‑DISC).
# * Format: NetCDF‑4 with three relevant bands: `Soil_Moisture`, `Quality`,
#   `Surface_Flag`.
# * Temporal coverage: 2012‑07‑02 → near‑real‑time (≈ 3–4‑day latency).

# The module exposes two high‑level helpers:

# ```python
# >>> from soil_moisture_downloader import fetch_lprm_amsr2, to_parquet
# >>> ds = fetch_lprm_amsr2(years=range(2018, 2023))
# >>> to_parquet(ds, Path("Data/GeoData/SoilMoisture/soil_moisture.parquet"))
# ```

# The returned *xarray.Dataset* contains the daily **SM** values for Morocco
# clipped to the country bounding box (−13.5…‑1.0 lon, 20…36 lat).

# Dependencies
# ------------
# * `xarray`, `rasterio`, `netCDF4`, `requests`, `tqdm`, `pandas`, `pyproj`.

# Environment
# -----------
# Set `NASA_GESDISC_TOKEN` in `~/.netrc` **or** as an environment variable if the
# data portal requires authentication (occasionally needed for bulk download).

# CLI usage
# ---------
# ```
# python soil_moisture_downloader.py 2020 2021 2022
# ```
# This will fetch 2020‑2022 data and write/update the cached parquet.
# """

# from __future__ import annotations

# import calendar
# import os
# import shutil
# from datetime import date, datetime, timedelta
# from pathlib import Path
# from typing import Iterable, List, Literal, Optional

# import pandas as pd
# import requests
# import xarray as xr
# from tqdm import tqdm

# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------

# # Root cache dir – mirror NDVI downloader pattern
# ROOT_DIR = Path("Data/GeoData/SoilMoisture")
# ROOT_DIR.mkdir(parents=True, exist_ok=True)

# # Morocco bounding box (approx.)
# LON_MIN, LON_MAX = -13.5, -1.0
# LAT_MIN, LAT_MAX = 20.0, 36.0

# # Base URL pattern (GES DISC HTTPS endpoint)
# BASE_URL = (
#     "https://opendap.cr.usgs.gov/opendap/hyrax/GES_DISC/LPRM/SM_AP/"
#     "{year}/{doy:03d}/LPRM_AMSR2_DS_A_SOILM3_{year}{doy:03d}.nc4"
# )

# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------

# def _doys_in_year(year: int) -> List[int]:
#     """Return list of day‑of‑year indices accounting for leap years."""
#     return list(range(1, 367 if calendar.isleap(year) else 366))


# def _download_nc(year: int, doy: int, out_dir: Path) -> Path:
#     """Download a single NetCDF file if not cached. Return local path."""
#     out_dir.mkdir(parents=True, exist_ok=True)
#     fname = out_dir / f"soil_{year}_{doy:03d}.nc4"
#     if fname.exists():
#         return fname

#     url = BASE_URL.format(year=year, doy=doy)
#     with requests.get(url, stream=True, timeout=60) as r:
#         r.raise_for_status()
#         with open(fname, "wb") as f:
#             shutil.copyfileobj(r.raw, f)
#     return fname


# def _subset_to_bbox(ds: xr.Dataset) -> xr.Dataset:
#     """Clip global grid to Morocco bbox."""
#     return ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MAX, LAT_MIN))


# def _open_and_clean(nc_path: Path) -> xr.Dataset:
#     """Open NetCDF, subset, rename variables, add `time` coordinate."""
#     ds = xr.open_dataset(nc_path, mask_and_scale=True).load()
#     ds = _subset_to_bbox(ds)

#     # Keep only the SM variable and rename for clarity
#     if "Soil_Moisture" not in ds:
#         raise KeyError(f"Soil_Moisture var missing in {nc_path}")

#     ds = ds[["Soil_Moisture"]].rename({"Soil_Moisture": "SoilMoisture"})

#     # Attach the timestamp from filename (00:00 UTC)
#     y, doy = map(int, nc_path.stem.split("_")[1:3])
#     timestamp = datetime.fromordinal(date(y, 1, 1).toordinal() + doy - 1)
#     ds = ds.assign_coords(time=("time", [timestamp]))

#     # Convert 0.25° grid to float32 to save RAM
#     ds["SoilMoisture"] = ds["SoilMoisture"].astype("float32")
#     return ds

# # -----------------------------------------------------------------------------
# # Public API
# # -----------------------------------------------------------------------------

# def fetch_lprm_amsr2(
#     years: Iterable[int], *,
#     force_download: bool = False,
#     show_progress: bool = True,
# ) -> xr.Dataset:
#     """Fetch daily LPRM‑AMSR2 soil‑moisture for the given years.

#     Parameters
#     ----------
#     years : Iterable[int]
#         Years to download (e.g. `range(2018, 2024)`).
#     force_download : bool, default False
#         If *True* re‑download files even if they exist in cache.
#     show_progress : bool, default True
#         Show tqdm progress bars.

#     Returns
#     -------
#     xarray.Dataset
#         Concatenated daily dataset (`time`, `lat`, `lon`).
#     """
#     cache_dir = ROOT_DIR / "_netcdf"
#     if force_download and cache_dir.exists():
#         shutil.rmtree(cache_dir)
#     cache_dir.mkdir(parents=True, exist_ok=True)

#     daily_dsets = []
#     outer_iter = tqdm(years, desc="Years", disable=not show_progress)
#     for y in outer_iter:
#         inner_iter = _doys_in_year(y)
#         if show_progress:
#             inner_iter = tqdm(inner_iter, leave=False, desc=f"{y}")
#         for doy in inner_iter:
#             try:
#                 nc_file = _download_nc(y, doy, cache_dir)
#             except requests.HTTPError as e:
#                 # Many days are missing before launch date; skip 404s quietly
#                 if e.response.status_code == 404:
#                     continue
#                 raise
#             ds_day = _open_and_clean(nc_file)
#             daily_dsets.append(ds_day)

#     if not daily_dsets:
#         raise RuntimeError("No soil‑moisture files were downloaded!")

#     # Merge along the time dimension
#     soil_ds = xr.concat(daily_dsets, dim="time")
#     soil_ds = soil_ds.sortby("time")
#     return soil_ds


# def to_parquet(ds: xr.Dataset, out_path: Path) -> Path:
#     """Serialise the DS to Parquet (tidy format) for fast point lookup."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     # xarray → DataFrame with MultiIndex → flat tidy df
#     df = ds["SoilMoisture"].to_dataframe().reset_index()
#     df.to_parquet(out_path, index=False)
#     return out_path

# # -----------------------------------------------------------------------------
# # CLI for convenience
# # -----------------------------------------------------------------------------

# if __name__ == "__main__":
#     import sys
#     from pathlib import Path

#     if len(sys.argv) < 2:
#         print("Usage: python soil_moisture_downloader.py <year> [<year> …]")
#         sys.exit(1)

#     yrs = [int(y) for y in sys.argv[1:]]
#     ds = fetch_lprm_amsr2(yrs)
#     out_fp = ROOT_DIR / "soil_moisture.parquet"
#     to_parquet(ds, out_fp)
#     print(f"✅ Saved parquet → {out_fp.relative_to(Path.cwd())}")
# -*- coding: utf-8 -*-
"""soil_moisture_downloader.py

Download and process LPRM AMSR-2 soil-moisture grids for Morocco.

* Product: **LPRM_AMSR2_DS_A_SOILM3** (V6) – daily, 0.25° × 0.25° global grids
  released by the Department of Hydrology and Geo-Environmental Sciences, VU
  University Amsterdam, and hosted at NASA Goddard (GES-DISC).
* Format: NetCDF-4 with three relevant bands: `Soil_Moisture`, `Quality`,
  `Surface_Flag`.
* Temporal coverage: 2012-07-02 → near-real-time (≈ 3–4-day latency).

This module exposes two top-level helpers:
    fetch_lprm_amsr2(years=…) → xarray.Dataset
    to_parquet(dataset, out_path) → Path

and, for backward compatibility with pipeline.py, a class wrapper:

    class SoilMoistureDownloader:
        .fetch(years)   → calls fetch_lprm_amsr2(years)
        .to_parquet(ds, out_path) → calls to_parquet(ds, out_path)

Environment
-----------
Set `NASA_GESDISC_TOKEN` in `~/.netrc` **or** as an environment variable if the
data portal requires authentication (occasionally needed for bulk download).

CLI usage
---------
    python soil_moisture_downloader.py 2020 2021 2022
"""
from __future__ import annotations

import calendar
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------

# Root cache dir – mirror NDVI downloader pattern
ROOT_DIR = Path("Data/GeoData/SoilMoisture")
ROOT_DIR.mkdir(parents=True, exist_ok=True)

# Morocco bounding box (approx.)
LON_MIN, LON_MAX = -13.5, -1.0
LAT_MIN, LAT_MAX = 20.0, 36.0

# Base URL pattern (GES DISC HTTPS endpoint)
BASE_URL = (
    "https://opendap.cr.usgs.gov/opendap/hyrax/GES_DISC/LPRM/SM_AP/"
    "{year}/{doy:03d}/LPRM_AMSR2_DS_A_SOILM3_{year}{doy:03d}.nc4"
)

# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------

def _doys_in_year(year: int) -> list[int]:
    """Return list of day-of-year indices accounting for leap years."""
    return list(range(1, 367 if calendar.isleap(year) else 366))


def _download_nc(year: int, doy: int, out_dir: Path) -> Path:
    """Download a single NetCDF file if not cached. Returns local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"soil_{year}_{doy:03d}.nc4"
    if fname.exists():
        return fname

    url = BASE_URL.format(year=year, doy=doy)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return fname


def _subset_to_bbox(ds: xr.Dataset) -> xr.Dataset:
    """Clip global grid to Morocco bounding box."""
    return ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MAX, LAT_MIN))


def _open_and_clean(nc_path: Path) -> xr.Dataset:
    """Open NetCDF, subset, keep Soil_Moisture band, rename & add `time`."""
    ds = xr.open_dataset(nc_path, mask_and_scale=True).load()
    ds = _subset_to_bbox(ds)

    # Keep only the SM variable and rename for clarity
    if "Soil_Moisture" not in ds:
        raise KeyError(f"Soil_Moisture var missing in {nc_path}")

    ds = ds[["Soil_Moisture"]].rename({"Soil_Moisture": "SoilMoisture"})

    # Attach the timestamp from filename (00:00 UTC)
    y, doy = map(int, nc_path.stem.split("_")[1:3])
    timestamp = datetime.fromordinal(date(y, 1, 1).toordinal() + doy - 1)
    ds = ds.assign_coords(time=("time", [timestamp]))

    # Convert to float32 to save RAM
    ds["SoilMoisture"] = ds["SoilMoisture"].astype("float32")
    return ds


# -------------------------------------------------------------------------------
# Public Functions
# -------------------------------------------------------------------------------

def fetch_lprm_amsr2(
    years: Iterable[int],
    *,
    force_download: bool = False,
    show_progress: bool = True,
) -> xr.Dataset:
    """Fetch daily LPRM-AMSR2 soil-moisture for the given years.

    Parameters
    ----------
    years : Iterable[int]
        Years to download (e.g. `range(2018, 2024)`).
    force_download : bool, default False
        If True, re-download files even if they exist in cache.
    show_progress : bool, default True
        Show tqdm progress bars.

    Returns
    -------
    xarray.Dataset
        Concatenated daily dataset (`time`, `lat`, `lon`).
    """
    cache_dir = ROOT_DIR / "_netcdf"
    if force_download and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    daily_dsets: list[xr.Dataset] = []
    outer_iter = tqdm(years, desc="Years", disable=not show_progress)
    for y in outer_iter:
        inner_days = _doys_in_year(y)
        if show_progress:
            inner_days = tqdm(inner_days, leave=False, desc=f"{y}")
        for doy in inner_days:
            try:
                nc_file = _download_nc(y, doy, cache_dir)
            except requests.HTTPError as e:
                # Many days missing pre-launch; skip 404 quietly
                if e.response.status_code == 404:
                    continue
                raise
            ds_day = _open_and_clean(nc_file)
            daily_dsets.append(ds_day)

    if not daily_dsets:
        raise RuntimeError("No soil-moisture files were downloaded!")

    # Merge along the time dimension
    soil_ds = xr.concat(daily_dsets, dim="time")
    soil_ds = soil_ds.sortby("time")
    return soil_ds


def to_parquet(ds: xr.Dataset, out_path: Path) -> Path:
    """Serialise the Dataset to Parquet (tidy format) for fast lookup."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert xarray → DataFrame with MultiIndex → flat tidy DataFrame
    df = ds["SoilMoisture"].to_dataframe().reset_index()
    df.to_parquet(out_path, index=False)
    return out_path


# -------------------------------------------------------------------------------
# Backwards-compatibility class
# -------------------------------------------------------------------------------

class SoilMoistureDownloader:
    """Minimal wrapper so that `pipeline.py` can still do:
         sm_dl = SoilMoistureDownloader()
         sm_df = sm_dl.fetch(...)
         sm_dl.to_parquet(sm_df, out_path)
    """

    def __init__(self) -> None:
        # no per-instance state required
        pass

    def fetch(
        self,
        years: Iterable[int],
        *,
        force_download: bool = False,
        show_progress: bool = True,
    ) -> xr.Dataset:
        """Fetch the Xarray Dataset for all requested years."""
        return fetch_lprm_amsr2(years, force_download=force_download, show_progress=show_progress)

    def to_parquet(self, ds: xr.Dataset, out_path: Path) -> Path:
        """Write the Dataset to Parquet in tidy (time,lat,lon) format."""
        return to_parquet(ds, out_path)


# -------------------------------------------------------------------------------
# CLI convenience
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python soil_moisture_downloader.py <year> [<year> …]")
        sys.exit(1)

    yrs = [int(y) for y in sys.argv[1:]]
    ds = fetch_lprm_amsr2(yrs)
    out_fp = ROOT_DIR / "soil_moisture.parquet"
    to_parquet(ds, out_fp)
    print(f"✅ Saved parquet → {out_fp.relative_to(Path.cwd())}")
