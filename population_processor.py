# -*- coding: utf-8 -*-
"""population_processor.py

Extracts gridded population density for Morocco from the GPW v4 revision 11
2.5-arc-minute NetCDF files and converts them to an analysis-ready Parquet
format suitable for spatial joins with fire/control points.

• **Data source**: CIESIN / Columbia University – "Gridded Population of the
  World, Version 4 (GPWv4): Population Density, Revision 11" – 2.5-arc-minute
  (~5 km) resolution global NetCDF. DOI: 10.7927/H45Q4T5F
• **Years available**: 2000, 2005, 2010, 2015, 2020.

The processor keeps a single **wide** dataframe with columns:
lat lon year_2000 year_2005 year_2010 year_2015 year_2020



All density values are stored as `float32` persons / km². Missing or masked
cells are dropped.

A new convenience function `slice_gpw()` is included so that existing
`pipeline.py` (which does `from population_processor import slice_gpw`) will work
unchanged.
"""
from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

try:
    import rioxarray  # noqa: F401  # ensures CRS API is activated
except ImportError:  # pragma: no cover
    raise ImportError(
        "population_processor.py requires the 'rioxarray' package.\n"
        "Install with: pip install rioxarray"
    )

import requests

try:
    # local project config (optional)
    import config  # type: ignore

    DATA_ROOT = Path(config.DIRS["data_root"])
except Exception:  # pragma: no cover
    DATA_ROOT = Path("Data")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GPW_BASE_URL = (
    "https://sedac.ciesin.columbia.edu/downloads/data/gpw-v4/"
    "gpw-v4-population-density-rev11_2020_2pt5_min_nc/"
)
GPW_FILENAME_TEMPLATE = "gpw_v4_population_density_rev11_{year}_2pt5_min.nc"

# Morocco bounding box (approx):
# lat  20.5–36.2 N, lon −13.5 – −0.9 W → convert to 0–360 if needed later
LAT_MIN, LAT_MAX = 20.5, 36.2
LON_MIN, LON_MAX = -13.5, -0.9  # keep in −180…180 for consistency

CACHE_DIR = DATA_ROOT / "GeoData" / "Population"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Year layers available in GPW v4 Rev 11 2.5-arc-min product
YEARS_AVAILABLE = (2000, 2005, 2010, 2015, 2020)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _download_if_missing(years: List[int]) -> List[Path]:
    """Ensure the NetCDF for each *year* is stored locally; download otherwise."""
    local_paths: List[Path] = []
    for y in years:
        fname = GPW_FILENAME_TEMPLATE.format(year=y)
        url = GPW_BASE_URL + fname + ".gz"
        local_nc = CACHE_DIR / fname
        gzip_path = local_nc.with_suffix(local_nc.suffix + ".gz")

        if local_nc.exists():
            local_paths.append(local_nc)
            continue

        logger.info("Downloading GPW population density %s …", y)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(gzip_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 18):
                    fh.write(chunk)

        # Extract .gz → .nc
        logger.info("Extracting %s …", gzip_path.name)
        with tarfile.open(gzip_path, "r:gz") as tf:
            tf.extractall(path=CACHE_DIR)
        gzip_path.unlink()  # cleanup

        if not local_nc.exists():
            raise FileNotFoundError(f"Extraction failed for {gzip_path}")

        local_paths.append(local_nc)

    return local_paths


def _subset_and_tidy(nc_path: Path, year: int) -> pd.DataFrame:
    """Open a single NetCDF file, slice to Morocco bbox, and return tidy DF."""
    # xarray open; the variable is named e.g. "population_density"
    ds = xr.open_dataset(nc_path)  # type: ignore[arg-type]

    # Ensure lon dimension is −180…180 instead of 0…360
    if (ds["lon"] > 180).any():
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")

    # Spatial subset
    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    var_name = [v for v in ds.data_vars][0]
    da = ds[var_name]

    # Mask values < 0 or > (some huge number) – GPW uses −9999 for nodata
    da = da.where(da >= 0)

    # Convert to DataFrame
    df = da.to_dataframe(name=f"year_{year}").reset_index()

    # Drop NaNs
    df = df.dropna(subset=[f"year_{year}"])

    # Store as float32
    df[f"year_{year}"] = df[f"year_{year}"].astype("float32")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PopulationProcessor:
    """Fetch and convert GPW population grids for Morocco into Parquet."""

    def __init__(self, years: List[int] | None = None):
        if years is None:
            years = list(YEARS_AVAILABLE)
        invalid = set(years) - set(YEARS_AVAILABLE)
        if invalid:
            raise ValueError(
                f"GPW v4 only provides {YEARS_AVAILABLE}; invalid: {sorted(invalid)}"
            )
        self.years = sorted(years)

    # ---------------------------------------------------------------------
    def fetch(self) -> pd.DataFrame:
        """Download (if needed) and return Morocco-subset population density."""
        nc_paths = _download_if_missing(self.years)

        dfs = [_subset_and_tidy(p, y) for p, y in zip(nc_paths, self.years)]

        # Merge on lat/lon (outer join, then back-fill NaNs with −1 so we know)
        df_merged = dfs[0]
        for dfi in dfs[1:]:
            df_merged = df_merged.merge(dfi, on=["lat", "lon"], how="outer")

        # Drop cells that are NaN for *all* years (shouldn't happen)
        year_cols = [f"year_{y}" for y in self.years]
        df_merged = df_merged.dropna(how="all", subset=year_cols)

        # Replace remaining NaNs with 0 (uninhabited cells)
        df_merged[year_cols] = df_merged[year_cols].fillna(0.0).astype("float32")

        # Round lat/lon to 4 decimals (≈11 m) to reduce file size deterministically
        df_merged["lat"] = df_merged["lat"].round(4)
        df_merged["lon"] = df_merged["lon"].round(4)

        return df_merged

    # ------------------------------------------------------------------
    def to_parquet(self, df: pd.DataFrame | None = None, *, overwrite: bool = False) -> Path:
        """Save the tidy dataframe to Parquet; returns the file path."""
        if df is None:
            df = self.fetch()

        fname = CACHE_DIR / "morocco_population.parquet"
        if fname.exists() and not overwrite:
            logger.info(
                "%s already exists; skipping write (use overwrite=True)",
                fname.name,
            )
            return fname

        fname.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fname, index=False)
        logger.info("Wrote %s (%s rows)", fname.name, len(df))
        return fname


def slice_gpw(years: List[int] | None = None) -> pd.DataFrame:
    """
    Convenience wrapper so that `pipeline.py` (which does `from population_processor
    import slice_gpw`) continues working. Returns the full merged DataFrame for Morocco.

    Arguments:
        years: list of int (e.g. [2010, 2015]), or None for all available years.
    """
    proc = PopulationProcessor(years)
    return proc.fetch()


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Slice GPWv4 population for Morocco → Parquet"
    )
    parser.add_argument(
        "years",
        nargs="*",
        type=int,
        default=list(YEARS_AVAILABLE),
        help="Years to pull (default: all available)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing parquet"
    )
    args = parser.parse_args()

    proc = PopulationProcessor(args.years)
    df = proc.fetch()
    proc.to_parquet(df, overwrite=args.overwrite)

    print(df.head())
