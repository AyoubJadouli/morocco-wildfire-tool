# -*- coding: utf-8 -*-
"""pipeline.py

End‑to‑end orchestrator that builds the **Morocco Wildfire Prediction Dataset**
(272 columns, balanced 50 / 50 fire vs. non‑fire) from scratch.

The script is intentionally *idempotent*: every major artefact is cached on
disk. If the file already exists (and `--force` isn’t passed), that step is
skipped. This means you can resume a partially‑completed run without repeating
time‑consuming downloads.

Run from repo root:
    $ python pipeline.py --years 2010 2024 \
        --out Data/FinalDataset/morocco_wildfire_prediction_dataset.parquet

Dependencies: the other modules in this repo plus
    bigquery‑client, requests, aiohttp, xarray, rasterio, netCDF4,
    holidays, geopy, scipy, pyproj.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

import config
from downloader import DataDownloader  # FIRMS
from weather_downloader import WeatherDownloader
from ndvi_downloader import fetch_mod13a2, to_parquet as ndvi_to_parquet
from soil_moisture_downloader import SoilMoistureDownloader
from population_processor import slice_gpw
from temporal_features import add_lag_features, add_rollups
from spatial_join import SpatialJoiner
from geoutils import nearest_station, distance_to_coast
from augment_data import jitter_fire_points
from control_sampler import sample_no_fire
from holiday_utils import mark_holidays
from data_processor import DataProcessor

# ---------------------------------------------------------------------------
# logging
logger = logging.getLogger("pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# helpers

def ensure_directories() -> None:
    """Create directory tree declared in config.DIRS."""
    for name, path in config.DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("✅ All directories ensured.")


def download_geography() -> None:
    """Download Natural‑Earth shapefiles unless already present."""
    shp_dir = config.DIRS["shapefiles"]
    countries_shp = shp_dir / "ne_10m_admin_0_countries_mar.shp"
    coast_shp = shp_dir / "ne_10m_coastline.shp"
    if countries_shp.exists() and coast_shp.exists():
        logger.info("🌍 Geographic data already present – skipping download.")
        return

    downloader = DataDownloader()
    logger.info("🌍 Downloading Natural‑Earth shapefiles…")
    downloader.download_geographic_data()
    logger.info("✅ Geographic data ready.")


# ---------------------------------------------------------------------------
# main pipeline logic

def run(years: Sequence[int], output: Path, fire_only: bool = False, force: bool = False) -> None:
    """Execute full pipeline for the given year range.

    Parameters
    ----------
    years : list[int]
        Inclusive list of years to process (e.g., range(2010, 2024)).
    output : Path
        Final parquet path.
    fire_only : bool, default False
        If True, skip non‑fire control generation (for debugging speed‑ups).
    force : bool, default False
        Re‑compute every step even if cached artefacts exist.
    """

    ensure_directories()
    download_geography()

    # ------------------------------------------------------------------ FIRMS
    dl = DataDownloader()
    fire_parquet = config.DIRS["firms"] / "all_firms_data_Morocco_combined.parquet"
    if fire_parquet.exists() and not force:
        logger.info("🔥 FIRMS parquet cached – loading")
        fires_df = pd.read_parquet(fire_parquet)
    else:
        logger.info("🔥 Downloading FIRMS CSVs %s…", list(years))
        dl.download_firms_data(years)
        fires_df = dl.process_firms_data()

    # Keep only years requested (download helper always fetches 2010‑2023)
    fires_df = fires_df[fires_df["acq_date"].dt.year.isin(years)]

    # ------------------------------------------------------------ Weather / GSOD
    wdl = WeatherDownloader()
    met_df = wdl.fetch_daily(years=list(years))  # tidy per‑station daily
    met_df = wdl.interpolate(met_df)
    met_df_expanded = add_rollups(add_lag_features(met_df.copy(), [
        "average_temperature", "maximum_temperature", "minimum_temperature"
    ], lags=range(1, 16)), [
        "average_temperature", "maximum_temperature", "minimum_temperature"
    ], windows={"weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365})

    # ------------------------------------------------------------------ NDVI
    ndvi_parquet = config.DIRS["ndvi"] / "morocco_ndvi_data.parquet"
    if not ndvi_parquet.exists() or force:
        logger.info("🌱 NDVI parquet missing – downloading tiles…")
        ndvi_df = fetch_mod13a2(years=list(years))
        ndvi_to_parquet(ndvi_df, ndvi_parquet)
    else:
        ndvi_df = pd.read_parquet(ndvi_parquet)

    # ------------------------------------------------------- Soil Moisture
    sm_parquet = config.DIRS["soil_moisture"] / "soil_moisture.parquet"
    if not sm_parquet.exists() or force:
        logger.info("💧 Soil‑moisture parquet missing – downloading…")
        sm_dl = SoilMoistureDownloader()
        sm_df = sm_dl.fetch(years=list(years))
        sm_dl.to_parquet(sm_df, sm_parquet)
    else:
        sm_df = pd.read_parquet(sm_parquet)

    # -------------------------------------------------------- Population
    pop_parquet = config.DIRS["population"] / "morocco_population.parquet"
    if not pop_parquet.exists() or force:
        logger.info("👥 Population parquet missing – slicing GPW…")
        pop_df = slice_gpw()
        pop_df.to_parquet(pop_parquet, index=False)
    else:
        pop_df = pd.read_parquet(pop_parquet)

    # ----------------------------------------------------- Spatial joins & enrich
    spjoin = SpatialJoiner()
    logger.info("📍 Attaching nearest weather station to fire pixels…")
    fires_df["station_name"] = nearest_station(fires_df, met_df_expanded)
    fires_df = fires_df.merge(
        met_df_expanded,
        on=["station_name", "acq_date"],
        how="left",
        suffixes=("", "_wx"),
    )

    logger.info("🌊 Computing distance to sea…")
    coastline = (config.DIRS["shapefiles"] / "ne_10m_coastline.shp")
    fires_df["sea_distance"] = fires_df.apply(
        lambda r: distance_to_coast(r.latitude, r.longitude, coastline), axis=1
    )

    logger.info("🌱 Joining NDVI…")
    fires_df = spjoin.nearest_grid_value(fires_df, ndvi_df, "NDVI", "NDVI")

    logger.info("💧 Joining soil‑moisture…")
    fires_df = spjoin.nearest_grid_value(fires_df, sm_df, "SoilMoisture", "SoilMoisture")

    logger.info("👥 Joining population density…")
    fires_df = spjoin.nearest_grid_value(fires_df, pop_df, "pop_density", "pop_density")

    # ---------------------------------------------------- Holiday / calendar
    fires_df = mark_holidays(fires_df)

    # ---------------------------------------------------- Augmentation & control
    fires_df = jitter_fire_points(fires_df)
    if not fire_only:
        logger.info("⚖️  Generating non‑fire control points…")
        control_df = sample_no_fire(
            n=len(fires_df),
            date_min=fires_df["acq_date"].min(),
            date_max=fires_df["acq_date"].max(),
            polygon=DataProcessor().morocco_shape.geometry.iloc[0],
        )
        control_df = spjoin.nearest_grid_value(control_df, ndvi_df, "NDVI", "NDVI")
        control_df = spjoin.nearest_grid_value(control_df, sm_df, "SoilMoisture", "SoilMoisture")
        control_df = spjoin.nearest_grid_value(control_df, pop_df, "pop_density", "pop_density")
        control_df = mark_holidays(control_df)
        combined_df = pd.concat([fires_df, control_df], ignore_index=True).sample(frac=1.0)
    else:
        combined_df = fires_df

    # ---------------------------------------------------- optimise & export
    float_cols = combined_df.select_dtypes("float64").columns
    combined_df[float_cols] = combined_df[float_cols].astype("float32")

    output.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output, index=False)
    logger.info("🎉 Pipeline complete → %s (rows=%d, cols=%d)", output, *combined_df.shape)


# ---------------------------------------------------------------------------
# CLI

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Morocco wildfire dataset.")
    p.add_argument("years", nargs="*", type=int, default=list(range(2010, 2024)), help="Years to include (e.g. 2018 2019 2020)")
    p.add_argument("--out", type=Path, default=config.DIRS["final"] / "morocco_wildfire_prediction_dataset.parquet", help="Output parquet path")
    p.add_argument("--fire-only", action="store_true", help="Skip non‑fire control generation (faster, imbalanced)")
    p.add_argument("--force", action="store_true", help="Recompute all steps even if cached")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    run(years=args.years, output=args.out, fire_only=args.fire_only, force=args.force)
