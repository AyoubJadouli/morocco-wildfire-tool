# -*- coding: utf-8 -*-
"""weather_downloader.py

Pulls daily GSOD weather observations for Moroccan stations (2010‑present),
interpolates missing values, and caches them as Parquet so that downstream
steps can broadcast station‑level features to fire/control points.

Designed to work primarily via Google BigQuery (public dataset
`bigquery-public-data.noaa_gsod`), with an automatic fallback to NOAA's FTP
if a BigQuery client/credentials are not available.

Example
-------
>>> from weather_downloader import WeatherDownloader
>>> wd = WeatherDownloader()
>>> wd.run(years=range(2010, 2024))

This will populate:
    Data/Weather_Noaa/morocco_stations.feather
    Data/Weather_Noaa/Morocco/<station_id>_interpolated.parquet
    Data/Weather_Noaa/ExpandedMorocco/<station_id>_expanded.parquet

The expanded files are optional; they hold lag / rolling aggregates if you
already imported ``temporal_features.add_lag_features``.  Otherwise only the
interpolated daily files are produced.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    # BigQuery is preferred because it is ~10× faster and free for the
    # first terabyte per month.  Credentials are optional for the public
    # GSOD dataset (anon access) but a GCP project is required so that the
    # API can bill the free quota to *something*.
    from google.cloud import bigquery  # type: ignore

    _BQ_AVAILABLE = True
except ImportError:  # pragma: no cover – BigQuery optional
    _BQ_AVAILABLE = False

try:
    import geopandas as gpd  # type: ignore

    from shapely.geometry import Point  # type: ignore
except ImportError:  # pragma: no cover – geopandas optional for station meta
    gpd = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover – tqdm optional
    def tqdm(x, **kwargs):  # type: ignore
        return x

# --- Local project imports --------------------------------------------------
try:
    import config  # expects DIRS dict
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "weather_downloader.py expects a config.py with a DIRS dict containing "
        " 'DATA' and sub‑folders.'"
    ) from exc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------


class WeatherDownloader:
    """Download + clean GSOD for Morocco.

    Parameters
    ----------
    data_root : Path-like | None
        Base directory that holds ``Data/Weather_Noaa``.  If *None*, uses
        ``config.DIRS['DATA']``.
    use_bigquery : bool | None
        Force/disable BigQuery.  If *None*, autodetect based on library.
    project_id : str | None
        GCP project ID.  Required if *use_bigquery* is *True* and you have
        no application‑default credentials that already specify a project.
    """

    #: bounding box for mainland Morocco (deg) – loose, includes enclaves
    BBOX = (-13.5, 20.5, -1.0, 36.5)  # min_lon, min_lat, max_lon, max_lat

    def __init__(
        self,
        data_root: Optional[os.PathLike] = None,
        *,
        use_bigquery: Optional[bool] = None,
        project_id: Optional[str] = None,
    ) -> None:
        self.data_root = Path(data_root or config.DIRS["DATA"]).expanduser()
        self.weather_dir = self.data_root / "Weather_Noaa"
        self.weather_dir.mkdir(parents=True, exist_ok=True)
        (self.weather_dir / "Morocco").mkdir(exist_ok=True)

        if use_bigquery is None:
            use_bigquery = _BQ_AVAILABLE
        self.use_bigquery = use_bigquery
        self.project_id = project_id

        if self.use_bigquery:
            if not _BQ_AVAILABLE:
                raise RuntimeError("google‑cloud‑bigquery is not installed.")
            self.bq_client = bigquery.Client(project=project_id)

    # ---------------------------------------------------------------------
    # Station list helpers
    # ---------------------------------------------------------------------

    def fetch_station_list(self) -> pd.DataFrame | "gpd.GeoDataFrame":
        """Return GSOD stations located inside Morocco's bbox.
        Cached to ``morocco_stations.feather``.
        """
        dest = self.weather_dir / "morocco_stations.feather"
        if dest.exists():
            logger.info("▶︎ Loading cached station list → %s", dest)
            return pd.read_feather(dest)

        if self.use_bigquery:
            logger.info("🔎 Querying GSOD station metadata from BigQuery…")
            q = f"""
            SELECT
                usaf || '-' || wban            AS station_id,
                CAST(latitude AS FLOAT64)      AS lat,
                CAST(longitude AS FLOAT64)     AS lon,
                elevation                      AS elevation_m,
                name,
                country
            FROM `bigquery-public-data.noaa_gsod.stations`
            WHERE longitude BETWEEN {self.BBOX[0]} AND {self.BBOX[2]}
              AND latitude  BETWEEN {self.BBOX[1]} AND {self.BBOX[3]}
              AND country = 'MO'
            """
            df = self.bq_client.query(q).to_dataframe()
        else:  # fallback using NOAA TSV (much slower)
            logger.info("📥 Downloading station list TSV via NOAA FTP…")
            import requests, io, zipfile  # local import to avoid hard dep

            url = (
                "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text))
            df = raw.copy()
            df["station_id"] = df["USAF"].str.zfill(6) + "-" + df["WBAN"].str.zfill(5)
            df.rename(
                columns={"LAT": "lat", "LON": "lon", "STATION NAME": "name"},
                inplace=True,
            )
            df = df[(df.lon.between(self.BBOX[0], self.BBOX[2])) & (df.lat.between(self.BBOX[1], self.BBOX[3]))]

        df.reset_index(drop=True, inplace=True)
        dest.parent.mkdir(exist_ok=True)
        df.to_feather(dest)
        logger.info("✅ %d stations cached → %s", len(df), dest)

        if gpd is not None:
            gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326")
            return gdf
        return df

    # ---------------------------------------------------------------------
    # Daily observations
    # ---------------------------------------------------------------------

    _COLS_MAP = {
        "temp": "average_temperature",
        "max": "maximum_temperature",
        "min": "minimum_temperature",
        "dewp": "dew_point_temperature",
        "prcp": "precipitation",
        "wdsp": "wind_speed",
        "gust": "max_wind_gust",
        "snow": "snow_depth",
        "station_pressure": "station_pressure",
        "sea_level_pressure": "sea_level_pressure",
    }

    def _query_year(self, year: int, station_ids: List[str]) -> pd.DataFrame:
        """Internal: query one `gsodYYYY` table for multiple stations."""
        ids_str = ",".join(f"'{sid}'" for sid in station_ids)
        q = f"""
            SELECT
                usaf || '-' || wban AS station_id,
                PARSE_DATE('%Y%m%d', CAST(year || month || day AS STRING)) AS date,
                temp, max, min, dewp, prcp, wdsp, gust, snow,
                stp AS station_pressure, slp AS sea_level_pressure
            FROM `bigquery-public-data.noaa_gsod.gsod{year}`
            WHERE CONCAT(usaf,'-',wban) IN ({ids_str})
        """
        return self.bq_client.query(q).to_dataframe()

    def fetch_daily(self, years: Iterable[int] | range) -> pd.DataFrame:
        """Download GSOD daily observations for *years* and return one DataFrame."""
        stations_df = self.fetch_station_list()
        station_ids = stations_df["station_id"].tolist()

        if self.use_bigquery:
            all_rows: List[pd.DataFrame] = []
            for y in tqdm(years, desc="GSOD"):
                logger.info("📅 Fetching %s…", y)
                df_year = self._query_year(y, station_ids)
                all_rows.append(df_year)
            obs = pd.concat(all_rows, ignore_index=True)
        else:  # FTP fallback – one file per station‑year (slow!)
            import gzip
            import io
            import requests

            rows = []
            for sid in tqdm(station_ids, desc="Stations"):
                usaf, wban = sid.split("-")
                for y in years:
                    fn = f"{usaf}-{wban}-{y}.op.gz"
                    url = (
                        "https://www.ncei.noaa.gov/pub/data/gsod/" f"{y}/{fn}"
                    )
                    try:
                        resp = requests.get(url, timeout=10)
                        if resp.status_code != 200:
                            continue  # station did not report that year
                        with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
                            df_year = pd.read_csv(
                                gz,
                                names=[
                                    "station_id",
                                    "wban",
                                    "date",
                                    "temp",
                                    "temp_cnt",
                                    "dewp",
                                    "dewp_cnt",
                                    "station_pressure",
                                    "station_pressure_cnt",
                                    "sea_level_pressure",
                                    "sea_level_pressure_cnt",
                                    "max",
                                    "max_flag",
                                    "min",
                                    "min_flag",
                                    "prcp",
                                    "prcp_flag",
                                    "snow",
                                    "ice_on_ground",
                                    "frshtt",
                                    "wdsp",
                                    "wdsp_cnt",
                                    "gust",
                                    "max_temp_flag",
                                    "min_temp_flag",
                                    "tobs",
                                    "time",
                                    "mean_wind_dir",
                                    "max_wind_dir",
                                    "dir_cnt",
                                ],
                                parse_dates=["date"],
                                na_values=[99.9, 9999.9, 999.9, 99.99, 999.99],
                            )
                            df_year["station_id"] = sid
                        rows.append(df_year[[c for c in df_year.columns if c in self._COLS_MAP or c == "date" or c == "station_id"]])
                    except Exception as exc:  # pragma: no cover
                        logger.warning("⚠️  Could not fetch %s: %s", url, exc)
            if not rows:
                raise RuntimeError("No GSOD files could be downloaded from FTP.")
            obs = pd.concat(rows, ignore_index=True)

        # Rename → human‑friendly
        obs.rename(columns=self._COLS_MAP, inplace=True)
        # Convert Fahrenheit to °C; inches to mm where necessary
        if "average_temperature" in obs.columns:
            obs["average_temperature"] = (obs["average_temperature"] - 32) * (5 / 9)
        if "maximum_temperature" in obs.columns:
            obs["maximum_temperature"] = (obs["maximum_temperature"] - 32) * (5 / 9)
        if "minimum_temperature" in obs.columns:
            obs["minimum_temperature"] = (obs["minimum_temperature"] - 32) * (5 / 9)
        if "dew_point_temperature" in obs.columns:
            obs["dew_point_temperature"] = (obs["dew_point_temperature"] - 32) * (5 / 9)
        if "precipitation" in obs.columns:
            obs["precipitation"] = obs["precipitation"] * 25.4  # inches → mm

        obs.sort_values(["station_id", "date"], inplace=True)
        obs.reset_index(drop=True, inplace=True)
        return obs

    # ---------------------------------------------------------------------
    # Interpolation & saving
    # ---------------------------------------------------------------------

    _NUMERIC_COLS = [
        "average_temperature",
        "maximum_temperature",
        "minimum_temperature",
        "dew_point_temperature",
        "precipitation",
        "wind_speed",
        "max_wind_gust",
        "station_pressure",
        "sea_level_pressure",
    ]

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill small gaps (<5 days) by linear interpolation; drop long gaps."""
        df = df.copy()
        df.set_index("date", inplace=True)
        df = df.asfreq("D")
        df[self._NUMERIC_COLS] = df[self._NUMERIC_COLS].interpolate(
            method="linear", limit=4, limit_direction="both"
        )
        df.reset_index(inplace=True)
        return df

    # ---------------------------------------------------------------------
    # Orchestrator helpers
    # ---------------------------------------------------------------------

    def _write_station(self, sid: str, sdf: pd.DataFrame) -> None:
        out = self.weather_dir / "Morocco" / f"{sid}_interpolated.parquet"
        sdf.to_parquet(out, index=False)
        logger.debug("💾 %s", out.relative_to(self.data_root))

    # ---------------------------------------------------------------------

    def run(self, years: Iterable[int] | range, *, interpolate: bool = True) -> None:
        """High‑level – download, interpolate, save parquet per station."""
        obs = self.fetch_daily(years)

        for sid, sdf in tqdm(obs.groupby("station_id"), desc="Stations", total=obs["station_id"].nunique()):
            if interpolate:
                sdf = self.interpolate(sdf)
            self._write_station(sid, sdf)

        logger.info("🏁 Weather download finished – %d stations, %d rows.", obs["station_id"].nunique(), len(obs))


# ---------------------------------------------------------------------------
# CLI helper (python -m weather_downloader 2010 2011 …)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download & clean GSOD for Morocco.")
    parser.add_argument("years", nargs="*", type=int, default=[], help="Years to fetch (default: 2010‑<today‑1>)")
    parser.add_argument("--no‑bq", dest="use_bigquery", action="store_false", help="Disable BigQuery (use NOAA FTP)")
    parser.add_argument("--project", help="GCP project ID (BigQuery)")
    args = parser.parse_args()

    if not args.years:
        this_year = _dt.date.today().year
        args.years = list(range(2010, this_year))

    wd = WeatherDownloader(use_bigquery=args.use_bigquery, project_id=args.project)
    wd.run(years=args.years)
