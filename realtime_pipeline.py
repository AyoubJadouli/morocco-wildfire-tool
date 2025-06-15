# realtime_pipeline.py

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import numpy as np
import duckdb
import geopandas as gpd

from fetchers.gfs import run as fetch_gfs  # ⛅ NOAA GFS NRT fetcher
from ndvi_downloader import fetch_mod13a2, to_parquet as ndvi_to_parquet  # 🌱 MODIS NDVI
from soil_moisture_downloader import fetch_lprm_amsr2, to_parquet as sm_to_parquet  # 💧 SMAP SM
from population_processor import slice_gpw  # 👥 GPW pop density
from weather_downloader import WeatherDownloader  # for station list
from geoutils import nearest_station, build_coast_kdtree, distance_to_coast
from spatial_join import nearest_grid_value, SpatialJoiner
from holiday_utils import mark_holidays
from temporal_features import add_lag_features, add_rollups

from feature_store.rolling_buffer import RollingBuffer  # per‐var parquet buffer
import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("realtime_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path.cwd()
# where we store rolling weather buffers:
WEATHER_BUFFER_ROOT = ROOT / "Data" / "feature_db"

# where we store the latest NDVI/SM/Population tables
NDVI_PARQUET      = ROOT / "Data" / "GeoData" / "NDVI" / "morocco_ndvi_nrt.parquet"
SM_PARQUET        = ROOT / "Data" / "GeoData" / "SoilMoisture" / "soil_moisture_nrt.parquet"
POPULATION_PARQ   = ROOT / "Data" / "GeoData" / "Population" / "morocco_population.parquet"

# shapefiles for coastline (for sea_distance)
COAST_SHAPEFILE   = ROOT / "Data" / "GeoData" / "SHP" / "ne_10m_coastline.shp"

# NRT “cache” directories (create if missing)
for p in [
    WEATHER_BUFFER_ROOT,
    NDVI_PARQUET.parent,
    SM_PARQUET.parent,
    POPULATION_PARQ.parent,
]:
    p.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1) UPDATE NRT FEEDS
# ─────────────────────────────────────────────────────────────────────────────
def update_nrt_feeds(target_date: datetime):
    """
    1a) GFS: fetch latest cycle and append to RollingBuffer (temperature, RH, wind, precip).
    1b) NDVI: fetch last 16‐day MOD13A2 composite that includes target_date.
    1c) Soil Moisture: fetch SMAP LPRM-AMSR2 dataset for last 2 weeks (to ensure at least one new).
    1d) Population: slice once (static) if missing.
    """

    log.info("── Updating GFS weather buffer ──")
    # fetchers/gfs.py -> writes into WEATHER_BUFFER_ROOT/<var>/dt=YYYY-MM-DD.parquet
    fetch_gfs()

    log.info("── Updating NDVI (MODIS) ──")
    # MODIS NDVI is given at 16‐day cadence; find the composite period containing target_date:
    year = target_date.year
    # (the fetch_mod13a2 helper will skip missing composites gracefully)
    df_ndvi = fetch_mod13a2(years=[year])
    ndvi_to_parquet(df_ndvi, NDVI_PARQUET)

    log.info("── Updating Soil Moisture (SMAP) ──")
    # fetch daily LPRM AMSR2 for this year; convert to parquet
    ds_sm = fetch_lprm_amsr2([year])
    sm_to_parquet(ds_sm, SM_PARQUET)

    log.info("── Ensuring Population grid is ready (static) ──")
    if not POPULATION_PARQ.exists():
        pop_df = slice_gpw()
        pop_df.to_parquet(POPULATION_PARQ, index=False)
    else:
        log.info("Population Parquet already exists, skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# 2) LOAD STATIC TABLES
# ─────────────────────────────────────────────────────────────────────────────
def load_static_tables():
    """Load NDVI, Soil Moisture, Population DataFrames (into wide Pandas frames)."""
    ndvi_df = pd.read_parquet(NDVI_PARQUET)
    sm_df   = pd.read_parquet(SM_PARQUET)
    pop_df  = pd.read_parquet(POPULATION_PARQ)
    return ndvi_df, sm_df, pop_df

# ─────────────────────────────────────────────────────────────────────────────
# 3) PREPARE COASTLINE KD-TREE (for fast distance_to_coast)
# ─────────────────────────────────────────────────────────────────────────────
_coast_tree = None
_coast_xyz  = None

def prepare_coast_kdtree():
    global _coast_tree, _coast_xyz
    if _coast_tree is None:
        coast_gdf = gpd.read_file(COAST_SHAPEFILE)
        _coast_tree, _coast_xyz = build_coast_kdtree(coast_gdf)
    return _coast_tree, _coast_xyz

# ─────────────────────────────────────────────────────────────────────────────
# 4) LOAD WEATHER BUFFER + QUERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
class WeatherFeatureBuilder:
    """
    Wraps RollingBuffer to extract lag and rolling features from GFS variables.
    """

    def __init__(self, buffer_root: Path):
        self.rb = RollingBuffer(buffer_root)
        # these correspond to the names used in fetchers/gfs.py: 
        #   'temperature', 'rh', 'u_wind', 'v_wind', 'precip'
        self.vars = {
            "temperature": "average_temperature",
            "rh":          "relative_humidity",
            "u_wind":      "u_wind",
            "v_wind":      "v_wind",
            "precip":      "precipitation",
        }

    def _load_recent(self, var_name: str, since: datetime, until: datetime) -> pl.DataFrame:
        """
        Load all parquet files for a given variable between dates [since, until],
        returning a Polars DataFrame with columns: 'latitude','longitude','timestamp',<var_raw>.
        """
        root = WEATHER_BUFFER_ROOT / var_name
        candidates = list(root.glob(f"dt=*.parquet"))
        frames = []
        for p in candidates:
            day = pd.to_datetime(p.stem.split("=")[1]).date()
            if since.date() <= day <= until.date():
                frames.append(pl.read_parquet(str(p)))
        if not frames:
            return pl.DataFrame([])  # empty
        return pl.concat(frames)

    def build_weather_lag_columns(self, points_df: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """
        Given a Pandas points_df with columns ['acq_date','latitude','longitude'], return
        a new DataFrame that has for each point *only the weather-related columns*, i.e.:
          - average_temperature_lag_1 … _lag_15
          - maximum_temperature_lag_1 … _lag_15
          - minimum_temperature_lag_1 … _lag_15
          - precipitation_lag_1 … _lag_15
          - (and similarly for wind_speed, max_sustained_wind_speed, wind_gust, dew_point, fog, thunder)
          - plus weekly / monthly / quarterly / yearly means for each of those
          - plus last_1_year / last_2_year / last_3_year seasonal comparators
        The output DataFrame has the same index as points_df.
        """
        out = pd.DataFrame(index=points_df.index)

        since_15 = now - timedelta(days=15)
        # --- Load the raw GFS fields from RollingBuffer ---
        temp_df = self._load_recent("temperature", since_15, now).to_pandas()
        precp_df = self._load_recent("precip", since_15, now).to_pandas()

        temp_df["date"] = temp_df["timestamp"].dt.floor("D")
        temp_daily = temp_df.groupby(["latitude", "longitude", "date"])["temperature"].agg(
            avg_temp="mean", max_temp="max", min_temp="min"
        ).reset_index()

        precp_df["date"] = precp_df["timestamp"].dt.floor("D")
        precp_daily = precp_df.groupby(["latitude", "longitude", "date"])["precip"].agg(
            daily_precip="sum"
        ).reset_index()

        weather_daily = (
            temp_daily.merge(precp_daily, on=["latitude", "longitude", "date"], how="outer")
            .fillna(0.0)
        )

        coords = weather_daily[["latitude", "longitude"]].drop_duplicates()
        xyz = SpatialJoiner(coords.rename(columns={"latitude":"lat","longitude":"lon"}), value_col="lat")
        tree = xyz._tree
        grid_lat = coords["latitude"].values
        grid_lon = coords["longitude"].values

        weather_daily["grid_idx"] = xyz.query(
            coords.rename(columns={"latitude":"lat","longitude":"lon"}),
            new_name="grid_idx"
        )["grid_idx"].values

        wd_by_grid = {
            idx: sub.sort_values("date").set_index("date")
            for idx, sub in weather_daily.groupby("grid_idx")
        }

        def extract_weather_lags(row):
            lat, lon = row["latitude"], row["longitude"]
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            xyz_pt = np.column_stack([
                np.cos(lat_rad)*np.cos(lon_rad),
                np.cos(lat_rad)*np.sin(lon_rad),
                np.sin(lat_rad),
            ])
            dist_chord, idx = tree.query(xyz_pt, k=1)
            idx = idx[0]

            df_cell = wd_by_grid.get(idx)
            if df_cell is None:
                return pd.Series(dtype="float32")

            d0 = row["acq_date"].floor("D")
            lags = {}
            for lag in range(1,16):
                date_lag = d0 - timedelta(days=lag)
                if date_lag in df_cell.index:
                    lags[f"average_temperature_lag_{lag}"] = df_cell.at[date_lag, "avg_temp"]
                    lags[f"maximum_temperature_lag_{lag}"] = df_cell.at[date_lag, "max_temp"]
                    lags[f"minimum_temperature_lag_{lag}"] = df_cell.at[date_lag, "min_temp"]
                    lags[f"precipitation_lag_{lag}"]       = df_cell.at[date_lag, "daily_precip"]
                else:
                    lags[f"average_temperature_lag_{lag}"] = np.nan
                    lags[f"maximum_temperature_lag_{lag}"] = np.nan
                    lags[f"minimum_temperature_lag_{lag}"] = np.nan
                    lags[f"precipitation_lag_{lag}"]       = np.nan

            windows = {"weekly":7, "monthly":30, "quarterly":90, "yearly":365}
            for name, wsize in windows.items():
                window_df = df_cell.loc[(df_cell.index >= (d0 - timedelta(days=wsize))) & (df_cell.index < d0)]
                if not window_df.empty:
                    lags[f"average_temperature_{name}_mean"] = window_df["avg_temp"].mean()
                    lags[f"maximum_temperature_{name}_mean"] = window_df["max_temp"].mean()
                    lags[f"minimum_temperature_{name}_mean"] = window_df["min_temp"].mean()
                    lags[f"precipitation_{name}_mean"]       = window_df["daily_precip"].mean()
                else:
                    for v in ["average_temperature","maximum_temperature","minimum_temperature","precipitation"]:
                        lags[f"{v}_{name}_mean"] = np.nan

            for n in (1,2,3):
                date_past = d0 - timedelta(days=365*n)
                if date_past in df_cell.index:
                    lags[f"average_temperature_last_{n}_year"] = df_cell.at[date_past, "avg_temp"]
                    lags[f"maximum_temperature_last_{n}_year"] = df_cell.at[date_past, "max_temp"]
                    lags[f"minimum_temperature_last_{n}_year"] = df_cell.at[date_past, "min_temp"]
                    lags[f"precipitation_last_{n}_year"]       = df_cell.at[date_past, "daily_precip"]
                else:
                    lags[f"average_temperature_last_{n}_year"] = np.nan
                    lags[f"maximum_temperature_last_{n}_year"] = np.nan
                    lags[f"minimum_temperature_last_{n}_year"] = np.nan
                    lags[f"precipitation_last_{n}_year"]       = np.nan

                past_month_df = df_cell.loc[
                    (df_cell.index >= (date_past - timedelta(days=30))) & (df_cell.index < date_past)
                ]
                for v in ["average_temperature","maximum_temperature","minimum_temperature","precipitation"]:
                    key = f"{v}_last_{n}_year_monthly_mean"
                    if not past_month_df.empty:
                        if "avg_temp" in past_month_df and v=="average_temperature":
                            lags[key] = past_month_df["avg_temp"].mean()
                        elif "max_temp" in past_month_df and v=="maximum_temperature":
                            lags[key] = past_month_df["max_temp"].mean()
                        elif "min_temp" in past_month_df and v=="minimum_temperature":
                            lags[key] = past_month_df["min_temp"].mean()
                        elif "daily_precip" in past_month_df and v=="precipitation":
                            lags[key] = past_month_df["daily_precip"].mean()
                        else:
                            lags[key] = np.nan
                    else:
                        lags[key] = np.nan

            return pd.Series(lags, dtype="float32")

        lag_df = points_df.apply(extract_weather_lags, axis=1)
        return lag_df.fillna(0.0)

# ─────────────────────────────────────────────────────────────────────────────
# 5) MAIN “BUILD FEATURES” FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_table(points_df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """
    Given points_df with ['acq_date','latitude','longitude'], return a new DataFrame
    that has all 272 columns (except 'is_fire', since real-time has no labels).
    """
    df = points_df.copy().reset_index(drop=True)
    log.info("Building features for %d points at %s …", len(df), now.isoformat())

    # 5a) Calendar / holiday features
    df = mark_holidays(df, country="MA", date_col="acq_date")
    # Now df has: is_holiday, is_weekend, day_of_week, day_of_year

    # 5b) Attach nearest GSOD‐station (static metadata only)
    wd = WeatherDownloader()
    stations = wd.fetch_station_list()
    if isinstance(stations, gpd.GeoDataFrame):
        stations = pd.DataFrame(stations.drop(columns="geometry"))
    df = nearest_station(df, stations, station_lat_col="lat", station_lon_col="lon")

    # 5c) Compute sea_distance
    coast_tree, coast_xyz = prepare_coast_kdtree()
    def _sea_dist(r):
        return distance_to_coast(r.latitude, r.longitude, coast_tree, coast_xyz)
    df["sea_distance"] = df.apply(_sea_dist, axis=1)

    # 5d) Join NDVI & SoilMoisture & Population
    ndvi_df, sm_df, pop_df = load_static_tables()
    spjoin = SpatialJoiner(grid_df=ndvi_df.rename(columns={"lat":"lat","lon":"lon"}), value_col="NDVI")
    df = spjoin(df, new_name="NDVI")
    spjoin_sm = SpatialJoiner(grid_df=sm_df.rename(columns={"lat":"lat","lon":"lon"}), value_col="SoilMoisture")
    df = spjoin_sm(df, new_name="SoilMoisture")
    spjoin_pop = SpatialJoiner(grid_df=pop_df.rename(columns={"lat":"lat","lon":"lon"}), value_col="year_2020")
    df["pop_density"] = spjoin_pop(df)["year_2020"]

    # 5e) Weather lags & roll‐ups via GFS buffer
    wb = WeatherFeatureBuilder(WEATHER_BUFFER_ROOT)
    weather_feats = wb.build_weather_lag_columns(df, now)
    df = pd.concat([df, weather_feats], axis=1)

    # 5f) Re‐compute weekly/monthly/quarterly/yearly means for non‐temperature vars
    # (Extend the same pattern as temperature above for wind_speed, dew_point, fog, thunder, etc.)

    # 5g) Column ordering: match the offline schema exactly
    desired_order = [
        "acq_date", "latitude", "longitude", "is_holiday", "day_of_week",
        "day_of_year", "is_weekend", "NDVI", "SoilMoisture", "sea_distance",
        "station_lat", "station_lon"
    ]

    # Temperature lags
    for lag in range(1, 16):
        desired_order.append(f"average_temperature_lag_{lag}")
    for lag in range(1, 16):
        desired_order.append(f"maximum_temperature_lag_{lag}")
    for lag in range(1, 16):
        desired_order.append(f"minimum_temperature_lag_{lag}")

    # Precipitation lags
    for lag in range(1, 16):
        desired_order.append(f"precipitation_lag_{lag}")

    # (Add other GFS-derived lags here: snow_depth, wind_speed, max_sustained_wind_speed, wind_gust, dew_point, fog, thunder)

    # Weekly means
    desired_order += [
        "average_temperature_weekly_mean", "maximum_temperature_weekly_mean",
        "minimum_temperature_weekly_mean", "precipitation_weekly_mean"
    ]

    # Last n-year comparators
    for n in (1, 2, 3):
        desired_order += [
            f"average_temperature_last_{n}_year",
            f"maximum_temperature_last_{n}_year",
            f"minimum_temperature_last_{n}_year",
            f"precipitation_last_{n}_year",
        ]
        desired_order += [
            f"average_temperature_last_{n}_year_monthly_mean",
            f"maximum_temperature_last_{n}_year_monthly_mean",
            f"minimum_temperature_last_{n}_year_monthly_mean",
            f"precipitation_last_{n}_year_monthly_mean",
        ]

    # Monthly means
    desired_order += [
        "average_temperature_monthly_mean", "maximum_temperature_monthly_mean",
        "minimum_temperature_monthly_mean", "precipitation_monthly_mean"
    ]

    # Last n-years monthly-means
    for n in (1, 2, 3):
        desired_order += [
            f"average_temperature_last_{n}_year_monthly_mean",
            f"maximum_temperature_last_{n}_year_monthly_mean",
            f"minimum_temperature_last_{n}_year_monthly_mean",
            f"precipitation_last_{n}_year_monthly_mean",
        ]

    # Quarterly & yearly means
    desired_order += [
        "average_temperature_quarterly_mean", "maximum_temperature_quarterly_mean",
        "minimum_temperature_quarterly_mean", "precipitation_quarterly_mean",
        "average_temperature_yearly_mean", "maximum_temperature_yearly_mean",
        "minimum_temperature_yearly_mean", "precipitation_yearly_mean"
    ]

    # (Continue adding snow_depth, wind_gust, dew_point, fog, thunder in the same pattern…)

    # In real-time we do not include 'is_fire' (no labels).

    # Add missing columns as zeros
    for col in desired_order:
        if col not in df.columns:
            df[col] = 0.0

    df = df[desired_order].astype(np.float32, errors="ignore")
    log.info("Built feature table with shape %s", df.shape)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 6) EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run as: python realtime_pipeline.py 2025-06-01
    """
    import sys
    if len(sys.argv) != 2:
        print("Usage: python realtime_pipeline.py YYYY-MM-DD")
        sys.exit(1)

    date_str = sys.argv[1]
    now = datetime.fromisoformat(date_str)

    # ─── 6a) Update NRT feeds ───
    update_nrt_feeds(now)

    # ─── 6b) Sample control points (or supply your own) ───
    from control_sampler import sample_no_fire
    from data_processor import DataProcessor
    dp = DataProcessor()
    polygon = dp.morocco_shape.geometry.unary_union
    control_df = sample_no_fire(
        n=1000,
        date_min=now.date(),
        date_max=now.date(),
        polygon=polygon,
        seed=42
    )

    # ─── 6c) Build features ───
    feature_df = build_feature_table(control_df, now)

    # ─── 6d) Write to Parquet ───
    out_path = ROOT / "Data" / "FinalDataset" / f"morocco_nrt_features_{date_str}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_path, index=False)
    log.info("✅ Real‐time feature table written to %s", out_path)
