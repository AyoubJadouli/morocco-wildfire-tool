# """
# 🔄 Data Processing and Integration
# """
# import pandas as pd
# import numpy as np
# import geopandas as gpd
# from shapely.geometry import Point
# from typing import Optional, Dict, List
# import logging
# from datetime import datetime
# import config

# logger = logging.getLogger(__name__)

# class DataProcessor:
#     """🔄 Handles data processing and feature engineering"""
    
#     def __init__(self):
#         self.morocco_shape = None
#         self._load_morocco_shape()
    
#     def _load_morocco_shape(self):
#         """🗺️ Load Morocco boundary shapefile"""
#         try:
#             shp_path = config.DIRS['shapefiles'] / 'ne_10m_admin_0_countries_mar.shp'
#             if shp_path.exists():
#                 world = gpd.read_file(shp_path)
#                 self.morocco_shape = world[world['ADMIN'] == 'Morocco']
#         except Exception as e:
#             logger.warning(f"⚠️ Could not load Morocco shapefile: {e}")
    
#     def filter_morocco_points(self, df: pd.DataFrame) -> pd.DataFrame:
#         """🗺️ Filter points within Morocco boundaries"""
#         if self.morocco_shape is None:
#             logger.warning("⚠️ Morocco shape not loaded, returning all points")
#             return df
        
#         # Convert to GeoDataFrame
#         gdf = gpd.GeoDataFrame(
#             df,
#             geometry=gpd.points_from_xy(df.longitude, df.latitude)
#         )
        
#         # Filter points within Morocco
#         morocco_geom = self.morocco_shape.geometry.iloc[0]
#         mask = gdf.geometry.within(morocco_geom)
        
#         return df[mask].reset_index(drop=True)
    
#     def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """📅 Add temporal features"""
#         df = df.copy()
        
#         # Ensure datetime
#         df['acq_date'] = pd.to_datetime(df['acq_date'])
        
#         # Add temporal features
#         df['day_of_week'] = df['acq_date'].dt.dayofweek + 1
#         df['day_of_year'] = df['acq_date'].dt.dayofyear
#         df['month'] = df['acq_date'].dt.month
#         df['year'] = df['acq_date'].dt.year
#         df['is_weekend'] = df['acq_date'].dt.dayofweek.isin([5, 6]).astype(int)
        
#         # Season
#         df['season'] = df['month'].apply(lambda x: 
#             'winter' if x in [12, 1, 2] else
#             'spring' if x in [3, 4, 5] else
#             'summer' if x in [6, 7, 8] else 'fall'
#         )
        
#         return df
    
#     def add_lag_features(self, df: pd.DataFrame, 
#                         weather_cols: List[str], 
#                         lags: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
#         """🔄 Add lagged weather features"""
#         df = df.copy()
        
#         # Sort by date
#         df = df.sort_values('acq_date')
        
#         # Add lag features
#         for col in weather_cols:
#             if col in df.columns:
#                 for lag in lags:
#                     df[f'{col}_lag_{lag}'] = df.groupby(['latitude', 'longitude'])[col].shift(lag)
        
#         return df
    
#     def add_rolling_features(self, df: pd.DataFrame, 
#                            weather_cols: List[str],
#                            windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
#         """📊 Add rolling statistics"""
#         df = df.copy()
        
#         for col in weather_cols:
#             if col in df.columns:
#                 for window in windows:
#                     df[f'{col}_rolling_mean_{window}'] = (
#                         df.groupby(['latitude', 'longitude'])[col]
#                         .transform(lambda x: x.rolling(window, min_periods=1).mean())
#                     )
#                     df[f'{col}_rolling_std_{window}'] = (
#                         df.groupby(['latitude', 'longitude'])[col]
#                         .transform(lambda x: x.rolling(window, min_periods=1).std())
#                     )
        
#         return df
    
#     def calculate_fire_risk_index(self, df: pd.DataFrame) -> pd.DataFrame:
#         """🔥 Calculate fire risk index"""
#         df = df.copy()
        
#         # Simple fire risk index based on available features
#         risk_factors = []
        
#         if 'average_temperature' in df.columns:
#             risk_factors.append((df['average_temperature'] - 20) / 20)
        
#         if 'precipitation' in df.columns:
#             risk_factors.append(1 - (df['precipitation'] / df['precipitation'].max()))
        
#         if 'wind_speed' in df.columns:
#             risk_factors.append(df['wind_speed'] / df['wind_speed'].max())
        
#         if 'NDVI' in df.columns:
#             risk_factors.append(1 - df['NDVI'])
        
#         if 'SoilMoisture' in df.columns:
#             risk_factors.append(1 - (df['SoilMoisture'] / 100))
        
#         if risk_factors:
#             df['fire_risk_index'] = np.mean(risk_factors, axis=0)
#         else:
#             df['fire_risk_index'] = 0.5
        
#         return df
    
#     def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
#         """🔄 Complete processing pipeline"""
#         logger.info("🔄 Starting data processing pipeline...")
        
#         # Filter Morocco points
#         df = self.filter_morocco_points(df)
#         logger.info(f"✅ Filtered to {len(df)} points within Morocco")
        
#         # Add temporal features
#         df = self.add_temporal_features(df)
#         logger.info("✅ Added temporal features")
        
#         # Add weather-related features if available
#         weather_cols = ['average_temperature', 'precipitation', 'wind_speed', 
#                        'SoilMoisture', 'NDVI']
#         available_weather = [col for col in weather_cols if col in df.columns]
        
#         if available_weather:
#             df = self.add_lag_features(df, available_weather)
#             df = self.add_rolling_features(df, available_weather)
#             logger.info("✅ Added lag and rolling features")
        
#         # Calculate fire risk index
#         df = self.calculate_fire_risk_index(df)
#         logger.info("✅ Calculated fire risk index")
        
#         # Drop rows with too many NaN values
#         df = df.dropna(thresh=len(df.columns) * 0.5)
        
#         logger.info(f"✅ Processing complete. Final shape: {df.shape}")
#         return df
    
#     def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.parquet"):
#         """💾 Save processed data to parquet"""
#         output_path = config.DIRS['output'] / filename
#         df.to_parquet(output_path, index=False)
#         logger.info(f"✅ Saved processed data to {output_path}")
#         return output_path


# -*- coding: utf-8 -*-
"""data_processor.py

Central hub that glues together FIRMS fire pixels, NOAA weather, environmental
grids (NDVI, soil‑moisture, population, sea‑distance), and calendar features
to produce the *analysis‑ready* dataframe used for modelling.

*Fully backward‑compatible* – all public methods that existed in the original
implementation are preserved, but additional helpers have been added so the
class can now orchestrate the full fourteen‑step pipeline described in
`project-structure.md`.

External dependencies (other in‑repo modules)
--------------------------------------------
* `geoutils.nearest_station`, `geoutils.distance_to_coast`
* `temporal_features.add_lag_features`, `temporal_features.add_rollups`
* `holiday_utils.mark_holidays`
* `spatial_join.nearest_grid_value`
* `augment_data.jitter_fire_points`
* `control_sampler.sample_no_fire`

If any of those modules are missing, the extra methods will raise a friendly
`ImportError` so the legacy subset of functionality keeps working.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (wrapped so legacy users can still use the class)
# ---------------------------------------------------------------------------
try:
    from geoutils import nearest_station, distance_to_coast
    from temporal_features import add_lag_features, add_rollups
    from holiday_utils import mark_holidays
    from spatial_join import nearest_grid_value
    from augment_data import jitter_fire_points
    from control_sampler import sample_no_fire

    _HAS_FULL_STACK = True
except ImportError as e:
    logger.warning(
        "🔸 Some advanced pipeline modules are missing – full feature set will\n"
        "    not be available until they are added: %s",
        e,
    )
    _HAS_FULL_STACK = False


class DataProcessor:
    """🔄 Handles data processing and feature engineering for fire prediction"""

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(self):
        self.morocco_shape: Optional[gpd.GeoDataFrame] = None
        self._load_morocco_shape()

    def _load_morocco_shape(self) -> None:
        """🗺️ Load Morocco boundary shapefile into memory (4326)."""
        shp_path = config.DIRS["shapefiles"] / "ne_10m_admin_0_countries_mar.shp"
        try:
            world = gpd.read_file(shp_path)
            self.morocco_shape = world[world["ADMIN"] == "Morocco"].to_crs(4326)
            logger.info("🌍 Morocco boundary loaded (%d polygon%s)", len(self.morocco_shape), "s" if len(self.morocco_shape) != 1 else "")
        except Exception as exc:  # pragma: no cover – file missing in some dev envs
            logger.warning("⚠️ Could not load Morocco shapefile (%s). Spatial filters disabled.", exc)
            self.morocco_shape = None

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------

    def filter_morocco_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """🗺️ Keep only points lying inside Morocco’s land polygon."""
        if self.morocco_shape is None or df.empty:
            return df.reset_index(drop=True)

        # Build GeoSeries lazily to avoid creating full GeoDataFrame when not necessary.
        pts = gpd.GeoSeries(gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
        morocco_geom = self.morocco_shape.geometry.unary_union
        mask = pts.within(morocco_geom)
        return df.loc[mask].reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Calendar / temporal features
    # ---------------------------------------------------------------------

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """📅 Adds calendar‑based columns (incl. holiday/weekend)."""
        if df.empty:
            return df

        df = df.copy()
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

        df["day_of_week"] = df["acq_date"].dt.dayofweek + 1  # Mon=1 … Sun=7
        df["day_of_year"] = df["acq_date"].dt.dayofyear
        df["month"] = df["acq_date"].dt.month
        df["year"] = df["acq_date"].dt.year
        df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)

        # Public holidays (if advanced stack available)
        if _HAS_FULL_STACK:
            df = mark_holidays(df, country="MA")
        else:
            df["is_holiday"] = 0

        # Season label useful for EDA
        df["season"] = df["month"].map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"})
        return df

    # ---------------------------------------------------------------------
    # Weather lags & roll‑ups – thin wrappers around temporal_features module
    # ---------------------------------------------------------------------

    def add_lag_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        lags: List[int] | range = range(1, 16),
        entity_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Thin wrapper to keep old signature. Delegates to temporal_features."""
        if not _HAS_FULL_STACK:
            logger.warning("⚠️ temporal_features missing – lag features not added.")
            return df

        return add_lag_features(df, cols=cols, lags=lags, entity_cols=entity_cols or ["station_name"])

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        windows: Optional[dict[str, int]] = None,
        entity_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Weekly / monthly / quarterly means & stds."""
        if not _HAS_FULL_STACK:
            logger.warning("⚠️ temporal_features missing – rolling features not added.")
            return df

        if windows is None:
            windows = {"weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365}
        return add_rollups(df, cols=cols, windows=windows, entity_cols=entity_cols or ["station_name"])

    # ---------------------------------------------------------------------
    # Station & environmental joins
    # ---------------------------------------------------------------------

    def attach_nearest_station(self, df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
        """Add columns `station_name`, `station_lat`, `station_lon` to each row."""
        if not _HAS_FULL_STACK:
            logger.warning("⚠️ geoutils missing – cannot attach nearest station.")
            return df
        return df.join(nearest_station(df, stations), how="left")

    def add_sea_distance(self, df: pd.DataFrame, coastline_ls: "LineString") -> pd.DataFrame:
        """Compute `sea_distance` in km for each row."""
        if not _HAS_FULL_STACK:
            logger.warning("⚠️ geoutils missing – sea_distance not computed.")
            df["sea_distance"] = np.nan
            return df

        df = df.copy()
        df["sea_distance"] = [distance_to_coast(lat, lon, coastline_ls) for lat, lon in zip(df.latitude, df.longitude)]
        return df

    def join_environmental_grids(
        self,
        df: pd.DataFrame,
        ndvi_df: Optional[pd.DataFrame] = None,
        soil_df: Optional[pd.DataFrame] = None,
        pop_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Spatially joins NDVI, soil‑moisture, and population density grids."""
        if not _HAS_FULL_STACK:
            logger.warning("⚠️ spatial_join missing – environmental grids not joined.")
            return df

        joined = df.copy()
        if ndvi_df is not None:
            joined = nearest_grid_value(joined, ndvi_df, "NDVI", "NDVI")
        if soil_df is not None:
            joined = nearest_grid_value(joined, soil_df, "Soil_Moisture", "SoilMoisture")
        if pop_df is not None:
            joined = nearest_grid_value(joined, pop_df, "pop_density", "population_density")
        return joined

    # ---------------------------------------------------------------------
    # Fire & control point generators (wrappers)
    # ---------------------------------------------------------------------

    def augment_fire_points(
        self,
        df_fire: pd.DataFrame,
        n_points: int = 100,
        radius_m: int = 300,
    ) -> pd.DataFrame:
        if not _HAS_FULL_STACK:
            logger.warning("⚠️ augment_data missing – fire augmentation skipped.")
            return df_fire
        return jitter_fire_points(df_fire, n=n_points, radius_m=radius_m)

    def generate_non_fire_points(
        self,
        n_points: int,
        date_min: datetime,
        date_max: datetime,
    ) -> pd.DataFrame:
        if not _HAS_FULL_STACK:
            logger.warning("⚠️ control_sampler missing – cannot generate control points.")
            return pd.DataFrame()

        if self.morocco_shape is None:
            raise RuntimeError("Morocco polygon is required to sample control points but could not be loaded.")

        polygon = self.morocco_shape.geometry.unary_union
        return sample_no_fire(n_points, date_min=date_min, date_max=date_max, polygon=polygon)

    # ---------------------------------------------------------------------
    # Simple composite risk index (legacy method kept)
    # ---------------------------------------------------------------------

    def calculate_fire_risk_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔥 Quick‑and‑dirty risk metric for dashboards / sanity checks."""
        df = df.copy()
        risk_terms = []

        if "average_temperature" in df.columns:
            risk_terms.append((df["average_temperature"] - 20) / 20)
        if "precipitation" in df.columns and df["precipitation"].max() > 0:
            risk_terms.append(1 - (df["precipitation"] / df["precipitation"].max()))
        if "wind_speed" in df.columns and df["wind_speed"].max() > 0:
            risk_terms.append(df["wind_speed"] / df["wind_speed"].max())
        if "NDVI" in df.columns:
            risk_terms.append(1 - df["NDVI"])
        if "SoilMoisture" in df.columns and df["SoilMoisture"].max() > 0:
            risk_terms.append(1 - df["SoilMoisture"] / df["SoilMoisture"].max())

        df["fire_risk_index"] = np.mean(risk_terms, axis=0) if risk_terms else 0.5
        return df

    # ---------------------------------------------------------------------
    # Legacy one‑shot processing method (extended but backward compatible)
    # ---------------------------------------------------------------------

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901  (complexity – accepted)
        """High‑level convenience wrapper for notebooks / demos.

        Steps:
        1. Spatial filter to Morocco
        2. Calendar features (+ holidays)
        3. Lag & roll‑up weather statistics (if columns present)
        4. Very rough *fire_risk_index* (for dashboards)
        5. Drop rows with >50 % NaNs.
        """
        logger.info("🔄 Starting *quick* data‑processing pipeline… (%d records)", len(df))

        df = self.filter_morocco_points(df)
        df = self.add_temporal_features(df)

        weather_cols = [c for c in [
            "average_temperature",
            "precipitation",
            "maximum_temperature",
            "minimum_temperature",
            "wind_speed",
            "SoilMoisture",
            "NDVI",
        ] if c in df.columns]

        if weather_cols and _HAS_FULL_STACK:
            df = self.add_lag_features(df, cols=weather_cols)
            df = self.add_rolling_features(df, cols=weather_cols)
        else:
            logger.info("ℹ️ No weather columns or temporal_features missing – skipping lag/roll‑ups.")

        df = self.calculate_fire_risk_index(df)

        # Light NaN cleansing (≥50 % non‑null required)
        df = df.dropna(thresh=int(len(df.columns) * 0.5))
        logger.info("✅ Processing complete – final shape: %s", df.shape)
        return df.reset_index(drop=True)

    # ---------------------------------------------------------------------
    # I/O convenience wrappers
    # ---------------------------------------------------------------------

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.parquet") -> Path:
        """💾 Save dataframe to Parquet inside configured output dir."""
        output_dir = config.DIRS.get("output", Path.cwd())
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        df.to_parquet(path, index=False)
        logger.info("💾 Data saved → %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path
