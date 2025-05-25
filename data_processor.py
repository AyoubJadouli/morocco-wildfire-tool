"""
🔄 Data Processing and Integration
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Optional, Dict, List
import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)

class DataProcessor:
    """🔄 Handles data processing and feature engineering"""
    
    def __init__(self):
        self.morocco_shape = None
        self._load_morocco_shape()
    
    def _load_morocco_shape(self):
        """🗺️ Load Morocco boundary shapefile"""
        try:
            shp_path = config.DIRS['shapefiles'] / 'ne_10m_admin_0_countries_mar.shp'
            if shp_path.exists():
                world = gpd.read_file(shp_path)
                self.morocco_shape = world[world['ADMIN'] == 'Morocco']
        except Exception as e:
            logger.warning(f"⚠️ Could not load Morocco shapefile: {e}")
    
    def filter_morocco_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """🗺️ Filter points within Morocco boundaries"""
        if self.morocco_shape is None:
            logger.warning("⚠️ Morocco shape not loaded, returning all points")
            return df
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude)
        )
        
        # Filter points within Morocco
        morocco_geom = self.morocco_shape.geometry.iloc[0]
        mask = gdf.geometry.within(morocco_geom)
        
        return df[mask].reset_index(drop=True)
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """📅 Add temporal features"""
        df = df.copy()
        
        # Ensure datetime
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        
        # Add temporal features
        df['day_of_week'] = df['acq_date'].dt.dayofweek + 1
        df['day_of_year'] = df['acq_date'].dt.dayofyear
        df['month'] = df['acq_date'].dt.month
        df['year'] = df['acq_date'].dt.year
        df['is_weekend'] = df['acq_date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Season
        df['season'] = df['month'].apply(lambda x: 
            'winter' if x in [12, 1, 2] else
            'spring' if x in [3, 4, 5] else
            'summer' if x in [6, 7, 8] else 'fall'
        )
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, 
                        weather_cols: List[str], 
                        lags: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
        """🔄 Add lagged weather features"""
        df = df.copy()
        
        # Sort by date
        df = df.sort_values('acq_date')
        
        # Add lag features
        for col in weather_cols:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df.groupby(['latitude', 'longitude'])[col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, 
                           weather_cols: List[str],
                           windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """📊 Add rolling statistics"""
        df = df.copy()
        
        for col in weather_cols:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = (
                        df.groupby(['latitude', 'longitude'])[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    )
                    df[f'{col}_rolling_std_{window}'] = (
                        df.groupby(['latitude', 'longitude'])[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).std())
                    )
        
        return df
    
    def calculate_fire_risk_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔥 Calculate fire risk index"""
        df = df.copy()
        
        # Simple fire risk index based on available features
        risk_factors = []
        
        if 'average_temperature' in df.columns:
            risk_factors.append((df['average_temperature'] - 20) / 20)
        
        if 'precipitation' in df.columns:
            risk_factors.append(1 - (df['precipitation'] / df['precipitation'].max()))
        
        if 'wind_speed' in df.columns:
            risk_factors.append(df['wind_speed'] / df['wind_speed'].max())
        
        if 'NDVI' in df.columns:
            risk_factors.append(1 - df['NDVI'])
        
        if 'SoilMoisture' in df.columns:
            risk_factors.append(1 - (df['SoilMoisture'] / 100))
        
        if risk_factors:
            df['fire_risk_index'] = np.mean(risk_factors, axis=0)
        else:
            df['fire_risk_index'] = 0.5
        
        return df
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔄 Complete processing pipeline"""
        logger.info("🔄 Starting data processing pipeline...")
        
        # Filter Morocco points
        df = self.filter_morocco_points(df)
        logger.info(f"✅ Filtered to {len(df)} points within Morocco")
        
        # Add temporal features
        df = self.add_temporal_features(df)
        logger.info("✅ Added temporal features")
        
        # Add weather-related features if available
        weather_cols = ['average_temperature', 'precipitation', 'wind_speed', 
                       'SoilMoisture', 'NDVI']
        available_weather = [col for col in weather_cols if col in df.columns]
        
        if available_weather:
            df = self.add_lag_features(df, available_weather)
            df = self.add_rolling_features(df, available_weather)
            logger.info("✅ Added lag and rolling features")
        
        # Calculate fire risk index
        df = self.calculate_fire_risk_index(df)
        logger.info("✅ Calculated fire risk index")
        
        # Drop rows with too many NaN values
        df = df.dropna(thresh=len(df.columns) * 0.5)
        
        logger.info(f"✅ Processing complete. Final shape: {df.shape}")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.parquet"):
        """💾 Save processed data to parquet"""
        output_path = config.DIRS['output'] / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ Saved processed data to {output_path}")
        return output_path