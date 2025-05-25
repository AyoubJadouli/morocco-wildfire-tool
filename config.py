"""
⚙️ Configuration and Constants for Morocco Wildfire Tool
"""
import os
from pathlib import Path

# 🗺️ Morocco Bounding Box
MOROCCO_BBOX = {
    'min_lon': -17.10464433,
    'min_lat': 20.76691315,
    'max_lon': -1.03199947,
    'max_lat': 35.92651915
}

# 📁 Directory Structure
BASE_DIR = Path.cwd() / "data"
DIRS = {
    'raw': BASE_DIR / 'raw',
    'processed': BASE_DIR / 'processed',
    'output': BASE_DIR / 'output',
    'cache': BASE_DIR / 'cache',
    'shapefiles': BASE_DIR / 'shapefiles'
}

# 🌐 Data Sources
DATA_SOURCES = {
    'coastline': "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip",
    'countries': "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries_mar.zip",
    'firms_base': "https://firms.modaps.eosdis.nasa.gov/data/country/",
}

# 📊 Dataset Features
FEATURE_COLUMNS = [
    'acq_date', 'latitude', 'longitude', 'is_holiday', 'day_of_week',
    'day_of_year', 'is_weekend', 'NDVI', 'SoilMoisture', 'sea_distance',
    'station_lat', 'station_lon', 'average_temperature', 'maximum_temperature',
    'minimum_temperature', 'precipitation', 'wind_speed', 'dew_point',
    'is_fire'
]

# 🎨 Visualization Settings
MAP_STYLE = 'open-street-map'
COLORSCALES = {
    'fire': 'Reds',
    'ndvi': 'Greens',
    'population': 'Viridis',
    'temperature': 'RdBu'
}

def setup_directories():
    """Create required directories"""
    for dir_path in DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)