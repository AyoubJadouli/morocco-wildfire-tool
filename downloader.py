"""
📥 Data Downloading Utilities
"""
import requests
import zipfile
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging
from tqdm import tqdm
import config

logger = logging.getLogger(__name__)

class DataDownloader:
    """📥 Handles downloading data from various sources"""
    
    def __init__(self):
        config.setup_directories()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_file(self, url: str, output_path: Path, 
                     chunk_size: int = 8192) -> bool:
        """📥 Download file with progress bar"""
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def download_shapefiles(self) -> bool:
        """🗺️ Download geographic shapefiles"""
        for name, url in [
            ('coastline', config.DATA_SOURCES['coastline']),
            ('countries', config.DATA_SOURCES['countries'])
        ]:
            output_file = config.DIRS['shapefiles'] / f"{name}.zip"
            
            if output_file.exists():
                logger.info(f"✅ {name} already downloaded")
                continue
            
            logger.info(f"📥 Downloading {name}...")
            if self.download_file(url, output_file):
                # Extract zip
                with zipfile.ZipFile(output_file, 'r') as zip_ref:
                    zip_ref.extractall(config.DIRS['shapefiles'])
                output_file.unlink()  # Remove zip
                logger.info(f"✅ {name} extracted")
        
        return True
    
    def download_firms_data(self, years: List[int], 
                          country: str = "Morocco") -> pd.DataFrame:
        """🔥 Download FIRMS wildfire data"""
        all_data = []
        
        for year in years:
            for data_type in ['modis', 'viirs-snpp']:
                url = f"{config.DATA_SOURCES['firms_base']}{data_type}/{year}/{data_type}_{year}_{country}.csv"
                output_file = config.DIRS['raw'] / f"{data_type}_{year}_{country}.csv"
                
                if output_file.exists():
                    logger.info(f"✅ Loading cached {data_type} {year}")
                    df = pd.read_csv(output_file)
                else:
                    logger.info(f"📥 Downloading {data_type} {year}")
                    if self.download_file(url, output_file):
                        df = pd.read_csv(output_file)
                    else:
                        continue
                
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        return pd.DataFrame()
    
    def download_sample_data(self) -> pd.DataFrame:
        """📊 Generate sample data for testing"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample data
        n_samples = 1000
        dates = [datetime(2020, 1, 1) + timedelta(days=x) 
                for x in np.random.randint(0, 1095, n_samples)]
        
        data = {
            'acq_date': dates,
            'latitude': np.random.uniform(config.MOROCCO_BBOX['min_lat'], 
                                        config.MOROCCO_BBOX['max_lat'], n_samples),
            'longitude': np.random.uniform(config.MOROCCO_BBOX['min_lon'], 
                                         config.MOROCCO_BBOX['max_lon'], n_samples),
            'NDVI': np.random.uniform(0, 1, n_samples),
            'SoilMoisture': np.random.uniform(0, 100, n_samples),
            'average_temperature': np.random.uniform(10, 40, n_samples),
            'precipitation': np.random.exponential(5, n_samples),
            'wind_speed': np.random.uniform(0, 30, n_samples),
            'is_fire': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        }
        
        return pd.DataFrame(data)