# """
# 🚀 Morocco Wildfire Data Tool - Gradio UI
# """
# import gradio as gr
# import pandas as pd
# import plotly.express as px
# from pathlib import Path
# import logging
# from datetime import datetime
# import json

# from data_processor import DataProcessor
# from downloader import DataDownloader
# from visualizer import Visualizer
# import config

# # Setup logging
# logging.basicConfig(level=logging.INFO, 
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Initialize components
# downloader = DataDownloader()
# processor = DataProcessor()
# visualizer = Visualizer()

# # Global data storage
# current_data = None

# def load_data(file):
#     """📂 Load data from uploaded file"""
#     global current_data
    
#     if file is None:
#         return "❌ No file uploaded", None, None
    
#     try:
#         if file.name.endswith('.parquet'):
#             current_data = pd.read_parquet(file.name)
#         elif file.name.endswith('.csv'):
#             current_data = pd.read_csv(file.name)
#         else:
#             return "❌ Unsupported file format. Please upload CSV or Parquet.", None, None
        
#         # Generate summary
#         summary = f"""
#         ✅ Data loaded successfully!
        
#         📊 Dataset Information:
#         - Shape: {current_data.shape}
#         - Columns: {len(current_data.columns)}
#         - Date Range: {current_data['acq_date'].min() if 'acq_date' in current_data.columns else 'N/A'} to {current_data['acq_date'].max() if 'acq_date' in current_data.columns else 'N/A'}
#         """
        
#         # Show first few rows
#         preview = current_data.head(10)
        
#         return summary, preview, gr.update(visible=True)
    
#     except Exception as e:
#         return f"❌ Error loading file: {str(e)}", None, gr.update(visible=False)

# def download_sample_data():
#     """📥 Download sample data"""
#     global current_data
    
#     try:
#         current_data = downloader.download_sample_data()
        
#         summary = f"""
#         ✅ Sample data generated!
        
#         📊 Dataset Information:
#         - Shape: {current_data.shape}
#         - Columns: {len(current_data.columns)}
#         """
        
#         return summary, current_data.head(10), gr.update(visible=True)
    
#     except Exception as e:
#         return f"❌ Error: {str(e)}", None, gr.update(visible=False)

# def download_firms_data(start_year, end_year):
#     """🔥 Download FIRMS wildfire data"""
#     global current_data
    
#     try:
#         years = list(range(int(start_year), int(end_year) + 1))
#         current_data = downloader.download_firms_data(years)
        
#         if current_data.empty:
#             return "❌ No data downloaded", None, gr.update(visible=False)
        
#         summary = f"""
#         ✅ FIRMS data downloaded!
        
#         📊 Dataset Information:
#         - Shape: {current_data.shape}
#         - Years: {start_year} - {end_year}
#         - Fire events: {len(current_data)}
#         """
        
#         return summary, current_data.head(10), gr.update(visible=True)
    
#     except Exception as e:
#         return f"❌ Error: {str(e)}", None, gr.update(visible=False)

# def process_data():
#     """🔄 Process the loaded data"""
#     global current_data
    
#     if current_data is None:
#         return "❌ No data loaded", None
    
#     try:
#         # Store original column count
#         original_cols = len(current_data.columns)
        
#         # Process data
#         processed_data = processor.process_dataset(current_data.copy())
#         current_data = processed_data
        
#         summary = f"""
#         ✅ Data processed successfully!
        
#         📊 Processed Dataset:
#         - Shape: {processed_data.shape}
#         - New features added: {len(processed_data.columns) - original_cols}
#         - Missing values: {processed_data.isnull().sum().sum()}
#         """
        
#         return summary, processed_data.head(10)
    
#     except Exception as e:
#         return f"❌ Error: {str(e)}", None

# def create_map_visualization(color_feature):
#     """🗺️ Create map visualization"""
#     if current_data is None:
#         return None
    
#     try:
#         return visualizer.create_map_plot(current_data, color_col=color_feature)
#     except Exception as e:
#         logger.error(f"Map visualization error: {e}")
#         return None

# def create_time_series_visualization(feature):
#     """📈 Create time series visualization"""
#     if current_data is None:
#         return None
    
#     try:
#         return visualizer.create_time_series_plot(current_data, feature)
#     except Exception as e:
#         logger.error(f"Time series error: {e}")
#         return None

# def create_distribution_visualization(feature):
#     """📊 Create distribution visualization"""
#     if current_data is None:
#         return None
    
#     try:
#         return visualizer.create_feature_distribution(current_data, feature)
#     except Exception as e:
#         logger.error(f"Distribution error: {e}")
#         return None

# def create_correlation_visualization(features):
#     """🔥 Create correlation heatmap"""
#     if current_data is None:
#         return None
    
#     try:
#         feature_list = [f.strip() for f in features.split(',')] if features else None
#         return visualizer.create_correlation_heatmap(current_data, feature_list)
#     except Exception as e:
#         logger.error(f"Correlation error: {e}")
#         return None

# def export_data(format_type):
#     """💾 Export processed data"""
#     if current_data is None:
#         return "❌ No data to export"
    
#     try:
#         # Create output directory if it doesn't exist
#         config.DIRS['output'].mkdir(parents=True, exist_ok=True)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         if format_type == "Parquet":
#             filename = f"morocco_wildfire_data_{timestamp}.parquet"
#             path = config.DIRS['output'] / filename
#             current_data.to_parquet(path, index=False)
#         else:  # CSV
#             filename = f"morocco_wildfire_data_{timestamp}.csv"
#             path = config.DIRS['output'] / filename
#             current_data.to_csv(path, index=False)
        
#         return f"✅ Data exported to: {path}"
    
#     except Exception as e:
#         return f"❌ Export error: {str(e)}"

# def get_feature_list():
#     """Get list of available features"""
#     if current_data is None:
#         return []
#     return current_data.columns.tolist()

# def update_feature_dropdowns():
#     """Update feature dropdown choices"""
#     if current_data is None:
#         return gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
    
#     numeric_cols = current_data.select_dtypes(include=['number']).columns.tolist()
#     all_cols = current_data.columns.tolist()
    
#     return (
#         gr.update(choices=all_cols),  # map feature
#         gr.update(choices=numeric_cols),  # time series feature
#         gr.update(choices=numeric_cols)   # distribution feature
#     )

# # 🎨 Create Gradio Interface
# with gr.Blocks(title="🔥 Morocco Wildfire Data Tool", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("""
#     # 🔥 Morocco Wildfire Data Tool
    
#     This tool provides comprehensive data processing and visualization capabilities for Morocco wildfire prediction.
    
#     ## Features:
#     - 📥 Download FIRMS wildfire data
#     - 🔄 Process and integrate multiple data sources
#     - 🗺️ Interactive map visualizations
#     - 📊 Statistical analysis and visualizations
#     - 💾 Export processed datasets
#     """)
    
#     with gr.Tabs():
#         # 📥 Data Loading Tab
#         with gr.Tab("📥 Data Loading"):
#             with gr.Row():
#                 with gr.Column():
#                     gr.Markdown("### Upload Data")
#                     file_upload = gr.File(label="Upload CSV or Parquet file")
#                     upload_btn = gr.Button("📂 Load Data", variant="primary")
                    
#                     gr.Markdown("### Download Data")
#                     sample_btn = gr.Button("📊 Generate Sample Data")
                    
#                     with gr.Row():
#                         start_year = gr.Number(value=2020, label="Start Year", precision=0)
#                         end_year = gr.Number(value=2023, label="End Year", precision=0)
#                     firms_btn = gr.Button("🔥 Download FIRMS Data")
                
#                 with gr.Column():
#                     load_status = gr.Textbox(label="Status", lines=5)
#                     data_preview = gr.Dataframe(label="Data Preview", max_rows=10)
            
#             # Event handlers
#             upload_btn.click(
#                 load_data, 
#                 inputs=[file_upload], 
#                 outputs=[load_status, data_preview]
#             ).then(
#                 update_feature_dropdowns,
#                 outputs=[map_feature, ts_feature, dist_feature]
#             )
            
#             sample_btn.click(
#                 download_sample_data, 
#                 outputs=[load_status, data_preview]
#             ).then(
#                 update_feature_dropdowns,
#                 outputs=[map_feature, ts_feature, dist_feature]
#             )
            
#             firms_btn.click(
#                 download_firms_data, 
#                 inputs=[start_year, end_year], 
#                 outputs=[load_status, data_preview]
#             ).then(
#                 update_feature_dropdowns,
#                 outputs=[map_feature, ts_feature, dist_feature]
#             )
        
#         # 🔄 Data Processing Tab
#         with gr.Tab("🔄 Data Processing"):
#             with gr.Row():
#                 with gr.Column():
#                     process_btn = gr.Button("🔄 Process Data", variant="primary")
#                     process_status = gr.Textbox(label="Processing Status", lines=5)
                
#                 with gr.Column():
#                     processed_preview = gr.Dataframe(label="Processed Data Preview", max_rows=10)
            
#             process_btn.click(
#                 process_data, 
#                 outputs=[process_status, processed_preview]
#             ).then(
#                 update_feature_dropdowns,
#                 outputs=[map_feature, ts_feature, dist_feature]
#             )
        
#         # 🗺️ Map Visualization Tab
#         with gr.Tab("🗺️ Map Visualization"):
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     map_feature = gr.Dropdown(
#                         choices=["is_fire", "NDVI", "average_temperature", "fire_risk_index"],
#                         value="is_fire",
#                         label="Color Feature"
#                     )
#                     map_btn = gr.Button("🗺️ Create Map", variant="primary")
                
#                 with gr.Column(scale=3):
#                     map_plot = gr.Plot(label="Map Visualization")
            
#             map_btn.click(create_map_visualization, inputs=[map_feature], outputs=[map_plot])
        
#         # 📊 Statistical Analysis Tab
#         with gr.Tab("📊 Statistical Analysis"):
#             with gr.Row():
#                 with gr.Column():
#                     gr.Markdown("### Time Series Analysis")
#                     ts_feature = gr.Dropdown(
#                         choices=["average_temperature", "NDVI", "precipitation", "wind_speed"],
#                         value="average_temperature",
#                         label="Feature"
#                     )
#                     ts_btn = gr.Button("📈 Create Time Series")
#                     ts_plot = gr.Plot(label="Time Series")
                
#                 with gr.Column():
#                     gr.Markdown("### Distribution Analysis")
#                     dist_feature = gr.Dropdown(
#                         choices=["average_temperature", "NDVI", "precipitation", "wind_speed"],
#                         value="NDVI",
#                         label="Feature"
#                     )
#                     dist_btn = gr.Button("📊 Create Distribution")
#                     dist_plot = gr.Plot(label="Distribution")
            
#             gr.Markdown("### Correlation Analysis")
#             corr_features = gr.Textbox(
#                 label="Features (comma-separated, leave empty for all)",
#                 placeholder="NDVI, average_temperature, precipitation"
#             )
#             corr_btn = gr.Button("🔥 Create Correlation Heatmap")
#             corr_plot = gr.Plot(label="Correlation Heatmap")
            
#             # Event handlers
#             ts_btn.click(create_time_series_visualization, inputs=[ts_feature], outputs=[ts_plot])
#             dist_btn.click(create_distribution_visualization, inputs=[dist_feature], outputs=[dist_plot])
#             corr_btn.click(create_correlation_visualization, inputs=[corr_features], outputs=[corr_plot])
        
#         # 💾 Export Tab
#         with gr.Tab("💾 Export Data"):
#             with gr.Row():
#                 with gr.Column():
#                     export_format = gr.Radio(
#                         choices=["Parquet", "CSV"],
#                         value="Parquet",
#                         label="Export Format"
#                     )
#                     export_btn = gr.Button("💾 Export Data", variant="primary")
#                     export_status = gr.Textbox(label="Export Status", lines=3)
            
#             export_btn.click(export_data, inputs=[export_format], outputs=[export_status])
    
#     gr.Markdown("""
#     ---
#     ### 📚 Documentation
    
#     **Data Sources:**
#     - 🔥 FIRMS (Fire Information for Resource Management System)
#     - 🌍 Natural Earth shapefiles
#     - 🌿 NDVI data from NASA MODIS
#     - 🌡️ Weather data from NOAA
    
#     **Processing Pipeline:**
#     1. Load raw data from various sources
#     2. Filter points within Morocco boundaries
#     3. Add temporal features (day of week, season, etc.)
#     4. Add lag features for weather variables
#     5. Calculate fire risk index
#     6. Export processed dataset
    
#     **Output Format:**
#     - 272 features including temporal aggregates, lag variables, and environmental indicators
#     - Balanced dataset with fire and non-fire events
#     """)

# # Launch the app
# if __name__ == "__main__":
#     demo.launch(share=True, debug=True)

import gradio as gr
import pandas as pd
import plotly.express as px
from pathlib import Path
import logging
from datetime import datetime
import json

from data_processor import DataProcessor
from downloader import DataDownloader
from visualizer import Visualizer
import config

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize components
downloader = DataDownloader()
processor = DataProcessor()
visualizer = Visualizer()

# Global data storage
current_data = None

def load_data(file):
    """📂 Load data from uploaded file"""
    global current_data
    
    if file is None:
        return "❌ No file uploaded", None, None
    
    try:
        if file.name.endswith('.parquet'):
            current_data = pd.read_parquet(file.name)
        elif file.name.endswith('.csv'):
            current_data = pd.read_csv(file.name)
        else:
            return "❌ Unsupported file format. Please upload CSV or Parquet.", None, None
        
        summary = f"""
        ✅ Data loaded successfully!
        
        📊 Dataset Information:
        - Shape: {current_data.shape}
        - Columns: {len(current_data.columns)}
        - Date Range: {current_data['acq_date'].min() if 'acq_date' in current_data.columns else 'N/A'} to {current_data['acq_date'].max() if 'acq_date' in current_data.columns else 'N/A'}
        """
        
        preview = current_data.head(10)
        return summary, preview, gr.update(visible=True)
    
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", None, gr.update(visible=False)

def download_sample_data():
    """📥 Download sample data"""
    global current_data
    try:
        current_data = downloader.download_sample_data()
        summary = f"""
        ✅ Sample data generated!
        
        📊 Dataset Information:
        - Shape: {current_data.shape}
        - Columns: {len(current_data.columns)}
        """
        return summary, current_data.head(10), gr.update(visible=True)
    except Exception as e:
        return f"❌ Error: {str(e)}", None, gr.update(visible=False)

def download_firms_data(start_year, end_year):
    """🔥 Download FIRMS wildfire data"""
    global current_data
    try:
        years = list(range(int(start_year), int(end_year) + 1))
        current_data = downloader.download_firms_data(years)
        if current_data.empty:
            return "❌ No data downloaded", None, gr.update(visible=False)
        summary = f"""
        ✅ FIRMS data downloaded!
        
        📊 Dataset Information:
        - Shape: {current_data.shape}
        - Years: {start_year} - {end_year}
        - Fire events: {len(current_data)}
        """
        return summary, current_data.head(10), gr.update(visible=True)
    except Exception as e:
        return f"❌ Error: {str(e)}", None, gr.update(visible=False)

def process_data():
    """🔄 Process the loaded data"""
    global current_data
    if current_data is None:
        return "❌ No data loaded", None
    try:
        original_cols = len(current_data.columns)
        processed_data = processor.process_dataset(current_data.copy())
        current_data = processed_data
        summary = f"""
        ✅ Data processed successfully!
        
        📊 Processed Dataset:
        - Shape: {processed_data.shape}
        - New features added: {len(processed_data.columns) - original_cols}
        - Missing values: {processed_data.isnull().sum().sum()}
        """
        return summary, processed_data.head(10)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def create_map_visualization(color_feature):
    if current_data is None:
        return None
    try:
        return visualizer.create_map_plot(current_data, color_col=color_feature)
    except Exception as e:
        logger.error(f"Map visualization error: {e}")
        return None

def create_time_series_visualization(feature):
    if current_data is None:
        return None
    try:
        return visualizer.create_time_series_plot(current_data, feature)
    except Exception as e:
        logger.error(f"Time series error: {e}")
        return None

def create_distribution_visualization(feature):
    if current_data is None:
        return None
    try:
        return visualizer.create_feature_distribution(current_data, feature)
    except Exception as e:
        logger.error(f"Distribution error: {e}")
        return None

def create_correlation_visualization(features):
    if current_data is None:
        return None
    try:
        feature_list = [f.strip() for f in features.split(',')] if features else None
        return visualizer.create_correlation_heatmap(current_data, feature_list)
    except Exception as e:
        logger.error(f"Correlation error: {e}")
        return None

def export_data(format_type):
    if current_data is None:
        return "❌ No data to export"
    try:
        config.DIRS['output'].mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format_type == "Parquet":
            path = config.DIRS['output'] / f"morocco_wildfire_data_{timestamp}.parquet"
            current_data.to_parquet(path, index=False)
        else:
            path = config.DIRS['output'] / f"morocco_wildfire_data_{timestamp}.csv"
            current_data.to_csv(path, index=False)
        return f"✅ Data exported to: {path}"
    except Exception as e:
        return f"❌ Export error: {str(e)}"

def update_feature_dropdowns():
    if current_data is None:
        return gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
    numeric_cols = current_data.select_dtypes(include=['number']).columns.tolist()
    all_cols = current_data.columns.tolist()
    return (
        gr.update(choices=all_cols),
        gr.update(choices=numeric_cols),
        gr.update(choices=numeric_cols)
    )

with gr.Blocks(title="🔥 Morocco Wildfire Data Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# 🔥 Morocco Wildfire Data Tool
    Provides tools for analyzing Morocco wildfire data.""")

    with gr.Tabs():
        with gr.Tab("📥 Data Loading"):
            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(label="Upload CSV or Parquet file")
                    upload_btn = gr.Button("📂 Load Data")
                    sample_btn = gr.Button("📊 Generate Sample Data")
                    start_year = gr.Number(value=2020, label="Start Year", precision=0)
                    end_year = gr.Number(value=2023, label="End Year", precision=0)
                    firms_btn = gr.Button("🔥 Download FIRMS Data")
                with gr.Column():
                    load_status = gr.Textbox(label="Status", lines=5)
                    data_preview = gr.Dataframe(label="Data Preview", max_height=300)

        with gr.Tab("🔄 Data Processing"):
            process_btn = gr.Button("🔄 Process Data")
            process_status = gr.Textbox(label="Processing Status", lines=5)
            processed_preview = gr.Dataframe(label="Processed Data Preview", max_height=300)

        with gr.Tab("🗺️ Map Visualization"):
            map_feature = gr.Dropdown(choices=[], label="Color Feature")
            map_btn = gr.Button("🗺️ Create Map")
            map_plot = gr.Plot(label="Map")

        with gr.Tab("📊 Statistical Analysis"):
            ts_feature = gr.Dropdown(choices=[], label="Time Series Feature")
            ts_btn = gr.Button("📈 Create Time Series")
            ts_plot = gr.Plot()
            dist_feature = gr.Dropdown(choices=[], label="Distribution Feature")
            dist_btn = gr.Button("📊 Create Distribution")
            dist_plot = gr.Plot()
            corr_features = gr.Textbox(label="Correlation Features (comma-separated)")
            corr_btn = gr.Button("🔥 Create Correlation Heatmap")
            corr_plot = gr.Plot()

        with gr.Tab("💾 Export"):
            export_format = gr.Radio(choices=["Parquet", "CSV"], value="Parquet", label="Format")
            export_btn = gr.Button("💾 Export Data")
            export_status = gr.Textbox(label="Export Status", lines=2)

    upload_btn.click(load_data, inputs=[file_upload], outputs=[load_status, data_preview])\
        .then(update_feature_dropdowns, outputs=[map_feature, ts_feature, dist_feature])
    sample_btn.click(download_sample_data, outputs=[load_status, data_preview])\
        .then(update_feature_dropdowns, outputs=[map_feature, ts_feature, dist_feature])
    firms_btn.click(download_firms_data, inputs=[start_year, end_year], outputs=[load_status, data_preview])\
        .then(update_feature_dropdowns, outputs=[map_feature, ts_feature, dist_feature])
    process_btn.click(process_data, outputs=[process_status, processed_preview])\
        .then(update_feature_dropdowns, outputs=[map_feature, ts_feature, dist_feature])
    map_btn.click(create_map_visualization, inputs=[map_feature], outputs=[map_plot])
    ts_btn.click(create_time_series_visualization, inputs=[ts_feature], outputs=[ts_plot])
    dist_btn.click(create_distribution_visualization, inputs=[dist_feature], outputs=[dist_plot])
    corr_btn.click(create_correlation_visualization, inputs=[corr_features], outputs=[corr_plot])
    export_btn.click(export_data, inputs=[export_format], outputs=[export_status])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
