import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
from typing import List, Dict, Tuple
import threading

# Simulated data for Morocco regions
MOROCCO_REGIONS = {
    "Tanger-Tétouan-Al Hoceïma": {"lat": 35.7595, "lon": -5.8340},
    "Oriental": {"lat": 34.6814, "lon": -1.9086},
    "Fès-Meknès": {"lat": 34.0331, "lon": -5.0003},
    "Rabat-Salé-Kénitra": {"lat": 34.0209, "lon": -6.8416},
    "Béni Mellal-Khénifra": {"lat": 32.3373, "lon": -6.3498},
    "Casablanca-Settat": {"lat": 33.5731, "lon": -7.5898},
    "Marrakech-Safi": {"lat": 31.6295, "lon": -7.9811},
    "Drâa-Tafilalet": {"lat": 31.9314, "lon": -4.4343},
    "Souss-Massa": {"lat": 30.4278, "lon": -9.5981},
    "Guelmim-Oued Noun": {"lat": 28.9870, "lon": -10.0574},
    "Laâyoune-Sakia El Hamra": {"lat": 27.1536, "lon": -13.2033},
    "Dakhla-Oued Ed-Dahab": {"lat": 23.6847, "lon": -15.9580}
}

# Available models
MODELS = {
    "🔥 XGBoost Fire Predictor v2.1": "xgboost_v2.1",
    "🌲 Random Forest Ensemble v1.8": "rf_ensemble_v1.8",
    "🧠 Neural Network Deep Fire v3.0": "nn_deepfire_v3.0",
    "🚀 LightGBM Fast Predictor v1.5": "lightgbm_v1.5",
    "🎯 Ensemble Meta-Model v2.0": "ensemble_meta_v2.0"
}

class MockDataStreamer:
    """Simulate real-time data streaming"""
    def __init__(self):
        self.is_streaming = False
        self.data_buffer = []
        self.fire_predictions = []
        
    def start_streaming(self):
        """Start simulated data streaming"""
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_data)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
    def stop_streaming(self):
        """Stop data streaming"""
        self.is_streaming = False
        
    def _stream_data(self):
        """Simulate continuous data updates"""
        while self.is_streaming:
            # Generate mock sensor data
            new_data = {
                "timestamp": datetime.now(),
                "weather_stations": random.randint(45, 52),
                "ndvi_coverage": random.uniform(0.75, 0.95),
                "active_fires": random.randint(0, 5),
                "soil_moisture": random.uniform(0.15, 0.45)
            }
            self.data_buffer.append(new_data)
            
            # Generate fire risk predictions
            self._generate_predictions()
            
            time.sleep(2)  # Update every 2 seconds
            
    def _generate_predictions(self):
        """Generate mock fire risk predictions"""
        predictions = []
        for region, coords in MOROCCO_REGIONS.items():
            # Add some randomness around region centers
            for i in range(random.randint(1, 3)):
                lat = coords["lat"] + random.uniform(-0.5, 0.5)
                lon = coords["lon"] + random.uniform(-0.5, 0.5)
                risk = random.uniform(0.1, 0.95)
                
                predictions.append({
                    "latitude": lat,
                    "longitude": lon,
                    "region": region,
                    "risk_probability": risk,
                    "temperature": random.uniform(25, 45),
                    "humidity": random.uniform(10, 60),
                    "wind_speed": random.uniform(5, 30),
                    "ndvi": random.uniform(0.2, 0.8)
                })
        
        # Sort by risk and keep top predictions
        self.fire_predictions = sorted(predictions, key=lambda x: x["risk_probability"], reverse=True)[:50]

# Initialize data streamer
data_streamer = MockDataStreamer()

def download_historical_data(region, start_date_str, end_date_str):
    """Simulate downloading historical data"""
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except:
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
    
    # Simulate download progress
    status_md = "### 📥 Downloading Historical Data\n\n"
    status_md += f"**Region:** {region}\n"
    status_md += f"**Period:** {start_date} to {end_date}\n\n"
    status_md += "**Progress:**\n"
    
    steps = [
        "🌡️ Fetching weather station data...",
        "🛰️ Downloading MODIS/VIIRS fire data...",
        "🌿 Retrieving NDVI composites...",
        "💧 Getting soil moisture data...",
        "🏔️ Loading terrain features...",
        "✅ Data download complete!"
    ]
    
    progress_text = ""
    for i, step in enumerate(steps):
        progress_text += f"{step}\n"
        yield status_md + progress_text, None, None
        time.sleep(0.5)
    
    # Generate mock historical data
    num_records = random.randint(10000, 50000)
    
    final_status = f"### ✅ Download Complete!\n\n"
    final_status += f"**Total Records:** {num_records:,}\n"
    final_status += f"**Time Period:** {(end_date - start_date).days} days\n"
    final_status += f"**File Size:** {random.randint(50, 200)} MB\n\n"
    final_status += "Data saved to: `/data/historical/morocco_wildfire_2010_2023.parquet`"
    
    return final_status, None, None

def toggle_realtime_monitoring(enable, model, region, sparse_points):
    """Toggle real-time monitoring"""
    if enable:
        data_streamer.start_streaming()
        status = "### 🟢 Real-Time Monitoring Active\n\n"
        status += f"**Model:** {model}\n"
        status += f"**Region:** {region}\n"
        status += f"**Sparse Sampling:** {'Enabled' if sparse_points else 'Disabled'}\n\n"
        status += "**Data Streams:**\n"
        status += "- 🌡️ Weather stations: **CONNECTED**\n"
        status += "- 🛰️ FIRMS satellite: **CONNECTED**\n"
        status += "- 🌿 NDVI service: **CONNECTED**\n"
        status += "- 💧 Soil moisture: **CONNECTED**\n"
        
        return status, gr.update(interactive=False), gr.update(interactive=True)
    else:
        data_streamer.stop_streaming()
        status = "### 🔴 Real-Time Monitoring Stopped\n\n"
        status += "All data streams disconnected."
        
        return status, gr.update(interactive=True), gr.update(interactive=False)

def update_predictions():
    """Update fire risk predictions"""
    if not data_streamer.is_streaming:
        return None, None, "⚠️ Real-time monitoring is not active"
    
    # Get latest predictions
    predictions = data_streamer.fire_predictions[:10]  # Top 10
    
    if not predictions:
        return None, None, "Waiting for data..."
    
    # Create map
    fig = go.Figure()
    
    # Add Morocco regions as background
    for region, coords in MOROCCO_REGIONS.items():
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[coords["lon"]],
            lat=[coords["lat"]],
            marker=dict(size=8, color="lightgray"),
            text=region,
            name="Regions",
            showlegend=False
        ))
    
    # Add predictions with color scale
    lats = [p["latitude"] for p in predictions]
    lons = [p["longitude"] for p in predictions]
    risks = [p["risk_probability"] for p in predictions]
    
    # Create hover text
    hover_texts = []
    for p in predictions:
        text = f"<b>Risk: {p['risk_probability']:.1%}</b><br>"
        text += f"Region: {p['region']}<br>"
        text += f"Temp: {p['temperature']:.1f}°C<br>"
        text += f"Humidity: {p['humidity']:.1f}%<br>"
        text += f"Wind: {p['wind_speed']:.1f} km/h"
        hover_texts.append(text)
    
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=lons,
        lat=lats,
        marker=dict(
            size=[r * 30 + 10 for r in risks],  # Size based on risk
            color=risks,
            colorscale="Hot",
            cmin=0,
            cmax=1,
            colorbar=dict(title="Fire Risk"),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name="Fire Risk"
    ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=31.7917, lon=-7.0926),  # Morocco center
            zoom=5
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Create dataframe for table
    df_data = []
    for i, p in enumerate(predictions):
        df_data.append({
            "🏆 Rank": i + 1,
            "📍 Location": f"{p['latitude']:.4f}, {p['longitude']:.4f}",
            "🌍 Region": p['region'],
            "🔥 Risk": f"{p['risk_probability']:.1%}",
            "🌡️ Temp": f"{p['temperature']:.1f}°C",
            "💧 Humidity": f"{p['humidity']:.1f}%",
            "💨 Wind": f"{p['wind_speed']:.1f} km/h",
            "🌿 NDVI": f"{p['ndvi']:.2f}"
        })
    
    df = pd.DataFrame(df_data)
    
    # Status update
    status = f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return fig, df, status

# Create Gradio interface
with gr.Blocks(title="🔥 Morocco Wildfire NRT Monitor", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🔥 Morocco Wildfire Near Real-Time Monitoring System
    
    Monitor and predict wildfire risks across Morocco using real-time satellite and weather data.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Configuration")
            
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="🤖 Select Model",
                interactive=True
            )
            
            region_selector = gr.Dropdown(
                choices=["All Morocco"] + list(MOROCCO_REGIONS.keys()),
                value="All Morocco",
                label="🌍 Select Region",
                interactive=True
            )
            
            sparse_sampling = gr.Checkbox(
                label="📍 Enable Sparse Point Sampling",
                value=True,
                info="Reduce computation by sampling sparse points"
            )
            
            gr.Markdown("### 📅 Historical Data")
            
            with gr.Row():
                start_date = gr.Textbox(
                    label="Start Date",
                    value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                    placeholder="YYYY-MM-DD"
                )
                end_date = gr.Textbox(
                    label="End Date",
                    value=datetime.now().strftime("%Y-%m-%d"),
                    placeholder="YYYY-MM-DD"
                )
            
            download_btn = gr.Button("📥 Download Historical Data", variant="primary")
            
            gr.Markdown("### 🔴 Real-Time Monitoring")
            
            realtime_btn = gr.Button("▶️ Start Real-Time Monitoring", variant="primary")
            stop_btn = gr.Button("⏹️ Stop Monitoring", variant="stop", interactive=False)
            
        with gr.Column(scale=2):
            status_display = gr.Markdown("### 📡 System Status\n\nSystem idle. Click 'Start Real-Time Monitoring' to begin.")
            
            with gr.Tab("🗺️ Risk Map"):
                risk_map = gr.Plot(label="Fire Risk Predictions")
                
            with gr.Tab("📊 Risk Table"):
                risk_table = gr.DataFrame(
                    label="Top 10 High-Risk Locations",
                    interactive=False
                )
            
            update_status = gr.Markdown("⏳ Waiting to start...")
    
    # Auto-refresh for real-time updates
    timer = gr.Timer(3.0, active=False)
    
    # Event handlers
    download_btn.click(
        fn=download_historical_data,
        inputs=[region_selector, start_date, end_date],
        outputs=[status_display, risk_map, risk_table]
    )
    
    realtime_btn.click(
        fn=lambda m, r, s: toggle_realtime_monitoring(True, m, r, s),
        inputs=[model_selector, region_selector, sparse_sampling],
        outputs=[status_display, realtime_btn, stop_btn]
    ).then(
        lambda: gr.update(active=True),
        outputs=[timer]
    )
    
    stop_btn.click(
        fn=lambda m, r, s: toggle_realtime_monitoring(False, m, r, s),
        inputs=[model_selector, region_selector, sparse_sampling],
        outputs=[status_display, realtime_btn, stop_btn]
    ).then(
        lambda: gr.update(active=False),
        outputs=[timer]
    )
    
    timer.tick(
        fn=update_predictions,
        outputs=[risk_map, risk_table, update_status]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)