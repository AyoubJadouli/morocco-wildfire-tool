"""
📊 Visualization Components
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import numpy as np
from typing import Optional, Dict, Any
import config

class Visualizer:
    """📊 Handles all visualization tasks"""
    
    def __init__(self):
        self.default_mapbox_style = config.MAP_STYLE
        self.colorscales = config.COLORSCALES
    
    def create_map_plot(self, df: pd.DataFrame, 
                       color_col: Optional[str] = None,
                       title: str = "Morocco Wildfire Data") -> go.Figure:
        """🗺️ Create interactive map plot"""
        
        # Default color column
        if color_col is None:
            color_col = 'is_fire' if 'is_fire' in df.columns else None
        
        # Create base map
        if color_col:
            fig = px.scatter_mapbox(
                df,
                lat='latitude',
                lon='longitude',
                color=color_col,
                title=title,
                mapbox_style=self.default_mapbox_style,
                zoom=5,
                center={"lat": 31.7917, "lon": -7.0926},  # Morocco center
                height=600,
                color_continuous_scale=self.colorscales.get(color_col, 'Viridis')
            )
        else:
            fig = px.scatter_mapbox(
                df,
                lat='latitude',
                lon='longitude',
                title=title,
                mapbox_style=self.default_mapbox_style,
                zoom=5,
                center={"lat": 31.7917, "lon": -7.0926},
                height=600
            )
        
        # Add Morocco boundary
        fig.update_layout(
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            showlegend=True
        )
        
        return fig
    
    def create_time_series_plot(self, df: pd.DataFrame, 
                               value_col: str,
                               title: Optional[str] = None) -> go.Figure:
        """📈 Create time series plot"""
        if 'acq_date' not in df.columns:
            return go.Figure().add_annotation(
                text="No date column found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Aggregate by date
        df_agg = df.groupby('acq_date')[value_col].agg(['mean', 'std']).reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=df_agg['acq_date'],
            y=df_agg['mean'],
            mode='lines',
            name='Mean',
            line=dict(color='blue', width=2)
        ))
        
        # Add confidence interval
        if 'std' in df_agg.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df_agg['acq_date'], df_agg['acq_date'][::-1]]),
                y=pd.concat([df_agg['mean'] + df_agg['std'], 
                           (df_agg['mean'] - df_agg['std'])[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
        
        fig.update_layout(
            title=title or f"{value_col} over Time",
            xaxis_title="Date",
            yaxis_title=value_col,
            height=400
        )
        
        return fig
    
    def create_feature_distribution(self, df: pd.DataFrame, 
                                  feature: str) -> go.Figure:
        """📊 Create feature distribution plot"""
        if feature not in df.columns:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Distribution", "Box Plot"),
            specs=[[{"type": "histogram"}, {"type": "box"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df[feature], name="Distribution", nbinsx=30),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df[feature], name="Box Plot"),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Distribution of {feature}",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                 features: Optional[list] = None) -> go.Figure:
        """🔥 Create correlation heatmap"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if features:
            numeric_cols = [col for col in features if col in numeric_cols]
        
        if len(numeric_cols) < 2:
            return go.Figure()
        
        # Calculate correlation
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            height=600,
            width=800
        )
        
        return fig
    
    def create_fire_risk_map(self, df: pd.DataFrame) -> folium.Map:
        """🔥 Create fire risk heatmap using Folium"""
        # Create base map
        m = folium.Map(
            location=[31.7917, -7.0926],  # Morocco center
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add heatmap layer
        if 'fire_risk_index' in df.columns:
            heat_data = [[row['latitude'], row['longitude'], row['fire_risk_index']] 
                        for idx, row in df.iterrows()]
            
            plugins.HeatMap(heat_data).add_to(m)
        
        # Add fire points if available
        if 'is_fire' in df.columns:
            fire_points = df[df['is_fire'] == 1]
            for idx, row in fire_points.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='red',
                    fill=True,
                    fillColor='red'
                ).add_to(m)
        
        return m
    
    def create_summary_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """📊 Create summary statistics dashboard"""
        summary = {
            'total_records': len(df),
            'date_range': f"{df['acq_date'].min()} to {df['acq_date'].max()}" if 'acq_date' in df.columns else "N/A",
            'fire_events': df['is_fire'].sum() if 'is_fire' in df.columns else 0,
            'avg_temperature': df['average_temperature'].mean() if 'average_temperature' in df.columns else None,
            'avg_ndvi': df['NDVI'].mean() if 'NDVI' in df.columns else None,
            'features': list(df.columns),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary