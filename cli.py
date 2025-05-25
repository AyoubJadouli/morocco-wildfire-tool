"""
🖥️ Command Line Interface for Morocco Wildfire Tool
"""
import click
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from data_processor import DataProcessor
from downloader import DataDownloader
from visualizer import Visualizer
import config

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """🔥 Morocco Wildfire Data Tool CLI"""
    config.setup_directories()

@cli.command()
@click.option('--years', '-y', multiple=True, type=int, 
              help='Years to download (e.g., -y 2020 -y 2021)')
@click.option('--country', '-c', default='Morocco', 
              help='Country name for FIRMS data')
def download(years, country):
    """📥 Download FIRMS wildfire data"""
    downloader = DataDownloader()
    
    # Download shapefiles first
    click.echo("📥 Downloading shapefiles...")
    downloader.download_shapefiles()
    
    # Download FIRMS data
    if years:
        click.echo(f"📥 Downloading FIRMS data for years: {years}")
        df = downloader.download_firms_data(list(years), country)
        
        if not df.empty:
            output_path = config.DIRS['raw'] / f"firms_{country}_downloaded.parquet"
            df.to_parquet(output_path)
            click.echo(f"✅ Downloaded {len(df)} records to {output_path}")
        else:
            click.echo("❌ No data downloaded")
    else:
        click.echo("⚠️ No years specified. Use -y option.")

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), 
              help='Input data file (CSV or Parquet)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path')
def process(input, output):
    """🔄 Process wildfire data"""
    if not input:
        click.echo("❌ Please specify input file with -i option")
        return
    
    processor = DataProcessor()
    
    # Load data
    click.echo(f"📂 Loading data from {input}")
    if input.endswith('.parquet'):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)
    
    click.echo(f"📊 Loaded {len(df)} records")
    
    # Process data
    click.echo("🔄 Processing data...")
    processed_df = processor.process_dataset(df)
    
    # Save output
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = config.DIRS['output'] / f"processed_data_{timestamp}.parquet"
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(output, index=False)
    
    click.echo(f"✅ Processed data saved to {output}")
    click.echo(f"📊 Final shape: {processed_df.shape}")

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), 
              help='Input data file')
@click.option('--type', '-t', 
              type=click.Choice(['map', 'timeseries', 'distribution', 'correlation']),
              default='map', help='Visualization type')
@click.option('--feature', '-f', help='Feature to visualize')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path for visualization')
def visualize(input, type, feature, output):
    """📊 Create visualizations"""
    if not input:
        click.echo("❌ Please specify input file with -i option")
        return
    
    visualizer = Visualizer()
    
    # Load data
    click.echo(f"📂 Loading data from {input}")
    if input.endswith('.parquet'):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)
    
    # Create visualization
    click.echo(f"📊 Creating {type} visualization...")
    
    if type == 'map':
        fig = visualizer.create_map_plot(df, color_col=feature)
    elif type == 'timeseries':
        if not feature:
            click.echo("❌ Please specify feature with -f option")
            return
        fig = visualizer.create_time_series_plot(df, feature)
    elif type == 'distribution':
        if not feature:
            click.echo("❌ Please specify feature with -f option")
            return
        fig = visualizer.create_feature_distribution(df, feature)
    elif type == 'correlation':
        fig = visualizer.create_correlation_heatmap(df)
    
    # Save or show
    if output:
        fig.write_html(output)
        click.echo(f"✅ Visualization saved to {output}")
    else:
        fig.show()

@cli.command()
def sample():
    """📊 Generate sample data"""
    downloader = DataDownloader()
    
    click.echo("📊 Generating sample data...")
    df = downloader.download_sample_data()
    
    output_path = config.DIRS['output'] / "sample_data.parquet"
    df.to_parquet(output_path)
    
    click.echo(f"✅ Sample data saved to {output_path}")
    click.echo(f"📊 Shape: {df.shape}")

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), 
              help='Input data file')
def info(input):
    """ℹ️ Show dataset information"""
    if not input:
        click.echo("❌ Please specify input file with -i option")
        return
    
    # Load data
    if input.endswith('.parquet'):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)
    
    click.echo("\n📊 Dataset Information:")
    click.echo(f"Shape: {df.shape}")
    click.echo(f"Columns: {list(df.columns)}")
    click.echo(f"\nData types:")
    click.echo(df.dtypes.to_string())
    click.echo(f"\nMissing values:")
    click.echo(df.isnull().sum().to_string())
    
    if 'acq_date' in df.columns:
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        click.echo(f"\nDate range: {df['acq_date'].min()} to {df['acq_date'].max()}")
    
    if 'is_fire' in df.columns:
        fire_count = df['is_fire'].sum()
        click.echo(f"\nFire events: {fire_count} ({fire_count/len(df)*100:.2f}%)")

if __name__ == '__main__':
    cli()