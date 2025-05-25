from setuptools import setup, find_packages

setup(
    name="morocco-wildfire-tool",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "geopandas>=0.10.0",
        "folium>=0.12.0",
        "netCDF4>=1.5.0",
        "gradio>=5.0.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "shapely>=1.7.0",
        "pyarrow>=5.0.0",  # For parquet support
        "openpyxl>=3.0.0",  # For Excel support
        "scipy>=1.7.0",     # For spatial calculations
        "haversine>=2.5.0", # For distance calculations
    ],
    entry_points={
        "console_scripts": [
            "morocco-data=morocco_wildfire_tool.cli:main",
        ],
    },
    author="Ayoub JADOULI - DevOps & ML Expert",
    author_email="ayoub@jadouli.com",
    description="A tool for processing and visualizing wildfire data in Morocco",
    keywords="wildfire, morocco, data-processing, visualization, gradio",
    url="https://github.com/ayoubjadouli/morocco-wildfire-tool",
    project_urls={
        "Bug Tracker": "https://github.com/ayoubjadouli/morocco-wildfire-tool/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
)