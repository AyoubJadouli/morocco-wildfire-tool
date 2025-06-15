morocco-wildfire/
├── cli.py
├── main.py
├── pipeline.py
├── config.py
├── downloader.py
├── weather_downloader.py
├── ndvi_downloader.py
├── soil_moisture_downloader.py
├── population_processor.py
├── data_processor.py
├── temporal_features.py
├── geoutils.py
├── augment_data.py
├── control_sampler.py
├── spatial_join.py
├── holiday_utils.py
├── visualizer.py
├── tests/
│   ├── test_weather.py
│   ├── test_temporal_features.py
│   └── …
├── docs/
│   └── README_pipeline.md
├── requirements.txt
└── Data/
    ├── GeoData/
    │   ├── SHP/
    │   ├── NDVI/
    │   ├── SoilMoisture/
    │   └── Population/
    ├── Weather_Noaa/
    │   ├── Morocco/
    │   └── ExpandedMorocco/
    ├── WildFireHist/
    │   └── FIRMS/
    └── FinalDataset/


Below is a **concise, architecture-level map** of the current repository.
Read top-to-bottom to reconstruct the execution flow without opening the code.

---

### `cli.py` — Command-line front-end

*Imports* `argparse`, `downloader`, `data_processor`, `visualizer`.
*Provides* a single `main()` entry that routes sub-commands (`download …`, `process …`, `plot …`) to the relevant modules, making the toolkit usable from Bash scripts.

| Fn / Class        | Signature     | Responsibility                       | Key steps & side-effects                                                          | Depends on                                                       |
| ----------------- | ------------- | ------------------------------------ | --------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| \`main(argv\:list | None)->None\` | Parse argv, dispatch to sub-command. | • Build `argparse` tree  • Instantiate helper classes  • Pass user flags through. | I/O: stdout, log; calls Downloader / DataProcessor / Visualizer. |

---

### `main.py` — Legacy demo driver

Imports `DataDownloader`, `DataProcessor`, `Visualizer`; wires them end-to-end for a single interactive run (kept for backward compatibility).

| Fn                 | Signature               | Responsibility                                     | Notes |
| ------------------ | ----------------------- | -------------------------------------------------- | ----- |
| `run_demo()->None` | One-shot demo pipeline. | Reads user input → downloads FIRMS → simple plots. |       |

---

### `pipeline.py` — Full production pipeline orchestrator

Imports *every* processing module; coordinates all 18 steps with cache checks.

| Fn / Class                                                      | Signature                      | Responsibility                                                                                                | Key steps                             | Depends on |
| --------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------- |
| `run(years:Iterable[int], output:Path, force:bool=False)->Path` | Build the 272-feature parquet. | • Bootstrap dirs  • Call each Downloader  • Temporal/Spatial joins  • Augment & balance data  • Save parquet. | Heavy disk I/O, network (NASA, NOAA). |            |
| `if __name__…`                                                  | CLI wrapper.                   | Parses CLI → calls `run()`.                                                                                   |                                       |            |

---

### `config.py` — Central settings

Declares `DIRS` (typed `Path`s), API tokens, bounding boxes, and logging config.

---

### `downloader.py` — FIRMS fire-pixel getter

Loops 2010-present, downloads MODIS & VIIRS CSVs into `Data/WildFireHist/FIRMS/`.

\| Class | `DataDownloader(years:list[int])` | Download raw fire CSVs. |
\| Method | `download()->list[Path]` | • Build URL  • Stream to disk  • Log progress. |

---

### `weather_downloader.py` — GSOD weather acquisition

Uses BigQuery (fallback FTP) to fetch daily GSOD, interpolates gaps, caches per-station Parquet.

| Class                                    | `WeatherDownloader(force:bool=False)` | End-to-end weather pull. |
| ---------------------------------------- | ------------------------------------- | ------------------------ |
| `fetch_station_list()->gpd.GeoDataFrame` | Query BQ, filter Morocco.             |                          |
| `fetch_daily(years)->pd.DataFrame`       | Pull GSOD rows, concat years.         |                          |
| `interpolate(df)->pd.DataFrame`          | Resample + linear interp, °F→°C.      |                          |
| `save_parquets(df)->list[Path]`          | One file per station.                 |                          |

---

### `ndvi_downloader.py` — MODIS MOD13A2 16-day NDVI

Downloads HDF tiles, extracts NDVI band, returns tidy DF.

\| Fn | `fetch_mod13a2(years:list[int])->pd.DataFrame` | Ensure tiles, stack, scale –> NDVI ∈\[-1,1]. |
\| Fn | `to_parquet(df,path)->Path` | Persist processed grid. |

---

### `soil_moisture_downloader.py` — LPRM AMSR-2 daily soil moisture

Streams NetCDF from NASA GES-DISC, quality-filters, averages to 16-day means.

\| Class | `SoilMoistureDownloader(token:str)` ||
\| `fetch(years)->pd.DataFrame` | Download, subset Morocco. |
\| `to_parquet(df,path)` | Save grid parquet. |

---

### `population_processor.py` — GPW v4 population density slicer

Cuts 2.5-arc-min NetCDF to Morocco bbox and converts to DF.

\| Fn | `slice_gpw(country_iso='MAR')->pd.DataFrame` | • Auto-retrieve file  • Crop  • Melt lat/lon/value. |

---

### `data_processor.py` — Central feature-engineering hub

Loads shapefile, enriches fire/control points with calendar, lags, roll-ups, environmental grids, risk index.

| Class                                   | `DataProcessor()`                   | Master transformer. |
| --------------------------------------- | ----------------------------------- | ------------------- |
| `filter_morocco_points(df)`             | Point-in-polygon mask.              |                     |
| `add_temporal_features(df)`             | DOW/DOY/weekend/month/season.       |                     |
| `add_lag_features(df,cols,lags)`        | Delegates to `temporal_features`.   |                     |
| `add_rolling_features(df,cols,windows)` | Rolling mean/std.                   |                     |
| `attach_nearest_station(df,stations)`   | KD-tree join (geoutils).            |                     |
| `join_environmental_features(df,grids)` | Uses `SpatialJoiner`.               |                     |
| `augment_fire_points(df)`               | Wraps `augment_data`.               |                     |
| `generate_non_fire_points(n)`           | Wraps `control_sampler`.            |                     |
| `calculate_fire_risk_index(df)`         | Simple composite score.             |                     |
| `build_final_dataset(...)`              | Orchestrates all helpers → parquet. |                     |

---

### `temporal_features.py` — Generic lag & roll-up utilities

Pure-pandas helpers reused by both weather and core processor.

\| Fn | `add_lag_features(df,cols,lags)->pd.DataFrame` | Vectorised group-by shift. |
\| Fn | `add_rollups(df,cols,windows)->pd.DataFrame` | Rolling mean/min/max/std. |

---

### `geoutils.py` — Spatial math helpers

KD-tree nearest station, haversine distance to coast.

\| Fn | `nearest_station(fire_df,station_df)->pd.Series` | Returns station idx per row. |
\| Fn | `distance_to_coast(lat,lon,coast)->float` | Geodesic km. |

---

### `augment_data.py` — Fire-pixel jittering

Scatters 100 pseudo-points in a 300 m radius around each fire for class balance.

\| Fn | `jitter_fire_points(df,n=100,radius_m=300)->pd.DataFrame` | Adds `parent_fire_id`. |

---

### `control_sampler.py` — No-fire random sampler

Uniformly draws lat/lon within Morocco polygon and random dates.

\| Fn | `sample_no_fire(n,date_min,date_max,polygon)->pd.DataFrame` | Returns rows with `is_fire=0`. |

---

### `spatial_join.py` — Fast nearest-grid lookup

3-D KD-tree join from point set → environmental grid.

\| Class | `SpatialJoiner(grid_df, value_col)` | Pre-computes KD-tree. |
\| `query(points_df)->pd.Series` | Returns nearest value (+distance opt.). |
\| Fn | `nearest_grid_value(points_df, grid_df, value_col, new_name)` | One-shot helper. |

---

### `holiday_utils.py` — Calendar enrichment

Wraps `holidays` / `workalendar` to mark Moroccan holidays.

\| Fn | `mark_holidays(df,country='MA')->pd.DataFrame` | Adds `is_holiday`. |

---

### `visualizer.py` — Plotly dashboards

Quick EDA plots (map scatter, time-series).

---

### `setup.py` / `requirements.txt` — Packaging

Defines install metadata; pins libs (`pandas`, `xarray`, `rasterio`, `scipy`, …).

---

### `project-structure.md` — Developer docs

Explains folder hierarchy & intermediate artefacts.

---

### `tests/` — PyTest suite

Golden-file checks for each helper; CI uses it for regression safety.

---

### `docs/README_pipeline.md` — End-to-end description

Markdown walk-through of all 18 steps plus usage snippets.

---

## High-level execution flow

1. **CLI / `pipeline.py`** invoked → ensures directories (`config.DIRS`).
2. **Downloaders** fetch: FIRMS fire CSVs, GSOD weather, MODIS NDVI, LPRM soil-moisture, GPW population.
3. **Weather** interpolated & expanded (`temporal_features`).
4. **DataProcessor** takes raw fire points → augments, adds no-fire control, attaches nearest station, environmental grids (`spatial_join`), temporal & holiday features, risk index.
5. Result shuffled, down-cast to `float32`, written as **`Data/FinalDataset/morocco_wildfire_prediction_dataset.parquet`**.
