# # # -*- coding: utf-8 -*-
# # """
# # 🔥 Morocco Wildfire Pipeline – Gradio 5.32 Web UI
# # ------------------------------------------------
# # Interactive front‑end that exposes **all major pipeline stages** as separate
# # Tabs so power‑users can run/download/inspect every artefact – or fire the full
# # pipeline with one click.

# # Compatible with *gradio 5.32*.

# #     • Tab 1 – Directories → create folders declared in `config.DIRS`.
# #     • Tab 2 – Geography   → Natural‑Earth shapefiles (countries + coastline).
# #     • Tab 3 – FIRMS       → download raw fire CSVs (MODIS + VIIRS).
# #     • Tab 4 – Weather     → GSOD daily + interpolation + Parquet per station.
# #     • Tab 5 – NDVI        → MODIS MOD13A2 tiles → tidy Parquet.
# #     • Tab 6 – Soil‑moist. → LPRM AMSR‑2 daily grids → Parquet.
# #     • Tab 7 – Population  → GPW v4 slice → Parquet.
# #     • Tab 8 – Build       → run the **full pipeline** (`pipeline.run`).
# #     • Tab 9 – Preview     → quick EDA / download link for the final dataset.

# # The heavy‑lifting remains in the underlying modules; the UI only orchestrates
# # and surfaces progress & sample previews.
# # """
# # from __future__ import annotations

# # import logging
# # from datetime import datetime
# # from pathlib import Path
# # from typing import List

# # import gradio as gr
# # import pandas as pd

# # import config
# # from downloader import DataDownloader
# # from weather_downloader import WeatherDownloader
# # from ndvi_downloader import fetch_mod13a2, to_parquet as ndvi_to_parquet
# # from soil_moisture_downloader import fetch_lprm_amsr2, to_parquet as sm_to_parquet
# # from population_processor import PopulationProcessor
# # import pipeline  # full orchestrator
# # from visualizer import Visualizer

# # # ───────────────────────── logging ────────────────────────── #
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
# #     datefmt="%Y‑%m‑%d %H:%M:%S",
# # )
# # log = logging.getLogger("ui")

# # # ───────────────────────── helpers ────────────────────────── #

# # def _year_range(start: int, end: int) -> List[int]:
# #     if end < start:
# #         start, end = end, start
# #     return list(range(int(start), int(end) + 1))


# # def _format_df(df: pd.DataFrame, n: int = 10) -> str:
# #     """Pretty‑print first *n* rows as Markdown table."""
# #     if df.empty:
# #         return "*(empty dataframe)*"
# #     return df.head(n).to_markdown(index=False)


# # # ──────────────────────── shared objects ──────────────────────── #

# # downloader = DataDownloader()
# # visualiser = Visualizer()

# # # Gradio State objects keep shared artefact paths
# # state_final_dataset = gr.State(value=None)  # type: ignore

# # # ───────────────────────── tab callbacks ────────────────────────── #

# # # 1️⃣ Directories -------------------------------------------------- #

# # def cb_setup_dirs() -> str:
# #     config.setup_directories()
# #     return "✅ Directory tree created / verified at **data/**"


# # # 2️⃣ Geography ---------------------------------------------------- #

# # def cb_download_geography() -> str:
# #     log.info("🌍 Downloading Natural‑Earth shapefiles…")
# #     downloader.download_shapefiles()
# #     return "✅ Natural‑Earth coastline & country borders ready."


# # # 3️⃣ FIRMS fire data --------------------------------------------- #

# # def cb_download_firms(start_year: int, end_year: int) -> tuple[str, str]:
# #     years = _year_range(start_year, end_year)
# #     log.info("🔥 Downloading FIRMS for %s…", years)
# #     df = downloader.download_firms_data(years)
# #     if df.empty:
# #         return "⚠️ No FIRMS rows downloaded (check years & connectivity).", ""

# #     out_path = config.DIRS["raw"] / f"firms_{years[0]}_{years[-1]}.parquet"
# #     out_path.parent.mkdir(parents=True, exist_ok=True)
# #     df.to_parquet(out_path, index=False)
# #     msg = f"✅ Downloaded **{len(df):,}** fire rows → `{out_path}`"
# #     preview = _format_df(df)
# #     return msg, preview


# # # 4️⃣ Weather ------------------------------------------------------ #

# # def cb_download_weather(start_year: int, end_year: int) -> str:
# #     years = _year_range(start_year, end_year)
# #     wd = WeatherDownloader()
# #     wd.run(years=years)  # internal cache prevents duplicates
# #     return "✅ GSOD weather pulled, interpolated & cached under `Data/Weather_Noaa/`."


# # # 5️⃣ NDVI --------------------------------------------------------- #

# # def cb_download_ndvi(start_year: int, end_year: int) -> str:
# #     years = _year_range(start_year, end_year)
# #     log.info("🌱 Fetching MOD13A2 NDVI tiles…")
# #     df = fetch_mod13a2(years)
# #     out_path = config.DIRS["cache"] / "morocco_ndvi_data.parquet"
# #     ndvi_to_parquet(df, out_path)
# #     return f"✅ NDVI grid saved → `{out_path}` ({len(df):,} rows)."


# # # 6️⃣ Soil moisture ------------------------------------------------ #

# # def cb_download_soil(start_year: int, end_year: int) -> str:
# #     years = _year_range(start_year, end_year)
# #     df = fetch_lprm_amsr2(years)
# #     out_path = config.DIRS["cache"] / "soil_moisture.parquet"
# #     sm_to_parquet(df, out_path)
# #     return f"✅ Soil‑moisture grid saved → `{out_path}` ({len(df):,} rows)."


# # # 7️⃣ Population density ------------------------------------------ #

# # def cb_download_population() -> str:
# #     proc = PopulationProcessor()
# #     df = proc.fetch()
# #     out_path = proc.to_parquet(df)
# #     return f"✅ Population grid saved → `{out_path}` ({len(df):,} rows)."


# # # 8️⃣ Full pipeline ------------------------------------------------ #

# # def cb_run_pipeline(start_year: int, end_year: int, state_ds_path: gr.State):
# #     years = _year_range(start_year, end_year)
# #     out_path = (
# #         config.DIRS.get("output", Path.cwd())
# #         / f"morocco_wildfire_prediction_dataset_{years[0]}_{years[-1]}.parquet"
# #     )
# #     pipeline.run(years=years, output=out_path)
# #     state_ds_path.value = out_path  # stash for preview tab
# #     return f"🎉 Pipeline complete – dataset written to `{out_path}`"


# # # 9️⃣ Preview & download ------------------------------------------ #

# # def cb_preview_dataset(state_ds_path: Path | None):
# #     if not state_ds_path or not Path(state_ds_path).exists():
# #         return "⚠️ Run the pipeline first.", gr.Dataframe(pd.DataFrame())

# #     df = pd.read_parquet(state_ds_path)
# #     summary = (
# #         f"**Rows**: {len(df):,} | **Columns**: {df.shape[1]}\n\n"
# #         f"Date range: {df['acq_date'].min()} → {df['acq_date'].max()}"
# #     )
# #     return summary, gr.Dataframe(df.head(20), height=400)


# # def cb_download_file(state_ds_path: Path | None):
# #     if state_ds_path and Path(state_ds_path).exists():
# #         return state_ds_path
# #     return None


# # # ──────────────────────────── UI layout ──────────────────────────── #
# # with gr.Blocks(title="🔥 Morocco Wildfire Pipeline", theme=gr.themes.Soft()) as demo:
# #     gr.Markdown("""
# #     # 🔥 Morocco Wildfire Dataset Builder
# #     Step‑by‑step interface to download, process, and combine all environmental
# #     layers into the **272‑feature machine‑learning table**.
# #     """)

# #     with gr.Tabs():
# #         # 1 – Directories
# #         with gr.Tab("1️⃣ Directories"):
# #             dir_btn = gr.Button("Create / Verify directory tree", variant="primary")
# #             dir_status = gr.Markdown()
# #             dir_btn.click(cb_setup_dirs, outputs=dir_status)

# #         # 2 – Geography
# #         with gr.Tab("2️⃣ Geography"):
# #             geo_btn = gr.Button("Download Natural‑Earth shapefiles", variant="primary")
# #             geo_status = gr.Markdown()
# #             geo_btn.click(cb_download_geography, outputs=geo_status)

# #         # 3 – FIRMS fire data
# #         with gr.Tab("3️⃣ Fire data (FIRMS)"):
# #             with gr.Row():
# #                 start_firms = gr.Number(value=2010, precision=0, label="Start year")
# #                 end_firms = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
# #             fire_btn = gr.Button("Download FIRMS")
# #             fire_status = gr.Markdown()
# #             fire_preview = gr.Markdown()
# #             fire_btn.click(cb_download_firms, inputs=[start_firms, end_firms], outputs=[fire_status, fire_preview])

# #         # 4 – Weather
# #         with gr.Tab("4️⃣ Weather (GSOD)"):
# #             with gr.Row():
# #                 w_start = gr.Number(value=2010, precision=0, label="Start year")
# #                 w_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
# #             weather_btn = gr.Button("Download + interpolate GSOD")
# #             weather_status = gr.Markdown()
# #             weather_btn.click(cb_download_weather, inputs=[w_start, w_end], outputs=weather_status)

# #         # 5 – NDVI
# #         with gr.Tab("5️⃣ NDVI"):
# #             with gr.Row():
# #                 n_start = gr.Number(value=2010, precision=0, label="Start year")
# #                 n_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
# #             ndvi_btn = gr.Button("Fetch MOD13A2 tiles")
# #             ndvi_status = gr.Markdown()
# #             ndvi_btn.click(cb_download_ndvi, inputs=[n_start, n_end], outputs=ndvi_status)

# #         # 6 – Soil moisture
# #         with gr.Tab("6️⃣ Soil‑moisture"):
# #             with gr.Row():
# #                 s_start = gr.Number(value=2012, precision=0, label="Start year (AMSR‑2 launch 2012)")
# #                 s_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
# #             soil_btn = gr.Button("Fetch LPRM AMSR‑2")
# #             soil_status = gr.Markdown()
# #             soil_btn.click(cb_download_soil, inputs=[s_start, s_end], outputs=soil_status)

# #         # 7 – Population
# #         with gr.Tab("7️⃣ Population density"):
# #             pop_btn = gr.Button("Slice GPW v4 (all years)")
# #             pop_status = gr.Markdown()
# #             pop_btn.click(cb_download_population, outputs=pop_status)

# #         # 8 – Build final dataset
# #         with gr.Tab("8️⃣ Build dataset"):
# #             with gr.Row():
# #                 p_start = gr.Number(value=2010, precision=0, label="Start year")
# #                 p_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
# #             run_btn = gr.Button("Run full pipeline", variant="primary")
# #             build_status = gr.Markdown()
# #             run_btn.click(cb_run_pipeline, inputs=[p_start, p_end, state_final_dataset], outputs=build_status)

# #         # 9 – Preview / Download
# #         with gr.Tab("9️⃣ Preview & download"):
# #             prev_btn = gr.Button("Refresh preview")
# #             summary_md = gr.Markdown()
# #             df_preview = gr.Dataframe()
# #             prev_btn.click(cb_preview_dataset, inputs=[state_final_dataset], outputs=[summary_md, df_preview])

# #             download_btn = gr.DownloadButton("⬇️ Download Parquet", label="Download Parquet")
# #             download_btn.click(cb_download_file, inputs=[state_final_dataset], outputs=download_btn)

# #     gr.Markdown("""---\n© 2025 WildFreDataTool – MIT License""")

# # # ─────────────────────────── launch ─────────────────────────── #
# # if __name__ == "__main__":
# #     demo.launch(show_error=True, inbrowser=False)


# # -*- coding: utf-8 -*-
# """
# 🔥 Morocco Wildfire Pipeline – Gradio 5.32 Web UI
# ------------------------------------------------
# Interactive front-end that exposes all major pipeline stages and
# adds visualization tabs for maps, time series, distributions, and correlations.

# Compatible with gradio 5.32.
# """
# from __future__ import annotations

# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import List

# import gradio as gr
# import pandas as pd

# import config
# from downloader import DataDownloader
# from weather_downloader import WeatherDownloader
# from ndvi_downloader import fetch_mod13a2, to_parquet as ndvi_to_parquet
# from soil_moisture_downloader import fetch_lprm_amsr2, to_parquet as sm_to_parquet
# from population_processor import PopulationProcessor
# import pipeline  # full orchestrator
# from visualizer import Visualizer

# # ───────────────────────── logging ────────────────────────── #
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# log = logging.getLogger("ui")

# # ───────────────────────── helpers ────────────────────────── #

# def _year_range(start: int, end: int) -> List[int]:
#     if end < start:
#         start, end = end, start
#     return list(range(int(start), int(end) + 1))


# def _format_df(df: pd.DataFrame, n: int = 10) -> str:
#     """Pretty-print first *n* rows as Markdown table."""
#     if df.empty:
#         return "*(empty dataframe)*"
#     return df.head(n).to_markdown(index=False)


# # ─────────────────────── shared objects ─────────────────────── #

# downloader = DataDownloader()
# visualiser = Visualizer()

# # Gradio State objects keep shared artifact paths
# state_final_dataset = gr.State(value=None)  # type: ignore

# # ───────────────────────── tab callbacks ────────────────────────── #

# # 1️⃣ Directories -------------------------------------------------- #

# def cb_setup_dirs() -> str:
#     config.setup_directories()
#     return "✅ Directory tree created / verified at **data/**"


# # 2️⃣ Geography ---------------------------------------------------- #

# def cb_download_geography() -> str:
#     log.info("🌍 Downloading Natural-Earth shapefiles…")
#     downloader.download_shapefiles()
#     return "✅ Natural-Earth coastline & country borders ready."


# # 3️⃣ FIRMS fire data --------------------------------------------- #

# def cb_download_firms(start_year: int, end_year: int) -> tuple[str, str]:
#     years = _year_range(start_year, end_year)
#     log.info("🔥 Downloading FIRMS for %s…", years)
#     df = downloader.download_firms_data(years)
#     if df.empty:
#         return "⚠️ No FIRMS rows downloaded (check years & connectivity).", ""

#     out_path = config.DIRS["raw"] / f"firms_{years[0]}_{years[-1]}.parquet"
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_parquet(out_path, index=False)
#     msg = f"✅ Downloaded **{len(df):,}** fire rows → `{out_path}`"
#     preview = _format_df(df)
#     return msg, preview


# # 4️⃣ Weather ------------------------------------------------------ #

# def cb_download_weather(start_year: int, end_year: int) -> str:
#     years = _year_range(start_year, end_year)
#     wd = WeatherDownloader()
#     wd.run(years=years)  # internal cache prevents duplicates
#     return "✅ GSOD weather pulled, interpolated & cached under `Data/Weather_Noaa/`."


# # 5️⃣ NDVI --------------------------------------------------------- #

# def cb_download_ndvi(start_year: int, end_year: int) -> str:
#     years = _year_range(start_year, end_year)
#     log.info("🌱 Fetching MOD13A2 NDVI tiles…")
#     df = fetch_mod13a2(years)
#     out_path = config.DIRS["cache"] / "morocco_ndvi_data.parquet"
#     ndvi_to_parquet(df, out_path)
#     return f"✅ NDVI grid saved → `{out_path}` ({len(df):,} rows)."


# # 6️⃣ Soil moisture ------------------------------------------------ #

# def cb_download_soil(start_year: int, end_year: int) -> str:
#     years = _year_range(start_year, end_year)
#     df = fetch_lprm_amsr2(years)
#     out_path = config.DIRS["cache"] / "soil_moisture.parquet"
#     sm_to_parquet(df, out_path)
#     return f"✅ Soil-moisture grid saved → `{out_path}` ({len(df):,} rows)."


# # 7️⃣ Population density ------------------------------------------ #

# def cb_download_population() -> str:
#     proc = PopulationProcessor()
#     df = proc.fetch()
#     out_path = proc.to_parquet(df)
#     return f"✅ Population grid saved → `{out_path}` ({len(df):,} rows)."


# # 8️⃣ Full pipeline ------------------------------------------------ #

# def cb_run_pipeline(start_year: int, end_year: int, state_ds_path: gr.State):
#     years = _year_range(start_year, end_year)
#     out_path = (
#         config.DIRS.get("output", Path.cwd())
#         / f"morocco_wildfire_prediction_dataset_{years[0]}_{years[-1]}.parquet"
#     )
#     pipeline.run(years=years, output=out_path)
#     state_ds_path.value = out_path  # stash for preview & visualization tabs
#     return f"🎉 Pipeline complete – dataset written to `{out_path}`"


# # 9️⃣ Preview & download ------------------------------------------ #

# def cb_preview_dataset(state_ds_path: Path | None):
#     if not state_ds_path or not Path(state_ds_path).exists():
#         return "⚠️ Run the pipeline first.", gr.Dataframe(pd.DataFrame())

#     df = pd.read_parquet(state_ds_path)
#     summary = (
#         f"**Rows**: {len(df):,} | **Columns**: {df.shape[1]}\n\n"
#         f"Date range: {df['acq_date'].min()} → {df['acq_date'].max()}"
#     )
#     return summary, gr.Dataframe(df.head(20), height=400)


# def cb_download_file(state_ds_path: Path | None):
#     if state_ds_path and Path(state_ds_path).exists():
#         return state_ds_path
#     return None


# # 🔟 Visualization callbacks -------------------------------------- #

# # Use the predefined FEATURE_COLUMNS from config for dropdown choices
# FEATURE_CHOICES = config.FEATURE_COLUMNS.copy()

# def cb_map_viz(color_feature: str, state_ds_path: Path | None):
#     if not state_ds_path or not Path(state_ds_path).exists():
#         return None
#     df = pd.read_parquet(state_ds_path)
#     return visualiser.create_map_plot(df, color_col=color_feature)


# def cb_ts_viz(feature: str, state_ds_path: Path | None):
#     if not state_ds_path or not Path(state_ds_path).exists():
#         return None
#     df = pd.read_parquet(state_ds_path)
#     return visualiser.create_time_series_plot(df, feature)


# def cb_dist_viz(feature: str, state_ds_path: Path | None):
#     if not state_ds_path or not Path(state_ds_path).exists():
#         return None
#     df = pd.read_parquet(state_ds_path)
#     return visualiser.create_feature_distribution(df, feature)


# def cb_corr_viz(features: List[str], state_ds_path: Path | None):
#     if not state_ds_path or not Path(state_ds_path).exists():
#         return None
#     df = pd.read_parquet(state_ds_path)
#     return visualiser.create_correlation_heatmap(df, features)


# # ──────────────────────────── UI layout ──────────────────────────── #
# with gr.Blocks(title="🔥 Morocco Wildfire Pipeline", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("""
#     # 🔥 Morocco Wildfire Dataset Builder & Visualizer
#     Step-by-step interface to download, process, combine all environmental layers into the **272-feature ML table**, 
#     and explore the resulting dataset via interactive visualizations.
#     """)

#     with gr.Tabs():
#         # 1 – Directories
#         with gr.Tab("1️⃣ Directories"):
#             dir_btn = gr.Button("Create / Verify directory tree", variant="primary")
#             dir_status = gr.Markdown()
#             dir_btn.click(cb_setup_dirs, outputs=dir_status)

#         # 2 – Geography
#         with gr.Tab("2️⃣ Geography"):
#             geo_btn = gr.Button("Download Natural-Earth shapefiles", variant="primary")
#             geo_status = gr.Markdown()
#             geo_btn.click(cb_download_geography, outputs=geo_status)

#         # 3 – FIRMS fire data
#         with gr.Tab("3️⃣ Fire data (FIRMS)"):
#             with gr.Row():
#                 start_firms = gr.Number(value=2010, precision=0, label="Start year")
#                 end_firms = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
#             fire_btn = gr.Button("Download FIRMS")
#             fire_status = gr.Markdown()
#             fire_preview = gr.Markdown()
#             fire_btn.click(
#                 cb_download_firms,
#                 inputs=[start_firms, end_firms],
#                 outputs=[fire_status, fire_preview]
#             )

#         # 4 – Weather
#         with gr.Tab("4️⃣ Weather (GSOD)"):
#             with gr.Row():
#                 w_start = gr.Number(value=2010, precision=0, label="Start year")
#                 w_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
#             weather_btn = gr.Button("Download + interpolate GSOD")
#             weather_status = gr.Markdown()
#             weather_btn.click(
#                 cb_download_weather,
#                 inputs=[w_start, w_end],
#                 outputs=weather_status
#             )

#         # 5 – NDVI
#         with gr.Tab("5️⃣ NDVI"):
#             with gr.Row():
#                 n_start = gr.Number(value=2010, precision=0, label="Start year")
#                 n_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
#             ndvi_btn = gr.Button("Fetch MOD13A2 tiles")
#             ndvi_status = gr.Markdown()
#             ndvi_btn.click(
#                 cb_download_ndvi,
#                 inputs=[n_start, n_end],
#                 outputs=ndvi_status
#             )

#         # 6 – Soil moisture
#         with gr.Tab("6️⃣ Soil-moisture"):
#             with gr.Row():
#                 s_start = gr.Number(value=2012, precision=0, label="Start year (AMSR-2 launch 2012)")
#                 s_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
#             soil_btn = gr.Button("Fetch LPRM AMSR-2")
#             soil_status = gr.Markdown()
#             soil_btn.click(
#                 cb_download_soil,
#                 inputs=[s_start, s_end],
#                 outputs=soil_status
#             )

#         # 7 – Population
#         with gr.Tab("7️⃣ Population density"):
#             pop_btn = gr.Button("Slice GPW v4 (all years)")
#             pop_status = gr.Markdown()
#             pop_btn.click(cb_download_population, outputs=pop_status)

#         # 8 – Build final dataset
#         with gr.Tab("8️⃣ Build dataset"):
#             with gr.Row():
#                 p_start = gr.Number(value=2010, precision=0, label="Start year")
#                 p_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
#             run_btn = gr.Button("Run full pipeline", variant="primary")
#             build_status = gr.Markdown()
#             run_btn.click(
#                 cb_run_pipeline,
#                 inputs=[p_start, p_end, state_final_dataset],
#                 outputs=build_status
#             )

#         # 9 – Preview / Download
#         with gr.Tab("9️⃣ Preview & download"):
#             prev_btn = gr.Button("Refresh preview")
#             summary_md = gr.Markdown()
#             df_preview = gr.Dataframe()
#             prev_btn.click(
#                 cb_preview_dataset,
#                 inputs=[state_final_dataset],
#                 outputs=[summary_md, df_preview]
#             )

#             download_btn = gr.DownloadButton("⬇️ Download Parquet", label="Download Parquet")
#             download_btn.click(
#                 cb_download_file,
#                 inputs=[state_final_dataset],
#                 outputs=download_btn
#             )

#         # 🔟 Visualization
#         with gr.Tab("🔟 Visualization"):
#             gr.Markdown("### Explore the final dataset if available")

#             with gr.Tabs():
#                 # Map visualization
#                 with gr.Tab("🗺️ Map"):
#                     map_feature = gr.Dropdown(
#                         choices=FEATURE_CHOICES,
#                         value="is_fire",
#                         label="Color Feature"
#                     )
#                     map_btn = gr.Button("🗺️ Create Map")
#                     map_plot = gr.Plot(label="Map Visualization")
#                     map_btn.click(
#                         cb_map_viz,
#                         inputs=[map_feature, state_final_dataset],
#                         outputs=[map_plot]
#                     )

#                 # Time series visualization
#                 with gr.Tab("📈 Time Series"):
#                     ts_feature = gr.Dropdown(
#                         choices=FEATURE_CHOICES,
#                         value="average_temperature",
#                         label="Feature"
#                     )
#                     ts_btn = gr.Button("📈 Create Time Series")
#                     ts_plot = gr.Plot(label="Time Series")
#                     ts_btn.click(
#                         cb_ts_viz,
#                         inputs=[ts_feature, state_final_dataset],
#                         outputs=[ts_plot]
#                     )

#                 # Distribution visualization
#                 with gr.Tab("📊 Distribution"):
#                     dist_feature = gr.Dropdown(
#                         choices=FEATURE_CHOICES,
#                         value="NDVI",
#                         label="Feature"
#                     )
#                     dist_btn = gr.Button("📊 Create Distribution")
#                     dist_plot = gr.Plot(label="Distribution")
#                     dist_btn.click(
#                         cb_dist_viz,
#                         inputs=[dist_feature, state_final_dataset],
#                         outputs=[dist_plot]
#                     )

#                 # Correlation heatmap
#                 with gr.Tab("🔥 Correlation"):
#                     corr_features = gr.CheckboxGroup(
#                         choices=FEATURE_CHOICES,
#                         label="Select Features"
#                     )
#                     corr_btn = gr.Button("🔥 Create Correlation Heatmap")
#                     corr_plot = gr.Plot(label="Correlation Heatmap")
#                     corr_btn.click(
#                         cb_corr_viz,
#                         inputs=[corr_features, state_final_dataset],
#                         outputs=[corr_plot]
#                     )

#     gr.Markdown("""---\n© 2025 WildFreDataTool – MIT License""")

# # ─────────────────────────── launch ─────────────────────────── #
# if __name__ == "__main__":
#     demo.launch(show_error=True, inbrowser=False)


# -*- coding: utf-8 -*-
"""
🔥 Morocco Wildfire Pipeline – Gradio 5.32 Web UI
------------------------------------------------
Interactive front-end that exposes all major pipeline stages and
adds visualization tabs for maps, time series, distributions, and correlations.

Compatible with gradio 5.32.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import gradio as gr
import pandas as pd

import config
from downloader import DataDownloader
from weather_downloader import WeatherDownloader
from ndvi_downloader import fetch_mod13a2, to_parquet as ndvi_to_parquet
from soil_moisture_downloader import fetch_lprm_amsr2, to_parquet as sm_to_parquet
from population_processor import PopulationProcessor
import pipeline  # full orchestrator
from visualizer import Visualizer

# ───────────────────────── logging ────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ui")

# ───────────────────────── helpers ────────────────────────── #

def _year_range(start: int, end: int) -> List[int]:
    if end < start:
        start, end = end, start
    return list(range(int(start), int(end) + 1))


def _format_df(df: pd.DataFrame, n: int = 10) -> str:
    """Pretty-print first *n* rows as Markdown table."""
    if df.empty:
        return "*(empty dataframe)*"
    return df.head(n).to_markdown(index=False)


# ─────────────────────── shared objects ─────────────────────── #

downloader = DataDownloader()
visualiser = Visualizer()

# Gradio State objects keep shared artifact paths
state_final_dataset = gr.State(value=None)  # type: ignore

# ───────────────────────── tab callbacks ────────────────────────── #

# 1️⃣ Directories -------------------------------------------------- #

def cb_setup_dirs() -> str:
    config.setup_directories()
    return "✅ Directory tree created / verified at **data/**"


# 2️⃣ Geography ---------------------------------------------------- #

def cb_download_geography() -> str:
    log.info("🌍 Downloading Natural-Earth shapefiles…")
    downloader.download_shapefiles()
    return "✅ Natural-Earth coastline & country borders ready."


# 3️⃣ FIRMS fire data --------------------------------------------- #

def cb_download_firms(start_year: int, end_year: int) -> tuple[str, str]:
    years = _year_range(start_year, end_year)
    log.info("🔥 Downloading FIRMS for %s…", years)
    df = downloader.download_firms_data(years)
    if df.empty:
        return "⚠️ No FIRMS rows downloaded (check years & connectivity).", ""

    out_path = config.DIRS["raw"] / f"firms_{years[0]}_{years[-1]}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    msg = f"✅ Downloaded **{len(df):,}** fire rows → `{out_path}`"
    preview = _format_df(df)
    return msg, preview


# 4️⃣ Weather ------------------------------------------------------ #

def cb_download_weather(start_year: int, end_year: int) -> str:
    years = _year_range(start_year, end_year)
    wd = WeatherDownloader()
    wd.run(years=years)  # internal cache prevents duplicates
    return "✅ GSOD weather pulled, interpolated & cached under `Data/Weather_Noaa/`."


# 5️⃣ NDVI --------------------------------------------------------- #

def cb_download_ndvi(start_year: int, end_year: int) -> str:
    years = _year_range(start_year, end_year)
    log.info("🌱 Fetching MOD13A2 NDVI tiles…")
    df = fetch_mod13a2(years)
    out_path = config.DIRS["cache"] / "morocco_ndvi_data.parquet"
    ndvi_to_parquet(df, out_path)
    return f"✅ NDVI grid saved → `{out_path}` ({len(df):,} rows)."


# 6️⃣ Soil moisture ------------------------------------------------ #

def cb_download_soil(start_year: int, end_year: int) -> str:
    years = _year_range(start_year, end_year)
    df = fetch_lprm_amsr2(years)
    out_path = config.DIRS["cache"] / "soil_moisture.parquet"
    sm_to_parquet(df, out_path)
    return f"✅ Soil-moisture grid saved → `{out_path}` ({len(df):,} rows)."


# 7️⃣ Population density ------------------------------------------ #

def cb_download_population() -> str:
    proc = PopulationProcessor()
    df = proc.fetch()
    out_path = proc.to_parquet(df)
    return f"✅ Population grid saved → `{out_path}` ({len(df):,} rows)."


# 8️⃣ Full pipeline ------------------------------------------------ #

def cb_run_pipeline(start_year: int, end_year: int, state_ds_path: gr.State):
    years = _year_range(start_year, end_year)
    out_path = (
        config.DIRS.get("output", Path.cwd())
        / f"morocco_wildfire_prediction_dataset_{years[0]}_{years[-1]}.parquet"
    )
    pipeline.run(years=years, output=out_path)
    state_ds_path.value = out_path  # stash for preview & visualization tabs
    return f"🎉 Pipeline complete – dataset written to `{out_path}`"


# 9️⃣ Preview & download ------------------------------------------ #

def cb_preview_dataset(state_ds_path: Path | None):
    if not state_ds_path or not Path(state_ds_path).exists():
        return "⚠️ Run the pipeline first.", gr.Dataframe(pd.DataFrame())

    df = pd.read_parquet(state_ds_path)
    summary = (
        f"**Rows**: {len(df):,} | **Columns**: {df.shape[1]}\n\n"
        f"Date range: {df['acq_date'].min()} → {df['acq_date'].max()}"
    )
    return summary, gr.Dataframe(df.head(20), height=400)


def cb_download_file(state_ds_path: Path | None):
    if state_ds_path and Path(state_ds_path).exists():
        return state_ds_path
    return None


# 🔟 Visualization callbacks -------------------------------------- #

# Use the predefined FEATURE_COLUMNS from config for dropdown choices
FEATURE_CHOICES = config.FEATURE_COLUMNS.copy()

def cb_map_viz(color_feature: str, state_ds_path: Path | None):
    if not state_ds_path or not Path(state_ds_path).exists():
        return None
    df = pd.read_parquet(state_ds_path)
    return visualiser.create_map_plot(df, color_col=color_feature)


def cb_ts_viz(feature: str, state_ds_path: Path | None):
    if not state_ds_path or not Path(state_ds_path).exists():
        return None
    df = pd.read_parquet(state_ds_path)
    return visualiser.create_time_series_plot(df, feature)


def cb_dist_viz(feature: str, state_ds_path: Path | None):
    if not state_ds_path or not Path(state_ds_path).exists():
        return None
    df = pd.read_parquet(state_ds_path)
    return visualiser.create_feature_distribution(df, feature)


def cb_corr_viz(features: List[str], state_ds_path: Path | None):
    if not state_ds_path or not Path(state_ds_path).exists():
        return None
    df = pd.read_parquet(state_ds_path)
    return visualiser.create_correlation_heatmap(df, features)


# ──────────────────────────── UI layout ──────────────────────────── #
with gr.Blocks(title="🔥 Morocco Wildfire Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔥 Morocco Wildfire Dataset Builder & Visualizer
    Step-by-step interface to download, process, combine all environmental layers into the **272-feature ML table**, 
    and explore the resulting dataset via interactive visualizations.
    """)

    with gr.Tabs():
        # 1 – Directories
        with gr.Tab("1️⃣ Directories"):
            dir_btn = gr.Button("Create / Verify directory tree", variant="primary")
            dir_status = gr.Markdown()
            dir_btn.click(cb_setup_dirs, outputs=dir_status)

        # 2 – Geography
        with gr.Tab("2️⃣ Geography"):
            geo_btn = gr.Button("Download Natural-Earth shapefiles", variant="primary")
            geo_status = gr.Markdown()
            geo_btn.click(cb_download_geography, outputs=geo_status)

        # 3 – FIRMS fire data
        with gr.Tab("3️⃣ Fire data (FIRMS)"):
            with gr.Row():
                start_firms = gr.Number(value=2010, precision=0, label="Start year")
                end_firms = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
            fire_btn = gr.Button("Download FIRMS")
            fire_status = gr.Markdown()
            fire_preview = gr.Markdown()
            fire_btn.click(
                cb_download_firms,
                inputs=[start_firms, end_firms],
                outputs=[fire_status, fire_preview]
            )

        # 4 – Weather
        with gr.Tab("4️⃣ Weather (GSOD)"):
            with gr.Row():
                w_start = gr.Number(value=2010, precision=0, label="Start year")
                w_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
            weather_btn = gr.Button("Download + interpolate GSOD")
            weather_status = gr.Markdown()
            weather_btn.click(
                cb_download_weather,
                inputs=[w_start, w_end],
                outputs=weather_status
            )

        # 5 – NDVI
        with gr.Tab("5️⃣ NDVI"):
            with gr.Row():
                n_start = gr.Number(value=2010, precision=0, label="Start year")
                n_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
            ndvi_btn = gr.Button("Fetch MOD13A2 tiles")
            ndvi_status = gr.Markdown()
            ndvi_btn.click(
                cb_download_ndvi,
                inputs=[n_start, n_end],
                outputs=ndvi_status
            )

        # 6 – Soil moisture
        with gr.Tab("6️⃣ Soil-moisture"):
            with gr.Row():
                s_start = gr.Number(value=2012, precision=0, label="Start year (AMSR-2 launch 2012)")
                s_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
            soil_btn = gr.Button("Fetch LPRM AMSR-2")
            soil_status = gr.Markdown()
            soil_btn.click(
                cb_download_soil,
                inputs=[s_start, s_end],
                outputs=soil_status
            )

        # 7 – Population
        with gr.Tab("7️⃣ Population density"):
            pop_btn = gr.Button("Slice GPW v4 (all years)")
            pop_status = gr.Markdown()
            pop_btn.click(cb_download_population, outputs=pop_status)

        # 8 – Build final dataset
        with gr.Tab("8️⃣ Build dataset"):
            with gr.Row():
                p_start = gr.Number(value=2010, precision=0, label="Start year")
                p_end = gr.Number(value=datetime.utcnow().year - 1, precision=0, label="End year")
            run_btn = gr.Button("Run full pipeline", variant="primary")
            build_status = gr.Markdown()
            run_btn.click(
                cb_run_pipeline,
                inputs=[p_start, p_end, state_final_dataset],
                outputs=build_status
            )

        # 9 – Preview / Download
        with gr.Tab("9️⃣ Preview & download"):
            prev_btn = gr.Button("Refresh preview")
            summary_md = gr.Markdown()
            df_preview = gr.Dataframe()
            prev_btn.click(
                cb_preview_dataset,
                inputs=[state_final_dataset],
                outputs=[summary_md, df_preview]
            )

            download_btn = gr.DownloadButton("⬇️ Download Parquet", label="Download Parquet")
            download_btn.click(
                cb_download_file,
                inputs=[state_final_dataset],
                outputs=download_btn
            )

        # 🔟 Visualization
        with gr.Tab("🔟 Visualization"):
            gr.Markdown("### Explore the final dataset if available")

            with gr.Tabs():
                # Map visualization
                with gr.Tab("🗺️ Map"):
                    map_feature = gr.Dropdown(
                        choices=FEATURE_CHOICES,
                        value="is_fire",
                        label="Color Feature"
                    )
                    map_btn = gr.Button("🗺️ Create Map")
                    map_plot = gr.Plot(label="Map Visualization")
                    map_btn.click(
                        cb_map_viz,
                        inputs=[map_feature, state_final_dataset],
                        outputs=[map_plot]
                    )

                # Time series visualization
                with gr.Tab("📈 Time Series"):
                    ts_feature = gr.Dropdown(
                        choices=FEATURE_CHOICES,
                        value="average_temperature",
                        label="Feature"
                    )
                    ts_btn = gr.Button("📈 Create Time Series")
                    ts_plot = gr.Plot(label="Time Series")
                    ts_btn.click(
                        cb_ts_viz,
                        inputs=[ts_feature, state_final_dataset],
                        outputs=[ts_plot]
                    )

                # Distribution visualization
                with gr.Tab("📊 Distribution"):
                    dist_feature = gr.Dropdown(
                        choices=FEATURE_CHOICES,
                        value="NDVI",
                        label="Feature"
                    )
                    dist_btn = gr.Button("📊 Create Distribution")
                    dist_plot = gr.Plot(label="Distribution")
                    dist_btn.click(
                        cb_dist_viz,
                        inputs=[dist_feature, state_final_dataset],
                        outputs=[dist_plot]
                    )

                # Correlation heatmap
                with gr.Tab("🔥 Correlation"):
                    corr_features = gr.CheckboxGroup(
                        choices=FEATURE_CHOICES,
                        label="Select Features"
                    )
                    corr_btn = gr.Button("🔥 Create Correlation Heatmap")
                    corr_plot = gr.Plot(label="Correlation Heatmap")
                    corr_btn.click(
                        cb_corr_viz,
                        inputs=[corr_features, state_final_dataset],
                        outputs=[corr_plot]
                    )

    gr.Markdown("""---\n© 2025 WildFreDataTool – MIT License""")

# ─────────────────────────── launch ─────────────────────────── #
if __name__ == "__main__":
    demo.launch(show_error=True, inbrowser=False)
