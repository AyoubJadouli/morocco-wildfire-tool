# fetchers/gfs.py

from pathlib import Path
from datetime import datetime, timedelta
import requests
import xarray as xr
import cfgrib
import polars as pl
import logging

from feature_store.rolling_buffer import RollingBuffer

log = logging.getLogger("fetchers.gfs")
log.setLevel(logging.INFO)


# NOAA GFS 0.25° data root
GFS_ROOT = "https://www.ncei.noaa.gov/data/global-forecast-system/access"
BBOX     = (-14, 21, -1, 36)   # West, South, East, North for Morocco
VARS     = {
    "t2m": "temperature",
    "r2m": "rh",
    "u10": "u_wind",
    "v10": "v_wind",
    "tp":  "precip",
}


def latest_cycle(attempts: int = 0) -> datetime:
    """
    Return the most recent GFS cycle (00, 06, 12, 18 UTC) < 2 h old.
    If attempts > 0, back off by `6 * attempts` hours.
    """
    now = datetime.utcnow() - timedelta(hours=2 + 6 * attempts)
    # Choose the largest cycle hour ≤ now.hour among [0,6,12,18]
    cycle_hour = max(h for h in (0, 6, 12, 18) if h <= now.hour)
    return now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)


def download_tile(code: str, dt: datetime) -> xr.Dataset:
    """
    Download a single GRIB2 file for variable 'code' at datetime dt, then clip to Morocco.
    Returns an xarray.Dataset with dims (time, lat, lon).
    Throws HTTPError if 404 or other client error.
    """
    ymd = dt.strftime("%Y%m%d")
    cyc = f"{dt:%H}"
    url = f"{GFS_ROOT}/{ymd}/{cyc}/{code}.grib2"
    log.info("Downloading GFS %s for %s → %s", code, dt.isoformat(), url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tmpf = Path("/tmp") / f"gfs_{code}_{ymd}{cyc}.grib2"
    tmpf.write_bytes(r.content)
    ds = xr.open_dataset(tmpf, engine="cfgrib")
    # Clip to the Morocco bounding box
    ds = ds.sel(
        longitude = slice(BBOX[0], BBOX[2]),
        latitude  = slice(BBOX[3], BBOX[1])   # latitude is likely descending
    )
    return ds


def run():
    """
    ‣ Attempt to fetch the latest GFS cycle (00/06/12/18 UTC < 2 h old).
    ‣ If the first attempt returns 404, back off 6 h and retry (up to 4 times).
    ‣ Once a valid cycle is found, download each variable in VARS, convert to Polars,
      and append into RollingBuffer under Data/feature_db/<var>/dt=YYYY-MM-DD.parquet.
    """
    max_attempts = 4
    ds = None
    used_dt = None

    # Try up to max_attempts cycles, backing off 6 h each time
    for attempt in range(max_attempts):
        dt_try = latest_cycle(attempts=attempt)
        try:
            # Just test one variable to confirm this cycle exists
            _ = download_tile(next(iter(VARS)), dt_try)
            used_dt = dt_try
            log.info("Found valid GFS cycle at %s after %d attempt(s)", used_dt.isoformat(), attempt)
            break
        except requests.HTTPError as e:
            log.warning(
                "GFS cycle %s not available (HTTP %d). Retrying with previous cycle...",
                dt_try.isoformat(), e.response.status_code
            )
            continue

    if used_dt is None:
        log.error("No valid GFS cycle found in last %d attempts. Skipping GFS fetch.", max_attempts)
        return

    # Now download each variable for the confirmed dt
    rb = RollingBuffer(Path("Data/feature_db"))

    for code, name in VARS.items():
        try:
            ds = download_tile(code, used_dt)
            # Convert to DataFrame
            df = ds.to_dataframe()[[code]].rename(columns={code: name})
            df = df.reset_index()[["latitude", "longitude", "time", name]]
            df = df.rename(columns={"time": "timestamp"})
            # Append into RollingBuffer
            rb.append(name, df)
            log.info("Appended %s data for %s (%d rows)", name, used_dt.date(), len(df))
        except requests.HTTPError as e:
            log.error(
                "Failed to download GFS %s for cycle %s (HTTP %d). Skipping this variable.",
                code, used_dt.isoformat(), e.response.status_code
            )
        except Exception as e:
            log.exception("Unexpected error fetching GFS %s for %s: %s", code, used_dt.isoformat(), e)
