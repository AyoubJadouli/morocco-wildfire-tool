# feature_store/rolling_buffer.py

from pathlib import Path
import polars as pl
import pandas as pd

class RollingBuffer:
    """
    Append-only Parquet store for time-partitioned variables.

    Directory layout under `root/`:
        root/
          temperature/
            dt=2025-06-01.parquet
            dt=2025-06-02.parquet
          rh/
            dt=2025-06-01.parquet
          precip/
            dt=2025-06-01.parquet
    Each Parquet file contains all rows for that variable on a given date.
    """

    def __init__(self, root: Path, partition_key: str = "dt"):
        """
        Args:
            root: Path to the base directory where buffers live (e.g. Data/feature_db).
            partition_key: Folder prefix used for each date partition (default "dt").
        """
        self.root = root
        self.partition_key = partition_key
        self.root.mkdir(parents=True, exist_ok=True)

    def _date_str(self, ts) -> str:
        """
        Convert a timestamp (pd.Timestamp or datetime) to "YYYY-MM-DD" string.
        """
        if isinstance(ts, (pd.Timestamp,)):
            return ts.date().isoformat()
        # If ts is a Python datetime:
        return pd.to_datetime(ts).date().isoformat()

    def _path(self, var: str, ts) -> Path:
        """
        Given a variable name and a timestamp, return the Path to the partition file:
            root/var/dt=YYYY-MM-DD.parquet
        """
        date_str = self._date_str(ts)
        return self.root / var / f"{self.partition_key}={date_str}.parquet"

    def append(self, var: str, df: pl.DataFrame | pd.DataFrame):
        """
        Append a batch of rows for variable `var` into the daily Parquet partition.

        Args:
            var: Name of the variable (e.g. "temperature", "precipitation", "rh").
            df: A Polars or Pandas DataFrame containing at least:
                - a column named "timestamp" (datetime-like) indicating when the sample was taken
                - any other measurement columns for that variable
                - "latitude" and "longitude" columns if spatial context is needed downstream

        Behavior:
          1. Determine partition date from the first row's "timestamp".
          2. Create folder root/var/ if it does not exist.
          3. Write or append `df` into the file root/var/dt=YYYY-MM-DD.parquet using Polars.
        """
        # Convert a Pandas DataFrame to Polars if needed:
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        # Extract partition path based on first row's timestamp:
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column to partition by date.")

        ts0 = df.select("timestamp").to_pandas()["timestamp"].iloc[0]
        part_path = self._path(var, ts0)
        part_path.parent.mkdir(parents=True, exist_ok=True)

        # Write or append the Polars DataFrame to Parquet:
        # - If file exists, use append=True
        # - If not, write a new file
        if part_path.exists():
            pl.write_parquet(df, part_path, mode="append")
        else:
            pl.write_parquet(df, part_path)

    def read_range(self, var: str, start_date: str, end_date: str) -> pl.DataFrame:
        """
        Read and concatenate all daily Parquet files for `var` between start_date and end_date inclusive.

        Args:
            var: Variable name (folder under root).
            start_date: "YYYY-MM-DD" string.
            end_date:   "YYYY-MM-DD" string.

        Returns:
            A single Polars DataFrame containing all rows for those dates, sorted by timestamp.
        """
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        folder = self.root / var
        if not folder.exists():
            return pl.DataFrame([])

        dfs = []
        for part_file in folder.glob(f"{self.partition_key}=*.parquet"):
            date_str = part_file.stem.split("=")[1]
            part_date = pd.to_datetime(date_str).date()
            if start <= part_date <= end:
                dfs.append(pl.read_parquet(part_file))

        if not dfs:
            return pl.DataFrame([])
        return pl.concat(dfs).sort("timestamp")
