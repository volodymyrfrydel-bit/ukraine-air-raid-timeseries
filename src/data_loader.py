"""
Data loading module for Ukrainian air raid sirens dataset.

Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
File used: datasets/volunteer_data_en.csv (volunteer-collected, oblast-level,
covers Feb 25 2022 onward — one day earlier than the official dataset).

Note on regions: 25 distinct values, including "Kyiv City" (a city with
special administrative status, not an oblast) alongside "Kyivska oblast".
Crimea and the permanent Luhansk siren (active since Apr 2022) are NOT
included in this dataset (per upstream README) — this is expected, not
a data quality issue.
"""

from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "volunteer_data_en.csv"

EXPECTED_COLUMNS = ["region", "started_at", "finished_at", "naive"]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw volunteer air raid siren CSV and parse timestamps.

    Returns a DataFrame with:
        region       : str, oblast/city name
        started_at   : datetime64[ns, UTC]
        finished_at  : datetime64[ns, UTC]
        naive        : bool, True if finished_at is an approximation
                       (started_at + 30 min, no real end-of-alert message)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Download it from https://raw.githubusercontent.com/Vadimkin/"
            "ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"
        )

    df = pd.read_csv(path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df["naive"] = df["naive"].astype(bool)

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run basic sanity checks and return a summary dict.
    Does not raise — intended for a quick diagnostic print during EDA.
    """
    duration = df["finished_at"] - df["started_at"]

    summary = {
        "n_records": len(df),
        "n_regions": df["region"].nunique(),
        "date_range": (df["started_at"].min(), df["started_at"].max()),
        "n_duplicates": df.duplicated().sum(),
        "n_nulls": int(df.isnull().sum().sum()),
        "pct_naive": round(df["naive"].mean() * 100, 2),
        "n_negative_duration": int((duration < pd.Timedelta(0)).sum()),
        "n_zero_duration": int((duration == pd.Timedelta(0)).sum()),
        "max_duration_hours": round(duration.max().total_seconds() / 3600, 2),
    }
    return summary


if __name__ == "__main__":
    df = load_raw_data()
    summary = validate_data(df)
    for key, value in summary.items():
        print(f"{key}: {value}")
