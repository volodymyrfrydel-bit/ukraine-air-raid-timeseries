"""
Preprocessing module: cleaning and feature engineering for the
Ukrainian air raid sirens dataset (volunteer_data_en.csv).

Decisions made here (agreed with the project owner, see README for
full reasoning):

1. Zero-duration records (started_at == finished_at) are dropped.
   All 5 such records in the dataset have naive=False, meaning the
   source system received an explicit end-of-alert signal at the
   same second as the start — physically implausible, treated as a
   source-side technical artifact. Negligible share of data (~0.005%).

2. Long-duration records (>180 min) are KEPT, not dropped.
   None of them are naive=True, i.e. they are not "default +30min"
   fill-ins — the source received a real end signal, just much later.
   Many cluster around known intense combat periods (e.g. early March
   2022 in Zaporizka oblast). Treated as genuine signal, not noise.

3. Because long-duration events skew the mean, duration statistics
   should always be reported with the median alongside (or instead
   of) the mean.
"""

import pandas as pd
import numpy as np

ZERO_DURATION_THRESHOLD = pd.Timedelta(0)
LONG_DURATION_THRESHOLD_MIN = 180  # flag only, not used for exclusion


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop zero-duration records. Returns a copy.
    """
    duration = df["finished_at"] - df["started_at"]
    zero_mask = duration == ZERO_DURATION_THRESHOLD

    n_dropped = int(zero_mask.sum())
    if n_dropped > 0:
        print(f"[clean_data] Dropping {n_dropped} zero-duration record(s) "
              f"as source-side artifacts (see preprocessing.py docstring).")

    cleaned = df.loc[~zero_mask].copy()
    return cleaned


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns used throughout the analysis and dashboard:

        duration_min     : float, alert duration in minutes
        is_long_duration  : bool, flag for >180 min (informational only,
                             NOT used to exclude records — see module docstring)
        date              : date (no time) of started_at, for daily aggregation
        hour              : hour of day (0-23) of started_at, UTC
        weekday           : day name (Monday, Tuesday, ...) of started_at
        month             : year-month string (YYYY-MM) of started_at
        is_night          : bool, True if started_at hour is in [22, 6) UTC
                             approx. local Kyiv time is UTC+2/+3, used as a
                             rough proxy for "night-time alert" filtering
    """
    out = df.copy()

    out["duration_min"] = (out["finished_at"] - out["started_at"]).dt.total_seconds() / 60
    out["is_long_duration"] = out["duration_min"] > LONG_DURATION_THRESHOLD_MIN

    out["date"] = out["started_at"].dt.date
    out["hour"] = out["started_at"].dt.hour
    out["weekday"] = out["started_at"].dt.day_name()
    out["month"] = out["started_at"].dt.strftime("%Y-%m")

    # Night-time proxy: UTC 22:00-06:00 covers roughly 00:00-08:00 Kyiv time
    # in winter (UTC+2) and 01:00-09:00 in summer (UTC+3). This is a rough
    # approximation for filtering, not a precise local-time conversion.
    out["is_night"] = (out["hour"] >= 22) | (out["hour"] < 6)

    return out


def load_clean_dataset(raw_loader_fn) -> pd.DataFrame:
    """
    Convenience wrapper: load raw data via the given loader function,
    clean it, and add features. Keeps data_loader.py independent of
    preprocessing decisions.
    """
    raw = raw_loader_fn()
    cleaned = clean_data(raw)
    enriched = add_features(cleaned)
    return enriched


def daily_counts(df: pd.DataFrame, region: str = None, drop_incomplete_today: bool = True) -> pd.DataFrame:
    """
    Aggregate to a daily alert-count time series, suitable for Prophet
    (columns: ds, y) or for plotting.

    region: if provided, filter to a single region first. If None,
            aggregates across all of Ukraine.
    drop_incomplete_today: if True (default), drops the current date's
            row. The dataset is updated daily but the *current* day is
            always a partial day (data collection is ongoing), so its
            count is artificially low and would distort trend/forecast
            if treated as a complete observation.
    """
    data = df if region is None else df[df["region"] == region]

    counts = data.groupby("date").size().reset_index()
    counts.columns = ["ds", "y"]
    counts["ds"] = pd.to_datetime(counts["ds"])

    if drop_incomplete_today:
        today = pd.Timestamp.now(tz="UTC").date()
        counts = counts[counts["ds"].dt.date != today]

    return counts.sort_values("ds").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import load_raw_data

    df = load_clean_dataset(load_raw_data)
    print(f"\nFinal shape: {df.shape}")
    print(f"\nDuration stats (minutes):\n{df['duration_min'].describe()}")
    print(f"\nMedian duration: {df['duration_min'].median():.1f} min")
    print(f"Mean duration: {df['duration_min'].mean():.1f} min")
    print(f"\nLong-duration flagged: {df['is_long_duration'].sum()} "
          f"({df['is_long_duration'].mean()*100:.1f}%)")
    print(f"\nSample:\n{df.head(3)}")
