"""
Forecasting module using Prophet.

Key design decisions (validated via backtesting, see README):

1. Daily aggregation, not weekly. Daily gives more granularity and lets
   Prophet's own weekly-seasonality component reveal day-of-week patterns
   (e.g. weekends being quieter) automatically, rather than averaging
   them away.

2. Per-region models, not just a national aggregate. Backtested on
   8 regions across Ukraine (west/center/east/south/north):
   high-frequency, frontline-adjacent regions (Kharkivska, Dnipropetrovska)
   produced the BEST MAPE (16-20%), better than the national baseline
   (~21%). Low-frequency regions (Lvivska, Odeska) showed much worse
   MAPE (70-85%) -- but this is a known mathematical property of MAPE
   on small counts (e.g. predicting 1.4 vs an actual of 3 is a huge
   percentage error despite a small absolute error), not a sign the
   model performs worse there. This is why MAE is always reported
   alongside MAPE.

3. The current (in-progress) day is always excluded from training data
   -- see preprocessing.daily_counts(drop_incomplete_today=True). The
   dataset updates daily but "today" is always a partial day.

Honest limitation (for README / dashboard disclaimer): even the best
backtested MAPE here is ~16-20%, meaning day-to-day forecasts carry
real uncertainty. Day-to-day alert counts are driven by actual military
activity, which a historical calendar-based model cannot predict --
Prophet captures trend and weekly seasonality, not specific events.
The forecast should be read as an expected range, not a precise count.
"""

import pandas as pd
from prophet import Prophet
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DEFAULT_FORECAST_DAYS = 14
MIN_TRAINING_DAYS = 60  # below this, Prophet has too little signal to be meaningful


def fit_forecast_model(daily_df: pd.DataFrame) -> Prophet:
    """
    Fit a Prophet model on a daily (ds, y) series.
    Raises ValueError if there isn't enough data.
    """
    if len(daily_df) < MIN_TRAINING_DAYS:
        raise ValueError(
            f"Not enough data to forecast reliably: {len(daily_df)} days "
            f"(minimum {MIN_TRAINING_DAYS}). This region may have too few "
            f"recorded alerts for a meaningful daily forecast."
        )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
    )
    model.fit(daily_df)
    return model


def make_forecast(daily_df: pd.DataFrame, periods: int = DEFAULT_FORECAST_DAYS) -> pd.DataFrame:
    """
    Fit a model and return the forecast dataframe (history + future),
    with columns: ds, yhat, yhat_lower, yhat_upper.
    """
    model = fit_forecast_model(daily_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def backtest(daily_df: pd.DataFrame, holdout_days: int = 14) -> dict:
    """
    Train on all but the last `holdout_days`, predict that window,
    and compare against actuals. Returns MAE and MAPE plus the
    actual mean (for context -- MAPE is misleading on low-volume series).
    """
    if len(daily_df) < MIN_TRAINING_DAYS + holdout_days:
        raise ValueError("Not enough data for backtesting with this holdout size.")

    train = daily_df.iloc[:-holdout_days].copy()
    test = daily_df.iloc[-holdout_days:].copy()

    model = fit_forecast_model(train)
    future = model.make_future_dataframe(periods=holdout_days)
    forecast = model.predict(future)

    pred = forecast[["ds", "yhat"]].tail(holdout_days).reset_index(drop=True)
    comparison = test.reset_index(drop=True).merge(pred, on="ds")
    comparison["abs_error"] = (comparison["y"] - comparison["yhat"]).abs()

    mae = comparison["abs_error"].mean()
    # Guard against division by zero for days with y=0
    mape = (comparison["abs_error"] / comparison["y"].replace(0, pd.NA)).mean() * 100

    return {
        "MAE": round(mae, 2),
        "MAPE_%": round(mape, 1) if pd.notna(mape) else None,
        "actual_mean": round(test["y"].mean(), 2),
        "holdout_days": holdout_days,
        "comparison": comparison,
    }


def get_weekly_pattern(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the day-of-week seasonality component from a forecast
    (requires the forecast to include a 'weekly' column -- call
    model.predict() directly for this, not the trimmed make_forecast()
    output). Returns one row per weekday with the average seasonal
    effect (positive = busier than average, negative = quieter).
    """
    if "weekly" not in forecast_df.columns:
        raise ValueError("forecast_df must include the 'weekly' column "
                          "(use model.predict(future) directly, not make_forecast()).")

    df = forecast_df.copy()
    df["weekday"] = pd.to_datetime(df["ds"]).dt.day_name()
    pattern = df.groupby("weekday")["weekly"].mean()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return pattern.reindex(order)
