"""
Streamlit dashboard for the Ukrainian air raid alerts time series project.

Run with: streamlit run app.py

Design notes (see README.md for full reasoning behind each decision):
- Data: Vadimkin/ukrainian-air-raid-sirens-dataset, volunteer_data_en.csv
- Zero-duration records dropped as source artifacts; long-duration
  records (>180 min) kept as genuine signal.
- Today's (incomplete) day excluded from trend/forecast calculations.
- Forecasting: Prophet, daily aggregation, per-region models.
- Up to 5 regions can be compared at once (keeps the UI responsive,
  since each region trains its own Prophet model).

IMPORTANT DISCLAIMER shown in the app: this tool is for historical/
informational purposes only. It does NOT replace official air raid
alert systems (e.g. air alarms, alerts.in.ua) and must never be used
to decide whether to take shelter during an actual alert.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_raw_data
from preprocessing import load_clean_dataset, daily_counts
from forecasting import make_forecast, backtest, fit_forecast_model, get_weekly_pattern

MAX_REGIONS = 5
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_UA = {
    "Monday": "Пн", "Tuesday": "Вт", "Wednesday": "Ср", "Thursday": "Чт",
    "Friday": "Пт", "Saturday": "Сб", "Sunday": "Нд",
}

st.set_page_config(page_title="Повітряні тривоги в Україні — Time Series", layout="wide")


@st.cache_data(ttl=3600)
def get_data():
    raw = load_raw_data()
    return load_clean_dataset(lambda: raw)


@st.cache_data(ttl=3600)
def get_daily_series(region):
    df = get_data()
    return daily_counts(df, region=region)


@st.cache_data(ttl=3600)
def get_backtest(region):
    daily = get_daily_series(region)
    return backtest(daily, holdout_days=14)


@st.cache_data(ttl=3600)
def get_forecast_and_pattern(region, periods):
    daily = get_daily_series(region)
    model = fit_forecast_model(daily)
    future = model.make_future_dataframe(periods=periods)
    full_forecast = model.predict(future)
    trimmed = full_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    weekly_pattern = get_weekly_pattern(full_forecast)
    return trimmed, weekly_pattern


df = get_data()
all_regions = sorted(df["region"].unique())

st.title("🚨 Повітряні тривоги в Україні: аналіз часових рядів")

st.warning(
    "⚠️ **Дисклеймер:** цей інструмент створено для історичного аналізу та "
    "освітніх цілей. Він НЕ замінює офіційні системи попередження "
    "(сирени, застосунки типу alerts.in.ua) і не повинен використовуватись "
    "для прийняття рішення про укриття під час реальної тривоги.",
    icon="⚠️",
)

st.caption(
    f"Джерело даних: [Vadimkin/ukrainian-air-raid-sirens-dataset]"
    f"(https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) "
    f"(volunteer_data_en.csv). Період: {df['started_at'].min().date()} — "
    f"{df['started_at'].max().date()}. Сьогоднішній (неповний) день виключено "
    f"з розрахунків тренду та прогнозу."
)

# --- Region selection ---
st.sidebar.header("Налаштування")
default_regions = ["Kyiv City", "Kharkivska oblast", "Lvivska oblast"]
default_regions = [r for r in default_regions if r in all_regions]

selected_regions = st.sidebar.multiselect(
    f"Оберіть регіони (до {MAX_REGIONS})",
    options=all_regions,
    default=default_regions,
)

if len(selected_regions) > MAX_REGIONS:
    st.sidebar.error(f"Будь ласка, оберіть не більше {MAX_REGIONS} регіонів.")
    selected_regions = selected_regions[:MAX_REGIONS]

filter_night_only = st.sidebar.checkbox("Тільки нічні тривоги (≈22:00–06:00 UTC)", value=False)
filter_last_30_days = st.sidebar.checkbox("Тільки останні 30 днів", value=False)

forecast_horizon = st.sidebar.slider("Горизонт прогнозу (днів)", 7, 30, 14)

if not selected_regions:
    st.info("👈 Оберіть хоча б один регіон у боковій панелі, щоб побачити аналіз.")
    st.stop()

# --- Apply optional filters to the raw df for the "overview" section ---
filtered_df = df[df["region"].isin(selected_regions)].copy()
if filter_night_only:
    filtered_df = filtered_df[filtered_df["is_night"]]
if filter_last_30_days:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    filtered_df = filtered_df[filtered_df["started_at"] >= cutoff]

# === SECTION 1: Trend over time ===
st.header("1. Динаміка тривог у часі")

trend_fig = go.Figure()
for region in selected_regions:
    daily = get_daily_series(region)
    if filter_last_30_days:
        cutoff_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)).date()
        daily = daily[daily["ds"].dt.date >= cutoff_date]
    trend_fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode="lines", name=region))

trend_fig.update_layout(
    xaxis_title="Дата", yaxis_title="Кількість тривог/день",
    legend_title="Регіон", hovermode="x unified",
)
st.plotly_chart(trend_fig, width='stretch')

if filter_night_only:
    st.caption(
        "Примітка: фільтр 'нічні тривоги' застосовано лише до показників нижче "
        "(огляд, розподіл по годинах), а не до графіку тренду вище — тренд завжди "
        "показує повну денну кількість тривог для коректного порівняння з прогнозом."
    )

# === SECTION 2: Seasonality patterns ===
st.header("2. Сезонність: коли частіше трапляються тривоги")

col1, col2 = st.columns(2)

with col1:
    st.subheader("По годинах доби (UTC)")
    hourly = filtered_df.groupby(["region", "hour"]).size().reset_index(name="count")
    hour_fig = go.Figure()
    for region in selected_regions:
        sub = hourly[hourly["region"] == region]
        hour_fig.add_trace(go.Bar(x=sub["hour"], y=sub["count"], name=region))
    hour_fig.update_layout(barmode="group", xaxis_title="Година (UTC)", yaxis_title="Кількість тривог")
    st.plotly_chart(hour_fig, width='stretch')

with col2:
    st.subheader("По днях тижня")
    weekday_counts = filtered_df.groupby(["region", "weekday"]).size().reset_index(name="count")
    wd_fig = go.Figure()
    for region in selected_regions:
        sub = weekday_counts[weekday_counts["region"] == region].set_index("weekday").reindex(WEEKDAY_ORDER).reset_index()
        wd_fig.add_trace(go.Bar(x=[WEEKDAY_UA[d] for d in sub["weekday"]], y=sub["count"], name=region))
    wd_fig.update_layout(barmode="group", xaxis_title="День тижня", yaxis_title="Кількість тривог")
    st.plotly_chart(wd_fig, width='stretch')

# === SECTION 3: Duration ===
st.header("3. Тривалість тривог")
st.caption(
    "Медіана показана поряд із середнім, бо невелика частка дуже довгих тривог "
    "(переважно з прифронтових регіонів у періоди інтенсивних бойових дій) "
    "суттєво підвищує середнє значення без зміни типової тривоги."
)

duration_stats = (
    filtered_df.groupby("region")["duration_min"]
    .agg(median_min="median", mean_min="mean")
    .round(1)
    .reindex(selected_regions)
)
st.dataframe(duration_stats, width='stretch')

# === SECTION 4: Region comparison ===
st.header("4. Порівняння регіонів (загальна кількість тривог)")
region_totals = filtered_df.groupby("region").size().sort_values(ascending=False)
totals_fig = go.Figure(go.Bar(x=region_totals.index, y=region_totals.values))
totals_fig.update_layout(xaxis_title="Регіон", yaxis_title="Кількість тривог (з обраними фільтрами)")
st.plotly_chart(totals_fig, width='stretch')

# === SECTION 5: Forecast ===
st.header("5. Прогноз")
st.caption(
    "Прогноз побудовано окремою моделлю Prophet для кожного регіону "
    "(денна агрегація, тренд + тижнева + річна сезонність). Показано очікуваний "
    "діапазон (80% довірчий інтервал), не точне число — точність моделі обмежена, "
    "бо реальна кількість тривог залежить від воєнної ситуації, яку календарна "
    "модель не передбачає."
)

for region in selected_regions:
    with st.expander(f"📍 {region}", expanded=(len(selected_regions) <= 2)):
        try:
            forecast, weekly_pattern = get_forecast_and_pattern(region, forecast_horizon)
            bt = get_backtest(region)

            daily = get_daily_series(region)
            history_tail = daily.tail(60)
            future_part = forecast[forecast["ds"] > daily["ds"].max()]

            fc_fig = go.Figure()
            fc_fig.add_trace(go.Scatter(x=history_tail["ds"], y=history_tail["y"],
                                         mode="lines", name="Факт (останні 60 днів)", line=dict(color="gray")))
            fc_fig.add_trace(go.Scatter(x=future_part["ds"], y=future_part["yhat"],
                                         mode="lines", name="Прогноз", line=dict(color="firebrick")))
            fc_fig.add_trace(go.Scatter(
                x=pd.concat([future_part["ds"], future_part["ds"][::-1]]),
                y=pd.concat([future_part["yhat_upper"], future_part["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(178,34,34,0.15)", line=dict(color="rgba(0,0,0,0)"),
                name="80% довірчий інтервал", showlegend=True,
            ))
            fc_fig.update_layout(xaxis_title="Дата", yaxis_title="Кількість тривог/день")
            st.plotly_chart(fc_fig, width='stretch')

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE (backtest, 14 днів)", f"{bt['MAE']:.1f} тривог/день")
            m2.metric("MAPE (backtest)", f"{bt['MAPE_%']:.1f}%" if bt["MAPE_%"] else "н/д")
            m3.metric("Середнє у тестовому періоді", f"{bt['actual_mean']:.1f}/день")

            st.caption(
                "Високий MAPE при низькій середній кількості тривог/день — "
                "очікувана математична особливість метрики на малих числах, "
                "не показник 'поламаної' моделі. Дивіться MAE для абсолютної похибки."
            )

        except ValueError as e:
            st.error(f"Недостатньо даних для прогнозу: {e}")

st.divider()
st.caption(
    "Проєкт виконано як навчальний мініпроєкт (KSE AI Agentic Summer School). "
    "Код: [GitHub repo]. Дані оновлюються щодня волонтерами каналу eTryvoga."
)
