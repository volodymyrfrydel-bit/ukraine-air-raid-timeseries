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
import streamlit.components.v1 as components
import pandas as pd
import json
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

CHARTJS_PATH = Path(__file__).resolve().parent / "src" / "static" / "chart.umd.js"


def _load_chartjs_source() -> str:
    """
    Chart.js is bundled locally (src/static/chart.umd.js) rather than loaded
    from a CDN. This avoids any dependency on external network access at
    runtime (some sandboxed/offline environments block CDN domains), and
    avoids any risk of a CDN outage breaking the dashboard.
    """
    return CHARTJS_PATH.read_text()


def render_hourly_clock(hourly_by_region: dict, height: int = 480) -> str:
    """
    Builds the self-contained HTML/Chart.js widget for the hour-of-day
    clock chart. hourly_by_region: {region_name: [count for hour 0..23]}.
    Colors map low->high counts on a green->amber->red scale, computed
    per-region (each region's own max), so a quiet region doesn't look
    uniformly "red" just because its busiest hour is still low in
    absolute terms relative to a frontline region.
    """
    region_names = list(hourly_by_region.keys())
    data_json = json.dumps(hourly_by_region)
    buttons_html = "".join(
        f'<button class="clock-btn" data-region="{r}" '
        f'style="padding:6px 14px;border-radius:8px;border:0.5px solid rgba(128,128,128,0.4);'
        f'background:{"#e6f1fb" if i == 0 else "transparent"};'
        f'color:{"#0c447c" if i == 0 else "#444"};font-size:13px;cursor:pointer;margin-right:8px;">{r}</button>'
        for i, r in enumerate(region_names)
    )

    return f"""
    <div style="font-family:sans-serif;">
      <div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:12px;">{buttons_html}</div>
      <div style="position:relative;width:100%;height:{height-90}px;max-width:480px;margin:0 auto;">
        <canvas id="clockChart"></canvas>
      </div>
      <div style="display:flex;align-items:center;justify-content:center;gap:6px;margin-top:0.75rem;font-size:12px;color:#666;">
        <span>менше тривог</span>
        <span style="display:inline-block;width:90px;height:8px;border-radius:4px;background:linear-gradient(to right,#639922,#EF9F27,#E24B4A);"></span>
        <span>більше тривог</span>
      </div>
      <p style="text-align:center;font-size:12px;color:#999;margin-top:0.4rem;">Години у форматі UTC</p>
    </div>
    <script>{_load_chartjs_source()}</script>
    <script>
    const clockData = {data_json};
    const regionNames = {json.dumps(region_names)};
    const labels = Array.from({{length:24}}, (_, i) => i + ':00');

    function colorFor(v, max) {{
      const t = max > 0 ? v / max : 0;
      let r, g, b;
      if (t < 0.5) {{
        const k = t / 0.5;
        r = Math.round(99 + (239 - 99) * k);
        g = Math.round(153 + (167 - 153) * k);
        b = Math.round(34 + (39 - 34) * k);
      }} else {{
        const k = (t - 0.5) / 0.5;
        r = Math.round(239 + (226 - 239) * k);
        g = Math.round(167 + (75 - 167) * k);
        b = Math.round(39 + (74 - 39) * k);
      }}
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    const hourLabelPlugin = {{
      id: 'hourLabels',
      afterDraw(chart) {{
        const {{ ctx, chartArea }} = chart;
        const cx = (chartArea.left + chartArea.right) / 2;
        const cy = (chartArea.top + chartArea.bottom) / 2;
        const radius = Math.min(chartArea.right - chartArea.left, chartArea.bottom - chartArea.top) / 2 + 16;
        ctx.save();
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#888780';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (let i = 0; i < 24; i++) {{
          const displayHour = i === 0 ? 24 : i;
          const angle = (i / 24) * 2 * Math.PI - Math.PI / 2;
          const x = cx + radius * Math.cos(angle);
          const y = cy + radius * Math.sin(angle);
          ctx.fillText(displayHour, x, y);
        }}
        ctx.restore();
      }}
    }};

    const firstRegion = regionNames[0];
    const ctx = document.getElementById('clockChart');
    let chart = new Chart(ctx, {{
      type: 'polarArea',
      data: {{
        labels: labels,
        datasets: [{{
          data: clockData[firstRegion],
          backgroundColor: clockData[firstRegion].map(v => colorFor(v, Math.max(...clockData[firstRegion]))),
          borderColor: 'rgba(255,255,255,0.4)',
          borderWidth: 1
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        layout: {{ padding: 28 }},
        animation: {{ duration: 500, easing: 'easeOutQuart' }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            backgroundColor: 'rgba(40,40,38,0.92)',
            padding: 10,
            bodyFont: {{ size: 12 }},
            cornerRadius: 6,
            displayColors: false,
            callbacks: {{ label: (item) => item.label + ' \u2014 ' + item.raw + ' \u0442\u0440\u0438\u0432\u043e\u0433' }}
          }}
        }},
        scales: {{
          r: {{
            ticks: {{ display: false }},
            grid: {{ color: 'rgba(128,128,128,0.15)' }},
            angleLines: {{ color: 'rgba(128,128,128,0.15)' }}
          }}
        }}
      }},
      plugins: [hourLabelPlugin]
    }});

    document.querySelectorAll('.clock-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const region = btn.dataset.region;
        const vals = clockData[region];
        const max = Math.max(...vals);
        chart.data.datasets[0].data = vals;
        chart.data.datasets[0].backgroundColor = vals.map(v => colorFor(v, max));
        chart.update();
        document.querySelectorAll('.clock-btn').forEach(b => {{
          if (b === btn) {{ b.style.background = '#e6f1fb'; b.style.color = '#0c447c'; }}
          else {{ b.style.background = 'transparent'; b.style.color = '#444'; }}
        }});
      }});
    }});
    </script>
    """


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
    hourly_by_region = {}
    for region in selected_regions:
        sub = hourly[hourly["region"] == region].set_index("hour").reindex(range(24), fill_value=0)
        hourly_by_region[region] = sub["count"].tolist()
    components.html(render_hourly_clock(hourly_by_region), height=500)

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
