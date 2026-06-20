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
HAMMERJS_PATH = Path(__file__).resolve().parent / "src" / "static" / "hammer.min.js"
ZOOM_PLUGIN_PATH = Path(__file__).resolve().parent / "src" / "static" / "chartjs-plugin-zoom.min.js"


def _load_chartjs_source() -> str:
    """
    Chart.js is bundled locally (src/static/chart.umd.js) rather than loaded
    from a CDN. This avoids any dependency on external network access at
    runtime (some sandboxed/offline environments block CDN domains), and
    avoids any risk of a CDN outage breaking the dashboard.
    """
    return CHARTJS_PATH.read_text()


def _load_zoom_plugin_source() -> str:
    """
    chartjs-plugin-zoom (+ hammer.js, its touch/drag dependency) bundled
    locally for the same reason as Chart.js itself -- no CDN dependency.
    Used by the trend chart to support drag-to-pan and pinch/scroll-to-zoom
    on the time axis.
    """
    return HAMMERJS_PATH.read_text() + "\n" + ZOOM_PLUGIN_PATH.read_text()


REGION_COLORS = [
    "#185fa5", "#85b7eb", "#e24b4a", "#639922", "#854f0b",
]


def render_trend_chart(dates: list, series_by_region: dict, height: int = 460) -> str:
    """
    Builds the self-contained HTML/Chart.js line chart for the alert
    trend over time. dates: list of 'YYYY-MM-DD' strings (a single shared
    calendar axis -- all regions are reindexed onto this same axis with
    0-fill for days with no recorded alerts, so multiple regions compare
    correctly rather than each drawing its own sparse date axis).
    series_by_region: {region_name: [count per date]}.

    Includes:
    - Region toggle buttons (click to show/hide a line, same pattern as
      the hourly clock chart)
    - Quick time-range buttons (5y/1y/6m/3m/1m/1week/today)
    - Drag-to-pan and scroll/pinch-to-zoom on the time axis
      (chartjs-plugin-zoom, bundled locally)
    """
    region_names = list(series_by_region.keys())
    colors = {r: REGION_COLORS[i % len(REGION_COLORS)] for i, r in enumerate(region_names)}

    region_buttons_html = "".join(
        f'<button class="region-btn" data-region="{r}" data-idx="{i}" data-color="{colors[r]}" '
        f'style="padding:6px 14px;border-radius:8px;border:1.5px solid {colors[r]};'
        f'background:{colors[r]};color:#fff;font-weight:500;font-size:13px;cursor:pointer;margin-right:8px;margin-bottom:6px;">{r}</button>'
        for i, r in enumerate(region_names)
    )

    range_buttons = [
        ("5y", "5р"), ("1y", "1р"), ("6m", "6м"), ("3m", "3м"),
        ("1m", "1м"), ("1w", "тиждень"),
    ]
    range_buttons_html = "".join(
        f'<button class="range-btn" data-range="{key}" '
        f'style="padding:5px 12px;border-radius:8px;border:0.5px solid rgba(128,128,128,0.4);'
        f'background:transparent;color:#444;font-size:12px;cursor:pointer;margin-right:6px;margin-bottom:6px;">{label}</button>'
        for key, label in range_buttons
    )

    datasets_json = json.dumps([
        {
            "label": r,
            "data": series_by_region[r],
            "borderColor": colors[r],
            "borderWidth": 1.5,
            "pointRadius": 0,
            "tension": 0,
            "hidden": False,
        }
        for r in region_names
    ])

    return f"""
    <div style="font-family:sans-serif;">
      <div style="display:flex;flex-wrap:wrap;">{region_buttons_html}</div>
      <div style="display:flex;flex-wrap:wrap;margin-bottom:10px;">{range_buttons_html}</div>
      <div style="position:relative;width:100%;height:{height-110}px;">
        <canvas id="trendChart"></canvas>
      </div>
      <p style="text-align:center;font-size:11px;color:#999;margin-top:6px;">
        Перетягуйте графік для зсуву, колесо миші / жест масштабування для збільшення
      </p>
    </div>
    <script>{_load_chartjs_source()}</script>
    <script>{_load_zoom_plugin_source()}</script>
    <script>
    const trendDates = {json.dumps(dates)};
    const trendDatasets = {datasets_json};
    let activeRegions = new Set(trendDatasets.map(d => d.label));

    const trendCtx = document.getElementById('trendChart');
    const trendChart = new Chart(trendCtx, {{
      type: 'line',
      data: {{ labels: trendDates, datasets: trendDatasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{ mode: 'index', intersect: false }},
        animation: {{ duration: 300 }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            backgroundColor: 'rgba(40,40,38,0.92)',
            cornerRadius: 6,
            padding: 10
          }},
          zoom: {{
            pan: {{ enabled: true, mode: 'x' }},
            zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: 'x' }},
            limits: {{ x: {{ min: 0, max: trendDates.length - 1 }} }}
          }}
        }},
        scales: {{
          x: {{ ticks: {{ maxTicksLimit: 10, autoSkip: true }} }},
          y: {{ beginAtZero: true, title: {{ display: true, text: 'Кількість тривог/день' }} }}
        }}
      }}
    }});

    document.querySelectorAll('.region-btn').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        const region = btn.dataset.region;
        const idx = parseInt(btn.dataset.idx);
        const color = btn.dataset.color;
        if (activeRegions.has(region)) {{
          activeRegions.delete(region);
          btn.style.background = 'transparent';
          btn.style.color = color;
        }} else {{
          activeRegions.add(region);
          btn.style.background = color;
          btn.style.color = '#fff';
        }}
        trendChart.data.datasets[idx].hidden = !activeRegions.has(region);
        trendChart.update();
      }});
    }});

    function setTrendRange(rangeKey) {{
      const total = trendDates.length;
      let days;
      switch(rangeKey) {{
        case '5y': days = total; break;
        case '1y': days = 365; break;
        case '6m': days = 182; break;
        case '3m': days = 91; break;
        case '1m': days = 30; break;
        case '1w': days = 7; break;
        default: days = total;
      }}
      const startIdx = Math.max(0, total - days);
      trendChart.zoomScale('x', {{ min: startIdx, max: total - 1 }});
    }}

    document.querySelectorAll('.range-btn').forEach(btn => {{
      btn.addEventListener('click', () => setTrendRange(btn.dataset.range));
    }});
    </script>
    """


def render_forecast_cards(forecast_data: list) -> str:
    """
    Builds a row of small forecast cards, one per region, shown side by
    side (instead of a vertical list of expanders that hides most
    regions behind a click). Each card's percentage change is colored
    on a green-to-red scale based on magnitude (not direction alone --
    a small change in either direction stays neutral gray).

    The comparison window (last N days vs. next N forecasted days) uses
    N = forecast_horizon from the sidebar slider, so the slider actually
    changes what's displayed -- previously this was hardcoded to 7 days
    regardless of the slider, which made the slider a no-op from the
    user's perspective (caught in review).

    forecast_data: list of dicts, one per region:
        {region, last_14d_avg, pct_change, abs_change, mae, compare_window}
    """
    def pct_color(pct):
        abs_pct = abs(pct)
        if abs_pct < 5:
            return "#5F5E5A"  # neutral gray -- not a meaningful change
        elif abs_pct < 15:
            return "#854F0B"  # amber -- moderate change
        else:
            return "#A32D2D" if pct > 0 else "#3B6D11"  # red for increase, green for decrease

    cards_html = ""
    for d in forecast_data:
        arrow = "↑" if d["pct_change"] > 3 else ("↓" if d["pct_change"] < -3 else "→")
        color = pct_color(d["pct_change"])
        window = d["compare_window"]
        cards_html += f"""
        <div style="background:#fafafa;border:0.5px solid rgba(128,128,128,0.25);border-radius:10px;
                    padding:0.9rem;min-width:170px;flex:1;">
          <p style="font-size:13px;font-weight:500;margin:0 0 8px;color:#333;">{d['region']}</p>
          <p style="font-size:11px;color:#888;margin:0 0 2px;">Останні 14 днів</p>
          <p style="font-size:18px;font-weight:500;margin:0 0 10px;">{d['last_14d_avg']:.1f}/день</p>
          <p style="font-size:11px;color:#888;margin:0 0 2px;">Зміна, наступні {window} дн.</p>
          <p style="font-size:18px;font-weight:500;margin:0 0 10px;color:{color};">
            {arrow} {d['pct_change']:+.0f}% ({d['abs_change']:+.1f}/день)
          </p>
          <p style="font-size:11px;color:#888;margin:0 0 2px;">Похибка прогнозу</p>
          <p style="font-size:14px;font-weight:500;margin:0;">±{d['mae']:.1f}/день</p>
        </div>
        """

    return f"""
    <div style="display:flex;flex-wrap:wrap;gap:10px;font-family:sans-serif;">
      {cards_html}
    </div>
    """


def render_weekend_info(weekend_data: list) -> str:
    """
    Informational card grid: average alerts/day on weekends vs. weekdays
    for each selected region. Purely descriptive (no recommendation,
    no "safer to travel here" framing) -- see project README for why:
    using historical frequency to suggest where shelter/travel decisions
    should be made is out of scope, since it could create false
    confidence about actual real-time threat level.

    weekend_data: list of dicts {region, weekend_avg, weekday_avg}.
    """
    cards_html = ""
    for d in weekend_data:
        diff_pct = ((d["weekend_avg"] - d["weekday_avg"]) / d["weekday_avg"] * 100) if d["weekday_avg"] else 0
        cards_html += f"""
        <div style="background:#fafafa;border:0.5px solid rgba(128,128,128,0.25);border-radius:10px;
                    padding:0.9rem;min-width:170px;flex:1;">
          <p style="font-size:13px;font-weight:500;margin:0 0 8px;color:#333;">{d['region']}</p>
          <p style="font-size:11px;color:#888;margin:0 0 2px;">Вихідні (Сб-Нд), середнє/день</p>
          <p style="font-size:18px;font-weight:500;margin:0 0 10px;">{d['weekend_avg']:.1f}</p>
          <p style="font-size:11px;color:#888;margin:0 0 2px;">Будні, середнє/день</p>
          <p style="font-size:18px;font-weight:500;margin:0;">{d['weekday_avg']:.1f}</p>
        </div>
        """

    return f"""
    <div style="display:flex;flex-wrap:wrap;gap:10px;font-family:sans-serif;">
      {cards_html}
    </div>
    """


def render_weekday_chart(weekday_by_region: dict, height: int = 460) -> str:
    """
    Builds the self-contained HTML/Chart.js bar chart for day-of-week
    distribution, with region toggle buttons above it (same pattern as
    the hourly clock and trend charts). weekday_by_region:
    {region_name: [count for Mon..Sun]}.
    """
    region_names = list(weekday_by_region.keys())
    colors = {r: REGION_COLORS[i % len(REGION_COLORS)] for i, r in enumerate(region_names)}
    labels_ua = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]

    region_buttons_html = "".join(
        f'<button class="wd-region-btn" data-region="{r}" data-idx="{i}" data-color="{colors[r]}" '
        f'style="padding:6px 14px;border-radius:8px;border:1.5px solid {colors[r]};'
        f'background:{colors[r]};color:#fff;font-weight:500;font-size:13px;cursor:pointer;margin-right:8px;margin-bottom:6px;">{r}</button>'
        for i, r in enumerate(region_names)
    )

    datasets_json = json.dumps([
        {
            "label": r,
            "data": weekday_by_region[r],
            "backgroundColor": colors[r],
            "hidden": False,
        }
        for r in region_names
    ])

    return f"""
    <div style="font-family:sans-serif;">
      <div style="display:flex;flex-wrap:wrap;">{region_buttons_html}</div>
      <div style="position:relative;width:100%;height:{height-60}px;">
        <canvas id="weekdayChart"></canvas>
      </div>
    </div>
    <script>{_load_chartjs_source()}</script>
    <script>
    const wdLabels = {json.dumps(labels_ua)};
    const wdDatasets = {datasets_json};
    let wdActiveRegions = new Set(wdDatasets.map(d => d.label));

    const wdCtx = document.getElementById('weekdayChart');
    const wdChart = new Chart(wdCtx, {{
      type: 'bar',
      data: {{ labels: wdLabels, datasets: wdDatasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'День тижня' }} }},
          y: {{ beginAtZero: true, title: {{ display: true, text: 'Кількість тривог' }} }}
        }}
      }}
    }});

    document.querySelectorAll('.wd-region-btn').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        const region = btn.dataset.region;
        const idx = parseInt(btn.dataset.idx);
        const color = btn.dataset.color;
        if (wdActiveRegions.has(region)) {{
          wdActiveRegions.delete(region);
          btn.style.background = 'transparent';
          btn.style.color = color;
        }} else {{
          wdActiveRegions.add(region);
          btn.style.background = color;
          btn.style.color = '#fff';
        }}
        wdChart.data.datasets[idx].hidden = !wdActiveRegions.has(region);
        wdChart.update();
      }});
    }});
    </script>
    """


def render_duration_barrels(duration_by_region: dict, height: int = 220) -> str:
    """
    Builds the self-contained HTML/Chart.js horizontal bar chart for
    average alert duration per region. Only the mean is shown (not
    median) -- a single, intuitive number per region, since showing both
    confused non-technical users in review. Color gradient from green
    (shorter duration) to amber (longer duration), scaled to the max
    value among the currently selected regions.
    """
    region_names = list(duration_by_region.keys())
    values = list(duration_by_region.values())
    max_v = max(values) if values else 1

    def color_for(v):
        t = v / max_v if max_v else 0
        r = round(59 + (239 - 59) * t)
        g = round(109 + (167 - 109) * t)
        b = round(17 + (39 - 17) * t)
        return f"rgb({r},{g},{b})"

    colors = [color_for(v) for v in values]

    return f"""
    <div style="font-family:sans-serif;">
      <div style="position:relative;width:100%;height:{height-40}px;">
        <canvas id="durationChart" role="img" aria-label="Середня тривалість тривог по регіонах у хвилинах">
          {", ".join(f"{r}: {v:.1f} хв" for r, v in duration_by_region.items())}
        </canvas>
      </div>
      <div style="display:flex;align-items:center;justify-content:center;gap:6px;margin-top:8px;font-size:12px;color:#999;">
        <span>коротші</span>
        <span style="display:inline-block;width:90px;height:8px;border-radius:4px;background:linear-gradient(to right,#3B6D11,#EF9F27);"></span>
        <span>довші</span>
      </div>
    </div>
    <script>{_load_chartjs_source()}</script>
    <script>
    new Chart(document.getElementById('durationChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(region_names)},
        datasets: [{{ data: {json.dumps(values)}, backgroundColor: {json.dumps(colors)}, borderRadius: 6 }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ x: {{ title: {{ display: true, text: 'хвилин, в середньому' }} }} }}
      }}
    }});
    </script>
    """


def render_region_comparison(totals_by_region: dict, height: int = 220) -> str:
    """
    Builds the self-contained HTML/Chart.js horizontal bar chart for
    total alert count comparison across regions. Single neutral color
    (this is a count comparison, not a severity/intensity scale, so a
    gradient would be misleading here -- magnitude alone tells the story).
    """
    region_names = list(totals_by_region.keys())
    values = list(totals_by_region.values())

    return f"""
    <div style="font-family:sans-serif;">
      <div style="position:relative;width:100%;height:{height}px;">
        <canvas id="comparisonChart" role="img" aria-label="Порівняння загальної кількості тривог по регіонах">
          {", ".join(f"{r}: {v}" for r, v in totals_by_region.items())}
        </canvas>
      </div>
    </div>
    <script>{_load_chartjs_source()}</script>
    <script>
    new Chart(document.getElementById('comparisonChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(region_names)},
        datasets: [{{ data: {json.dumps(values)}, backgroundColor: '#378ADD', borderRadius: 6 }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ x: {{ title: {{ display: true, text: 'кількість тривог' }} }} }}
      }}
    }});
    </script>
    """


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

st.title("🚨 Повітряні тривоги в Україні від 02.2022 до сьогодні")

st.caption(
    f"Джерело даних: [Vadimkin/ukrainian-air-raid-sirens-dataset]"
    f"(https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) "
    f"(volunteer_data_en.csv). Період: {df['started_at'].min().date()} — "
    f"{df['started_at'].max().date()}. Сьогоднішній (неповний) день виключено "
    f"з розрахунків тренду та прогнозу."
)
st.caption(
    "⚠️ Історичні дані для аналізу, не замінюють офіційні сигнали тривоги."
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

forecast_horizon = st.sidebar.slider("Горизонт прогнозу (днів)", 7, 30, 14)
st.sidebar.caption(
    "Впливає на секцію 5 (Прогноз): порівнює останні N днів із наступними "
    "N прогнозованими днями, де N — обране тут число."
)

if not selected_regions:
    st.info("👈 Оберіть хоча б один регіон у боковій панелі, щоб побачити аналіз.")
    st.stop()

# --- Apply optional filters to the raw df for the "overview" section ---
filtered_df = df[df["region"].isin(selected_regions)].copy()
if filter_night_only:
    filtered_df = filtered_df[filtered_df["is_night"]]

# === SECTION 1: Trend over time ===
st.header("1. Динаміка тривог у часі")

# Build one shared calendar axis covering the full dataset (minus today,
# which is always incomplete -- see daily_counts). All regions are
# reindexed onto this same axis with 0-fill, so multiple lines compare
# correctly instead of each drawing its own sparse set of dates.
full_date_range = pd.date_range(
    df["started_at"].min().date(),
    (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date(),
    freq="D",
)
trend_series_by_region = {}
for region in selected_regions:
    daily = get_daily_series(region).set_index("ds")["y"]
    daily = daily.reindex(full_date_range, fill_value=0)
    trend_series_by_region[region] = daily.tolist()

trend_dates_str = full_date_range.strftime("%Y-%m-%d").tolist()
components.html(render_trend_chart(trend_dates_str, trend_series_by_region), height=470)

if filter_night_only:
    st.caption(
        "Примітка: фільтр 'нічні тривоги' застосовано лише до показників "
        "нижче (сезонність, тривалість, порівняння), а не до графіку тренду "
        "вище — тренд завжди показує повну денну кількість тривог за весь "
        "період для коректного порівняння з прогнозом. Для обмеження "
        "часового діапазону тренду використовуйте кнопки періоду над графіком."
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
    weekday_by_region = {}
    for region in selected_regions:
        sub = weekday_counts[weekday_counts["region"] == region].set_index("weekday").reindex(WEEKDAY_ORDER, fill_value=0)
        weekday_by_region[region] = sub["count"].tolist()
    components.html(render_weekday_chart(weekday_by_region), height=480)

# === SECTION 3: Duration ===
st.header("3. Тривалість тривог")

duration_means = (
    filtered_df.groupby("region")["duration_min"]
    .mean()
    .reindex(selected_regions)
)
components.html(render_duration_barrels(duration_means.to_dict()), height=220)
st.caption(
    "Показано лише середнє значення. Невелика частка дуже довгих тривог "
    "(переважно з прифронтових регіонів у періоди інтенсивних бойових дій) "
    "підвищує середнє порівняно з типовою тривогою — це врахована особливість "
    "даних, не помилка."
)

# === SECTION 4: Region comparison ===
st.header("4. Порівняння регіонів (загальна кількість тривог)")
region_totals = filtered_df.groupby("region").size().sort_values(ascending=False)
components.html(render_region_comparison(region_totals.to_dict()), height=220)

# === SECTION 5: Forecast ===
st.header("5. Прогноз")
st.caption(
    "Прогноз побудовано окремою моделлю Prophet для кожного регіону "
    "(денна агрегація, тренд + тижнева + річна сезонність). Показано очікуваний "
    "діапазон (80% довірчий інтервал), не точне число — точність моделі обмежена, "
    "бо реальна кількість тривог залежить від воєнної ситуації, яку календарна "
    "модель не передбачає."
)

forecast_cards_data = []
for region in selected_regions:
    try:
        forecast, weekly_pattern = get_forecast_and_pattern(region, forecast_horizon)
        bt = get_backtest(region)

        daily = get_daily_series(region)
        future_part = forecast[forecast["ds"] > daily["ds"].max()]

        # Compare like-for-like: average of the last N known days vs.
        # average of the next N forecasted days, where N = forecast_horizon
        # from the sidebar slider. Previously this was hardcoded to 7 days
        # regardless of the slider value, so moving the slider changed the
        # underlying Prophet forecast horizon but never changed what these
        # cards displayed -- a real bug caught in review.
        compare_window = min(forecast_horizon, len(daily))
        last_n_avg = daily["y"].tail(compare_window).mean()
        next_n_avg = future_part["yhat"].head(compare_window).mean()
        abs_change = next_n_avg - last_n_avg
        pct_change = (abs_change / last_n_avg * 100) if last_n_avg else 0

        forecast_cards_data.append({
            "region": region,
            "last_14d_avg": daily["y"].tail(14).mean(),
            "pct_change": pct_change,
            "abs_change": abs_change,
            "mae": bt["MAE"],
            "compare_window": compare_window,
        })
    except ValueError as e:
        st.error(f"{region}: недостатньо даних для прогнозу ({e})")

if forecast_cards_data:
    components.html(render_forecast_cards(forecast_cards_data), height=220)

st.caption(
    "**Похибка** — середня абсолютна помилка моделі на історичних даних "
    "(backtest): наскільки в середньому прогноз відхилявся від факту, "
    "у тривогах на день. Великий відсоток зміни при малій середній "
    "кількості тривог означає малу абсолютну різницю (показана в дужках) "
    "— кольори відображають саме % зміни, не абсолютну тривожність."
)

st.divider()
st.header("6. Вихідні vs будні дні")
st.caption(
    "Інформаційне порівняння середньої кількості тривог по днях тижня. "
    "Це історична статистика, не оцінка поточного ризику чи рекомендація "
    "щодо планування поїздок — реальна загроза завжди залежить від "
    "актуальної воєнної ситуації, а не від дня тижня."
)

if len(selected_regions) < 2:
    st.info("Оберіть щонайменше 2 регіони у боковій панелі, щоб побачити це порівняння.")
else:
    weekend_data = []
    for region in selected_regions:
        sub = df[df["region"] == region]
        n_weeks = max((sub["started_at"].max() - sub["started_at"].min()).days / 7, 1)
        weekend_total = len(sub[sub["weekday"].isin(["Saturday", "Sunday"])])
        weekday_total = len(sub[~sub["weekday"].isin(["Saturday", "Sunday"])])
        weekend_data.append({
            "region": region,
            "weekend_avg": (weekend_total / 2) / n_weeks,
            "weekday_avg": (weekday_total / 5) / n_weeks,
        })
    components.html(render_weekend_info(weekend_data), height=180)

st.divider()
st.caption(
    "Проєкт виконано як навчальний мініпроєкт (KSE AI Agentic Summer School). "
    "Код: [GitHub repo]. Дані оновлюються щодня волонтерами каналу eTryvoga."
)
