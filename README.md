# 🚨 Time Series Analysis: Air Raid Alerts in Ukraine

> Mini pet-project for **KSE AI Agentic Summer School** (Stage 2).
> Time series analysis, regional comparison, and short-term forecasting
> of air raid sirens in Ukraine, built with AI as the primary engineering tool.

**🔗 Live dashboard:** https://ukraine-air-raid-time.streamlit.app/

---

## 🇺🇦 Українська

### Що це таке і яку проблему вирішує

Цей проєкт перетворює сирий лог із 101 261 події (старт/кінець повітряної
тривоги по областях України, 25.02.2022 — сьогодні) на структуровану
картину: чи зростає інтенсивність тривог із часом, коли вони найчастіші
протягом доби/тижня, які регіони найбільш навантажені, і чи можна дати
короткостроковий прогноз.

Це не заміна офіційних систем попередження — це інструмент для
**інформованого планування**: коли краще запланувати поїздку,
як порівняти "ризик-профіль" різних областей, на що очікувати найближчі
1-2 тижні.

> ⚠️ **Важливо:** інструмент НЕ призначений і НЕ повинен використовуватись
> для прийняття рішення про укриття під час реальної тривоги. Під час
> тривоги завжди орієнтуйтесь на офіційні сигнали та йдіть в укриття.

### Дані

- **Джерело:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
  (MIT license, оновлюється щодня волонтерами каналу eTryvoga)
- **Файл:** `datasets/volunteer_data_en.csv`
- **Обсяг:** 101 266 записів → 101 261 після очищення, 25 регіонів,
  25.02.2022 — сьогодні
- Крим і "вічна" сирена Луганської області (з квітня 2022) **не входять**
  у датасет — це задокументована особливість джерела, не помилка нашого коду

### Архітектурні рішення (і чому саме так)

**1. Очищення аномалій тривалості**
- **5 записів з нульовою тривалістю** (start == finish, усі `naive=False`)
  видалено як технічний артефакт джерела — фізично неможлива подія.
- **Довгі тривоги (>180 хв, 3.5% записів)** залишено. Жодна з них не
  `naive=True` — тобто це не "дефолтне заповнення", а реальний відкладений
  сигнал відбою, часто в періоди інтенсивних бойових дій (наприклад,
  Запорізька область, березень 2022). Видалити їх означало б видалити
  сигнал, не шум.
- Через довгий хвіст довгих тривог, медіана (≈31 хв) суттєво нижча за
  середнє (≈50 хв). У самому дашборді (секція 3) показано лише середнє
  значення — показ обох метрик одночасно під час тестування плутав
  нетехнічних користувачів. Натомість це розходження прямо пояснюється
  в підписі під графіком.

**2. Виключення поточного дня**
Датасет оновлюється щодня, але "сьогодні" — завжди неповний день
(дані збираються по ходу доби). Якщо включити його в тренд/прогноз,
це виглядає як штучний "спад", якого нема в реальності. Тому
поточна дата завжди виключається з розрахунків тренду та з тренування
прогнозної моделі.

**3. Денна агрегація (не тижнева)**
Тижнева агрегація "змиває" патерн по днях тижня (наприклад, чи вихідні
спокійніші). Денний рівень детальніший і дозволяє моделі Prophet самій
виявити цей патерн через компонент тижневої сезонності.

**4. Регіональний рівень моделювання**
Протестовано Prophet на 8 регіонах (захід/центр/схід/південь/північ).
Результат — **контрінтуїтивний**: регіони з високою й стабільною
частотою тривог (Харківська, Дніпропетровська) дали **кращий** MAPE
(16–20%), ніж загальноукраїнський baseline (~21%), тоді як
малочастотні регіони (Львівська, Одеська) показали гірший MAPE
(70–85%). Причина — математична властивість метрики MAPE на малих
числах (помилка 1.4 vs факт 3 — це 53% похибки при тривіальній
абсолютній різниці), а не ознака "поганого" регіону. Тому в дашборді
завжди показано MAE поряд із MAPE.

**5. Чесна межа прогнозу**
Навіть найкращий backtested MAPE (~16-20%) означає реальну
невизначеність на рівні конкретного дня. Денна кількість тривог
залежить від реальної воєнної активності, яку календарна модель
(тренд + сезонність) принципово не передбачає. Прогноз варто читати
як **очікуваний діапазон**, не точне число.

**6. Стабільний обідній пік (≈10:00 UTC / 12:00–13:00 за київським часом)**
Перевірено окремо (не лише візуально): частка тривог о 10:00 UTC
перевищує рівномірний базовий рівень (4.17%) майже в усіх перевірених
регіонах (5.3–6.1% замість 4.17%), і цей ефект розподілений рівномірно
по 45 місяцях спостережень (2022–2026), а не сконцентрований у
кілька конкретних днів. Це стабільний, відтворюваний паттерн, не
артефакт малої вибірки чи шум.

### Структура репозиторію

```
ukraine-air-raid-timeseries/
├── README.md
├── requirements.txt
├── app.py                  # Streamlit-дашборд (точка входу)
├── data/raw/                # сирий CSV
├── src/
│   ├── data_loader.py       # завантаження + валідація
│   ├── preprocessing.py     # очищення, фічі, агрегація
│   └── forecasting.py       # Prophet: прогноз, backtest
└── outputs/figures/
```

### Технічна примітка: Chart.js вбудовано локально
Циферблатна діаграма (секція 2) використовує Chart.js, який зберігається
локально в `src/static/chart.umd.js` (≈200 КБ), а не завантажується з
CDN. Це усуває залежність від зовнішнього мережевого доступу під час
виконання та захищає від збоїв CDN.

### Як запустити

```bash
pip install -r requirements.txt
streamlit run app.py
```

Дані завантажуються автоматично під час першого запуску, якщо файл
відсутній (див. `ensure_raw_data()` в `data_loader.py`) — це важливо
для хмарного деплою (наприклад, Streamlit Community Cloud), де немає
можливості виконати окремий скрипт перед стартом застосунку.
`bash download_data.sh` лишається доступним для ручного попереднього
завантаження чи примусового оновлення файлу.

### Деплой

Застосунок задеплоєно на Streamlit Community Cloud і автоматично
оновлюється при кожному `git push` у гілку `main`.

### Що показує дашборд

1. Динаміка тривог у часі — інтерактивний графік (Chart.js) з кнопками
   вибору регіону, швидкими кнопками періоду (5р/1р/6м/3м/1м/тиждень)
   та перетягуванням/масштабуванням
2. Сезонність: циферблатна діаграма по годинах доби (UTC) та графік
   по днях тижня — обидва з перемикачами регіонів
3. Тривалість тривог — середнє значення по регіонах (градієнт
   зелений→жовтий за тривалістю)
4. Порівняння регіонів за загальною кількістю тривог
5. Прогноз — картки по кожному регіону: середнє за останні 14 днів,
   очікувана зміна (% + абсолютна різниця, з кольоровим кодуванням за
   магнітудою), похибка прогнозу
6. Вихідні vs будні — інформаційне порівняння (без рекомендацій
   щодо безпеки чи планування поїздок)

Фільтр: тільки нічні тривоги (≈22:00–06:00 UTC).

### Відома обмеженість

- `is_night` — приблизний проксі (UTC 22:00–06:00), не точна конвертація
  в київський час з урахуванням сезонного зсуву (+2/+3 UTC).
- Прогноз — орієнтовний; реальна точність залежить від регіону
  (див. пункт 4 вище).

---

## 🇬🇧 English

### What this is and what problem it solves

This project turns a raw log of 101,261 events (air raid siren start/end
times by Ukrainian oblast, Feb 25 2022 — present) into a structured
picture: is alert intensity rising over time, when are alerts most
frequent during the day/week, which regions are most affected, and
can a short-term forecast be produced.

This is not a replacement for official alert systems — it's a tool for
**informed planning**: when to schedule travel, how to compare the
"risk profile" of different regions, what to expect over the next
1-2 weeks.

> ⚠️ **Important:** this tool must NOT be used to decide whether to
> take shelter during an actual alert. Always follow official signals
> and go to shelter during a real alert.

### Data

- **Source:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
  (MIT license, updated daily by eTryvoga channel volunteers)
- **File:** `datasets/volunteer_data_en.csv`
- **Volume:** 101,266 records → 101,261 after cleaning, 25 regions,
  Feb 25 2022 — present
- Crimea and the permanent Luhansk siren (active since April 2022) are
  **not included** in this dataset — a documented upstream limitation,
  not a bug in this code.

### Architecture decisions (and why)

**1. Cleaning duration anomalies**
- **5 zero-duration records** (start == finish, all `naive=False`)
  dropped as a source-side technical artifact — physically implausible.
- **Long-duration alerts (>180 min, 3.5% of records)** kept. None are
  `naive=True`, meaning these are real delayed end-signals, not default
  fill-ins — often clustering around intense combat periods (e.g.
  Zaporizka oblast, March 2022). Dropping them would remove signal,
  not noise.
- Because of this long tail, the median (≈31 min) is notably lower
  than the mean (≈50 min). The dashboard itself (section 3) shows only
  the mean -- showing both metrics side by side confused non-technical
  reviewers during testing. This gap is instead explained directly in
  the caption under the chart.

**2. Excluding the current (in-progress) day**
The dataset updates daily, but "today" is always a partial day. Including
it would look like an artificial drop in trend/forecast charts. The
current date is always excluded from trend calculations and model training.

**3. Daily aggregation (not weekly)**
Weekly aggregation washes out day-of-week patterns. Daily granularity
lets Prophet's own weekly-seasonality component surface that pattern
automatically.

**4. Per-region modeling**
Tested Prophet on 8 regions (west/center/east/south/north). Result was
**counter-intuitive**: high-frequency, frontline-adjacent regions
(Kharkivska, Dnipropetrovska) produced a **better** MAPE (16–20%) than
the national baseline (~21%), while low-frequency regions (Lvivska,
Odeska) showed a worse MAPE (70–85%). This is a mathematical property
of MAPE on small counts (predicting 1.4 vs an actual of 3 is a 53%
error despite a trivial absolute gap), not evidence the model performs
worse there. The dashboard always shows MAE alongside MAPE for this reason.

**5. Honest forecast limitation**
Even the best backtested MAPE (~16-20%) implies real day-to-day
uncertainty. Daily alert counts are driven by actual military activity,
which a calendar-based model (trend + seasonality) fundamentally cannot
predict. The forecast should be read as an **expected range**, not an
exact number.

**6. A stable midday peak (≈10:00 UTC / 12:00-13:00 Kyiv time)**
Checked explicitly (not just visually): the share of alerts at 10:00 UTC
exceeds the uniform baseline (4.17%) in nearly every region tested
(5.3-6.1% vs. 4.17%), and the effect is spread evenly across 45 months
of observations (2022-2026) rather than concentrated on a handful of
specific days. This is a stable, reproducible pattern, not a small-
sample artifact or noise.

### Repository structure

See above (identical for both languages).

### Technical note: Chart.js bundled locally
The clock-face diagram (section 2) uses Chart.js, bundled locally at
`src/static/chart.umd.js` (~200KB) rather than loaded from a CDN. This
removes any runtime dependency on external network access and protects
against CDN outages.

### How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data downloads automatically on first run if missing (see
`ensure_raw_data()` in `data_loader.py`) -- important for cloud
deployments (e.g. Streamlit Community Cloud), where there's no way to
run a separate shell script before the app starts. `bash
download_data.sh` remains available for manually pre-fetching the
file or forcing a refresh.

### Deployment

Deployed on Streamlit Community Cloud, auto-updating on every
`git push` to `main`.
streamlit run app.py
```

### What the dashboard shows

1. Alert trend over time -- interactive chart (Chart.js) with region
   toggle buttons, quick time-range buttons (5y/1y/6m/3m/1m/1week), and
   drag-to-pan / scroll-to-zoom
2. Seasonality: a clock-face diagram for hour-of-day (UTC) and a bar
   chart for day-of-week, both with region toggles
3. Alert duration -- mean value per region (green-to-amber gradient by
   duration)
4. Region comparison by total alert count
5. Forecast -- per-region cards: average over the last 14 days,
   expected change (% plus absolute difference, color-coded by
   magnitude), forecast error
6. Weekend vs weekday -- informational comparison only (no safety or
   travel-planning recommendation)

Filter: night-only alerts (≈22:00-06:00 UTC).

### Known limitations

- `is_night` is an approximate proxy (UTC 22:00–06:00), not an exact
  conversion to Kyiv local time accounting for seasonal UTC offset (+2/+3).
- Forecast accuracy varies meaningfully by region (see decision #4 above).
