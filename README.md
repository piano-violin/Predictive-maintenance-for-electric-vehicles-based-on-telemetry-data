# 🔋 Предиктивное обслуживание электротранспорта на основе данных телеметрии

Система предиктивного обслуживания аккумуляторных батарей электробусов. Проект реализует полный pipeline: от анализа необработанных данных телеметрии до прогнозирования состояния здоровья (SOH) и остаточного ресурса (RUL) батарей с интерактивным дашбордом для операторов.

---

## 📌 Ключевые возможности

- **Анализ телеметрии CAN-шины** реального электробуса (Екатеринбург, №218, 35 сигналов, 15.9M записей)
- **EDA и очистка** датчика NASA Li-ion Battery Aging (33 батареи, 2765 циклов разряда)
- **Фильтр Калмана** для сглаживания шумных сигналов тока
- **Ансамблевое прогнозирование RUL**: Random Forest + XGBoost + LSTM (веса 0.3 / 0.3 / 0.4)
- **LSTM-авторегрессия** для out-of-sample прогноза SOH
- **Гибридный метод RUL V4** — комбинация пропорциональной оценки и локальной линейной экстраполяции
- **4-уровневая система предупреждений**: 🟢 Зелёный → 🟡 Жёлтый → 🟠 Оранжевый → 🔴 Красный
- **Streamlit-дашборд** для мониторинга состояния парка батарей в реальном времени

---

## 🗂️ Структура проекта

```
├── EDA_Infomatiks 218 Final.ipynb   # EDA реальной телеметрии CAN-шины (электробус #218)
├── battery_eda.ipynb                 # EDA и очистка датасета NASA → final_df_clean.csv
├── battery.ipynb                     # Базовые модели прогнозирования ёмкости (RF + LSTM)
├── battery_enhanced.ipynb            # Продвинутые модели SOH/RUL + ансамбль + система предупреждений
├── app2.py                           # Streamlit-дашборд (production)
├── models2/                          # Обученные модели и скейлеры
│   ├── rf_soh.pkl                    # Random Forest — SOH
│   ├── rf_rul.pkl                    # Random Forest — RUL
│   ├── xgb_rul.pkl                   # XGBoost — RUL
│   ├── lstm_soh_model.keras          # LSTM — SOH
│   ├── lstm_rul_model.keras          # LSTM — RUL
│   ├── scaler_*.pkl                  # MinMax-скейлеры
│   ├── model_config.json             # Конфигурация моделей
│   └── model_metrics.json            # Метрики качества
├── final_df_clean.csv                # Очищенный датасет (2765 строк, 33 батареи)

```

---

## 🔄 Pipeline

```
┌─────────────────────┐     ┌─────────────────┐     ┌──────────────────────────┐
│  battery_eda.ipynb  │ ──→ │   battery.ipynb  │     │  battery_enhanced.ipynb  │
│  (NASA .mat → CSV)  │     │  (baseline RF +  │     │  (SOH/RUL: RF + XGB +   │
│                     │     │   LSTM capacity) │     │   LSTM + ensemble)       │
└─────────────────────┘     └─────────────────┘     └────────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                        ┌────────────────┐
                                                        │    app2.py     │
                                                        │  (Streamlit    │
                                                        │   Dashboard)  │
                                                        └────────────────┘

┌───────────────────────────────┐
│  EDA_Infomatiks 218 Final.ipynb │  ←  отдельный анализ реальной телеметрии
│  (CAN-bus, фильтр Калмана)      │     электробуса (не входит в основной pipeline)
└───────────────────────────────┘
```

---

## 🚀 Установка и запуск

### Требования

- Python 3.12+
- CUDA (опционально, для ускорения LSTM на GPU)

### Установка

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>

pip install -r requirements.txt
```

### Зависимости

<details>
<summary>📋 Основные библиотеки</summary>

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
tensorflow
filterpy
streamlit
plotly
joblib
jupyter
```

</details>

### Запуск дашборда

```bash
streamlit run app2.py
```

Дашборд будет доступен по адресу `http://localhost:8501`.

---

## 📊 Данные

### Датасет 1: NASA Li-ion Battery Aging Dataset

| Параметр | Значение |
|----------|----------|
| Источник | NASA Ames Prognostics Center of Excellence (PCoE) |
| Батарее | B0005 – B0056 (33 шт.) |
| Записей после очистки | 2765 циклов разряда |
| Признаки | cycle, capacity, SOH, initial_capacity, degradation_rate и др. |
| Критерий EOL | Падение ёмкости на 30% от номинала |

### Датасет 2: CAN-шина электробуса №218

| Параметр | Значение |
|----------|----------|
| Город | Екатеринбург |
| Период | 2026-03-01 — 2026-04-30 |
| Записей | 15.9M |
| CAN-сигналов | 35 (напряжение, ток, скорость, температура, HVAC и др.) |
| ТОП-3 сигнала | VOLTAGE (41.6%), CURRENT (36%), VehicleSpeed (16%) |

---

## 🧠 Модели

### Прогнозирование SOH (State of Health, %)

| Модель | Описание |
|--------|----------|
| Random Forest | Ансамбль деревьев на 11 признаках (rolling stats, degradation rate) |
| LSTM | Рекуррентная сеть, window=20, per-battery последовательности |

### Прогнозирование RUL (Remaining Useful Life, циклы)

| Модель | Вес в ансамбле | Описание |
|--------|:-:|----------|
| Random Forest | 0.3 | Деревья на handcrafted признаках |
| XGBoost | 0.3 | Градиентный бустинг |
| LSTM | 0.4 | Последовательная модель, window=20 |

### Гибридный RUL V4

Метод комбинирует две оценки:

1. **Пропорциональная**: `RUL = (SOH_current − 70) / SOH_drop × elapsed`
2. **Локальная экстраполяция**: линейная регрессия по последним 20 циклам (при std > 0.05 и slope < −0.003)

Итоговый RUL = `0.5 × RUL_пропорциональный + 0.5 × RUL_локальный` (с ограничением max_allowed_rul и гарантией монотонности).

---

## 🚦 Система предупреждений

| Уровень | SOH | Цвет | Рекомендация |
|---------|-----|------|-------------|
| Зелёный | ≥ 85% | 🟢 | Нормальная эксплуатация |
| Жёлтый | 70% – 84% | 🟡 | Плановое обслуживание |
| Оранжевый | 60% – 69% | 🟠 | Ускоренная диагностика |
| Красный | < 60% | 🔴 | Замена батареи |

---

## 🔬 Технические детали

### Feature Engineering (11 признаков для продвинутых моделей)

| Признак | Описание |
|---------|----------|
| `cycle` | Номер цикла |
| `cycle_normalized` | Нормализованный цикл (0–1) |
| `cycle_squared` | Квадрат цикла (нелинейность) |
| `initial_capacity` | Начальная ёмкость батареи |
| `soh` | Текущее SOH |
| `soh_rmean_5` | Скользящее среднее SOH (5 циклов) |
| `soh_rmean_10` | Скользящее среднее SOH (10 циклов) |
| `soh_slope_10` | Наклон SOH за 10 циклов |
| `soh_drop` | Максимальное падение SOH |
| `degradation_rate` | Скорость деградации |
| `capacity` | Текущая ёмкость (Ah) |

### Ключевые подходы

- **Per-battery изоляция** последовательностей LSTM — предотвращает утечку данных между батареями
- **Монотонность RUL** — гарантия неубывания прогнозируемого RUL с увеличением цикла
- **Фильтр Калмана** (`filterpy`) — сглаживание шумного сигнала тока CAN-шины
- **VIF-анализ** — отбор признаков с учётом мультиколлинеарности
- **LSTM-авторегрессия** — out-of-sample прогноз SOH с пошаговым обновлением rolling-признаков

---

## 📝 Как воспроизвести

### 1. EDA и подготовка данных

```bash
jupyter notebook battery_eda.ipynb
```

> На выходе: `final_df_clean.csv` (2765 строк, 33 батареи)

### 2. Базовые модели

```bash
jupyter notebook battery.ipynb
```

> На выходе: модели прогнозирования ёмкости (RF + LSTM baseline)

### 3. Продвинутые модели SOH/RUL

```bash
jupyter notebook battery_enhanced.ipynb
```

> На выходе: обученные модели в `models2/`, метрики качества

### 4. Запуск дашборда

```bash
streamlit run app2.py
```

### (Опционально) Анализ реальной телеметрии

```bash
jupyter notebook "EDA_Infomatiks 218 Final.ipynb"
```

> Требуется файл `ekb_1/218_from_2026-03-01_to_2026-04-30.json` с данными CAN-шины

---

## 📄 Лицензия

Проект создан по заказу индустриального партнера. Датасет NASA PCoE распространяется на условиях оригинального источника.

---
<details>
<summary>🌐 English version</summary>

# 🔋 Predictive Maintenance for Electric Transport Based on Telemetry Data


A predictive maintenance system for electric bus batteries. The project implements a full pipeline: from raw CAN-bus telemetry analysis to State of Health (SOH) and Remaining Useful Life (RUL) prediction with an interactive operator dashboard.

## Key Features

- **CAN-bus telemetry analysis** of a real electric bus (Yekaterinburg, #218, 35 signals, 15.9M records)
- **EDA & cleaning** of NASA Li-ion Battery Aging Dataset (33 batteries, 2765 discharge cycles)
- **Kalman Filter** for noisy current signal smoothing
- **Ensemble RUL prediction**: Random Forest + XGBoost + LSTM (weights 0.3 / 0.3 / 0.4)
- **LSTM autoregression** for out-of-sample SOH forecasting
- **Hybrid RUL V4 method** — blended proportional + local linear extrapolation
- **4-level warning system**: 🟢 Green → 🟡 Yellow → 🟠 Orange → 🔴 Red
- **Streamlit dashboard** for real-time fleet battery monitoring

## Quick Start

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app2.py
```

## Models

| Target | Models | Ensemble |
|--------|--------|----------|
| SOH (%) | Random Forest, LSTM | — |
| RUL (cycles) | RF (0.3), XGBoost (0.3), LSTM (0.4) | Weighted average |

## Warning System

| Level | SOH Range | Action |
|-------|-----------|--------|
| 🟢 Green | ≥ 85% | Normal operation |
| 🟡 Yellow | 70–84% | Scheduled maintenance |
| 🟠 Orange | 60–69% | Accelerated diagnostics |
| 🔴 Red | < 60% | Battery replacement |

</details>
