"""
================================================================================
 ДАШБОРД ПРЕДИКТИВНОЙ АНАЛИТИКИ ЭЛЕКТРОТРАНСПОРТА  (v3)
 Прогнозирование SOH и RUL аккумуляторных батарей электробусов
================================================================================

ВКР: «Разработка системы предиктивной аналитики электротранспорта
      на основе данных телеметрии».

Соответствует battery_enhanced.ipynb:
  • Динамический RUL (calculate_soh_rul_v4).
  • 11 признаков (cycle, cycle_normalized, cycle_squared, initial_capacity,
    soh, soh_rmean_5/10, soh_slope_10, soh_drop, degradation_rate, capacity).
  • Модели: RF (SOH, RUL) + XGBoost (RUL) + 2 LSTM (SOH, RUL, window=20)
    + Ensemble RUL (0.3·RF + 0.3·XGB + 0.4·LSTM).
  • Система предупреждений по SOH (get_warning_level): 85/70/60 %.
  • Out-of-sample прогноз: LSTM-авторегрессия для SOH (вместо RF),
    RUL выводится из прогнозного SOH, гарантированно монотонно невозрастает.

Исправления v3 (относительно app1-1.py):
  1. Убрал MAX_EOL_HORIZON=300 — final_eol больше не искусственно ограничен.
  2. В warmup-ветке добавлен max_allowed_rul cap.
  3. elapsed = c - start_cycle (от argmax SOH), а не c - cycles[0].
  4. Локальная экстраполяция: std > 0.05, a < -0.003 (строгие критерии).
  5. Блендинг: rul_local=None → только rul_prop (не вынужденный 50/50).
  6. EOL-проверка после warmup (правильный порядок).
  7. Out-of-sample: RF → LSTM-авторегрессия для SOH.

Запуск:    streamlit run app2.py
Зависимости: streamlit, pandas, numpy, plotly, joblib, scikit-learn,
             tensorflow, xgboost.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# КОНСТАНТЫ (из model_config.json, значения по умолчанию из notebook)
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join("battery_dataset", "final_df_clean.csv")
MODELS_DIR = "models2"

FEATURE_COLS = [
    "cycle", "cycle_normalized", "cycle_squared", "initial_capacity",
    "soh", "soh_rmean_5", "soh_rmean_10", "soh_slope_10",
    "soh_drop", "degradation_rate", "capacity",
]
WINDOW_SIZE = 20
SOH_THRESHOLD_EOL = 70
WARMUP = 5
LOCAL_WINDOW = 20
MIN_WINDOW = 10
ENSEMBLE_WEIGHTS = {"rf": 0.3, "xgb": 0.3, "lstm": 0.4}

# Цвета зон состояния (консистентно во всем дашборде)
HEALTH_COLORS = {
    "ЗЕЛЁНЫЙ": "#16a34a", "ЖЁЛТЫЙ": "#eab308",
    "ОРАНЖЕВЫЙ": "#ea580c", "КРАСНЫЙ": "#dc2626",
}
HEALTH_ORDER = ["ЗЕЛЁНЫЙ", "ЖЁЛТЫЙ", "ОРАНЖЕВЫЙ", "КРАСНЫЙ"]
HEALTH_LABEL = {
    "ЗЕЛЁНЫЙ": "Зелёный (норма)",
    "ЖЁЛТЫЙ": "Жёлтый (внимание)",
    "ОРАНЖЕВЫЙ": "Оранжевый (риск)",
    "КРАСНЫЙ": "Красный (критично)",
}
HEALTH_ACTION = {
    "ЗЕЛЁНЫЙ": "Штатная эксплуатация",
    "ЖЁЛТЫЙ": "Плановое ТО",
    "ОРАНЖЕВЫЙ": "Срочная замена",
    "КРАСНЫЙ": "Немедленное отключение",
}
THEME_COLOR = "#0d9488"   # teal-600


# ---------------------------------------------------------------------------
# 1. ЗАГРУЗКА КОНФИГУРАЦИИ ИЗ model_config.json
# ---------------------------------------------------------------------------
def load_config():
    """Подгружает конфиг из models/model_config.json (если есть)."""
    path = os.path.join(MODELS_DIR, "model_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg
    return None


# ---------------------------------------------------------------------------
# 2. ЗАГРУЗКА ДАННЫХ
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Загрузка датасета телеметрии батарей…")
def load_dataset():
    """Загружает final_df_clean.csv (SOH в масштабе 0..1)."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Датасет не найден: `{DATA_FILE}`. Положите ваш "
                 "`final_df_clean.csv` в `battery_dataset/` или запустите "
                 "`python3 prepare_artifacts.py`.")
        st.stop()
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. ДИНАМИЧЕСКИЙ RUL (calculate_soh_rul_v4 из battery_enhanced.ipynb)
# ---------------------------------------------------------------------------
def calculate_soh_rul_v4(battery_df, soh_threshold=SOH_THRESHOLD_EOL,
                         warmup=WARMUP, local_window=LOCAL_WINDOW,
                         min_window=MIN_WINDOW):
    """Динамический RUL (v4): на каждом цикле EOL переоценивается.

    Гибрид пропорциональной оценки и rolling polyfit, с hard cap.
    Точное соответствие notebook battery_enhanced.ipynb (Часть 2).

    Исправления (v3, относительно app1-1.py):
      1. Убрал MAX_EOL_HORIZON — final_eol не ограничен искусственно.
      2. Warmup-ветка: rul ограничен max_allowed_rul.
      3. elapsed = c - start_cycle (от argmax SOH), а не c - cycles[0].
      4. Локальная экстраполяция: len>=3, std>0.05, a<-0.003.
      5. Блендинг: rul_local=None → только rul_prop.
      6. EOL-проверка после warmup-ветки (правильный порядок).
    """
    df = battery_df.copy().sort_values("cycle").reset_index(drop=True)

    # 1. Начальная емкость = медиана первых warmup циклов
    initial_capacity = df["capacity"].head(warmup).median()
    if initial_capacity <= 0:
        initial_capacity = df["capacity"].max()

    # 2. SOH в процентах
    df["soh"] = ((df["capacity"] / initial_capacity) * 100).clip(upper=100)

    cycles = df["cycle"].values
    sohs = df["soh"].values
    n = len(df)
    max_cycle = int(cycles.max())

    # 3. Финальная оценка EOL (по всем данным — для возврата)
    after_warmup = df.iloc[warmup:]
    eol_mask = after_warmup["soh"] < soh_threshold
    if eol_mask.any():
        final_eol = int(after_warmup.loc[eol_mask, "cycle"].iloc[0])
    else:
        tail = df.tail(local_window)
        if len(tail) >= 2 and tail["soh"].std() > 0:
            a, b = np.polyfit(tail["cycle"].values, tail["soh"].values, 1)
            if a < 0:
                final_eol = max(int(round((soh_threshold - b) / a)), max_cycle + 1)
            else:
                final_eol = max_cycle + 100
        else:
            final_eol = max_cycle + 100

    # 4. ДИНАМИЧЕСКИЙ RUL (V4 — гибрид)
    rul_values = []
    for i in range(n):
        c = int(cycles[i])
        s = float(sohs[i])

        # Максимально допустимый RUL: оставшиеся циклы + 50 (но не менее 50)
        max_allowed_rul = max(max_cycle - c + 50, 50)

        # На ранних циклах (мало данных) — берём финальную оценку
        if i < max(warmup, min_window):
            rul = min(max(final_eol - c, 0), max_allowed_rul)
            rul_values.append(rul)
            continue

        # --- Метод 1: Пропорциональная оценка ---
        max_soh_so_far = float(sohs[:i + 1].max())
        start_idx = int(np.argmax(sohs[:i + 1]))
        start_cycle = int(cycles[start_idx])
        elapsed = c - start_cycle

        if elapsed > 0 and max_soh_so_far > s:
            soh_drop = max_soh_so_far - s
            rul_prop = (s - soh_threshold) / soh_drop * elapsed
        else:
            rul_prop = max_allowed_rul  # ещё нет деградации

        rul_prop = max(0, min(int(round(rul_prop)), max_allowed_rul))

        # --- Метод 2: Локальная экстраполяция (только при значимом тренде) ---
        rul_local = None
        tail_n = min(local_window, i + 1)
        tail_soh = sohs[i - tail_n + 1: i + 1]
        tail_cyc = cycles[i - tail_n + 1: i + 1]

        if len(tail_soh) >= 3 and np.std(tail_soh) > 0.05:
            a, b = np.polyfit(tail_cyc, tail_soh, 1)
            if a < -0.003:  # значимый убывающий тренд
                eol_local = (soh_threshold - b) / a
                rul_local = max(0, min(int(round(eol_local - c)), max_allowed_rul))

        # --- Финальный RUL: средневзвешенное ---
        if rul_local is not None:
            rul_final = int(round(0.5 * rul_prop + 0.5 * rul_local))
        else:
            rul_final = rul_prop

        # Проверяем, достигнут ли EOL
        if s < soh_threshold and i >= warmup:
            rul_final = 0

        rul_values.append(rul_final)

    df["rul"] = rul_values
    df["initial_capacity"] = initial_capacity
    return df, float(initial_capacity), final_eol


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (add_engineering_features из Notebook)
# ---------------------------------------------------------------------------
def add_engineering_features(df):
    """Добавляет 11 признаков. df уже должен содержать soh (%), initial_capacity."""
    out = df.copy().sort_values(["battery_id", "cycle"]).reset_index(drop=True)
    max_cycle = out.groupby("battery_id")["cycle"].transform("max")
    out["cycle_normalized"] = out["cycle"] / max_cycle
    out["cycle_squared"] = out["cycle"] ** 2

    g = out.groupby("battery_id")["soh"]
    out["soh_rmean_5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
    out["soh_rmean_10"] = g.transform(lambda x: x.rolling(10, min_periods=1).mean())

    def _slope(x):
        n = len(x)
        if n < 3:
            return 0.0
        return float(np.polyfit(np.arange(n), x.values, 1)[0])
    out["soh_slope_10"] = g.transform(
        lambda x: x.rolling(10, min_periods=3).apply(_slope, raw=False))

    out["soh_drop"] = g.transform(lambda x: x.cummax() - x)
    out["degradation_rate"] = out["soh_drop"] / out["cycle"].clip(lower=1)
    return out


def build_enhanced_df(df_raw):
    """Полный датафрейм с динамическим SOH/RUL и 11 признаками для всех батарей."""
    parts, info_rows = [], []
    for bid in df_raw["battery_id"].unique():
        sub = df_raw[df_raw["battery_id"] == bid].copy()
        sub, init_cap, eol = calculate_soh_rul_v4(sub)
        parts.append(sub)
        info_rows.append({
            "battery_id": bid, "initial_capacity": round(init_cap, 4),
            "eol_cycle": eol, "total_cycles": int(sub["cycle"].max()),
            "final_soh": round(sub["soh"].iloc[-1], 2),
            "min_soh": round(sub["soh"].min(), 2),
            "max_soh": round(sub["soh"].max(), 2),
            "min_rul": int(sub["rul"].min()),
            "max_rul": int(sub["rul"].max()),
            "rul_volatility": round(float(sub["rul"].std()), 1),
            "achieved_eol": bool(eol <= sub["cycle"].max()),
        })
    enhanced = pd.concat(parts, ignore_index=True)
    enhanced = add_engineering_features(enhanced)
    battery_info = pd.DataFrame(info_rows)
    return enhanced, battery_info


# ---------------------------------------------------------------------------
# 5. ЗАГРУЗКА МОДЕЛЕЙ
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Загрузка моделей (RF + XGBoost + 2 LSTM)…")
def load_models():
    """Загружает все артефакты из models/. Возвращает dict."""
    def _load(path):
        return joblib.load(path) if os.path.exists(path) else None
    def _load_keras(path):
        return tf.keras.models.load_model(path) if os.path.exists(path) else None

    art = {}
    # RF
    art["rf_soh"] = _load(os.path.join(MODELS_DIR, "rf_soh.pkl"))
    art["rf_rul"] = _load(os.path.join(MODELS_DIR, "rf_rul.pkl"))
    art["xgb_rul"] = _load(os.path.join(MODELS_DIR, "xgb_rul.pkl"))
    # LSTM (2 отдельные модели)
    art["lstm_soh"] = _load_keras(os.path.join(MODELS_DIR, "lstm_soh_model.keras"))
    art["lstm_rul"] = _load_keras(os.path.join(MODELS_DIR, "lstm_rul_model.keras"))
    # Scaler'ы (4)
    art["scaler_X_soh"] = _load(os.path.join(MODELS_DIR, "scaler_X_soh.pkl"))
    art["scaler_y_soh"] = _load(os.path.join(MODELS_DIR, "scaler_y_soh.pkl"))
    art["scaler_X_rul"] = _load(os.path.join(MODELS_DIR, "scaler_X_rul.pkl"))
    art["scaler_y_rul"] = _load(os.path.join(MODELS_DIR, "scaler_y_rul.pkl"))
    # Базовая модель ёмкости (battery3.ipynb)
    art["rf_capacity"] = _load(os.path.join(MODELS_DIR, "rf_capacity.pkl"))
    art["lstm_capacity"] = _load_keras(os.path.join(MODELS_DIR, "lstm_capacity.keras"))
    art["scaler_X_capacity"] = _load(os.path.join(MODELS_DIR, "scaler_X_capacity.pkl"))
    art["scaler_y_capacity"] = _load(os.path.join(MODELS_DIR, "scaler_y_capacity.pkl"))
    # Метрики и конфиг
    metrics = {}
    for fn in ["model_metrics.json", "model_metrics_capacity.json"]:
        p = os.path.join(MODELS_DIR, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
                metrics.update(d if fn == "model_metrics.json" else {"capacity": d})
    art["metrics"] = metrics
    # Battery info (если сохранён)
    bi_path = os.path.join(MODELS_DIR, "battery_info.csv")
    art["battery_info"] = pd.read_csv(bi_path) if os.path.exists(bi_path) else None
    return art


# ---------------------------------------------------------------------------
# 6. СИСТЕМА ПРЕДУПРЕЖДЕНИЙ (get_warning_level из Notebook — только по SOH)
# ---------------------------------------------------------------------------
def get_warning_level(soh, rul=None):
    """Возвращает уровень предупреждения по SOH (RUL игнорируется)."""
    if soh >= 85:
        return "ЗЕЛЁНЫЙ"
    elif soh >= 70:
        return "ЖЁЛТЫЙ"
    elif soh >= 60:
        return "ОРАНЖЕВЫЙ"
    else:
        return "КРАСНЫЙ"


def alert_info(soh, rul=None):
    """Полная инфо-карточка: статус + сообщение + действие."""
    level = get_warning_level(soh, rul)
    messages = {
        "ЗЕЛЁНЫЙ": "Батарея в хорошем состоянии, деградация в норме.",
        "ЖЁЛТЫЙ": "Наблюдается заметная деградация. Усиленный мониторинг.",
        "ОРАНЖЕВЫЙ": "Значительная деградация. Запланировать обслуживание.",
        "КРАСНЫЙ": "Батарея близка к концу жизни. Срочная замена.",
    }
    return {
        "level": level,
        "status_label": HEALTH_LABEL[level],
        "message": messages[level],
        "action": HEALTH_ACTION[level],
        "soh": soh,
    }


# ---------------------------------------------------------------------------
# 7. ПРОГНОЗ МОДЕЛЯМИ
# ---------------------------------------------------------------------------
def predict_rf_soh(battery_data, models):
    return models["rf_soh"].predict(battery_data[FEATURE_COLS].values)


def predict_rf_rul(battery_data, models):
    return models["rf_rul"].predict(battery_data[FEATURE_COLS].values)


def predict_xgb_rul(battery_data, models):
    if models.get("xgb_rul") is None:
        return None
    return models["xgb_rul"].predict(battery_data[FEATURE_COLS].values)


def predict_lstm(model, scaler_X, scaler_y, battery_data, window=WINDOW_SIZE):
    """Прогноз одной LSTM со скользящим окном.

    Возвращает (cycles_aligned, predictions) — прогнозы смещены на `window`.
    """
    X = battery_data[FEATURE_COLS].values
    X_s = scaler_X.transform(X)
    if len(X_s) <= window:
        return np.array([]), np.array([])
    seqs = np.array([X_s[i:i + window] for i in range(len(X_s) - window)])
    pred_s = model.predict(seqs, verbose=0).flatten()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    cycles = battery_data["cycle"].values[window:]
    return cycles, pred


def predict_ensemble_rul(battery_data, models):
    """Ensemble RUL = 0.3*RF + 0.3*XGB + 0.4*LSTM (на наблюдаемых циклах)."""
    rf = predict_rf_rul(battery_data, models)
    xgb = predict_xgb_rul(battery_data, models)
    if models["lstm_rul"] is None:
        return None
    cyc_l, lstm = predict_lstm(models["lstm_rul"], models["scaler_X_rul"],
                               models["scaler_y_rul"], battery_data)
    if len(lstm) == 0:
        return None
    # Выравниваем по длине LSTM (она короче на window)
    n = len(lstm)
    rf_tail = rf[-n:]
    if xgb is not None:
        xgb_tail = xgb[-n:]
        ens = (ENSEMBLE_WEIGHTS["rf"] * rf_tail +
               ENSEMBLE_WEIGHTS["xgb"] * xgb_tail +
               ENSEMBLE_WEIGHTS["lstm"] * lstm)
    else:
        w = ENSEMBLE_WEIGHTS["rf"] + ENSEMBLE_WEIGHTS["xgb"]
        ens = (ENSEMBLE_WEIGHTS["rf"] / w * rf_tail +
               ENSEMBLE_WEIGHTS["lstm"] * lstm)
    return cyc_l, ens

def forecast_future_lstm(battery_data, init_cap, horizon, models):
    """Прогноз SOH и RUL на будущие циклы через LSTM-авторегрессию.

    SOH прогнозируется авторегрессионно через LSTM (window=20):
      на каждом шаге берём скользящее окно из 20 последних циклов,
      нормализуем через scaler_X, пропускаем через lstm_soh_model,
      обратное преобразование через scaler_y.

    RUL выводится из прогнозного SOH:
      eol_from_fact = last_cycle + last_rul
      eol_from_forecast = первый цикл, где прогнозный SOH < 70%
      eol_cycle = min(eol_from_fact, eol_from_forecast)
      fut_rul[i] = min(fut_rul[i], fut_rul[i-1])  # монотонность

    Возвращает:
      fut_cycles, fut_soh, fut_rul, use_lstm (bool)
    """
    lstm_model = models.get("lstm_soh")
    scaler_X = models.get("scaler_X_soh")
    scaler_y = models.get("scaler_y_soh")

    if lstm_model is None or scaler_X is None or scaler_y is None:
        # Fallback на RF если LSTM не загружена
        return _forecast_future_rf_fallback(battery_data, init_cap, horizon, models)

    hist = battery_data[["cycle", "capacity", "soh", "initial_capacity"]].copy()
    last_cycle = int(hist["cycle"].max())
    max_cycle_orig = last_cycle
    last_rul = int(battery_data["rul"].iloc[-1])
    eol_from_fact = last_cycle + last_rul

    # Строим таблицу признаков для истории (как в add_engineering_features)
    feat_df = hist.copy()
    feat_df["cycle_normalized"] = feat_df["cycle"] / max_cycle_orig
    feat_df["cycle_squared"] = feat_df["cycle"] ** 2
    s = feat_df["soh"]
    feat_df["soh_rmean_5"] = s.rolling(5, min_periods=1).mean()
    feat_df["soh_rmean_10"] = s.rolling(10, min_periods=1).mean()
    feat_df["soh_slope_10"] = s.rolling(10, min_periods=3).apply(
        lambda x: float(np.polyfit(np.arange(len(x)), x.values, 1)[0])
        if len(x) >= 3 else 0.0, raw=False)
    feat_df["soh_drop"] = s.cummax() - s
    feat_df["degradation_rate"] = feat_df["soh_drop"] / feat_df["cycle"].clip(lower=1)

    feature_cols = FEATURE_COLS
    window = WINDOW_SIZE

    # Масштабируем всю историю
    hist_scaled = scaler_X.transform(feat_df[feature_cols].values)

    # Начальное окно для LSTM (последние window строк истории)
    if len(hist_scaled) >= window:
        window_data = hist_scaled[-window:].copy()
    else:
        # Не хватает данных — fallback на RF
        return _forecast_future_rf_fallback(battery_data, init_cap, horizon, models)

    full_soh = list(feat_df["soh"].values)
    fut_cycles, fut_soh = [], []

    for step in range(horizon):
        c = last_cycle + 1 + step

        # Пересчитываем rolling-признаки
        last_soh = full_soh[-1]
        s_arr = np.array(full_soh)

        cycle_norm = c / max_cycle_orig
        cycle_sq = c ** 2
        rmean5 = s_arr[-5:].mean()
        rmean10 = s_arr[-10:].mean()
        if len(s_arr) >= 3:
            slope = float(np.polyfit(
                np.arange(len(s_arr[-10:])), s_arr[-10:], 1)[0])
        else:
            slope = 0.0
        cummax_val = float(np.maximum.accumulate(s_arr)[-1])
        soh_drop = cummax_val - last_soh
        deg_rate = soh_drop / max(c, 1)

        # Ёмкость: линейная экстраполяция из 5 последних
        cap_arr = list(feat_df["capacity"].values)
        if len(cap_arr) >= 5:
            x = np.arange(5)
            y = np.array(cap_arr[-5:])
            a, b = np.polyfit(x, y, 1)
            cap = max(a * 5 + b, 0.01)
        else:
            cap = cap_arr[-1]

        X_row = np.array([[c, cycle_norm, cycle_sq, init_cap,
                           last_soh, rmean5, rmean10, slope,
                           soh_drop, deg_rate, cap]])
        X_scaled = scaler_X.transform(X_row)

        # Сдвигаем окно
        window_data = np.vstack([window_data[1:], X_scaled])

        # Прогноз через LSTM
        pred_scaled = lstm_model.predict(
            window_data[np.newaxis, :, :], verbose=0).flatten()[0]
        new_soh = float(scaler_y.inverse_transform([[pred_scaled]])[0, 0])
        new_soh = max(0.0, min(100.0, new_soh))

        fut_cycles.append(c)
        fut_soh.append(new_soh)
        full_soh.append(new_soh)

        # Обновляем feat_df для rolling на следующем шаге
        new_feat = {fc: 0.0 for fc in feature_cols}
        new_feat["cycle"] = c
        new_feat["cycle_normalized"] = cycle_norm
        new_feat["cycle_squared"] = cycle_sq
        new_feat["initial_capacity"] = init_cap
        new_feat["soh"] = new_soh
        new_feat["soh_rmean_5"] = rmean5
        new_feat["soh_rmean_10"] = rmean10
        new_feat["soh_slope_10"] = slope
        new_feat["soh_drop"] = soh_drop
        new_feat["degradation_rate"] = deg_rate
        new_feat["capacity"] = cap
        feat_df = pd.concat(
            [feat_df, pd.DataFrame([new_feat])], ignore_index=True)

    fut_cycles = np.array(fut_cycles)
    fut_soh = np.array(fut_soh)

    # --- RUL из прогноза SOH (монотонно невозрастающий) ---
    eol_mask = fut_soh < SOH_THRESHOLD_EOL
    if eol_mask.any():
        eol_from_forecast = int(fut_cycles[eol_mask][0])
    else:
        tail_n = min(10, len(fut_soh))
        x = fut_cycles[-tail_n:]
        y = fut_soh[-tail_n:]
        if len(x) >= 2 and np.std(y) > 0:
            a, b = np.polyfit(x, y, 1)
            if a < 0:
                eol_from_forecast = int(round((SOH_THRESHOLD_EOL - b) / a))
            else:
                eol_from_forecast = eol_from_fact + 1000
        else:
            eol_from_forecast = eol_from_fact + 1000

    eol_cycle = min(eol_from_fact, eol_from_forecast)
    fut_rul = np.maximum(eol_cycle - fut_cycles, 0).astype(int)

    # Гарантируем монотонное невозрастание RUL
    for i in range(1, len(fut_rul)):
        fut_rul[i] = min(fut_rul[i], fut_rul[i - 1])

    return fut_cycles, fut_soh, fut_rul, True


def _forecast_future_rf_fallback(battery_data, init_cap, horizon, models):
    """RF-фоллбэк для out-of-sample прогноза (если LSTM не загружена)."""
    hist = battery_data[["cycle", "capacity", "soh"]].copy()
    last_cycle = int(hist["cycle"].max())
    last_rul = int(battery_data["rul"].iloc[-1])
    eol_from_fact = last_cycle + last_rul
    max_cycle = last_cycle

    cur_soh = hist["soh"].tolist()
    cur_cap = hist["capacity"].tolist()
    fut_cycles, fut_soh = [], []

    for step in range(horizon):
        c = last_cycle + 1 + step
        s_arr = np.array(cur_soh)
        rmean5 = s_arr[-5:].mean()
        rmean10 = s_arr[-10:].mean()
        slope = 0.0
        if len(s_arr) >= 3:
            slope = float(np.polyfit(
                np.arange(len(s_arr[-10:])), s_arr[-10:], 1)[0])
        cummax = float(np.maximum.accumulate(s_arr)[-1])
        soh_drop = cummax - cur_soh[-1]
        deg_rate = soh_drop / max(c, 1)
        if len(cur_cap) >= 5:
            x = np.arange(5)
            y = np.array(cur_cap[-5:])
            a, b = np.polyfit(x, y, 1)
            cap = a * 5 + b
        else:
            cap = cur_cap[-1]
        cur_cap.append(float(cap))

        X_row = np.array([[c, c / max_cycle, c ** 2, init_cap,
                           cur_soh[-1], rmean5, rmean10, slope,
                           soh_drop, deg_rate, cap]])
        new_soh = float(models["rf_soh"].predict(X_row)[0])
        new_soh = max(0.0, min(100.0, new_soh))
        fut_cycles.append(c)
        fut_soh.append(new_soh)
        cur_soh.append(new_soh)

    fut_cycles = np.array(fut_cycles)
    fut_soh = np.array(fut_soh)

    eol_mask = fut_soh < SOH_THRESHOLD_EOL
    if eol_mask.any():
        eol_from_forecast = int(fut_cycles[eol_mask][0])
    else:
        tail_n = min(10, len(fut_soh))
        x, y = fut_cycles[-tail_n:], fut_soh[-tail_n:]
        if len(x) >= 2 and np.std(y) > 0:
            a, b = np.polyfit(x, y, 1)
            eol_from_forecast = (int(round((SOH_THRESHOLD_EOL - b) / a))
                                if a < 0 else eol_from_fact + 1000)
        else:
            eol_from_forecast = eol_from_fact + 1000

    eol_cycle = min(eol_from_fact, eol_from_forecast)
    fut_rul = np.maximum(eol_cycle - fut_cycles, 0).astype(int)
    for i in range(1, len(fut_rul)):
        fut_rul[i] = min(fut_rul[i], fut_rul[i - 1])

    return fut_cycles, fut_soh, fut_rul, False




# ---------------------------------------------------------------------------
# 8. БАЗОВАЯ МОДЕЛЬ ЕМКОСТИ (battery.ipynb, lag=3)
# ---------------------------------------------------------------------------
CAP_FEATURES = ["cycle", "capacity_lag_1", "capacity_lag_2", "capacity_lag_3"]
CAP_LAGS = 3


def add_lags(df, lags=CAP_LAGS):
    t = df.copy()
    for lag in range(1, lags + 1):
        t[f"capacity_lag_{lag}"] = t.groupby("battery_id")["capacity"].shift(lag)
    return t


def predict_capacity_rf(battery_data, models):
    """Прогноз ёмкости через RF (lag=3)."""
    t = add_lags(battery_data[["battery_id", "cycle", "capacity"]])
    t = t.dropna()
    X = t[CAP_FEATURES].values
    pred = models["rf_capacity"].predict(X)
    return t["cycle"].values, pred


def predict_capacity_lstm(battery_data, models):
    """Прогноз ёмкости через LSTM (window=3)."""
    t = add_lags(battery_data[["battery_id", "cycle", "capacity"]])
    t = t.dropna()
    X = t[CAP_FEATURES].values
    X_s = models["scaler_X_capacity"].transform(X)
    seq = X_s[:, -CAP_LAGS:].reshape(-1, CAP_LAGS, 1)
    pred_s = models["lstm_capacity"].predict(seq, verbose=0).flatten()
    pred = models["scaler_y_capacity"].inverse_transform(pred_s.reshape(-1, 1)).flatten()
    return t["cycle"].values, pred


# ---------------------------------------------------------------------------
# 9. ВСПОМОГАТЕЛЬНЫЕ (графики, KPI)
# ---------------------------------------------------------------------------
def kpi_card(label, value, delta=None, color=THEME_COLOR):
    delta_html = (f'<div style="font-size:11px;color:#64748b;margin-top:2px;">{delta}</div>'
                  if delta else "")
    st.markdown(
        f"""<div style="border-left:4px solid {color};padding:4px 12px;margin-bottom:6px;
        background:#f8fafc;border-radius:0 6px 6px 0;">
        <div style="font-size:12px;color:#64748b;">{label}</div>
        <div style="font-size:22px;font-weight:700;color:#0f172a;line-height:1.2;">{value}</div>
        {delta_html}</div>""",
        unsafe_allow_html=True)


def health_donut(counts):
    fig = go.Figure(go.Pie(
        labels=[HEALTH_LABEL[k] for k in HEALTH_ORDER if counts.get(k, 0) > 0],
        values=[counts.get(k, 0) for k in HEALTH_ORDER if counts.get(k, 0) > 0],
        hole=0.6,
        marker=dict(colors=[HEALTH_COLORS[k] for k in HEALTH_ORDER if counts.get(k, 0) > 0]),
        textinfo="label+percent",
    ))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                      showlegend=True, legend=dict(font=dict(size=11)),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def status_badge(level):
    c = HEALTH_COLORS.get(level, "#64748b")
    return (f'<span style="background:{c};color:white;padding:4px 14px;'
            f'border-radius:999px;font-weight:600;font-size:14px;">'
            f'{HEALTH_LABEL.get(level, level)}</span>')


# ---------------------------------------------------------------------------
# 10. ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Предиктивная аналитика батарей · Дашборд оператора электротранспорта",
        page_icon="🔋", layout="wide", initial_sidebar_state="expanded",
    )

    st.markdown(f"""
    <h1 style="margin-bottom:0;color:{THEME_COLOR};">🔋 Предиктивная аналитика электротранспорта</h1>
    <p style="color:#64748b;margin-top:0;">Прогнозирование SOH и динамического RUL аккумуляторных
    батарей электробусов · ВКР: Науки о данных</p>
    <hr style="margin-top:8px;border:none;border-top:1px solid #e2e8f0;"/>
    """, unsafe_allow_html=True)

    # --- Загрузка ---
    df_raw = load_dataset()
    enhanced, battery_info = build_enhanced_df(df_raw)
    models = load_models()
    config = load_config()

    # Текущее состояние = последний цикл каждой батареи
    current_state = (
        enhanced.sort_values("cycle")
        .groupby("battery_id").last().reset_index()
        [["battery_id", "cycle", "capacity", "soh", "rul", "initial_capacity"]]
        .rename(columns={"cycle": "last_cycle", "capacity": "last_capacity",
                         "soh": "current_soh", "rul": "current_rul"})
        .merge(battery_info[["battery_id", "total_cycles", "eol_cycle",
                             "rul_volatility", "achieved_eol"]], on="battery_id")
    )
    current_state["level"] = current_state["current_soh"].apply(get_warning_level)
    current_state["action"] = current_state["level"].apply(lambda l: HEALTH_ACTION[l])

    # --- Боковая панель ---
    st.sidebar.markdown("## ⚙️ Панель оператора")
    battery_ids = sorted(df_raw["battery_id"].unique())
    selected = st.sidebar.selectbox("Выбор батареи", battery_ids,
                                    help="Идентификатор батареи (B0005…B0056)")
    show_lstm = st.sidebar.checkbox("Показывать LSTM", value=True)
    show_xgb = st.sidebar.checkbox("Показывать XGBoost RUL", value=True)
    show_ens = st.sidebar.checkbox("Показывать Ensemble RUL", value=True)
    horizon = st.sidebar.slider("Горизонт прогноза (циклы вперед)", 5, 100, 30, step=5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**О дашборде**")
    st.sidebar.caption(
        f"Данные: NASA Li-ion Battery Aging (33 батареи, {len(df_raw)} циклов).\n\n"
        "Модели: RF + XGBoost + 2 LSTM (window=20) + Ensemble RUL.\n\n"
        f"Признаков: {len(FEATURE_COLS)}. RUL — динамический.\n\n"
        "Пороги SOH: 85 / 70 / 60 %.")
    missing = [k for k in ["rf_soh", "rf_rul", "lstm_soh", "lstm_rul"]
               if models.get(k) is None]
    if missing:
        st.sidebar.warning(f"Не загружены: {', '.join(missing)}")

    # --- Вкладки ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚌 Обзор парка",
        "🔋 Детали батареи",
        "📈 Прогноз SOH/RUL",
        "🔋 Базовая модель емкости",
        "📊 Качество моделей",
        "🚨 Предупреждения",
    ])

    with tab1:
        render_park_overview(current_state, battery_info)
    with tab2:
        render_battery_details(enhanced, selected, battery_info)
    with tab3:
        render_forecast(enhanced, selected, models, battery_info, horizon,
                        show_lstm, show_xgb, show_ens)
    with tab4:
        render_capacity_model(df_raw, selected, models)
    with tab5:
        render_model_quality(enhanced, models, battery_info, config)
    with tab6:
        render_alerts(current_state)

    st.markdown(
        f'<hr style="border:none;border-top:1px solid #e2e8f0;margin-top:24px"/>'
        f'<p style="color:#94a3b8;font-size:12px;text-align:center;">'
        f'Дашборд предиктивной аналитики батарей электротранспорта · '
        f'ВКР «Разработка системы предиктивной аналитики электротранспорта '
        f'на основе данных телеметрии» · Streamlit + Plotly + TensorFlow + XGBoost</p>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ВКЛАДКА 1: ОБЗОР ПАРКА
# ---------------------------------------------------------------------------
def render_park_overview(current_state, battery_info):
    st.markdown("### 🚌 Обзор парка электробусов")

    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(current_state)
    avg_soh = current_state["current_soh"].mean()
    avg_rul = current_state["current_rul"].mean()
    n_critical = (current_state["level"].isin(["ОРАНЖЕВЫЙ", "КРАСНЫЙ"])).sum()
    n_replacement = (current_state["level"] == "КРАСНЫЙ").sum()

    with c1: kpi_card("Всего батарей", f"{total}", color=THEME_COLOR)
    with c2: kpi_card("Средний SOH парка", f"{avg_soh:.1f}%",
                      delta=f"±{current_state['current_soh'].std():.1f}", color=THEME_COLOR)
    with c3: kpi_card("Средний RUL парка", f"{avg_rul:.0f} цикл.", color=THEME_COLOR)
    with c4: kpi_card("В зоне риска", f"{n_critical}",
                      delta="оранж.+красн." if n_critical else "норма",
                      color=HEALTH_COLORS["ОРАНЖЕВЫЙ"])
    with c5: kpi_card("Требуют замены", f"{n_replacement}",
                      delta="срочно" if n_replacement else "нет",
                      color=HEALTH_COLORS["КРАСНЫЙ"])

    st.markdown("")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("#### Распределение по зонам состояния")
        counts = current_state["level"].value_counts().to_dict()
        st.plotly_chart(health_donut(counts), use_container_width=True)

    with col_right:
        st.markdown("#### Текущий SOH по батареям")
        cs = current_state.sort_values("current_soh")
        fig = go.Figure(go.Bar(
            x=cs["battery_id"], y=cs["current_soh"],
            marker_color=[HEALTH_COLORS[s] for s in cs["level"]],
            text=[f"{v:.0f}%" for v in cs["current_soh"]], textposition="outside",
        ))
        fig.add_hline(y=85, line_dash="dot", line_color="#16a34a",
                      annotation_text="норма 85%", annotation_position="top left")
        fig.add_hline(y=70, line_dash="dot", line_color="#ea580c",
                      annotation_text="EOL 70%")
        fig.add_hline(y=60, line_dash="dot", line_color="#dc2626",
                      annotation_text="критично 60%")
        fig.update_layout(height=320, margin=dict(t=20, b=0),
                          yaxis_title="SOH, %", xaxis_title="",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Сводная таблица по парку")
    show_df = current_state[[
        "battery_id", "total_cycles", "last_cycle", "initial_capacity",
        "last_capacity", "current_soh", "current_rul", "rul_volatility",
        "level", "action"
    ]].copy()
    show_df.columns = ["Батарея", "Циклов всего", "Последний цикл", "С0 (Ah)",
                       "Текущая емкость (Ah)", "SOH, %", "RUL, цикл.",
                       "Волатильность RUL", "Зона", "Действие"]
    st.dataframe(show_df, use_container_width=True, hide_index=True, height=400)


# ---------------------------------------------------------------------------
# ВКЛАДКА 2: ДЕТАЛИ БАТАРЕИ
# ---------------------------------------------------------------------------
def render_battery_details(enhanced, selected, battery_info):
    batt = enhanced[enhanced["battery_id"] == selected].sort_values("cycle").reset_index(drop=True)
    info = battery_info[battery_info["battery_id"] == selected].iloc[0]
    last = batt.iloc[-1]
    alert = alert_info(last["soh"], last["rul"])

    st.markdown(f"### 🔋 Батарея `{selected}`")
    st.markdown(status_badge(alert["level"]) +
                f' <span style="color:#64748b;margin-left:10px;">{alert["message"]}</span>'
                f' &nbsp;|&nbsp; <b>Действие:</b> {alert["action"]}',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Циклов наблюдений", f"{len(batt)}", color=THEME_COLOR)
    with c2: kpi_card("Начальная емкость", f"{info['initial_capacity']:.3f} Ah", color=THEME_COLOR)
    with c3: kpi_card("Текущая емкость", f"{last['capacity']:.3f} Ah", color=THEME_COLOR)
    with c4: kpi_card("Текущий SOH", f"{last['soh']:.1f}%",
                      color=HEALTH_COLORS[alert["level"]])
    with c5: kpi_card("Текущий RUL", f"{int(last['rul'])} цикл.",
                      color=HEALTH_COLORS[alert["level"]])

    if last["rul"] == 0:
        st.info(f"ℹ️ Батарея достигла конца жизни (EOL) — SOH опустился ниже "
                f"{SOH_THRESHOLD_EOL}% (цикл {info['eol_cycle']}).")
    else:
        st.success(f"✅ EOL (SOH<{SOH_THRESHOLD_EOL}%) прогнозируется на цикл "
                   f"~{info['eol_cycle']} (осталось ~{int(last['rul'])} циклов). "
                   f"Волатильность RUL: σ={info['rul_volatility']:.1f} (динамический расчет).")

    st.markdown("#### Кривые деградации (динамический RUL)")
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Емкость, Ah", "SOH, % (с зонами порогов)", "RUL, циклы (динамический)"))

    fig.add_trace(go.Scatter(x=batt["cycle"], y=batt["capacity"], mode="lines+markers",
                             name="Емкость", line=dict(color=THEME_COLOR, width=2),
                             marker=dict(size=4)), row=1, col=1)
    fig.add_hline(y=info["initial_capacity"] * 0.7, line_dash="dot",
                  line_color="#dc2626", annotation_text="EOL (0.7·C0)", row=1, col=1)

    fig.add_trace(go.Scatter(x=batt["cycle"], y=batt["soh"], mode="lines+markers",
                             name="SOH", line=dict(color=THEME_COLOR, width=2),
                             marker=dict(size=4), showlegend=False), row=1, col=2)
    for y0, y1, col in [(85, 105, HEALTH_COLORS["ЗЕЛЁНЫЙ"]),
                        (70, 85, HEALTH_COLORS["ЖЁЛТЫЙ"]),
                        (60, 70, HEALTH_COLORS["ОРАНЖЕВЫЙ"]),
                        (0, 60, HEALTH_COLORS["КРАСНЫЙ"])]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.08,
                      line_width=0, row=1, col=2)

    fig.add_trace(go.Scatter(x=batt["cycle"], y=batt["rul"], mode="lines+markers",
                             name="RUL", line=dict(color="#7c3aed", width=2),
                             marker=dict(size=4), showlegend=False), row=1, col=3)
    fig.add_hline(y=0, line_color="#dc2626", row=1, col=3)

    fig.update_layout(height=360, margin=dict(t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Детальная таблица циклов (последние 20)"):
        show = batt[["cycle", "capacity", "soh", "rul"]].tail(20).copy()
        show["health_state"] = show["soh"].apply(get_warning_level)
        show.columns = ["Цикл", "Емкость, Ah", "SOH, %", "RUL, цикл", "Зона"]
        st.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# ВКЛАДКА 3: ПРОГНОЗ SOH/RUL
# ---------------------------------------------------------------------------
def render_forecast(enhanced, selected, models, battery_info, horizon,
                    show_lstm, show_xgb, show_ens):
    batt = enhanced[enhanced["battery_id"] == selected].sort_values("cycle").reset_index(drop=True)
    info = battery_info[battery_info["battery_id"] == selected].iloc[0]

    st.markdown(f"### 📈 Прогноз SOH/RUL для батареи `{selected}`")
    st.caption("Сравнение факта с прогнозами RF / XGBoost / LSTM / Ensemble, "
               "а также авторегрессионный прогноз на будущие циклы.")

    has_rf = models["rf_soh"] is not None and models["rf_rul"] is not None
    has_lstm = show_lstm and models["lstm_soh"] is not None and models["lstm_rul"] is not None
    has_xgb = show_xgb and models.get("xgb_rul") is not None

    # --- SOH: факт vs RF vs LSTM ---
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=batt["cycle"], y=batt["soh"], mode="lines+markers",
                                 name="Факт SOH", line=dict(color="#0f172a", width=2.5)))
        if has_rf:
            ps = predict_rf_soh(batt, models)
            fig.add_trace(go.Scatter(x=batt["cycle"], y=ps, mode="lines",
                                     name="RF SOH", line=dict(color=THEME_COLOR, dash="dash")))
        if has_lstm:
            cyc, ps_l = predict_lstm(models["lstm_soh"], models["scaler_X_soh"],
                                     models["scaler_y_soh"], batt)
            if len(cyc):
                fig.add_trace(go.Scatter(x=cyc, y=ps_l, mode="lines",
                                         name="LSTM SOH", line=dict(color="#9333ea", dash="dot")))
        for y0, y1, col in [(85, 105, HEALTH_COLORS["ЗЕЛЁНЫЙ"]),
                            (70, 85, HEALTH_COLORS["ЖЁЛТЫЙ"]),
                            (60, 70, HEALTH_COLORS["ОРАНЖЕВЫЙ"]),
                            (0, 60, HEALTH_COLORS["КРАСНЫЙ"])]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.07, line_width=0)
        fig.update_layout(height=340, title="SOH: факт vs модели",
                          yaxis_title="SOH, %", xaxis_title="Цикл",
                          margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    # --- RUL: факт vs RF vs XGB vs LSTM vs Ensemble ---
    with sub_c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=batt["cycle"], y=batt["rul"], mode="lines+markers",
                                 name="Факт RUL", line=dict(color="#0f172a", width=2.5)))
        if has_rf:
            pr = predict_rf_rul(batt, models)
            fig.add_trace(go.Scatter(x=batt["cycle"], y=pr, mode="lines",
                                     name="RF RUL", line=dict(color=THEME_COLOR, dash="dash")))
        if has_xgb:
            px = predict_xgb_rul(batt, models)
            fig.add_trace(go.Scatter(x=batt["cycle"], y=px, mode="lines",
                                     name="XGB RUL", line=dict(color="#d97706", dash="dashdot")))
        if has_lstm:
            cyc, pr_l = predict_lstm(models["lstm_rul"], models["scaler_X_rul"],
                                     models["scaler_y_rul"], batt)
            if len(cyc):
                fig.add_trace(go.Scatter(x=cyc, y=pr_l, mode="lines",
                                         name="LSTM RUL", line=dict(color="#9333ea", dash="dot")))
        if show_ens:
            res = predict_ensemble_rul(batt, models)
            if res is not None:
                cyc_e, ens = res
                fig.add_trace(go.Scatter(x=cyc_e, y=ens, mode="lines",
                                         name="Ensemble RUL", line=dict(color="#dc2626", width=3)))
        fig.add_hline(y=0, line_color="#dc2626", line_dash="dot")
        fig.update_layout(height=340, title="RUL: факт vs модели (динамический)",
                          yaxis_title="RUL, циклы", xaxis_title="Цикл",
                          margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    # --- Прогноз на будущие циклы (авторегрессионный через LSTM) ---
    st.markdown(f"#### Прогноз на {horizon} циклов вперёд (LSTM-авторегрессия)")
    lstm_available = models.get("lstm_soh") is not None
    if lstm_available:
        fc, fs, fr, use_lstm = forecast_future_lstm(batt, info["initial_capacity"],
                                                    horizon, models)
        fc_c1, fc_c2 = st.columns(2)
        with fc_c1:
            fig = go.Figure()
            hist = batt.tail(30)
            fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["soh"], mode="lines+markers",
                                     name="Факт (история)", line=dict(color="#0f172a", width=2)))
            fig.add_trace(go.Scatter(x=fc, y=fs, mode="lines+markers",
                                     name="LSTM прогноз SOH",
                                     line=dict(color=THEME_COLOR, dash="dash")))
            fig.add_vline(x=batt["cycle"].max(), line_dash="dot", line_color="#94a3b8")
            for y0, y1, col in [(85, 105, HEALTH_COLORS["ЗЕЛЁНЫЙ"]),
                                (70, 85, HEALTH_COLORS["ЖЁЛТЫЙ"]),
                                (60, 70, HEALTH_COLORS["ОРАНЖЕВЫЙ"]),
                                (0, 60, HEALTH_COLORS["КРАСНЫЙ"])]:
                fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.07, line_width=0)
            fig.update_layout(height=340, title="Прогноз SOH (LSTM, авторегрессия)",
                              yaxis_title="SOH, %", xaxis_title="Цикл",
                              margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)
        with fc_c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul"], mode="lines+markers",
                                     name="Факт (история)", line=dict(color="#0f172a", width=2)))
            fig.add_trace(go.Scatter(x=fc, y=fr, mode="lines+markers",
                                     name="LSTM прогноз RUL",
                                     line=dict(color="#7c3aed", dash="dash")))
            fig.add_vline(x=batt["cycle"].max(), line_dash="dot", line_color="#94a3b8")
            fig.add_hline(y=0, line_color="#dc2626", line_dash="dot")
            fig.update_layout(height=340, title="Прогноз RUL (LSTM, авторегрессия)",
                              yaxis_title="RUL, циклы", xaxis_title="Цикл",
                              margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Таблица прогноза (LSTM, авторегрессионный)")
        pred_df = pd.DataFrame({
            "Цикл": fc, "SOH прогноз, %": np.round(fs, 1),
            "RUL прогноз, цикл": np.round(fr, 0).astype(int),
        })
        pred_df["Зона"] = [get_warning_level(s) for s in fs]
        st.dataframe(pred_df, use_container_width=True, hide_index=True, height=260)
    else:
        st.warning("LSTM-модель не загружена — авторегрессионный прогноз недоступен.")


# ---------------------------------------------------------------------------
# ВКЛАДКА 4: БАЗОВАЯ МОДЕЛЬ ЕМКОСТИ (battery.ipynb)
# ---------------------------------------------------------------------------
def render_capacity_model(df_raw, selected, models):
    st.markdown("### 🔋 Базовая модель прогноза емкости (lag=3)")
    st.caption("Из battery.ipynb: прогноз емкости текущего цикла по емкостям "
               "трех предыдущих циклов. RF + LSTM (window=3).")

    if models["rf_capacity"] is None:
        st.warning("Модели емкости не загружены.")
        return

    batt = df_raw[df_raw["battery_id"] == selected].sort_values("cycle").reset_index(drop=True)
    last = batt.iloc[-1]

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Циклов наблюдений", f"{len(batt)}", color=THEME_COLOR)
    with c2: kpi_card("Текущая емкость", f"{last['capacity']:.3f} Ah", color=THEME_COLOR)
    with c3: kpi_card("Начальная емкость", f"{batt['capacity'].head(5).median():.3f} Ah",
                      color=THEME_COLOR)

    # --- График: факт vs RF vs LSTM ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=batt["cycle"], y=batt["capacity"], mode="lines+markers",
                             name="Факт емкость", line=dict(color="#0f172a", width=2.5),
                             marker=dict(size=4)))
    cyc_rf, pred_rf = predict_capacity_rf(batt, models)
    fig.add_trace(go.Scatter(x=cyc_rf, y=pred_rf, mode="lines",
                             name="RF прогноз", line=dict(color=THEME_COLOR, dash="dash")))
    if models["lstm_capacity"] is not None:
        cyc_l, pred_l = predict_capacity_lstm(batt, models)
        fig.add_trace(go.Scatter(x=cyc_l, y=pred_l, mode="lines",
                                 name="LSTM прогноз", line=dict(color="#9333ea", dash="dot")))
    fig.add_hline(y=batt["capacity"].head(5).median() * 0.7, line_dash="dot",
                  line_color="#dc2626", annotation_text="EOL (0.7·C0)")
    fig.update_layout(height=380, title=f"Емкость батареи {selected}: факт vs модели",
                      yaxis_title="Емкость, Ah", xaxis_title="Цикл",
                      margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True)

    # --- Метрики базовой модели ---
    cap_m = models["metrics"].get("capacity")
    if cap_m:
        st.markdown("#### Метрики базовой модели")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Random Forest**")
            st.markdown(f"MAE: `{cap_m['rf_mae']:.4f}` Ah  ·  "
                        f"RMSE: `{cap_m['rf_rmse']:.4f}` Ah  ·  "
                        f"R²: `{cap_m['rf_r2']:.4f}`")
        with mc2:
            st.markdown("**LSTM**")
            st.markdown(f"MAE: `{cap_m['lstm_mae']:.4f}` Ah  ·  "
                        f"RMSE: `{cap_m['lstm_rmse']:.4f}` Ah  ·  "
                        f"R²: `{cap_m['lstm_r2']:.4f}`")
        st.caption(f"Признаки: `{', '.join(cap_m.get('features', CAP_FEATURES))}` · "
                   f"окно (timesteps): {cap_m.get('timesteps', CAP_LAGS)}")


# ---------------------------------------------------------------------------
# ВКЛАДКА 5: КАЧЕСТВО МОДЕЛЕЙ
# ---------------------------------------------------------------------------
def render_model_quality(enhanced, models, battery_info, config):
    st.markdown("### 📊 Качество моделей")
    metrics = models.get("metrics", {})

    st.markdown("#### Метрики на тестовой выборке (30% батарей)")
    rows = []
    for key, label in [("rf_soh", "RF · SOH"), ("lstm_soh", "LSTM · SOH"),
                       ("rf_rul", "RF · RUL"), ("xgb_rul", "XGBoost · RUL"),
                       ("lstm_rul", "LSTM · RUL"), ("ensemble_rul", "Ensemble · RUL")]:
        m = metrics.get(key)
        if m and "MAE" in m:
            rows.append({"Модель": label, "MAE": round(m["MAE"], 3),
                         "RMSE": round(m["RMSE"], 3), "R²": round(m["R2"], 3)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        best_soh = max([r for r in rows if "SOH" in r["Модель"]], key=lambda r: r["R²"])
        best_rul = max([r for r in rows if "RUL" in r["Модель"]], key=lambda r: r["R²"])
        st.success(f"**Лучшая модель SOH:** {best_soh['Модель']} (R²={best_soh['R²']:.3f})  ·  "
                   f"**Лучшая модель RUL:** {best_rul['Модель']} (R²={best_rul['R²']:.3f})")
    else:
        st.warning("Метрики не найдены.")

    # --- Базовая модель ---
    cap_m = metrics.get("capacity")
    if cap_m:
        with st.expander("📊 Метрики базовой модели емкости (lag=3)"):
            cap_rows = [
                {"Модель": "RF · Capacity", "MAE": round(cap_m["rf_mae"], 4),
                 "RMSE": round(cap_m["rf_rmse"], 4), "R²": round(cap_m["rf_r2"], 4)},
                {"Модель": "LSTM · Capacity", "MAE": round(cap_m["lstm_mae"], 4),
                 "RMSE": round(cap_m["lstm_rmse"], 4), "R²": round(cap_m["lstm_r2"], 4)},
            ]
            st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)

    # --- Scatter факт vs прогноз (RF SOH/RUL, весь датасет) ---
    st.markdown("#### Факт vs прогноз (Random Forest, весь датасет)")
    if models["rf_soh"] is not None and models["rf_rul"] is not None:
        X = enhanced[FEATURE_COLS].values
        ps = models["rf_soh"].predict(X)
        pr = models["rf_rul"].predict(X)
        sc1, sc2 = st.columns(2)
        with sc1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=enhanced["soh"], y=ps, mode="markers",
                                     marker=dict(color=THEME_COLOR, size=3, opacity=0.4),
                                     showlegend=False))
            lim = [min(enhanced["soh"].min(), ps.min()) - 2,
                   max(enhanced["soh"].max(), ps.max()) + 2]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                     line=dict(color="#94a3b8", dash="dash"), showlegend=False))
            fig.update_layout(title="SOH: факт vs RF-прогноз", height=340,
                              xaxis_title="Факт SOH, %", yaxis_title="Прогноз SOH, %",
                              margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with sc2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=enhanced["rul"], y=pr, mode="markers",
                                     marker=dict(color="#9333ea", size=3, opacity=0.4),
                                     showlegend=False))
            lim = [0, max(enhanced["rul"].max(), pr.max()) + 5]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                     line=dict(color="#94a3b8", dash="dash"), showlegend=False))
            fig.update_layout(title="RUL: факт vs RF-прогноз", height=340,
                              xaxis_title="Факт RUL, циклы", yaxis_title="Прогноз RUL, циклы",
                              margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # --- Feature importance ---
    st.markdown("#### Важность признаков (Random Forest)")
    if models["rf_soh"] is not None:
        try:
            fi = pd.DataFrame({
                "Признак": FEATURE_COLS,
                "Важность (SOH)": models["rf_soh"].feature_importances_,
                "Важность (RUL)": models["rf_rul"].feature_importances_,
            }).sort_values("Важность (SOH)", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=fi["Признак"], x=fi["Важность (SOH)"], orientation="h",
                                 name="SOH", marker_color=THEME_COLOR))
            fig.add_trace(go.Bar(y=fi["Признак"], x=fi["Важность (RUL)"], orientation="h",
                                 name="RUL", marker_color="#9333ea"))
            fig.update_layout(height=400, barmode="group",
                              xaxis_title="Важность", yaxis_title="",
                              margin=dict(t=20, b=20), paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Не удалось получить важность признаков: {e}")

    # --- Конфигурация ---
    if config:
        with st.expander("⚙️ Конфигурация обучения (model_config.json)"):
            st.json({
                "Признаки": config.get("feature_cols"),
                "Окно LSTM (window_size)": config.get("window_size"),
                "Порог EOL SOH, %": config.get("soh_threshold_eol"),
                "Warmup": config.get("warmup"),
                "Local window (RUL)": config.get("local_window"),
                "Min window (RUL)": config.get("min_window"),
                "Веса ensemble": config.get("ensemble_weights"),
                "Режим RUL": config.get("rul_mode", "dynamic_v4"),
                "Батареи в тесте": config.get("test_batteries"),
            })


# ---------------------------------------------------------------------------
# ВКЛАДКА 6: СИСТЕМА ПРЕДУПРЕЖДЕНИЙ
# ---------------------------------------------------------------------------
def render_alerts(current_state):
    st.markdown("### 🚨 Система предупреждений оператора")
    st.caption("Уровень предупреждения определяется **только по SOH** "
               "(как в battery_enhanced.ipynb). Пороги: 85 / 70 / 60 %.")

    statuses = ["Все"] + HEALTH_ORDER
    filt = st.multiselect("Фильтр по зоне", statuses, default=["Все"])
    cs = current_state.copy()
    if filt and "Все" not in filt:
        cs = cs[cs["level"].isin(filt)]
    cs = cs.sort_values(["level"], key=lambda s: s.map(
        {k: i for i, k in enumerate(HEALTH_ORDER)}))

    c1, c2, c3, c4 = st.columns(4)
    counts = current_state["level"].value_counts().to_dict()
    with c1: kpi_card("Зелёный (норма)", f"{counts.get('ЗЕЛЁНЫЙ', 0)}", color=HEALTH_COLORS["ЗЕЛЁНЫЙ"])
    with c2: kpi_card("Жёлтый (внимание)", f"{counts.get('ЖЁЛТЫЙ', 0)}", color=HEALTH_COLORS["ЖЁЛТЫЙ"])
    with c3: kpi_card("Оранжевый (риск)", f"{counts.get('ОРАНЖЕВЫЙ', 0)}", color=HEALTH_COLORS["ОРАНЖЕВЫЙ"])
    with c4: kpi_card("Красный (критично)", f"{counts.get('КРАСНЫЙ', 0)}", color=HEALTH_COLORS["КРАСНЫЙ"])

    st.markdown("")
    show = cs[["battery_id", "last_cycle", "current_soh", "current_rul",
               "level", "action"]].copy()
    show.columns = ["Батарея", "Посл. цикл", "SOH, %", "RUL, цикл", "Зона", "Действие"]
    styled = show.style.apply(
        lambda r: [f"background-color: {HEALTH_COLORS[r['Зона']]}22;"] * len(r), axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    st.markdown("#### Рекомендации по критическим батареям")
    crit = cs[cs["level"].isin(["ОРАНЖЕВЫЙ", "КРАСНЫЙ"])]
    if len(crit) == 0:
        st.success("✅ Критических батарей нет — парк в норме.")
    else:
        for _, r in crit.iterrows():
            alert = alert_info(r["current_soh"], r["current_rul"])
            icon = "🔴" if r["level"] == "КРАСНЫЙ" else "🟠"
            st.markdown(
                f"{icon} **{r['battery_id']}** — SOH {r['current_soh']:.1f}% / "
                f"RUL {int(r['current_rul'])} цикл. · *{alert['message']}* "
                f"→ **{alert['action']}**")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
