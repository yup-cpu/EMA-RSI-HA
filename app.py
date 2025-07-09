import streamlit as st
import pandas as pd
import numpy as np
from numba import njit
import io

# ------------------------
# Load dữ liệu
# ------------------------
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.set_index('Datetime', inplace=True)
    return df

# ------------------------
# Chỉ báo kỹ thuật
# ------------------------
@njit
def compute_ema(close, span):
    n = len(close)
    ema = np.empty(n)
    ema[0] = close[0]
    alpha = 2 / (span + 1)
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
    return ema

@njit
def compute_rsi(close, period):
    n = len(close)
    delta = np.empty(n)
    delta[0] = 0
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]

    gain = np.zeros(n)
    loss = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            gain[i] = delta[i]
        else:
            loss[i] = -delta[i]

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    for i in range(period + 1):
        rsi[i] = np.nan
    return rsi

@njit
def compute_ha(open_, high, low, close):
    n = len(open_)
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.empty(n)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_color = np.where(ha_close > ha_open, 1, 0)
    return ha_color

@njit
def detect_signals_sequential(ohlc, ema50, rsi, ha, rsi_lo=30, rsi_hi=70):
    n = len(ohlc)
    max_signals = n
    idxs = np.empty(max_signals, dtype=np.int32)
    types = np.empty(max_signals, dtype=np.int8)
    prices = np.empty(max_signals, dtype=np.float64)
    points = np.empty(max_signals, dtype=np.int8)
    count = 0

    for i in range(1, n):
        if np.isnan(rsi[i]):
            continue
        for j in range(4):
            price = ohlc[i, j]
            if price > ema50[i] and rsi_lo < rsi[i] < rsi_hi and ha[i - 1] == 0 and ha[i] == 1:
                idxs[count] = i
                types[count] = 1
                prices[count] = price
                points[count] = j
                count += 1
                break
            elif price < ema50[i] and rsi_lo < rsi[i] < rsi_hi and ha[i - 1] == 1 and ha[i] == 0:
                idxs[count] = i
                types[count] = 0
                prices[count] = price
                points[count] = j
                count += 1
                break
    return idxs[:count], types[:count], prices[:count], points[:count]

# ------------------------
# Streamlit App
# ------------------------
st.title("📈 Chiến lược giao dịch: EMA50 + RSI14 + Heiken Ashi")

uploaded_file = st.file_uploader("📂 Tải lên file CSV dữ liệu (không có header):", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    open_, high, low, close = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    ema = compute_ema(close, 50)
    rsi = compute_rsi(close, 14)
    ha = compute_ha(open_, high, low, close)

    valid = ~np.isnan(rsi)
    ohlc = np.stack([open_[valid], high[valid], low[valid], close[valid]], axis=1)
    ema, rsi, ha = ema[valid], rsi[valid], ha[valid]

    idxs, types, prices, points = detect_signals_sequential(ohlc, ema, rsi, ha)

    st.success(f"✅ Tổng tín hiệu: {len(idxs)}")
    st.info(f"🔼 BUY: {np.sum(types == 1)}")
    st.warning(f"🔽 SELL: {np.sum(types == 0)}")

    # Hiển thị bảng tín hiệu
    signal_df = pd.DataFrame({
        'Datetime': df.index[valid][idxs],
        'Type': np.where(types == 1, 'BUY', 'SELL'),
        'Price': prices,
        'Point': np.array(['Open', 'High', 'Low', 'Close'])[points]
    })

    st.dataframe(signal_df, use_container_width=True)