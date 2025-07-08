import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
from numba import njit

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Spread']
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    for col in ['Open', 'High', 'Low', 'Close', 'Spread']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.set_index('Datetime', inplace=True)
    return df

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
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1]*(period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1]*(period-1) + loss[i]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    for i in range(period+1):
        rsi[i] = np.nan
    return rsi

@njit
def compute_ha(open_, high, low, close):
    n = len(open_)
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.empty(n)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
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

def extract_drive_id(text):
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"^([a-zA-Z0-9_-]{25,})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

st.title("📈 Phát hiện tín hiệu giao dịch XAUUSD")

if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "load_state" not in st.session_state:
    st.session_state.load_state = "idle"

input_ready = False

with st.form(key="load_form"):
    data_source = st.selectbox("📅 Chọn nguồn dữ liệu", [
        "Tải file từ máy",
        "Từ Google Drive (đường dẫn)",
        "Từ Google Drive (link hoặc ID)"
    ])

    uploaded_file = None
    path = ""
    user_input = ""

    if data_source == "Tải file từ máy":
        uploaded_file = st.file_uploader("📂 Chọn file CSV", type=["csv"])
        input_ready = uploaded_file is not None
    elif data_source == "Từ Google Drive (đường dẫn)":
        path = st.text_input("📂 Nhập đường dẫn file")
        input_ready = bool(path)
    elif data_source == "Từ Google Drive (link hoặc ID)":
        user_input = st.text_input("🔗 Nhập link hoặc ID")
        input_ready = bool(user_input)

    submit = st.form_submit_button("📤 Tải và xử lý", disabled=not input_ready)

if submit:
    st.session_state.load_state = "loading"
    with st.spinner("Đang tải dữ liệu..."):
        try:
            if uploaded_file:
                df = load_data(uploaded_file)
                st.session_state.df = df
                st.success("✅ Đã tải dữ liệu từ máy")
            elif path and os.path.exists(path):
                df = load_data(path)
                st.session_state.df = df
                st.success("✅ Đã tải từ đường dẫn Drive")
            elif user_input:
                file_id = extract_drive_id(user_input)
                if file_id:
                    import gdown
                    url = f"https://drive.google.com/uc?id={file_id}"
                    local_path = "downloaded_data.csv"
                    gdown.download(url, local_path, quiet=True)
                    df = load_data(local_path)
                    st.session_state.df = df
                    st.success(f"✅ Tải thành công từ ID: {file_id}")
                else:
                    st.error("❌ Không nhận diện được ID")
            else:
                st.warning("⚠️ Vui lòng cung cấp dữ liệu")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
        finally:
            st.session_state.load_state = "done"

if st.session_state.df is not None:
    if st.button("🔍 Phân tích tín hiệu"):
        df = st.session_state.df
        open_, high, low, close = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        ema = compute_ema(close, 50)
        rsi = compute_rsi(close, 14)
        ha = compute_ha(open_, high, low, close)

        valid = ~np.isnan(rsi)
        valid_index = df.index[valid].to_numpy()

        ohlc = np.stack([open_[valid], high[valid], low[valid], close[valid]], axis=1)
        ema, rsi, ha = ema[valid], rsi[valid], ha[valid]

        idxs, types, prices, points = detect_signals_sequential(ohlc, ema, rsi, ha)

        if len(idxs) == 0:
            st.warning("⚠️ Không phát hiện tín hiệu")
        else:
            signal_df = pd.DataFrame({
                'Datetime': valid_index[idxs],
                'Type': ['BUY' if t == 1 else 'SELL' for t in types],
                'Price': prices,
                'Point (OHLC)': [['O', 'H', 'L', 'C'][p] for p in points]
            })

            st.session_state.results = {
                "valid_index": valid_index.astype(str).tolist(),
                "ema": ema.tolist(),
                "rsi": rsi.tolist(),
                "ha": ha.tolist(),
                "signals": {
                    "indexes": idxs.tolist(),
                    "types": types.tolist(),
                    "prices": prices.tolist(),
                    "points": points.tolist()
                }
            }

            st.success(f"✅ Tổng tín hiệu: {len(signal_df)}")
            st.dataframe(signal_df)