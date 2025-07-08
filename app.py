import streamlit as st
import pandas as pd
import numpy as np
from numba import njit
import io
import requests

# ----------------------------
# Phần tính toán kỹ thuật
# ----------------------------
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

# ----------------------------
# Load dữ liệu
# ----------------------------
def load_data_from_df(df):
    df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Spread']
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    for col in ['Open', 'High', 'Low', 'Close', 'Spread']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.set_index('Datetime', inplace=True)
    return df

# ----------------------------
# Giao diện Streamlit
# ----------------------------

st.set_page_config(page_title="Phân tích tín hiệu giao dịch", layout="wide")
st.title("📈 Phân tích tín hiệu BUY / SELL")

# Cách chọn
method = st.radio("📁 Chọn cách tải dữ liệu:", ["Tải từ máy (Upload)", "Nhập đường dẫn (URL)", "Google Drive ID"])

uploaded_file = None
df = None
url = ""

if method == "Tải từ máy (Upload)":
    uploaded_file = st.file_uploader("📂 Chọn file CSV", type=["csv"])

elif method == "Nhập đường dẫn (URL)":
    url = st.text_input("🔗 Nhập đường dẫn tới file CSV:")

elif method == "Google Drive ID":
    drive_id = st.text_input("🔑 Nhập Google Drive file ID:")
    if drive_id:
        url = f"https://drive.google.com/uc?id={drive_id}"

if st.button("📥 Load File"):
    try:
        if method == "Tải từ máy (Upload)" and uploaded_file:
            df = pd.read_csv(uploaded_file, header=None)
        elif method in ["Nhập đường dẫn (URL)", "Google Drive ID"] and url:
            response = requests.get(url)
            df = pd.read_csv(io.StringIO(response.text), header=None)
        else:
            st.warning("⚠️ Hãy chọn và nhập đúng thông tin để tải file.")
    except Exception as e:
        st.error(f"❌ Lỗi khi tải file: {e}")

# ----------------------------
# Phân tích sau khi tải thành công
# ----------------------------
if df is not None:
    st.success("✅ File đã được tải thành công!")
    df = load_data_from_df(df)
    st.dataframe(df.head())

    with st.spinner("⏳ Đang phân tích tín hiệu..."):
        open_, high, low, close = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        ema = compute_ema(close, 50)
        rsi = compute_rsi(close, 14)
        ha = compute_ha(open_, high, low, close)

        valid = ~np.isnan(rsi)
        ohlc = np.stack([open_[valid], high[valid], low[valid], close[valid]], axis=1)
        ema, rsi, ha = ema[valid], rsi[valid], ha[valid]

        idxs, types, prices, points = detect_signals_sequential(ohlc, ema, rsi, ha)

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Tổng tín hiệu", len(idxs))
    col2.metric("🔼 BUY", int(np.sum(types == 1)))
    col3.metric("🔽 SELL", int(np.sum(types == 0)))

    # Hiển thị bảng tín hiệu
    if len(idxs) > 0:
        df_result = pd.DataFrame({
            "Thời gian": df.index[valid][idxs],
            "Loại lệnh": ["BUY" if t == 1 else "SELL" for t in types],
            "Giá vào lệnh": prices,
            "Tại điểm": ["Open", "High", "Low", "Close"],
        })
        st.dataframe(df_result)
    else:
        st.info("ℹ️ Không có tín hiệu nào được phát hiện trong dữ liệu.")