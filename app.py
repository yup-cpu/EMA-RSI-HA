import streamlit as st
import pandas as pd
import numpy as np
from numba import njit
import re

# ---------------------
# Tải dữ liệu
# ---------------------
@st.cache_data
def load_data(source):
    df = pd.read_csv(source, header=None)
    df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Spread']
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    for col in ['Open', 'High', 'Low', 'Close', 'Spread']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.set_index('Datetime', inplace=True)
    return df

# ---------------------
# Các hàm kỹ thuật
# ---------------------
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

# ---------------------
# Giao diện Streamlit
# ---------------------
st.title("📊 Phân tích tín hiệu giao dịch XAUUSD")

st.markdown("### 📥 Chọn nguồn dữ liệu")
data_source = st.radio("Nguồn dữ liệu", ["Tải từ máy", "Google Drive (link hoặc ID)"])

uploaded_file = None
drive_link = ""

if data_source == "Tải từ máy":
    uploaded_file = st.file_uploader("📂 Chọn file CSV từ máy", type=["csv"])
elif data_source == "Google Drive (link hoặc ID)":
    drive_link = st.text_input("🔗 Nhập Google Drive link hoặc ID")

df = None
if uploaded_file:
    df = load_data(uploaded_file)
elif drive_link:
    try:
        import gdown
        file_id = extract_drive_id(drive_link)
        if file_id:
            url = f"https://drive.google.com/uc?id={file_id}"
            local_path = "temp.csv"
            gdown.download(url, local_path, quiet=True)
            df = load_data(local_path)
        else:
            st.error("❌ Không nhận diện được ID từ link.")
    except Exception as e:
        st.error(f"❌ Lỗi tải dữ liệu: {e}")

if df is not None:
    st.success("✅ Dữ liệu đã được tải thành công.")
    st.dataframe(df.head())

    if st.button("🚀 Phân tích tín hiệu"):
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
            st.warning("⚠️ Không phát hiện tín hiệu.")
        else:
            signal_df = pd.DataFrame({
                "Datetime": valid_index[idxs],
                "Type": ['BUY' if t == 1 else 'SELL' for t in types],
                "Price": prices,
                "Point (OHLC)": [ ['O', 'H', 'L', 'C'][p] for p in points ]
            })
            st.success(f"✅ Phát hiện {len(signal_df)} tín hiệu.")
            st.dataframe(signal_df)