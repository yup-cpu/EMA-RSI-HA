import streamlit as st
import pandas as pd
import numpy as np
from numba import njit
import requests
import io
import gdown

# ------------------------
# Chuẩn hóa link
# ------------------------
def normalize_url(url):
    if "drive.google.com" in url:
        if "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
        elif "/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
        else:
            raise ValueError("❌ Không tìm thấy ID hợp lệ từ Google Drive link.")
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    elif "dropbox.com" in url:
        return url.replace("?dl=0", "?dl=1")
    elif url.endswith(".csv"):
        return url
    else:
        raise ValueError("❌ Link không hợp lệ. Hãy nhập link Google Drive, Dropbox hoặc file .csv.")

# ------------------------
# Load dữ liệu
# ------------------------
def load_data_safe(file_like):
    try:
        content = file_like.read().decode("utf-8")
    except:
        content = file_like.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        else:
            raise ValueError("❌ Không thể giải mã nội dung file.")
    try:
        df = pd.read_csv(io.StringIO(content), header=None)
    except Exception as e:
        raise ValueError(f"❌ Lỗi khi đọc CSV: {str(e)}")

    if df.shape[1] != 7:
        raise ValueError(f"❌ Dữ liệu có {df.shape[1]} cột, cần đúng 7 cột: Date, Time, Open, High, Low, Close, Volume.")

    try:
        df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M', errors='coerce')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)
        df.set_index('Datetime', inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"❌ Lỗi xử lý dữ liệu: {str(e)}")

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
def compute_rsi_raw(close, period):
    n = len(close)
    delta = np.empty(n)
    delta[0] = 0
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]

    gain = np.empty(n)
    loss = np.empty(n)
    for i in range(n):
        if delta[i] > 0:
            gain[i] = delta[i]
            loss[i] = 0
        else:
            gain[i] = 0
            loss[i] = -delta[i]

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    sum_gain = 0.0
    sum_loss = 0.0
    for i in range(1, period + 1):
        sum_gain += gain[i]
        sum_loss += loss[i]
    avg_gain[period] = sum_gain / period
    avg_loss[period] = sum_loss / period

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
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
        if rsi[i] != rsi[i]:
            continue
        for j in range(4):
            price = ohlc[i, j]
            if price > ema50[i] and rsi_lo < rsi[i] < rsi_hi and ha[i - 1] == 0 and ha[i] == 1:
                idxs[count], types[count], prices[count], points[count] = i, 1, price, j
                count += 1
                break
            elif price < ema50[i] and rsi_lo < rsi[i] < rsi_hi and ha[i - 1] == 1 and ha[i] == 0:
                idxs[count], types[count], prices[count], points[count] = i, 0, price, j
                count += 1
                break
    return idxs[:count], types[:count], prices[:count], points[:count]

# ------------------------
# Giao diện Streamlit
# ------------------------
st.title("📈 Chiến lược giao dịch: EMA50 + RSI14 + Heiken Ashi")

# Trạng thái mặc định
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "data_source" not in st.session_state:
    st.session_state.data_source = None
if "df" not in st.session_state:
    st.session_state.df = None

option = st.radio("📥 Chọn cách nhập dữ liệu:", ["📂 Tải lên file CSV", "🌐 Link đến file CSV", "📝 Dán nội dung CSV"])

# Reset nếu đổi nguồn
if st.session_state.data_source != option:
    st.session_state.confirmed = False
    st.session_state.df = None
    st.session_state.data_source = option

# Biến lưu dữ liệu tạm
uploaded_file, url, content = None, "", ""

# Hiển thị vùng nhập tương ứng
if option == "📂 Tải lên file CSV":
    uploaded_file = st.file_uploader("Tải file CSV dữ liệu (không có header):", type=["csv"])
elif option == "🌐 Link đến file CSV":
    url = st.text_input("Dán link Google Drive / Dropbox / CSV:")
elif option == "📝 Dán nội dung CSV":
    content = st.text_area("Dán nội dung CSV (raw text):")

# ✅ Chỉ 1 nút duy nhất
if st.button("📥 Tải & Phân tích", disabled=st.session_state.confirmed):
    try:
        if option == "📂 Tải lên file CSV":
            if uploaded_file is not None:
                st.session_state.df = load_data_safe(uploaded_file)
            else:
                st.warning("⚠️ Vui lòng chọn file.")
                st.stop()

        elif option == "🌐 Link đến file CSV":
            if not url.strip():
                st.warning("⚠️ Vui lòng nhập link.")
                st.stop()
            norm_url = normalize_url(url)
            if "drive.google.com" in url:
                gdown.download(norm_url, "temp.csv", quiet=False)
                with open("temp.csv", "rb") as f:
                    st.session_state.df = load_data_safe(f)
            else:
                response = requests.get(norm_url)
                response.raise_for_status()
                st.session_state.df = load_data_safe(io.BytesIO(response.content))

        elif option == "📝 Dán nội dung CSV":
            if not content.strip():
                st.warning("⚠️ Vui lòng dán nội dung CSV.")
                st.stop()
            st.session_state.df = load_data_safe(io.StringIO(content))

        st.session_state.confirmed = True
        st.rerun()

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý dữ liệu: {str(e)}")

# Nếu đã xác nhận thành công
if st.session_state.confirmed:
    st.markdown("✅ **Dữ liệu đã được phân tích. Để phân tích lại, hãy thay đổi nguồn dữ liệu.**")

# Nếu có dữ liệu
if st.session_state.df is not None and st.session_state.confirmed:
    df = st.session_state.df

    open_, high, low, close = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    ema = compute_ema(close, 50)
    rsi = compute_rsi_raw(close, 14)
    rsi[:15] = np.nan
    ha = compute_ha(open_, high, low, close)

    valid = ~np.isnan(rsi)
    ohlc = np.stack([open_[valid], high[valid], low[valid], close[valid]], axis=1)
    ema, rsi, ha = ema[valid], rsi[valid], ha[valid]

    idxs, types, prices, points = detect_signals_sequential(ohlc, ema, rsi, ha)

    st.success(f"✅ Tổng tín hiệu: {len(idxs)}")
    st.info(f"🔼 BUY: {np.sum(types == 1)}")
    st.warning(f"🔽 SELL: {np.sum(types == 0)}")

    signal_df = pd.DataFrame({
        'Datetime': df.index[valid][idxs],
        'Type': np.where(types == 1, 'BUY', 'SELL'),
        'Price': prices,
        'Point': np.array(['Open', 'High', 'Low', 'Close'])[points]
    })

    st.dataframe(signal_df, use_container_width=True)