import streamlit as st
import pandas as pd
import numpy as np
import io
import requests

# Import các hàm từ core.py
from core import load_data, compute_ema, compute_rsi, compute_ha, detect_signals_sequential

# ----------------------------
# Giao diện Streamlit
# ----------------------------
st.set_page_config(page_title="Phân tích tín hiệu giao dịch", layout="wide")
st.title("📈 Phân tích tín hiệu BUY / SELL")

# Chọn phương thức tải dữ liệu
method = st.radio("📁 Chọn cách tải dữ liệu:", ["Tải từ máy (Upload)", "Nhập đường dẫn (URL)", "Google Drive ID"])
uploaded_file = None
url = ""
df = None

if method == "Tải từ máy (Upload)":
    uploaded_file = st.file_uploader("📂 Chọn file CSV", type=["csv"])

elif method == "Nhập đường dẫn (URL)":
    url = st.text_input("🔗 Nhập URL tới file CSV:")

elif method == "Google Drive ID":
    drive_id = st.text_input("🔑 Nhập Google Drive file ID:")
    if drive_id:
        url = f"https://drive.google.com/uc?id={drive_id}"

# Nút xử lý tải file
if st.button("📥 Load File"):
    try:
        if method == "Tải từ máy (Upload)" and uploaded_file:
            df = load_data(uploaded_file)
        elif method in ["Nhập đường dẫn (URL)", "Google Drive ID"] and url:
            response = requests.get(url)
            df = load_data(io.StringIO(response.text))
        else:
            st.warning("⚠️ Hãy chọn và nhập đúng thông tin để tải file.")
    except Exception as e:
        st.error(f"❌ Lỗi khi tải hoặc xử lý file: {e}")

# Phân tích sau khi tải thành công
if df is not None:
    st.success("✅ File đã được tải thành công!")
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

    # Hiển thị bảng kết quả
    if len(idxs) > 0:
        df_result = pd.DataFrame({
            "Thời gian": df.index[valid][idxs],
            "Loại lệnh": ["BUY" if t == 1 else "SELL" for t in types],
            "Giá vào lệnh": prices,
            "Tại điểm": [ ["Open", "High", "Low", "Close"][p] for p in points ]
        })
        st.dataframe(df_result)
    else:
        st.info("ℹ️ Không có tín hiệu nào được phát hiện trong dữ liệu.")