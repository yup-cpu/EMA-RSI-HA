import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import re

# Import từ core.py
from core import load_data, compute_ema, compute_rsi, compute_ha, detect_signals_sequential

# ----------------------------
# Giao diện Streamlit
# ----------------------------
st.set_page_config(page_title="Phân tích tín hiệu giao dịch", layout="wide")
st.title("📈 Phân tích tín hiệu BUY / SELL")

# Chọn cách tải dữ liệu
method = st.radio("📁 Chọn cách tải dữ liệu:", ["Tải từ máy (Upload)", "Nhập đường dẫn (URL)", "Google Drive ID"])
uploaded_file = None
url = ""
df = None

# Giao diện nhập dữ liệu
if method == "Tải từ máy (Upload)":
    uploaded_file = st.file_uploader("📂 Chọn file CSV", type=["csv"])

elif method == "Nhập đường dẫn (URL)":
    url = st.text_input("🔗 Nhập URL tới file CSV:")

elif method == "Google Drive ID":
    raw_input = st.text_input("🔑 Nhập Google Drive file ID hoặc liên kết:")
    # Tự động tách ID nếu dán link
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', raw_input)
    if match:
        drive_id = match.group(1)
    else:
        drive_id = raw_input.strip()
    if drive_id:
        url = f"https://drive.google.com/uc?id={drive_id}"

# Nút tải file
if st.button("📥 Load File"):
    try:
        if method == "Tải từ máy (Upload)" and uploaded_file:
            df = load_data(uploaded_file)

        elif method in ["Nhập đường dẫn (URL)", "Google Drive ID"] and url:
            response = requests.get(url)
            response.raise_for_status()  # Kiểm tra lỗi HTTP
            df = load_data(io.StringIO(response.text))

        else:
            st.warning("⚠️ Vui lòng cung cấp file hoặc đường dẫn hợp lệ.")

    except Exception as e:
        st.error(f"❌ Lỗi khi tải hoặc xử lý dữ liệu:\n\n{e}")

# Nếu có dữ liệu → xử lý tín hiệu
if df is not None:
    st.success("✅ File đã được tải thành công!")
    st.dataframe(df.head())

    with st.spinner("⏳ Đang phân tích tín hiệu..."):
        open_, high, low, close = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values

        ema = compute_ema(close, 50)
        rsi = compute_rsi(close, 14)
        ha = compute_ha(open_, high, low, close)

        # Chỉ dùng phần dữ liệu có đủ thông tin
        valid = ~np.isnan(rsi)
        ohlc = np.stack([open_[valid], high[valid], low[valid], close[valid]], axis=1)
        ema, rsi, ha = ema[valid], rsi[valid], ha[valid]

        idxs, types, prices, points = detect_signals_sequential(ohlc, ema, rsi, ha)

    # Thống kê tín hiệu
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Tổng tín hiệu", len(idxs))
    col2.metric("🔼 BUY", int(np.sum(types == 1)))
    col3.metric("🔽 SELL", int(np.sum(types == 0)))

    # Bảng kết quả
    if len(idxs) > 0:
        valid_index = df.index[valid]
        if np.any(idxs >= len(valid_index)):
            st.error("❌ Có chỉ số tín hiệu vượt quá số dòng hợp lệ. Vui lòng kiểm tra dữ liệu đầu vào.")
        else:
            df_result = pd.DataFrame({
                "Thời gian": valid_index[idxs],
                "Loại lệnh": ["BUY" if t == 1 else "SELL" for t in types],
                "Giá vào lệnh": prices,
                "Tại điểm": [ ["Open", "High", "Low", "Close"][p] for p in points ]
            })
            st.dataframe(df_result)
    else:
        st.info("ℹ️ Không có tín hiệu nào được phát hiện trong dữ liệu.")