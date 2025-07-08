# core.py – Xử lý dữ liệu và tín hiệu
import pandas as pd
import numpy as np
from numba import njit

def extract_drive_id(url_or_id: str):
    """
    Lọc ra Drive file ID từ cả ID hoặc link đầy đủ.
    """
    if "drive.google.com" in url_or_id:
        # Ví dụ: https://drive.google.com/file/d/1abcXYZ/view?usp=sharing
        parts = url_or_id.split("/")
        if "d" in parts:
            idx = parts.index("d")
            return parts[idx + 1]
        raise ValueError("❌ Không thể trích xuất ID từ đường dẫn Google Drive.")
    return url_or_id.strip()

def load_data(file_path):
    # Thử đọc với tiêu đề, nếu sai sẽ đọc không có tiêu đề
    try:
        df = pd.read_csv(file_path)
        if df.shape[1] != 7 or not all(col in df.columns for col in ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Không đúng định dạng tiêu đề, thử đọc lại với header=None")
    except:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] >= 7:
            df = df.iloc[:, :7]
            df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        else:
            raise ValueError(f"❌ File cần đủ 7 cột: ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], nhưng nhận được {df.shape[1]} cột.")

    # Gộp Date + Time thành Datetime
    try:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    except Exception as e:
        raise ValueError(f"❌ Lỗi khi xử lý Date + Time: {e}")

    # Ép kiểu số
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
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
        for j in range(4):  # O, H, L, C
            price = ohlc[i, j]
            if price > ema50[i] and rsi_lo < rsi[i] < rsi_hi and ha[i - 1] == 0 and ha[i] == 1:
                idxs[count] = i
                types[count] = 1  # BUY
                prices[count] = price
                points[count] = j
                count += 1
                break
            elif price < ema50[i] and rsi_lo < rsi[i] < rsi_hi and ha[i - 1] == 1 and ha[i] == 0:
                idxs[count] = i
                types[count] = 0  # SELL
                prices[count] = price
                points[count] = j
                count += 1
                break
    return idxs[:count], types[:count], prices[:count], points[:count]