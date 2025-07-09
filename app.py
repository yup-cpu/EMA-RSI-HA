import streamlit as st
import random

st.title("🎲 App kiểm tra tự động cập nhật")

quotes = [
    "Không có gì là không thể.",
    "Thành công không đến với người ngồi chờ.",
    "Hãy là chính mình, đừng sao chép người khác.",
    "Chỉ cần bạn không dừng lại, thì bạn vẫn đang tiến lên.",
    "Dám nghĩ, dám làm!"
]

if st.button("👉 Bấm để nhận một câu ngẫu nhiên"):
    st.success(random.choice(quotes))