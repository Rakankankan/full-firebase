import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import datetime
import time
import requests
import openai

# === Konfigurasi rahasia ===
firebase_url = "https://asap-99106-default-rtdb.asia-southeast1.firebasedatabase.app/"
telegram_bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
telegram_chat_id = st.secrets["TELEGRAM_CHAT_ID"]
openai.api_key = st.secrets["OPENROUTER_API_KEY"]

# === Inisialisasi Firebase ===
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': firebase_url
    })

ref = db.reference('asap')

# === Load data dari Firebase ===
data = ref.get()

st.title("🚬 Dashboard Deteksi Asap Rokok")
st.write("Data real-time dari sensor MQ2 & MQ135 melalui Firebase")

if data:
    df = pd.DataFrame.from_dict(data, orient='index')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp')
    st.line_chart(df.set_index('timestamp')['asap'])

    latest = df.iloc[-1]
    st.metric("📊 Nilai Asap Terakhir", value=int(latest['asap']), delta=None)

    # === Kirim ke Telegram setiap 5 menit ===
    if "last_sent" not in st.session_state:
        st.session_state.last_sent = datetime.datetime.min

    now = datetime.datetime.now()
    if (now - st.session_state.last_sent).total_seconds() >= 300:
        msg = f"🚨 Update Deteksi Asap 🚨\nNilai Asap: {int(latest['asap'])}\nWaktu: {latest['timestamp']}"
        telegram_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
        requests.post(telegram_url, data={"chat_id": telegram_chat_id, "text": msg})
        st.session_state.last_sent = now
        st.success("📤 Laporan berhasil dikirim ke Telegram.")

    # === Analisis AI dari OpenRouter: Gemini Flash 1.5 ===
    st.subheader("🧠 Analisis AI (Gemini Flash 1.5)")
    prompt = f"Analisis tren dari data asap berikut ini:\n{df['asap'].tolist()}\nBerikan ringkasan kondisi dan potensi bahaya."

    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "HTTP-Referer": "https://rakan-streamlit.cloud",  # nama domain/streamlit kamu
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-flash-1.5",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        output = response.json()['choices'][0]['message']['content']
        st.write(output)
    else:
        st.error("Gagal memanggil AI dari OpenRouter.")
else:
    st.warning("Belum ada data dari Firebase.")
