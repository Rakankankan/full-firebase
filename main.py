import streamlit as st
import requests
import time
import datetime
import pytz
import cv2
import torch
import asyncio
from telegram.ext import Application
from telegram import Bot
import logging
from PIL import Image
import numpy as np
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set page configuration
st.set_page_config(page_title="Dashboard Rokok", layout="centered")

# Konfigurasi logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Atur cache PyTorch
os.environ['TORCH_HOME'] = '/tmp/torch_hub'

# Zona waktu WIB (UTC+7)
WIB = pytz.timezone('Asia/Jakarta')

# Konfigurasi
TELEGRAM_TOKEN = "7941979379:AAEWGtlb87RYkvht8GzL8Ber29uosKo3e4s"
CHAT_ID = "5721363432"
FIREBASE_URL = "https://asap-99106-default-rtdb.asia-southeast1.firebasedatabase.app/sensor.json"
CAMERA_URL = "http://192.168.1.12:81/stream"
FIREBASE_SERVO_URL = "https://servo-control-f3c90-default-rtdb.asia-southeast1.firebasedatabase.app/servo.json"
GEMINI_API_KEY = "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"
GEMINI_MODEL = "google/gemini-2.0-flash-001"
NOTIFICATION_INTERVAL = 300  # 5 menit
ALERT_COOLDOWN = 60  # 1 menit

# Inisialisasi session state
if 'last_notify' not in st.session_state:
    st.session_state.last_notify = 0
if 'last_notification_time' not in st.session_state:
    st.session_state.last_notification_time = 0
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{"role": "system", "content": "Asisten deteksi merokok"}]
if 'last_servo_update' not in st.session_state:
    st.session_state.last_servo_update = 0
if 'prev_sudut' not in st.session_state:
    st.session_state.prev_sudut = 90
if 'model_cam' not in st.session_state:
    st.session_state.model_cam = None
if 'cam_running' not in st.session_state:
    st.session_state.cam_running = False

# Fungsi Telegram
async def send_telegram_message(message):
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
        logger.info("Pesan Telegram dikirim")
        return True
    except Exception as e:
        logger.error(f"Gagal mengirim pesan: {str(e)}")
        st.error(f"Gagal mengirim pesan: {str(e)}")
        return False

async def send_telegram_photo(photo, caption):
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption, parse_mode="Markdown")
        logger.info("Foto Telegram dikirim")
        return True
    except Exception as e:
        logger.error(f"Gagal mengirim foto: {str(e)}")
        st.error(f"Gagal mengirim foto: {str(e)}")
        return False

# Fungsi ambil data dari Firebase
def ambil_data():
    try:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        res = session.get(FIREBASE_URL, timeout=15)
        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"Gagal mengambil data: Status {res.status_code}")
            return None
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return None

# Fungsi kirim notifikasi Telegram
def kirim_telegram(data):
    if data is None:
        return
        
    timestamp = datetime.datetime.now(WIB).strftime("%H:%M:%S")
    level = data.get("asap", 0)
    status, emoji = ("Aman", "😊") if level < 1000 else ("Waspada", "😷") if level < 1200 else ("Bahaya", "🚨")

    pesan = f"""
🚭 *Notifikasi Deteksi Asap Rokok*  
🕒 *Waktu*: {timestamp} WIB  
💨 *Level Asap*: {level} {emoji}  
📊 *Status*: {status} {emoji}  

*🔍 Detail Sensor*:  
• 💨 *MQ2 (Asap Rokok)*: {data.get('mq2', 'N/A')}  
• 💨 *MQ135 (Asap Rokok)*: {data.get('mq135', 'N/A')}  
• 🌡️ *Suhu Lingkungan*: {data.get('suhu', 'N/A')} °C  
• 💧 *Kelembapan Udara*: {data.get('kelembapan', 'N/A')} %  

*📝 Catatan*:  
- *MQ2* mendeteksi asap rokok dan senyawa volatil.  
- *MQ135* mendeteksi asap rokok.  
- *Suhu & Kelembapan* memengaruhi distribusi asap.  
"""
    if level >= 1200:
        pesan += "\n🚨 *PERINGATAN KRITIS*: Tingkat asap sangat tinggi! Aktivitas merokok terdeteksi. Segera periksa lokasi! 🚭"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": pesan,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Gagal kirim Telegram: {e}")

# Load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
        return model
    except Exception as e:
        st.error(f"Gagal memuat YOLOv5: {str(e)}")
        return None

# Deteksi kamera
async def run_camera_detection(frame_placeholder, status_placeholder):
    try:
        cap = cv2.VideoCapture(CAMERA_URL)
        if not cap.isOpened():
            status_placeholder.error("Tidak dapat membuka kamera")
            return
            
        while st.session_state.cam_running:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.warning("Gagal membaca frame, mencoba lagi...")
                await asyncio.sleep(1)
                continue
                
            st.session_state.last_frame = frame.copy()
            
            if st.session_state.model_cam:
                results = st.session_state.model_cam(frame)
                rendered_frame = np.squeeze(results.render())
                df = results.pandas().xyxy[0]
                found_person = 'person' in df['name'].values
                found_smoke = 'smoke' in df['name'].values
            else:
                rendered_frame = frame
                found_person = found_smoke = False
                
            _, buffer = cv2.imencode('.jpg', rendered_frame)
            st.session_state.latest_frame = buffer.tobytes()
            
            if found_person and found_smoke:
                if time.time() - st.session_state.last_notify > ALERT_COOLDOWN:
                    caption = f"🚨 Peringatan: Merokok terdeteksi!\n🕒 Waktu: {datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')}"
                    await send_telegram_photo(st.session_state.latest_frame, caption)
                    st.session_state.last_notify = time.time()
                    status_placeholder.warning("Merokok terdeteksi!")
            else:
                status_placeholder.success("Tidak ada aktivitas merokok")
                
            frame_placeholder.image(rendered_frame, channels="BGR", use_container_width=True)
            await asyncio.sleep(0.1)
            
        cap.release()
    except Exception as e:
        status_placeholder.error(f"Error kamera: {str(e)}")
        if st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)

# Notifikasi periodik
async def send_periodic_notification(data):
    if data is None:
        return
        
    current_time = time.time()
    if current_time - st.session_state.last_notification_time >= NOTIFICATION_INTERVAL:
        logger.info("Mengirim notifikasi periodik...")
        caption = f"""
🚭 *Laporan Kondisi Ruangan*  
🕒 *Waktu*: {datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')}  
💨 *Asap Total*: {data.get('asap', 'N/A')} ({'Aman 😊' if data.get('asap', 0) < 1000 else 'Waspada 😷' if data.get('asap', 0) < 1200 else 'Bahaya 🚨'})  

*🔍 Detail Sensor*:  
• 💨 *MQ2 (Asap Rokok)*: {data.get('mq2', 'N/A')}  
• 💨 *MQ135 (Asap Rokok)*: {data.get('mq135', 'N/A')}  
• 🌡️ *Suhu*: {data.get('suhu', 'N/A')}°C  
• 💧 *Kelembapan*: {data.get('kelembapan', 'N/A')}%  
"""
        if st.session_state.latest_frame:
            await send_telegram_photo(st.session_state.latest_frame, caption)
        else:
            await send_telegram_message(caption + "\n⚠️ *Kamera tidak aktif*")
        st.session_state.last_notification_time = current_time
        logger.info("Notifikasi periodik dikirim")

# Gemini AI Chatbot
def get_gemini_response(messages):
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": GEMINI_MODEL, "messages": messages, "stream": False}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=10)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Error {response.status_code}"
    except requests.exceptions.RequestException:
        return "Error menghubungi AI"

def generate_chatbot_context(data):
    if data is None:
        return "Sistem sedang tidak dapat mengakses data sensor"
        
    return (
        f"Data sensor:\n"
        f"- Asap Total: {data.get('asap', 'N/A')}\n"
        f"- MQ2 (Asap Rokok): {data.get('mq2', 'N/A')}\n"
        f"- MQ135 (Asap Rokok): {data.get('mq135', 'N/A')}\n"
        f"- Suhu: {data.get('suhu', 'N/A')}°C\n"
        f"- Kelembapan: {data.get('kelembapan', 'N/A')}%\n"
        f"Waktu (WIB): {datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Jawab sebagai asisten deteksi merokok."
    )

# Async wrapper
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        return loop.create_task(coro)
    else:
        return loop.run_until_complete(coro)

# CSS Styling
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1e2a44;
            color: #e0e6f0;
        }
        .main-container {
            max-width: 1200px;
            margin: auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #3b82f6, #10b981);
            color: white;
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            font-size: 38px;
            font-weight: bold;
            margin-bottom: 30px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            animation: slideIn 0.5s ease-in;
        }
        .narasi {
            background-color: #2a3b5e;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 16px;
            color: #a3bffa;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .data-card {
            background-color: #334876;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px,'' 12px rgba(0,0,0,0.15);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .data-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }
        .data-label {
            font-size: 20px;
            font-weight: 600;
            color: #a3bffa;
        }
        .data-value {
            font-size: 28px;
            font-weight: bold;
            color: #60a5fa;
        }
        .status-badge {
            padding: 10px 16px;
            border-radius: 20px;
            font-size: 15px;
            font-weight: 500;
            display: inline-block;
            margin-top: 10px;
        }
        .status-danger { background-color: #f87171; color: #fef2f2; }
        .status-warning { background-color: #f59e0b; color: #fef2f2; }
        .status-success { background-color: #10b981; color: #f0fdf4; }
        .status-info { background-color: #3b82f6; color: #eff6ff; }
        .chat-container {
            max-height: 450px;
            overflow-y: auto;
            background-color: #2a3b5e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
        }
        .chat-message {
            padding: 12px 18px;
            border-radius: 10px;
            margin-bottom: 12px;
            max-width: 85%;
            animation: fadeInChat 0.3s ease-in;
        }
        .user-message {
            background-color: #60a5fa;
            color: white;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #e0e6f0;
            color: #1e2a44;
            margin-right: auto;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 500;
            transition: background-color 0.3s, transform 0.2s;
        }
        .stButton>button:hover {
            background-color: #2563eb;
            transform: scale(1.05);
        }
        .stSlider .st-bx {
            background-color: #3b82f6;
        }
        .stCheckbox label {
            color: #a3bffa;
        }
        .footer {
            text-align: center;
            color: #a3bffa;
            margin-top: 30px;
            font-size: 14px;
            opacity: 0.8;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 20px;
            margin: 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2a3b5e;
            border-bottom: 3px solid #3b82f6;
        }
        .stTabs [aria-selected="false"] {
            background-color: #1e2a44;
        }
        .tab-content {
            border-top: 1px solid #3b82f6;
            padding-top: 20px;
            margin-top: -1px;
        }
        .sensor-title {
            text-align: left;
            margin-bottom: 20px;
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        @keyframes fadeInChat {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .fade-in {
            animation: slideIn 0.5s ease-in;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="header">🚭 Dashboard Deteksi Asap & Lingkungan</div>', unsafe_allow_html=True)

    # Penjelasan Awal
    st.markdown("""
        <div class="narasi">
        ### ℹ️ Penjelasan Data Sensor:
        - **MQ2** mendeteksi asap dari rokok secara umum.
        - **MQ135** mendeteksi asap rokok.
        - **Suhu & Kelembapan** memengaruhi penyebaran asap.
        - **Asap Total (level)** adalah hasil integrasi data yang merepresentasikan potensi keberadaan rokok.
        
        **Status Deteksi:**
        - 😊 Aman: Level < 1000  
        - 😷 Waspada: 1000 ≤ Level < 1200  
        - 🚨 Bahaya: Level ≥ 1200
        </div>
    """, unsafe_allow_html=True)

    # Checkbox auto refresh
    auto_refresh = st.checkbox("🔄 Aktifkan Auto Refresh Data", value=True)

    # Tabs
    tab1, tab2 = st.tabs(["📊 Sensor IoT", "📸 Kamera ESP32"])

    # Tab Sensor IoT
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-title"><h3>🔍 Data Sensor Saat Ini</h3></div>', unsafe_allow_html=True)
        
        # Ambil dan tampilkan data
        data = ambil_data()
        if data:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="data-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="data-label">💨 MQ2 (Asap)</span>
                            <span class="data-value">{data.get('mq2', 'N/A')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="data-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="data-label">🌡️ Suhu (°C)</span>
                            <span class="data-value">{data.get('suhu', 'N/A')} °C</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="data-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="data-label">💨 MQ135 (Asap Rokok)</span>
                            <span class="data-value">{data.get('mq135', 'N/A')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="data-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="data-label">💧 Kelembapan (%)</span>
                            <span class="data-value">{data.get('kelembapan', 'N/A')} %</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Status Asap
            level = data.get("asap", 0)
            st.markdown("### 🧭 Status Deteksi Asap")
            status_class = "status-success" if level < 1000 else "status-warning" if level < 1200 else "status-danger"
            st.markdown(
                f"""
                <div class="data-card">
                    <div class="status-badge {status_class}">
                        Level Asap: {level} — Status: {'Aman 😊' if level < 1000 else 'Waspada 😷' if level < 1200 else 'Bahaya 🚨'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Grafik
            st.session_state.history.append({
                "asap": level,
                "suhu": data.get("suhu", 0),
                "kelembapan": data.get("kelembapan", 0)
            })

            st.markdown("### 📈 Grafik Tren Data")
            st.line_chart({
                "Asap": [x["asap"] for x in st.session_state.history],
                "Suhu": [x["suhu"] for x in st.session_state.history],
                "Kelembapan": [x["kelembapan"] for x in st.session_state.history]
            })

            # Kirim Telegram jika perlu
            now = time.time()
            if (now - st.session_state.last_notify > 300) or level >= 1200:
                kirim_telegram(data)
                st.session_state.last_notify = now

            # Notifikasi periodik
            run_async(send_periodic_notification(data))

        else:
            st.error("❌ Data tidak ditemukan dari Firebase.")

        # Chatbot AI
        st.markdown("### 💬 AI Chatbot")
        with st.form("chat_form", clear_on_submit=True):
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for msg in st.session_state.chat_messages[1:]:
                st.markdown(
                    f'<div class="chat-message {msg["role"]}-message">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            user_input = st.text_input("Tanya tentang kondisi ruangan...")
            if st.form_submit_button("Kirim"):
                st.session_state.chat_messages = [{
                    "role": "system",
                    "content": generate_chatbot_context(data)
                }]
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                with st.spinner("Menunggu AI..."):
                    response = get_gemini_response(st.session_state.chat_messages)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab Kamera ESP32
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("Deteksi Kamera Real-Time")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Mulai Deteksi", key="cam_start"):
                st.session_state.cam_running = True
                if st.session_state.model_cam is None:
                    st.session_state.model_cam = load_yolo_model()
                run_async(run_camera_detection(frame_placeholder, status_placeholder))
            else:
                st.session_state.cam_running = False
                if st.session_state.last_frame is not None:
                    frame_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
                    status_placeholder.info("Kamera dimatikan")

        with col2:
            st.checkbox("Auto-Refresh", value=True, key="cam_refresh")

        st.write("Kontrol Sudut Kamera")
        sudut = st.slider("Sudut Servo (0-180)", 0, 180, st.session_state.prev_sudut)
        current_time = time.time()
        if st.session_state.prev_sudut != sudut and current_time - st.session_state.last_servo_update > 1:
            try:
                response = requests.put(FIREBASE_SERVO_URL, json=sudut, timeout=5)
                if response.status_code == 200:
                    st.success(f"Servo diatur ke {sudut}°")
                    st.session_state.prev_sudut = sudut
                    st.session_state.last_servo_update = current_time
                else:
                    st.error(f"Gagal mengirim perintah ke Firebase: Status {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Terjadi kesalahan jaringan: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">Dibuat dengan ❤️ oleh Tim SIGMA BOYS</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
