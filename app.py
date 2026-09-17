import os
import json
import ssl
import torch
import whisper
import ollama
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# میانبر برای دور زدن خطای SSL
ssl._create_default_https_context = ssl._create_unverified_context

# تنظیمات اولیه صفحه
st.set_page_config(
    page_title="Hotel Farzam Berlin - AI Receptionist",
    page_icon="🏨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# عکس روز دروازه براندنبورگ از Unsplash
bg_image_url = "https://images.unsplash.com/photo-1599946347371-68eb71b16afc?q=80&w=2070&auto=format&fit=crop"

# --- استایل سفارشی CSS با تم روز و طلایی لوکس ---
st.markdown(f"""
<style>
    /* ۱. تصویر پس‌زمینه روز دروازه براندنبورگ */
    .stApp {{
        background: linear-gradient(rgba(15, 23, 42, 0.55), rgba(15, 23, 42, 0.65)), 
                    url("{bg_image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f8fafc;
        font-family: 'Georgia', 'Helvetica Neue', serif;
        font-style: italic;
    }}

    /* ۲. مخفی کردن عناصر پیش‌فرض Streamlit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ۳. عنوان هتل فرزام و زیرنویس */
    .main-header {{
        text-align: center;
        padding: 25px 0 10px 0;
    }}
    .main-header h1 {{
        font-size: 2.8rem;
        font-weight: 700;
        color: #fbbf24;
        letter-spacing: 2px;
        font-style: italic;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);
    }}
    .hotel-subtitle {{
        color: #fef08a;
        font-size: 1.2rem;
        margin-top: -5px;
        margin-bottom: 25px;
        font-style: italic;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.9);
    }}

    /* ۴. طلایی کردن نوشته‌های بخش انتخاب متد ورودی (Voice/Text) */
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] label p {{
        color: #fbbf24 !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        font-style: italic !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);
    }}

    /* ۵. طلایی کردن متن‌های راهنمای داخل کارت */
    .stCard p, .stCard label, .stCard div {{
        color: #fde047 !important;
        font-style: italic !important;
    }}

    /* ۶. مشکی کردن کامل متن تایپ‌شده داخل کادر Textarea */
    textarea, input, [data-baseweb="textarea"] textarea {{
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        font-style: normal !important;
        -webkit-text-fill-color: #000000 !important;
    }}

    /* ۷. استایل Placeholder (متن راهنمای قبل از تایپ) */
    textarea::placeholder {{
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        font-style: italic !important;
    }}

    /* ۸. طراحی کارت‌های شیشه‌ای */
    .stCard {{
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(251, 191, 36, 0.5);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
        margin-bottom: 20px;
    }}

    /* ۹. استایل دکمه اصلی */
    div.stButton > button:first-child {{
        background: linear-gradient(90deg, #d97706 0%, #b45309 100%);
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none;
        padding: 14px 28px;
        font-size: 1.2rem;
        font-weight: 600;
        font-style: italic;
        border-radius: 12px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(217, 119, 6, 0.4);
        transition: all 0.3s ease;
    }}
    div.stButton > button:first-child:hover {{
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        transform: translateY(-2px);
    }}

    /* ۱۰. عناوین کلی */
    h1, h2, h3, h4 {{
        color: #fbbf24 !important;
        font-style: italic !important;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: bold;
        color: #fbbf24;
        font-style: italic;
    }}
</style>
""", unsafe_allow_html=True)

# ۱. بارگذاری مدل تصحیح املا (Cached)
@st.cache_resource
def load_spelling_model():
    model_name = "oliverguhr/spelling-correction-german-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ۲. بارگذاری مدل صوتی Whisper (Cached)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

tokenizer, spelling_model = load_spelling_model()
whisper_model = load_whisper_model()

def clean_german_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = spelling_model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_hotel_request(clean_text):
    prompt = f"""
    Du bist ein KI-Rezeptionist im Hotel Farzam in Berlin. 
    Analysiere die folgende Gastanfrage und erstelle NUR ein gültiges JSON-Objekt.

    Anfrage des Gastes: "{clean_text}"

    Format:
    {{
      "department": "Housekeeping" | "Maintenance" | "Front Desk",
      "room_number": "Zimmernummer als String oder null",
      "issue_summary": "Kurze Zusammenfassung auf Deutsch"
    }}
    """
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        format="json"
    )
    return json.loads(response['message']['content'])

# --- هدر کیوسک Hotel Farzam ---
st.markdown("""
    <div class="main-header">
        <h1>🏨 HOTEL FARZAM BERLIN</h1>
        <p class="hotel-subtitle">Willkommen! Wie können wir Ihnen heute helfen?</p>
    </div>
""", unsafe_allow_html=True)

# --- بخش ورودی ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
input_mode = st.radio("Wählen Sie Ihre Eingabemethode / Choose Input Method:", ("🎤 Sprache (Voice)", "⌨️ Text"), horizontal=True)

user_raw_text = ""

if input_mode == "🎤 Sprache (Voice)":
    st.write("Bitte drücken Sie auf das Mikrofon und sprechen Sie Ihre Anfrage ein:")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#ef4444",
        neutral_color="#fbbf24",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        with st.spinner("Transkribiere Audio (Whisper)..."):
            whisper_result = whisper_model.transcribe("temp_audio.wav", language="de")
            user_raw_text = whisper_result["text"]

        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

        st.info(f"**Erkannter Text (Raw Speech):** {user_raw_text}")

else:
    user_raw_text = st.text_area("Ihre Nachricht / Your Request:", height=100, placeholder="z. B. Die Heizung im Zimmer 204 funktioniert nicht...")

st.markdown('</div>', unsafe_allow_html=True)

# --- پردازش و نمایش خروجی ---
if st.button("Anfrage verarbeiten / Process Request"):
    if not user_raw_text.strip():
        st.warning("Bitte sprechen Sie eine Nachricht ein oder geben Sie Text ein.")
    else:
        with st.spinner("Verarbeite Anfrage (NLP Pipeline)..."):
            cleaned_text = clean_german_text(user_raw_text)
            result = process_hotel_request(cleaned_text)

        st.success("Vielen Dank! Ihre Anfrage wurde erfolgreich an das Hotel-Team weitergeleitet.")

        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📝 Korrigierter Text")
            st.info(cleaned_text)
        with col2:
            st.markdown("### 🏷️ Routing Details")
            st.metric(label="Zuständige Abteilung", value=result.get("department", "N/A"))
            if result.get("room_number"):
                st.caption(f"📍 Zimmernummer: **{result.get('room_number')}**")

        with st.expander("🔍 System Details (JSON Payload for Hotel API)"):
            st.json(result)
        st.markdown('</div>', unsafe_allow_html=True)
