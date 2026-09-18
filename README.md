# 🏨 Hotel Farzam Berlin - AI Receptionist Kiosk

An offline-first, GDPR-compliant AI Receptionist Kiosk for **Hotel Farzam Berlin**. The system captures guest audio complaints/requests in German, applies seq2seq grammar and spelling correction to the transcribed speech, and routes structured JSON outputs to the respective hotel departments using a local LLM.

---

## 🌟 Key Features

* **Voice-to-Text Processing:** Uses OpenAI Whisper for accurate speech recognition.
* **German Grammar & Spelling Correction:** Corrects typos, dialectal artifacts, and speech recognition errors using `oliverguhr/spelling-correction-german-base`.
* **Departmental JSON Routing:** Extracts structured JSON metadata (`department`, `room_number`, `urgency`, `issue_summary`) via Ollama (`llama3`).
* **Offline-First & GDPR Compliant:** All models run locally on-device without third-party API exposure.
* **Custom Glassmorphism Kiosk UI:** High-contrast, gold-accented UI customized for kiosk display.

---

## 🛠️ Tech Stack

* **UI Framework:** Streamlit
* **Speech Recognition:** OpenAI Whisper
* **NLP & Grammar Correction:** HuggingFace Transformers (`oliverguhr/spelling-correction-german-base`)
* **LLM & Structuring:** Ollama (`llama3`)
* **Language:** Python 3.10+

---

## 🚀 Getting Started

### Prerequisites

1. Install **Ollama** and pull the Llama 3 model:
   ```bash
   ollama pull llama3
cat << 'EOF' > requirements.txt
streamlit>=1.30.0
torch>=2.0.0
transformers>=4.35.0
openai-whisper
ollama
pydantic>=2.0
sounddevice
numpy
scipy
