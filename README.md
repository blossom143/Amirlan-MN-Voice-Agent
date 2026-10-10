# Khan Bank Speech-to-Speech Voicebot Agent 🤖🏦

An end-to-end, live speech-to-speech AI voicebot assistant built for **Khan Bank**, featuring real-time microphone capture, Google Cloud Speech-to-Text V2 (Chirp 3), a LangGraph ReAct reasoning engine powered by Gemini 2.5 Flash with official web data grounding (`khanbank.com`), and natural Mongolian voice synthesis via the Narakeet REST API.

---

## 🛠️ Tech Stack

- **Audio Capture & Handling:** `sounddevice`, `scipy`
- **Speech-to-Text (STT):** Google Cloud Speech-to-Text V2 (`chirp_3`, language: `mn-MN`, endpoint: `us-speech.googleapis.com`)
- **Reasoning & Agent Architecture:** LangGraph ReAct Agent, LangChain, Vertex AI (`gemini-2.5-flash`)
- **Tools & Grounding:** DuckDuckGo Search restricted to official sources (`khanbank.com`)
- **Text-to-Speech (TTS):** Narakeet REST API (Targeting high-quality `.m4a` output with the official Mongolian male voice `Ganbaatar`)
- **Environment:** Python 3.10+, Google Cloud ADC (`GOOGLE_CLOUD_PROJECT`)

---

## 🔄 Agent Workflow Architecture

The voicebot operates on a structured **6-stage pipeline** designed to handle user inquiries securely and efficiently:

```
[Stage 1: Call Intake] ──> [Stage 2: Intent Understanding] ──> [Stage 3: Information Retrieval]
                                                                        │
[Stage 6: Human Escalation] <── [Stage 4: Decision Making] <────────────┘
        │
        ▼
[Stage 5: Response Generation (TTS & Audio Playback)]
```

1. **Stage 1: Call Intake**
   - Captures live user audio from the microphone via `sounddevice` and transcribes it into text using Google Cloud Chirp 3 (`mn-MN`).
2. **Stage 2: Intent Understanding**
   - Analyzes the transcribed text using Gemini 2.5 Flash to classify the user's intent (e.g., loan inquiries, account services, or operator requests).
3. **Stage 3: Information Retrieval**
   - Performs targeted web searches via DuckDuckGo filtered explicitly for official data from `khanbank.com`.
4. **Stage 4: Decision Making**
   - Uses the LangGraph ReAct engine to evaluate whether the retrieved data directly answers the user prompt or requires validation.
5. **Stage 5: Response Generation**
   - Formulates a professional, polite Mongolian response and synthesizes it into streaming `.m4a` audio using Narakeet (`Ganbaatar`).
6. **Stage 6: Human Escalation**
   - Intercepts requests where the user explicitly asks for a human operator or where information is restricted, triggering a safe escalation protocol.

---

## 🚀 Getting Started & How to Run

### 1. Prerequisites & Environment Setup
Ensure you have Python installed and your Google Cloud credentials / API keys configured as environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export NARAKEET_API_KEY="your-narakeet-api-key"
```

### 2. Install Dependencies
Install the required audio and AI packages:

```bash
pip install sounddevice scipy requests google-cloud-speech langchain-google-vertexai langchain-community langgraph --upgrade
```

### 3. Run the Voicebot Loop
Execute your Python script containing the integrated orchestration function:

```python
from your_script_name import run_khanbank_voicebot

# Start the interactive speech-to-speech loop
run_khanbank_voicebot()
```

When prompted in your terminal or Jupyter Notebook:
1. Press **ENTER** to speak your question into the microphone.
2. Listen to the AI agent reason through Khan Bank data and reply naturally in Mongolian via synthesized voice.
3. Type **`q`** and hit **ENTER** to exit the loop.

#TO DO NEXT
1. Model visibility/explainability
2. Voice options (man/woman etc.)
3. New UI
4. Guard Rail for prompt injection
5. Escalation tool function
6. Memory refresh -> Clean up files when chat ends