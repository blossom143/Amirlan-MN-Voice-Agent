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

### 🌐 APIs & Technical Specifications by Workflow Stage

The voicebot integrates distinct cloud and on-premise APIs across each stage:

| Stage | Name | Service / Provider | Model / Protocol | API Endpoint / SDK | Primary Role & Active Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | **Call Intake** | Google Cloud Speech-to-Text V2 | `chirp_3` (`mn-MN`) | `us-speech.googleapis.com`<br>`google.cloud.speech_v2` | Microphone capture, Mongolian STT, Pre-LLM PII regex masking (Credit Cards, PIN, RD, Account #) |
| **Stage 2** | **Intent Understanding** | Google Cloud Vertex AI | `gemini-2.5-flash` | Vertex AI Chat Endpoint<br>`langchain_google_vertexai` | 7-category taxonomy classification, Confidence scoring, Pre-LLM Prompt Injection & Abuse filters |
| **Stage 3** | **Information Retrieval** | DuckDuckGo & Khan Bank Portal | REST / Web Search | `DuckDuckGoSearchRun`<br>`khanbank.com` | Official domain web grounding (`site:khanbank.com`), Deep-link specific page URI resolution |
| **Stage 4** | **Decision Making** | LangGraph ReAct Engine | Gemini 2.5 Flash ReAct | LangGraph StateGraph<br>`SystemMessage` quarantine | ReAct reasoning, Strict Anti-Hallucination verification, Financial Transaction compliance checks |
| **Stage 5** | **Response Generation** | Narakeet Text-to-Speech | `Ankhmaa` (Default) / `Ganbaatar` | `POST api.narakeet.com/text-to-speech/m4a`<br>`requests.Session` TCP pool | Mongolian voice synthesis, Streaming `.m4a` chunk delivery, SHA-256 disk audio memoization |
| **Stage 6** | **Human Escalation** | Khan Bank Contact Center | VIP Priority Telephony | Hotline `1800-1917`<br>Intl `+976 7515-1917` | Circuit breaker fail-safe (2x unclear audio, confidence < 0.70, session turn limit > 25) |

---

### 📡 Stage-by-Stage API Architecture

#### 1. Stage 1: Call Intake (Speech-to-Text & Hardware Capture)
* **Cloud API:** Google Cloud Speech-to-Text V2 API
* **Endpoint:** `https://us-speech.googleapis.com/v2/projects/{project_id}/locations/us/recognizers/_:recognize`
* **Model:** `chirp_3` (Google's 2B parameter Foundation Model optimized for low-resource languages including Mongolian)
* **Language Code:** `mn-MN`
* **Internal Web Routes:**
  * `POST /api/voice/process` (Multipart `multipart/form-data` audio file upload from browser)
  * `POST /api/voice/record-desktop` (Local hardware microphone recording via `sounddevice` & `scipy.io.wavfile`)
* **Active Guardrail:** Immediate Pre-LLM PII Masking:
  * Credit Cards: `XXXX-XXXX-XXXX-1234`
  * PIN Codes: `[PIN ХАМГААЛАГДСАН]`
  * Register Numbers: `[РД ДАЛДАЛСАН]`
  * 10-digit Account Numbers: `XXXXXX1234`

#### 2. Stage 2: Intent Understanding & Classification
* **Cloud API:** Google Cloud Vertex AI / Gemini API
* **SDK:** `langchain_google_vertexai.ChatVertexAI`
* **Model:** `gemini-2.5-flash`
* **Internal Web Route:** `POST /api/text/process` (`{"text": "...", "voice": "Ankhmaa"}`)
* **Output Schema (JSON):**
  ```json
  {
    "category": "loan_inquiry | account_service | card_service | branch_atm | exchange_rate | human_escalation | general_inquiry",
    "topic": "String (Mongolian summary)",
    "confidence": 0.0 - 1.0,
    "is_escalation_needed": false,
    "search_keywords": "String (targeted keywords)"
  }
  ```
* **Active Guardrails:**
  * **Pre-LLM Prompt Injection Interception:** Regex filter intercepts jailbreak patterns (`ignore previous instructions`, `system prompt`, `дүрмээ март`, etc.) in 0.0s before invoking Gemini.
  * **Profanity & Abuse Filter:** Keyword and tone filter immediately escalates hostile speech to human agents.

#### 3. Stage 3: Information Retrieval & Source Grounding
* **Search API:** DuckDuckGo Search API (`duckduckgo-search` library via `DuckDuckGoSearchRun`)
* **Search Query:** `site:khanbank.com <keywords>`
* **Page-Level URI Resolver (`resolve_khanbank_page_uri`):** Resolves the exact page accessed on `khanbank.com` and displays it in the Agent Reasoning panel:
  * **Mortgage Loans:** `https://www.khanbank.com/mn/personal/loans/mortgage-loan`
  * **Salary Loans:** `https://www.khanbank.com/mn/personal/loans/salary-loan`
  * **Consumer Loans:** `https://www.khanbank.com/mn/personal/loans/consumer-loan`
  * **General Loans:** `https://www.khanbank.com/mn/personal/loans`
  * **Payment & Credit Cards:** `https://www.khanbank.com/mn/personal/cards`
  * **Savings & Deposits:** `https://www.khanbank.com/mn/personal/deposits`
  * **Digital Banking / App:** `https://www.khanbank.com/mn/personal/digital-banking`
  * **Branches, ATMs & Express Centers:** `https://www.khanbank.com/mn/locations`
  * **Foreign Exchange Rates:** `https://www.khanbank.com/mn/personal/rates`
  * **Tariffs & Fees:** `https://www.khanbank.com/mn/personal/tariffs`
  * **Contact & Call Center:** `https://www.khanbank.com/mn/personal/contact`

#### 4. Stage 4: Decision Making (ReAct & Policy Engine)
* **Agent Engine:** LangGraph ReAct decision cycle with Gemini 2.5 Flash
* **Prompt Protection Protocol:** Non-injectable role-separated prompting using `SystemMessage(content=immutable_persona)` and quarantined `HumanMessage(content=bank_data + user_query)`.
* **Decision Contract:** JSON (`{"decision": "answer" | "escalate", "reason": "..."}`)
* **Active Guardrails:**
  * **Strict Source Grounding (Anti-Hallucination):** If retrieved context from `khanbank.com` is missing or ungrounded, the engine rejects guessing and triggers Stage 6 operator fallback.
  * **Financial Compliance Boundary:** Blocks autonomous money transfers, fund withdrawals, account closures, and loan approvals, deflecting callers to the Khan Bank App or 1800-1917.
  * **Topic Steering:** Automatically redirects off-topic questions (politics, coding, entertainment) back to banking services.

#### 5. Stage 5: Response Generation & Streaming Voice Synthesis
* **TTS API:** Narakeet Text-to-Speech Streaming REST API
* **Endpoint:** `POST https://api.narakeet.com/text-to-speech/m4a?voice={voice}&speed=1.0`
* **Headers:** `x-api-key: {NARAKEET_API_KEY}`, `Content-Type: text/plain`
* **Voice Options:**
  * `Ankhmaa` (Default, expressive Mongolian female voice)
  * `Ganbaatar` (Official Mongolian male voice)
* **Streaming Audio Route:** `GET /audio/<filename>` (Streams `.m4a` files with HTTP 206 Partial Content & `audio/mp4` MIME type)
* **Optimization & Caching:**
  * **TCP Connection Pooling:** Persistent `requests.Session` eliminates repeated TLS handshakes (~180ms latency saving).
  * **Byte-Clamping:** Input text is clamped to 950 bytes to comply with synchronous streaming limits.
  * **SHA-256 Audio Memoization:** Responses are hashed and cached on disk (`output-audio/amirlan_cache_*`); repeated requests return in 0.00s.

#### 6. Stage 6: Human Escalation Protocol
* **Department:** Khan Bank Customer Service Center (Харилцагчийн үйлчилгээний төв)
* **Hotlines:** Domestic `1800-1917` (24/7 Toll-free) | International `+976 7515-1917`
* **Circuit Breakers:**
  * **Confidence Trigger:** Intent classification confidence score < 0.70 automatically routes to an operator.
  * **Consecutive Unclear Audio:** Two consecutive inaudible or empty speech inputs immediately trip the circuit breaker.
  * **Session Timeout / Turn Limit:** Exceeding 25 turns in a single session forces an automatic operator handoff.
* **Internal Session Cleanup API:** `POST /api/chat/clear` (Purges input recordings, resets circuit breaker counters).

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
1. Data streaming, data layer outside local files?
2. Improve quick prompt suggestions
3. Escalation tool function -> live voice recording


AI Layer
•	Speech-to-Text model
•	LLM Orchestration
•	Response Generation
•	Text-to-Speech model
•	Speech to speech model
•	STT/TTS & S2S model-уудыг хосуулах эсвэл аль нэгийг сонгож болно.
Enterprise Integration Layer
•	Knowledge Base (khanbank.com ашиглах)
•	Backend APIs
Monitoring Layer
•	Logs
•	Metrics
•	Evaluation
