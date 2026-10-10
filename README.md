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

#### 3. Stage 3: Information Retrieval & Source Grounding (Local FAISS RAG)
* **Retrieval Engine:** Localized FAISS vector store (`LocalKnowledgeRetriever` in `app/vector_store.py`) backed by `app/knowledge_base.txt`.
* **Latency:** Sub-millisecond execution (< 1ms vs. 2000-3000ms DuckDuckGo web search).
* **Grounding Document:** Direct official product embedding covering Mortgages (6%), Salary Loans, Term & Demand Deposits, Payment Cards (Classic, Signature, Platinum, Lost Card Lock), Digital Banking (Khan Bank App), and Branch/ATM networks.
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

#### 4. Stage 4: Decision Making (Streamlined ReAct Engine)
* **Agent Engine:** Streamlined single-pass LangGraph ReAct decision cycle with Gemini 2.5 Flash (`temperature=0.0`, `max_output_tokens=140`, `recursion_limit=2`).
* **Prompt Protection Protocol:** Non-injectable role-separated prompting using `SystemMessage(content=immutable_persona)` and quarantined `HumanMessage(content=bank_data + user_query)`.
* **Decision Contract:** JSON (`{"decision": "answer" | "escalate", "reason": "...", "response_text": "..."}`)
* **Active Guardrails:**
  * **Strict Source Grounding (Anti-Hallucination):** If retrieved context from local knowledge is missing or ungrounded, the engine rejects guessing and triggers Stage 6 operator fallback.
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
  * **Sentence-Level Chunking & Byte Clamping:** Clamps response text to ≤250 bytes (18-24 words, 1-2 spoken sentences) to keep Narakeet network stream transfer < 1.2s.
  * **Frequent Phrase Memoization:** Dedicated audio caching for standard greetings, closures, unclear audio, and compliance disclosures (0.00s latency).
  * **Multi-Level Response Memoization:** Full-turn query caching delivering instantaneous responses (< 0.02s) on repeated queries.
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

### 1. Local Python Setup

#### Prerequisites
Configure your Google Cloud credentials and API keys in your environment or `.env` file:

```bash
export GOOGLE_CLOUD_PROJECT="project-1d76ba00-25b9-47dd-8d3"
export GCP_LOCATION="us-central1"
export NARAKEET_API_KEY="your-narakeet-api-key"
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run Web Application
```bash
python run.py
```
Access the application at `http://127.0.0.1:5005`.

---

## 🐳 Docker Build & Deployment

The application includes a production-grade [Dockerfile](file:///Users/tsomorlig/Documents/mn-voicebot-agent/Dockerfile) and [.dockerignore](file:///Users/tsomorlig/Documents/mn-voicebot-agent/.dockerignore) engineered specifically for containerized and cloud environments.

### 🌟 Key Container Highlights
* **Low-Level Linux Audio Stack:** Installs `libportaudio2`, `portaudio19-dev`, `libasound2`, `libasound2-plugins`, `alsa-utils`, `libpulse0`, `libsndfile1`, and `ffmpeg` to prevent runtime wheel compilation failures.
* **Headless Dummy ALSA Sink:** Configures `/etc/asound.conf` with a null PCM device (`pcm.!default { type plug; slave.pcm "null"; }`), allowing `sounddevice` and PortAudio to initialize without physical sound cards (`/dev/snd`).
* **Production Gunicorn Server:** Uses Gunicorn with `gthread` workers, 4 threads, and an extended `--timeout 120` to accommodate multi-stage LLM reasoning and real-time TTS synthesis.
* **Dynamic Port Binding:** Binds automatically to Render's dynamic `${PORT:-5005}`.

---

### Option A: Local Docker Build & Run

#### 1. Build the Docker Image
```bash
docker build -t mn-voicebot:latest .
```

#### 2. Run the Container Locally
Run the container in detached mode, passing your local `.env` and mounting your Google Cloud Application Default Credentials (ADC):

```bash
docker run -d \
  --name mn-voicebot-local \
  -p 5005:5005 \
  --env-file .env \
  -v ~/.config/gcloud/application_default_credentials.json:/tmp/gcp_credentials.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_credentials.json \
  mn-voicebot:latest
```

> **Note:** If you have a dedicated Service Account key file, replace the `-v` mount path with the path to your `service_account.json`.

#### 3. Verify Health & Pipeline
Check that the container and audio drivers are running:
```bash
# Check container status
curl http://localhost:5005/api/status

# Test full 6-stage pipeline (LLM + Grounding + TTS)
curl -X POST http://localhost:5005/api/text/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Хаан банкны хадгаламжийн хүү хэд вэ?", "voice": "Ankhmaa"}'
```
Open [http://localhost:5005](http://localhost:5005) in your web browser.

#### 4. Managing the Local Container
```bash
# View live logs
docker logs -f mn-voicebot-local

# Stop the container
docker stop mn-voicebot-local

# Start the container
docker start mn-voicebot-local

# Remove the container
docker rm -f mn-voicebot-local
```

---

### Option B: Cloud Deployment to Render.com

1. **Push your code to GitHub:**
   ```bash
   git add Dockerfile .dockerignore requirements.txt app/voicebot.py
   git commit -m "feat: Add Docker deployment support"
   git push origin main
   ```

2. **Create a new Web Service on Render:**
   - Log into [Render Dashboard](https://dashboard.render.com).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Connect your GitHub repository (`mn-voicebot-agent`).
   - Select **Docker** as the Environment.

3. **Configure Environment Variables:**
   Under the **Environment** tab, set:
   | Variable | Value / Description |
   | :--- | :--- |
   | `NARAKEET_API_KEY` | Your Narakeet API key |
   | `GOOGLE_CLOUD_PROJECT` | Your GCP Project ID (e.g. `project-1d76ba00-25b9-47dd-8d3`) |
   | `GCP_LOCATION` | `us-central1` |
   | `GOOGLE_CREDENTIALS_JSON` | **Raw JSON string** of your Google Cloud Service Account key |

   *(The container automatically detects `GOOGLE_CREDENTIALS_JSON` and creates `/tmp/gcp_credentials.json` upon startup).*

4. **Health Check Path:**
   - Set the Health Check Path to `/api/status`.

5. **Deploy:**
   - Click **Deploy Web Service**. Render builds the image, starts Gunicorn, and serves the app over HTTPS.

---

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
