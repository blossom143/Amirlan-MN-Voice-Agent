# System Architecture: Khan Bank Mongolian Voicebot Agent ("Amirlan")

## 1. High-Level Architecture Overview

The **Khan Bank Mongolian Voicebot Agent** is an enterprise-grade, speech-to-speech AI customer service agent tailored for the Mongolian banking domain. The system combines high-fidelity Mongolian speech recognition, hybrid semantic retrieval, agentic reasoning, safety guardrails, and real-time dual-engine speech synthesis.

```mermaid
graph TD
    subgraph Client ["Client Tier (Browser Web Audio)"]
        Mic["Microphone Input"] --> VAD["Client-side VAD (Web Audio Analyser)"]
        VAD -->|"RMS > 0.026 (Speech Start)"| REC["MediaRecorder (WebM/Opus)"]
        REC -->|"Silence > 1350ms"| DISPATCH["Dispatch Turn (Fetch POST)"]
        AUDIO_PLAY["Audio Player (HTML5 Audio)"] <-->|"Barge-in Interruption"| VAD
    end

    subgraph Server ["Server Tier (Flask / Gunicorn App)"]
        API["Flask HTTP Engine (/api/voice/process)"]
        WARMUP["Engine Pre-Warmer (/api/warmup)"]
        GUARD["Multi-Layer Guardrail Engine"]
    end

    subgraph Pipeline ["Agentic Voice Pipeline (app/voicebot.py)"]
        S1["Stage 1: Speech-to-Text (Chirp 3 mn-MN)"]
        S2["Stage 2: Intent Classification (Fast Pattern + LLM Fallback)"]
        S3["Stage 3: Hybrid Retrieval (FAISS + BM25 + Glossary)"]
        S4["Stage 4: ReAct Decision (Gemini 2.5 Flash)"]
        S5["Stage 5: Response Generation & Dual TTS"]
        S6["Stage 6: Human Escalation Protocol"]
    end

    subgraph Cloud ["External AI Cloud Services"]
        GCP_STT["Google Cloud Speech-to-Text v2 (Chirp 3)"]
        VERTEX_LLM["Google Cloud Vertex AI (Gemini 2.5 Flash)"]
        GEMINI_TTS["Google GenAI Live TTS (gemini-3.1-flash-tts-preview)"]
        NARAKEET_TTS["Narakeet REST API (mn-MN Neural Voices)"]
    end

    DISPATCH --> API
    API --> GUARD
    GUARD --> S1
    S1 <--> GCP_STT
    S1 --> S2
    S2 -.->|"Fallback"| VERTEX_LLM
    S2 --> S3
    S3 --> S4
    S4 <--> VERTEX_LLM
    S4 --> S5
    S4 -.->|"Escalate"| S6
    S5 <--> GEMINI_TTS
    S5 <--> NARAKEET_TTS
    S5 --> API
    API --> AUDIO_PLAY
```

---

## 2. Component Architecture

### 2.1 Frontend & Audio Processing Tier
- **Web Audio Context & AnalyserNode**: Samples live microphone audio at 60 FPS using an FFT size of 256. Computes root-mean-square (RMS) energy to continuously classify user speech vs. background silence.
- **Client-Side Voice Activity Detection (VAD)**:
  - **RMS Speech Threshold**: `0.026`
  - **Speech Start Confirmation**: $\ge 3$ consecutive frames (> 50ms)
  - **Silence Timeout**: `1350ms`
  - **Minimum Valid Speech Duration**: `500ms` (discards false clicks/taps)
- **Barge-in Controller**: Monitors user speech while the audio element is actively playing. If user energy exceeds the speech threshold for 3 consecutive frames, assistant playback is immediately paused, reset to `0s`, and the recorder arms for an incoming query.
- **Visual Intelligence Stepper & 3D WebGL Orb**: Renders real-time pipeline status (Listening, Analyzing, Speaking, Interrupted, Escalated) and per-stage latency telemetry.

### 2.2 Server & Ingestion Tier
- **Application Engine**: Python 3.11 with Flask, served via Gunicorn in production (`Dockerfile`).
- **Engine Pre-Warmer (`/api/warmup`)**: Automatically triggered on `DOMContentLoaded` to initialize GCP SpeechClient, Vertex AI credentials, GenAI client, and load FAISS vector indices into memory prior to the user's first spoken turn.
- **Audio Lifecycle Management**:
  - `input-audio/`: Preserves incoming user voice queries with timestamped session identifiers.
  - `output-audio/`: Stores synthesized assistant responses for playback and caching.
  - Automatic cleanup on chat reset or browser unload.

### 2.3 Agentic Pipeline Tier (`app/voicebot.py`)

| Stage | Name | Technology | Primary Function | SLA Target |
|---|---|---|---|---|
| **1** | Call Intake | Google Cloud Chirp 3 (`mn-MN`) | Conformer-based speech-to-text with PII masking | < 2.0s |
| **2** | Intent Understanding | Fast Regex Classifier + Gemini 2.5 Flash | Classifies banking category, topic, confidence, and search keywords | < 0.05s (Fast) / ~1.5s (LLM) |
| **3** | Information Retrieval | FAISS + BM25 + Reciprocal Rank Fusion | Local hybrid retrieval over verified Khan Bank knowledge documents | < 0.01s |
| **4** | Decision Making | Single-Pass ReAct Engine (Gemini 2.5 Flash) | Reasoning with strict recursion limit and banking boundaries | ~1.2s |
| **5** | Response & TTS | Dual TTS: Gemini 3.1 Flash Preview / Narakeet | Streaming 24kHz linear PCM or high-fidelity neural M4A | ~1.5s (Stream) / ~2.5s (Batch) |
| **6** | Human Escalation | Protocol Engine | Routes complex transactions or unclear calls to 1800-1917 | Immediate |

---

## 3. Knowledge Base & Hybrid Retrieval Architecture

The RAG subsystem (`app/vector_store.py`) indexes official Khan Bank product guidelines, fee structures, interest rates, and loan terms across 5 core knowledge domains:

```mermaid
graph LR
    DOCS["Markdown Knowledge Base (app/knowledge_base/)"] --> CHUNKER["Semantic Section Chunker"]
    CHUNKER --> INDEX["Indexed Chunks (Doc Title + Section + Content)"]
    
    QUERY["User Spoken Query"] --> GLOSS["Colloquial Mongolian Glossary Expansion"]
    GLOSS --> HYBRID["Hybrid Search Engine"]
    
    INDEX --> BM25["Lexical Ranker (BM25 Okapi)"]
    INDEX --> FAISS["Vector Ranker (FAISS In-Memory)"]
    
    HYBRID --> BM25
    HYBRID --> FAISS
    BM25 --> RRF["Reciprocal Rank Fusion (k=60)"]
    FAISS --> RRF
    RRF --> TOP["Verified Grounded Context"]
```

### Key RAG Features:
1. **Colloquial Mongolian Glossary Expansion (`app/glossary.py`)**:
   - Maps everyday spoken terms to official banking terminology (e.g., *"байрны зээл"* $\rightarrow$ *"Орон сууцны ипотекийн 6 хувийн зээл"*, *"карт залгисан"* $\rightarrow$ *"АТМ карт хураагдсан үед хэрэгжүүлэх заавар"*).
2. **In-Memory Hybrid Search**:
   - **BM25 Okapi**: Fast token-frequency lexical matching.
   - **FAISS L2 / Cosine Index**: Dense semantic matching.
   - **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense scores using $RRF(d) = \sum \frac{1}{60 + r(d)}$.
3. **Execution Latency**: 100% in-memory; executes in under **10 milliseconds**.

---

## 4. Multi-Layer Guardrails Architecture

The system incorporates four independent guardrails that intercept hazardous or off-topic inputs prior to calling downstream LLMs or executing transactions:

```mermaid
flowchart TD
    INPUT["Incoming Raw Transcript"] --> G1["Guardrail 1: PII Masking"]
    G1 -->|"Masks Account & Card Digits"| G2["Guardrail 2: Prompt Injection / Jailbreak Filter"]
    G2 -->|Safe| G3["Guardrail 3: Profanity & Harassment Filter"]
    G2 -->|Blocked| RESP_SEC["Security Deflection Response"]
    G3 -->|Safe| G4["Guardrail 4: Financial Compliance Boundary"]
    G3 -->|Abuse Detected| ESC_ABUSE["Escalate to Human Operator"]
    G4 -->|Informational Query| PIPELINE["Proceed to Stages 2-5"]
    G4 -->|Transaction / Account Change| ESC_FIN["Financial Compliance Escalation"]
```

1. **PII Masking**: Regular expressions detect 16-digit PAN card numbers, 10-digit bank accounts, and national ID patterns, substituting them with `[MASKED_ACCOUNT]` or `[MASKED_CARD]`.
2. **Prompt Injection & Jailbreak Filter**: Detects adversarial jailbreaks (e.g., "ignore all previous instructions", "DAN mode", "system prompt reveal") and deflects without LLM invocation.
3. **Profanity & Harassment Filter**: Escalates callers exhibiting hostile or abusive language directly to human supervisor queues.
4. **Financial Compliance Boundary**: Prevents the autonomous agent from performing money transfers, password resets, or account closures, steering callers to verified Khan Bank digital channels or the 1800-1917 contact center.

---

## 5. Dual-Engine Speech Synthesis (TTS) Architecture

The architecture provides two complementary TTS pathways:

```mermaid
graph TD
    TEXT["Synthesized Mongolian Response Text"] --> ROUTE{"Selected Voice Engine"}
    
    subgraph StreamTTS ["Gemini 3.1 Flash Streaming TTS (Preview)"]
        ROUTE -->|"gemini:Kore / gemini:Puck"| GENAI["Google GenAI SDK (gemini-3.1-flash-tts-preview)"]
        GENAI --> CHUNKS["Chunked Linear PCM 24kHz"]
        CHUNKS --> PACK["RIFF Header Pack (44 bytes) + Streaming WAV"]
        PACK --> PLAY_STREAM["Sub-second Audio Delivery (TTFB ~0.76s)"]
    end
    
    subgraph BatchTTS ["Narakeet High-Fidelity Batch TTS"]
        ROUTE -->|"narakeet:Ankhmaa / narakeet:Ganbaatar"| NARA["Narakeet HTTPS API (api.narakeet.com)"]
        NARA --> M4A["Compressed M4A Audio Stream"]
        M4A --> PLAY_BATCH["Studio Quality Batch Playback (~2.4s)"]
    end
```

### Voice Options:
- **`gemini:Kore`** (Google Gemini 3.1 Flash TTS, Female, Streaming 24kHz Linear PCM)
- **`gemini:Puck`** (Google Gemini 3.1 Flash TTS, Male, Streaming 24kHz Linear PCM)
- **`narakeet:Ankhmaa`** (Narakeet Cloud Neural Voice, Female, Batch M4A)
- **`narakeet:Ganbaatar`** (Narakeet Cloud Neural Voice, Male, Batch M4A)

---

## 6. Containerization & Deployment

- **Base Image**: `python:3.11-slim` with system packages (`libsndfile1`, `ffmpeg`, `libportaudio2`, `curl`).
- **GCP Credential Injection**: Supports standard `GOOGLE_APPLICATION_CREDENTIALS` file mount or runtime decoding from `GOOGLE_CREDENTIALS_BASE64` / `GOOGLE_CREDENTIALS_JSON` environment variables.
- **Port Mapping**: Exposes TCP port `5005` (or `$PORT` for Cloud Run / Render).
- **Process Model**: Gunicorn WSGI server with multi-threaded worker configurations.
