# Operational Workflows: Khan Bank Voicebot Agent ("Amirlan")

## 1. End-to-End Voice Call Lifecycle

The continuous voice call operates through an asynchronous state machine driven by browser Web Audio API, client-side Voice Activity Detection (VAD), and server-side agentic processing.

```mermaid
sequenceDiagram
    autonumber
    actor User as Caller (Microphone)
    participant Client as Browser (app.js + Web Audio)
    participant Server as Flask / Gunicorn Server
    participant STT as Google Chirp 3 STT
    participant RAG as FAISS + BM25 RAG
    participant LLM as Vertex AI Gemini 2.5 Flash
    participant TTS as Speech Engine (Gemini / Narakeet)

    Note over Client,Server: Step 0: Page Load Warmup
    Client->>Server: POST /api/warmup (Background pre-warm)
    Server-->>Client: 200 OK (FAISS + GCP clients pre-loaded)

    Note over User,Client: Step 1: Call Initiation
    User->>Client: Clicks "Start Voice Call"
    Client->>Client: Acquire getUserMedia (echoCancellation: true)
    Client->>Client: Initialize AudioContext + AnalyserNode (FFT 256)
    Client->>Client: Start 60 FPS VAD Loop (RMS monitoring)

    Note over User,Client: Step 2: Speech Detection
    User->>Client: Speaks ("Хадгаламжийн хүү хэд вэ?")
    Client->>Client: RMS > 0.026 for 3+ consecutive frames
    Client->>Client: Start MediaRecorder (WebM/Opus)
    Client->>Client: Orb State: "Listening to you speak..."

    Note over User,Client: Step 3: Silence & Turn Dispatch
    User->>Client: Finishes speaking (Silence > 1350ms)
    Client->>Client: MediaRecorder.stop() -> onstop creates audioBlob
    Client->>Client: isTurnInFlight = true
    Client->>Server: POST /api/voice/process (audio, voice, is_call_mode)

    Note over Server,TTS: Step 4: Server-Side Pipeline Execution
    Server->>STT: Stage 1: Chirp 3 Speech-to-Text (mn-MN)
    STT-->>Server: Transcribed text ("Хадгаламжийн хүү хэд вэ?")
    Server->>Server: Stage 1 Safeguard: PII Masking
    Server->>RAG: Stage 2 & 3: Fast Intent + FAISS/BM25 Search
    RAG-->>Server: Verified Khan Bank knowledge chunk
    Server->>LLM: Stage 4: Single-Pass ReAct Reasoning (Gemini 2.5 Flash)
    LLM-->>Server: JSON decision="answer", response_text="..."
    Server->>TTS: Stage 5: Speech Synthesis (WAV stream / M4A)
    TTS-->>Server: Synthesized audio file (amirlan_cache_*.wav)
    Server-->>Client: JSON response {transcript, response_text, audio_url, stage_logs}

    Note over Client,User: Step 5: Playback & Auto-Rearm
    Client->>Client: isTurnInFlight = false
    Client->>Client: Apply exact stage logs to UI stepper
    Client->>User: Play synthesized audio response
    Client->>Client: Audio finishes ('ended') -> Re-arm VAD for next turn
```

---

## 2. 6-Stage Agentic Reasoning Loop

Every incoming customer turn is processed through a deterministic 6-stage pipeline in [`app/voicebot.py`](file:///Users/tsomorlig/Documents/mn-voicebot-agent/app/voicebot.py):

```mermaid
flowchart TD
    START([Turn Received]) --> S1[Stage 1: Call Intake & Transcription]
    S1 --> PII{PII Masking}
    PII --> CACHE_CHECK{Response Cache Hit?}
    
    CACHE_CHECK -->|Hit| CACHE_RET[Return Cached Response & Audio in <0.02s]
    CACHE_CHECK -->|Miss| UNCLEAR_CHECK{Transcript Unclear?}
    
    UNCLEAR_CHECK -->|3x Consecutive Unclear| S6_ESC[Stage 6: Operator Escalation 1800-1917]
    UNCLEAR_CHECK -->|First Unclear| REPROMPT[Prompt user to repeat question]
    UNCLEAR_CHECK -->|Clear Speech| GUARDS{Input Guardrails}
    
    GUARDS -->|Prompt Injection| DEFLECT[Security Deflection Response]
    GUARDS -->|Profanity / Abuse| ESC_ABUSE[Escalate to Human Supervisor]
    GUARDS -->|Financial Transaction| ESC_COMP[Escalate to Khan Bank Digital Channels]
    GUARDS -->|Pass| S2[Stage 2: Intent Classification]
    
    S2 --> S3[Stage 3: Information Retrieval FAISS + BM25]
    S3 --> S4[Stage 4: Decision Making ReAct Engine]
    
    S4 -->|Requires Human| S6[Stage 6: Escalation Protocol]
    S4 -->|Answerable| S5[Stage 5: Response Generation & Dual TTS]
    S6 --> S5
    
    S5 --> MEMOIZE[Store in In-Memory Response Cache]
    MEMOIZE --> FINISH([Deliver Response & Play Audio])
```

### Stage Details:

1. **Stage 1: Call Intake & Transcription**
   - Transcribes audio using Google Cloud Speech-to-Text v2 Chirp 3 (`mn-MN`).
   - Applies regex-based PII masking to remove account numbers, credit card digits, and citizen registration IDs.
2. **Stage 2: Intent Understanding**
   - High-speed pattern classifier checks keyword categories (`loan_inquiry`, `account_service`, `card_service`, `branch_atm`, `exchange_rate`, `human_escalation`).
   - If no pattern matches, falls back to a low-token Gemini 2.5 Flash classification prompt.
3. **Stage 3: Information Retrieval**
   - Expands query with colloquial Mongolian banking glossary (`app/glossary.py`).
   - Executes hybrid retrieval via in-memory BM25 + FAISS vector search (`app/vector_store.py`).
   - Returns verified context chunk with official Khan Bank source URL.
4. **Stage 4: Decision Making (ReAct Engine)**
   - Formulates single-pass prompt enforcing banking compliance, brevity (<24 words for voice), and strict grounding in retrieved knowledge.
   - Evaluates whether to answer directly (`"decision": "answer"`) or transfer to a specialist (`"decision": "escalate"`).
5. **Stage 5: Response Generation & TTS**
   - Synthesizes audio using either Google Gemini 3.1 Flash Streaming TTS (24kHz linear PCM) or Narakeet Neural Batch TTS (M4A).
   - Memoizes `{voice:query}` response pair in cache for instant sub-20ms hits on repeat questions.
6. **Stage 6: Human Escalation Protocol**
   - Generates structured routing ticket: contact hotline (`1800-1917`), department assignment, priority, and caller summary.

---

## 3. Barge-In & Interruption Handling Workflow

Barge-in allows callers to interrupt the AI voicebot naturally while it is speaking, mirroring a real human telephone call.

```mermaid
stateDiagram-v2
    [*] --> Idle: Call Started
    Idle --> Listening: VAD starts
    
    Listening --> UserSpeaking: RMS > 0.026 for 3+ frames
    UserSpeaking --> Thinking: Silence > 1350ms
    Thinking --> Speaking: Server delivers audio & starts playback
    
    state Speaking {
        [*] --> AssistantAudioPlaying
        AssistantAudioPlaying --> InterruptionDetected: User speaks (RMS > 0.026 for 3+ frames)
        InterruptionDetected --> AudioPaused: pause() + currentTime = 0
        AudioPaused --> [*]: Discard remaining playback
    }
    
    Speaking --> UserSpeaking: Barge-in triggered (Starts new user turn)
    Speaking --> Listening: Audio naturally finishes ('ended' event)
```

### Interruption State Mechanics (`app/static/js/app.js`):
1. **Detection Window**: While `currentlyPlayingAudio` is playing (`isAssistantSpeaking = true`).
2. **Trigger Condition**:
   - `rms > SPEECH_RMS_THRESHOLD` (0.026) for $\ge 3$ consecutive frames (~50ms).
3. **Action Executed**:
   ```javascript
   currentlyPlayingAudio.pause();
   currentlyPlayingAudio.currentTime = 0;
   bargeInTag.textContent = "⚡ Interrupted!";
   ```
4. **Transition to Recording**:
   - `isUserSpeaking = true`
   - `vadRecorder.start(100)` begins recording the new question immediately.
   - When user stops speaking (silence > 1350ms), the new question is dispatched to `/api/voice/process`.

---

## 4. Operational Circuit Breakers & Safeguards

The voicebot enforces safety circuit breakers to prevent infinite loops, hallucinated advice, and adversarial exploitation:

| Circuit Breaker | Trigger Condition | Automated Action | Target SLA |
|---|---|---|---|
| **Turn Limit Breaker** | Session turns $\ge 12$ | Automatic Stage 6 escalation to human agent (`1800-1917`) | Instant |
| **Consecutive Unclear Breaker** | 3 consecutive unclear audio turns | Automatic escalation to telephone operator | Instant |
| **Prompt Injection Breaker** | Jailbreak or system prompt reveal keyword | Blocks LLM call; delivers pre-compiled security deflection | < 0.01s |
| **Financial Boundary Breaker** | Autonomous money transfer, password reset | Escalates with instructions to use Khan Bank App or branch | Instant |
| **Off-Topic Query Steering** | General knowledge / non-banking queries | Polite deflection steering user back to Khan Bank services | < 0.01s |
