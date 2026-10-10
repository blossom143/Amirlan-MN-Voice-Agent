/**
 * AMIRLAN CONVERSATIONAL VOICE AI - CLIENT CONTROLLER
 * ====================================================
 * - Real-time AudioContext canvas visualizer
 * - Live step-by-step elapsed duration counters for each workflow stage
 * - Chat history with embedded replayable audio players for all responses
 * - Browser MediaRecorder + local hardware microphone + audio file upload
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM References ---
    const orbWrapper = document.getElementById("orbWrapper");
    const agentStateText = document.getElementById("agentStateText");
    const mainRecordBtn = document.getElementById("mainRecordBtn");
    const recordBtnLabel = document.getElementById("recordBtnLabel");
    const waveformCanvas = document.getElementById("waveformCanvas");
    const recordingTimer = document.getElementById("recordingTimer");
    const timerDigits = document.getElementById("timerDigits");

    // Mode Tabs
    const tabBrowserMic = document.getElementById("tabBrowserMic");
    const tabDesktopMic = document.getElementById("tabDesktopMic");
    const tabUpload = document.getElementById("tabUpload");
    const tabText = document.getElementById("tabText");
    const audioFileInput = document.getElementById("audioFileInput");
    const textInputWrapper = document.getElementById("textInputWrapper");
    const textQueryInput = document.getElementById("textQueryInput");
    const sendTextBtn = document.getElementById("sendTextBtn");

    // Stepper & Live Timers
    const totalTimeBadge = document.getElementById("totalTimeBadge");

    // Dialogue & Escalation
    const chatFeed = document.getElementById("chatFeed");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const escalationCard = document.getElementById("escalationCard");

    // Intelligence Drawer
    const intelCard = document.getElementById("intelCard");
    const intelToggleBtn = document.getElementById("intelToggleBtn");
    const intelBody = document.getElementById("intelBody");
    const intelIntentVal = document.getElementById("intelIntentVal");
    const intelDecisionVal = document.getElementById("intelDecisionVal");
    const intelSnippetVal = document.getElementById("intelSnippetVal");

    // --- State Variables ---
    let currentInputMode = "browser"; // 'browser' | 'desktop' | 'upload' | 'text'
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let recTimerInterval = null;
    let recSeconds = 0;

    // Canvas & Web Audio API
    let audioCtx = null;
    let analyserNode = null;
    let micStream = null;
    let animFrameId = null;
    const canvasCtx = waveformCanvas ? waveformCanvas.getContext("2d") : null;

    // Live Step Duration Stopwatch Engine
    let liveStepInterval = null;
    let currentActiveStep = null;
    let stepStartTime = 0;
    let pipelineStartTime = 0;
    let totalLiveInterval = null;

    // Global Active Audio Player tracker (only 1 plays at a time)
    let currentlyPlayingAudio = null;

    // =========================================================================
    // 1. INPUT MODE SELECTION
    // =========================================================================
    const modeTabs = [tabBrowserMic, tabDesktopMic, tabUpload, tabText];

    function setInputMode(mode) {
        currentInputMode = mode;
        modeTabs.forEach(tab => {
            if (tab) tab.classList.toggle("active", tab.dataset.mode === mode);
        });

        if (textInputWrapper) {
            textInputWrapper.style.display = (mode === "text") ? "flex" : "none";
        }

        if (mode === "browser") {
            recordBtnLabel.textContent = isRecording ? "Stop Recording" : "Start Voice Recording";
        } else if (mode === "desktop") {
            recordBtnLabel.textContent = "Hardware Mic (5s)";
        } else if (mode === "upload") {
            recordBtnLabel.textContent = "Select Audio File";
        } else if (mode === "text") {
            recordBtnLabel.textContent = "Send Text Query";
        }
    }

    if (tabBrowserMic) tabBrowserMic.addEventListener("click", () => setInputMode("browser"));
    if (tabDesktopMic) tabDesktopMic.addEventListener("click", () => setInputMode("desktop"));
    if (tabUpload) tabUpload.addEventListener("click", () => {
        setInputMode("upload");
        if (audioFileInput) audioFileInput.click();
    });
    if (tabText) tabText.addEventListener("click", () => setInputMode("text"));

    if (audioFileInput) {
        audioFileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) processUploadedAudioFile(file);
        });
    }

    // =========================================================================
    // 2. ORB STATE CONTROLLER
    // =========================================================================
    function setOrbState(state, statusText) {
        if (!orbWrapper) return;
        orbWrapper.classList.remove("state-listening", "state-analyzing", "state-speaking", "state-escalated");
        if (state) {
            orbWrapper.classList.add(`state-${state}`);
        }
        if (agentStateText && statusText) {
            agentStateText.textContent = statusText;
        }
    }

    // =========================================================================
    // 3. AUDIO VISUALIZER (Web Audio API)
    // =========================================================================
    function initVisualizer(stream) {
        if (!canvasCtx) return;
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyserNode = audioCtx.createAnalyser();
            analyserNode.fftSize = 128;
            const source = audioCtx.createMediaStreamSource(stream);
            source.connect(analyserNode);
            drawWaveform();
        } catch (err) {
            console.warn("Visualizer init error:", err);
        }
    }

    function drawWaveform() {
        if (!canvasCtx || !analyserNode) return;
        const width = waveformCanvas.width;
        const height = waveformCanvas.height;
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        analyserNode.getByteFrequencyData(dataArray);
        canvasCtx.clearRect(0, 0, width, height);

        const barWidth = (width / bufferLength) * 2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * (height - 8);

            const gradient = canvasCtx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, "rgba(255, 65, 78, 0.25)");
            gradient.addColorStop(0.5, "#ff8928");
            gradient.addColorStop(1, "#ff414e");

            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(x, (height - barHeight) / 2, barWidth - 2, barHeight || 2);
            x += barWidth;
        }

        animFrameId = requestAnimationFrame(drawWaveform);
    }

    function stopVisualizer() {
        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
        if (canvasCtx) {
            canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        }
    }

    // =========================================================================
    // 4. LIVE STEP TIMER STOPWATCH ENGINE
    // =========================================================================
    function startPipelineTimers() {
        resetStepper();
        pipelineStartTime = performance.now();

        if (totalLiveInterval) clearInterval(totalLiveInterval);
        totalLiveInterval = setInterval(() => {
            const totalSec = ((performance.now() - pipelineStartTime) / 1000).toFixed(2);
            if (totalTimeBadge) totalTimeBadge.textContent = `${totalSec}s`;
        }, 60);

        startStepLiveTimer(1);
    }

    function startStepLiveTimer(stepNum) {
        if (liveStepInterval) clearInterval(liveStepInterval);
        currentActiveStep = stepNum;
        stepStartTime = performance.now();

        // Highlight active step
        for (let i = 1; i < stepNum; i++) {
            const prev = document.getElementById(`step${i}`);
            if (prev) {
                prev.classList.remove("active");
                prev.classList.add("completed");
            }
        }
        const currentEl = document.getElementById(`step${stepNum}`);
        if (currentEl) currentEl.classList.add("active");

        const timerEl = document.getElementById(`timerStep${stepNum}`);
        const metaEl = document.getElementById(`step${stepNum}Meta`);
        if (metaEl) metaEl.textContent = "Processing...";

        liveStepInterval = setInterval(() => {
            const elapsed = ((performance.now() - stepStartTime) / 1000).toFixed(1);
            if (timerEl) timerEl.textContent = `${elapsed}s`;
        }, 50);
    }

    function stopPipelineTimers() {
        if (liveStepInterval) {
            clearInterval(liveStepInterval);
            liveStepInterval = null;
        }
        if (totalLiveInterval) {
            clearInterval(totalLiveInterval);
            totalLiveInterval = null;
        }
    }

    function resetStepper() {
        stopPipelineTimers();
        for (let i = 1; i <= 6; i++) {
            const stepEl = document.getElementById(`step${i}`);
            const timerEl = document.getElementById(`timerStep${i}`);
            const metaEl = document.getElementById(`step${i}Meta`);
            if (stepEl) {
                stepEl.classList.remove("active", "completed", "escalated");
            }
            if (timerEl) {
                timerEl.textContent = "0.0s";
            }
            if (metaEl && i !== 6) {
                metaEl.textContent = "Idle";
            }
        }
        if (totalTimeBadge) totalTimeBadge.textContent = "0.00s";
    }

    function applyBackendStepLogs(stageLogs, isEscalated, totalSec) {
        stopPipelineTimers();

        stageLogs.forEach(log => {
            const stepEl = document.getElementById(`step${log.stage}`);
            const timerEl = document.getElementById(`timerStep${log.stage}`);
            const metaEl = document.getElementById(`step${log.stage}Meta`);

            if (stepEl) {
                stepEl.classList.remove("active");
                if (log.status === "triggered") {
                    stepEl.classList.add("escalated");
                } else {
                    stepEl.classList.add("completed");
                }
            }
            if (timerEl) {
                timerEl.textContent = `${log.duration}s`;
            }
            if (metaEl) {
                metaEl.textContent = log.details || "Finished";
            }
        });

        if (totalTimeBadge && totalSec) {
            totalTimeBadge.textContent = `${totalSec}s`;
        }
    }

    // =========================================================================
    // 5. BROWSER MICROPHONE RECORDING (Phone & PC)
    // =========================================================================
    async function startBrowserRecording() {
        primeAudioContext();
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];

            let mimeType = "audio/webm";
            if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
                mimeType = "audio/webm;codecs=opus";
            } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
                mimeType = "audio/mp4";
            }

            mediaRecorder = new MediaRecorder(micStream, { mimeType });

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                sendAudioBlobToPipeline(audioBlob);
                if (micStream) {
                    micStream.getTracks().forEach(track => track.stop());
                }
                stopVisualizer();
            };

            mediaRecorder.start(200);
            isRecording = true;

            setOrbState("listening", "Listening • Speak now...");
            mainRecordBtn.classList.add("recording");
            recordBtnLabel.textContent = "Stop Recording";

            initVisualizer(micStream);
            startRecTimer();

        } catch (err) {
            console.error("Mic access error:", err);
            alert("Microphone access is required to capture voice.");
            isRecording = false;
        }
    }

    function stopBrowserRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            stopRecTimer();
            setOrbState("analyzing", "Transcribing & Reasoning (Chirp 3 • Gemini)...");
            mainRecordBtn.classList.remove("recording");
            recordBtnLabel.textContent = "Processing...";
        }
    }

    function startRecTimer() {
        recSeconds = 0;
        if (recordingTimer) recordingTimer.style.display = "inline-flex";
        recTimerInterval = setInterval(() => {
            recSeconds++;
            const m = String(Math.floor(recSeconds / 60)).padStart(2, "0");
            const s = String(recSeconds % 60).padStart(2, "0");
            if (timerDigits) timerDigits.textContent = `${m}:${s}`;
            if (recSeconds >= 45) stopBrowserRecording();
        }, 1000);
    }

    function stopRecTimer() {
        if (recTimerInterval) {
            clearInterval(recTimerInterval);
            recTimerInterval = null;
        }
        if (recordingTimer) recordingTimer.style.display = "none";
    }

    // =========================================================================
    // 6. PIPELINE API EXECUTION
    // =========================================================================
    async function sendAudioBlobToPipeline(audioBlob) {
        startPipelineTimers();

        const formData = new FormData();
        const extension = audioBlob.type.includes("mp4") ? "mp4" : "webm";
        formData.append("audio", audioBlob, `speech_input.${extension}`);

        // Progress simulation for live feedback while waiting for server response
        simulateLiveStepTransitions();

        try {
            const res = await fetch("/api/voice/process", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            handlePipelineResponse(data);
        } catch (err) {
            console.error("Pipeline network error:", err);
            stopPipelineTimers();
            appendUserMessage("[Voice Input]");
            appendAssistantMessage("Network error occurred: " + err.message);
            setOrbState("", "Ready to Listen");
            recordBtnLabel.textContent = "Start Voice Recording";
        }
    }

    async function triggerHardwareRecording() {
        primeAudioContext();
        setOrbState("listening", "Recording from Hardware Mic (5 seconds)...");
        startPipelineTimers();

        simulateLiveStepTransitions();

        try {
            const res = await fetch("/api/voice/record-desktop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ duration: 5 })
            });
            const data = await res.json();
            handlePipelineResponse(data);
        } catch (err) {
            console.error("Hardware mic error:", err);
            stopPipelineTimers();
            appendAssistantMessage("Hardware microphone error: " + err.message);
            setOrbState("", "Ready to Listen");
        }
    }

    async function processUploadedAudioFile(file) {
        primeAudioContext();
        setInputMode("browser");
        setOrbState("analyzing", `Processing audio: ${file.name}...`);
        startPipelineTimers();

        const formData = new FormData();
        formData.append("audio", file, file.name);

        simulateLiveStepTransitions();

        try {
            const res = await fetch("/api/voice/process", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            handlePipelineResponse(data);
        } catch (err) {
            console.error("Uploaded file error:", err);
            stopPipelineTimers();
            appendAssistantMessage("Error processing file: " + err.message);
            setOrbState("", "Ready to Listen");
        }
    }

    async function sendTextQuery(text) {
        if (!text.trim()) return;
        primeAudioContext();
        setOrbState("analyzing", "Analyzing query (Gemini 2.5 Flash)...");
        startPipelineTimers();
        startStepLiveTimer(2);

        simulateLiveStepTransitions(2);

        try {
            const res = await fetch("/api/text/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text.trim() })
            });
            const data = await res.json();
            handlePipelineResponse(data);
            if (textQueryInput) textQueryInput.value = "";
        } catch (err) {
            console.error("Text process error:", err);
            stopPipelineTimers();
            appendUserMessage(text);
            appendAssistantMessage("Error: " + err.message);
            setOrbState("", "Ready to Listen");
        }
    }

    // Simulates visual stage progression ticks while awaiting response
    function simulateLiveStepTransitions(startStep = 1) {
        let step = startStep;
        const interval = setInterval(() => {
            if (!pipelineStartTime) {
                clearInterval(interval);
                return;
            }
            if (step < 5) {
                step++;
                startStepLiveTimer(step);
            }
        }, 3200);
    }

    // =========================================================================
    // 7. PIPELINE RESPONSE HANDLER
    // =========================================================================
    function handlePipelineResponse(data) {
        recordBtnLabel.textContent = "Start Voice Recording";

        if (!data.success) {
            stopPipelineTimers();
            appendAssistantMessage("Error: " + (data.error || "Unknown pipeline error"));
            setOrbState("", "Ready to Listen");
            return;
        }

        // 1. Add User speech transcript to chat
        const userText = data.transcript || "[Empty Audio]";
        appendUserMessage(userText);

        // 2. Add Assistant response with integrated Replay Audio Player
        appendAssistantMessage(data.response_text, data.intent, data.audio_url);

        // 3. Apply exact recorded backend stage duration logs
        if (data.stage_logs) {
            applyBackendStepLogs(data.stage_logs, data.is_escalated, data.execution_time_sec);
        }

        // 4. Update Reasoning drawer
        updateIntelligenceDrawer(data);

        // 5. Handle Human Escalation
        if (data.is_escalated && data.escalation) {
            setOrbState("escalated", "Escalated to Operator • 1800-1917");
            if (escalationCard) {
                escalationCard.style.display = "block";
                escalationCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        } else {
            if (escalationCard) escalationCard.style.display = "none";
        }

        // 6. Autoplay the synthesized response audio
        if (data.audio_url) {
            autoplayNewestAudio(data.audio_url);
        } else {
            if (!data.is_escalated) {
                setOrbState("", "Ready to Listen");
            }
        }
    }

    // =========================================================================
    // 8. CHAT HISTORY WITH EMBEDDED REPLAYABLE AUDIO PLAYERS
    // =========================================================================
    function appendUserMessage(text) {
        if (!chatFeed) return;
        const timeNow = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

        const row = document.createElement("div");
        row.className = "message-row row-user";
        row.innerHTML = `
            <div class="msg-card">
                <div class="msg-header">
                    <span class="speaker-title">You</span>
                    <span class="msg-timestamp">${timeNow}</span>
                </div>
                <div class="msg-content">${escapeHtml(text)}</div>
            </div>
        `;
        chatFeed.appendChild(row);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function appendAssistantMessage(text, intentData = null, audioUrl = null) {
        if (!chatFeed) return;
        const timeNow = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const intentLabel = intentData ? (intentData.topic || intentData.category) : "Verified Response";

        const row = document.createElement("div");
        row.className = "message-row row-assistant";

        // Build audio player widget HTML if audio is available
        let audioPlayerHtml = "";
        const playerId = "audio_" + Math.random().toString(36).substring(2, 9);

        if (audioUrl) {
            audioPlayerHtml = `
                <div class="bubble-audio-player" id="${playerId}_container">
                    <button class="bubble-play-btn" id="${playerId}_btn" title="Replay response">
                        <svg class="icon-play" viewBox="0 0 24 24" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3"/>
                        </svg>
                        <svg class="icon-pause" viewBox="0 0 24 24" fill="currentColor" style="display: none;">
                            <rect x="6" y="4" width="4" height="16"/>
                            <rect x="14" y="4" width="4" height="16"/>
                        </svg>
                    </button>
                    <input type="range" class="bubble-seeker" id="${playerId}_seek" min="0" max="100" value="0">
                    <span class="bubble-time" id="${playerId}_time">0:00</span>
                    <a href="${audioUrl}" download class="bubble-download-btn" title="Download audio (.m4a)">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                    </a>
                    <audio id="${playerId}_audio" src="${audioUrl}" preload="auto" style="display: none;"></audio>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="msg-card">
                <div class="msg-header">
                    <div class="avatar-badge">A</div>
                    <span class="speaker-title">Amirlan AI</span>
                    <span class="msg-timestamp">${timeNow}</span>
                </div>
                <div class="msg-content">${escapeHtml(text)}</div>
                ${audioPlayerHtml}
                <div class="msg-footer">
                    <span class="grounded-tag">✓ Grounded</span>
                    <span class="intent-chip">${intentLabel}</span>
                </div>
            </div>
        `;

        chatFeed.appendChild(row);
        chatFeed.scrollTop = chatFeed.scrollHeight;

        // Attach Replay Controller to this message's audio player
        if (audioUrl) {
            setupBubbleAudioPlayer(playerId, audioUrl);
        }
    }

    function setupBubbleAudioPlayer(playerId, audioUrl) {
        const audioEl = document.getElementById(`${playerId}_audio`);
        const playBtn = document.getElementById(`${playerId}_btn`);
        const seekEl = document.getElementById(`${playerId}_seek`);
        const timeEl = document.getElementById(`${playerId}_time`);

        if (!audioEl || !playBtn) return;

        const iconPlay = playBtn.querySelector(".icon-play");
        const iconPause = playBtn.querySelector(".icon-pause");

        playBtn.addEventListener("click", () => {
            if (audioEl.paused) {
                // Pause any other playing audio
                if (currentlyPlayingAudio && currentlyPlayingAudio !== audioEl) {
                    currentlyPlayingAudio.pause();
                }
                currentlyPlayingAudio = audioEl;

                audioEl.play().then(() => {
                    setOrbState("speaking", "Speaking • Ganbaatar voice");
                    iconPlay.style.display = "none";
                    iconPause.style.display = "block";
                }).catch(err => {
                    console.warn("Playback error:", err);
                });
            } else {
                audioEl.pause();
                iconPlay.style.display = "block";
                iconPause.style.display = "none";
                setOrbState("", "Ready to Listen");
            }
        });

        audioEl.ontimeupdate = () => {
            if (audioEl.duration) {
                const prog = (audioEl.currentTime / audioEl.duration) * 100;
                seekEl.value = prog;
                timeEl.textContent = formatSecs(audioEl.currentTime);
            }
        };

        seekEl.addEventListener("input", () => {
            if (audioEl.duration) {
                audioEl.currentTime = (seekEl.value / 100) * audioEl.duration;
            }
        });

        audioEl.onended = () => {
            iconPlay.style.display = "block";
            iconPause.style.display = "none";
            seekEl.value = 0;
            timeEl.textContent = formatSecs(audioEl.duration || 0);
            setOrbState("", "Ready to Listen");
            if (currentlyPlayingAudio === audioEl) currentlyPlayingAudio = null;
        };

        audioEl.onloadedmetadata = () => {
            timeEl.textContent = formatSecs(audioEl.duration);
        };
    }

    function autoplayNewestAudio(audioUrl) {
        // Find newest audio in feed
        const allAudios = chatFeed.querySelectorAll("audio");
        if (allAudios.length > 0) {
            const latestAudio = allAudios[allAudios.length - 1];
            if (currentlyPlayingAudio && currentlyPlayingAudio !== latestAudio) {
                currentlyPlayingAudio.pause();
            }
            currentlyPlayingAudio = latestAudio;

            latestAudio.play().then(() => {
                setOrbState("speaking", "Speaking • Narakeet Audio");
                const btn = latestAudio.parentElement.querySelector(".bubble-play-btn");
                if (btn) {
                    btn.querySelector(".icon-play").style.display = "none";
                    btn.querySelector(".icon-pause").style.display = "block";
                }
            }).catch(e => {
                console.warn("Autoplay deferred:", e);
                setOrbState("", "Response Ready • Click Play in Chat");
            });
        }
    }

    function formatSecs(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s < 10 ? "0" : ""}${s}`;
    }

    function primeAudioContext() {
        // Unlock user gesture for subsequent background autoplay
        try {
            const silent = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA");
            silent.play().catch(() => {});
        } catch (_) {}
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.innerText = text;
        return div.innerHTML;
    }

    // =========================================================================
    // 9. INTELLIGENCE DRAWER & CHIPS
    // =========================================================================
    function updateIntelligenceDrawer(data) {
        if (!intelCard) return;
        intelCard.style.display = "block";

        if (intelIntentVal && data.intent) {
            intelIntentVal.textContent = `${data.intent.topic} (${data.intent.category})`;
        }
        if (intelDecisionVal && data.decision) {
            intelDecisionVal.textContent = `${data.decision.decision.toUpperCase()} - ${data.decision.reason}`;
        }
        if (intelSnippetVal && data.retrieval) {
            intelSnippetVal.textContent = data.retrieval.search_results || data.retrieval.verified_context;
        }
    }

    if (intelToggleBtn && intelBody) {
        intelToggleBtn.addEventListener("click", () => {
            const isClosed = intelBody.style.display === "none";
            intelBody.style.display = isClosed ? "block" : "none";
            const arrow = intelToggleBtn.querySelector(".intel-toggle-arrow");
            if (arrow) arrow.textContent = isClosed ? "▲" : "▼";
        });
    }

    if (clearChatBtn && chatFeed) {
        clearChatBtn.addEventListener("click", () => {
            chatFeed.innerHTML = "";
            appendAssistantMessage("Dialogue history cleared. Ask a new inquiry to begin.");
        });
    }

    // Preset Prompt Chips
    document.querySelectorAll(".prompt-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.dataset.query;
            if (query) sendTextQuery(query);
        });
    });

    // Record / Submit Button Listeners
    if (mainRecordBtn) {
        mainRecordBtn.addEventListener("click", () => {
            if (currentInputMode === "browser") {
                if (isRecording) {
                    stopBrowserRecording();
                } else {
                    startBrowserRecording();
                }
            } else if (currentInputMode === "desktop") {
                triggerHardwareRecording();
            } else if (currentInputMode === "upload") {
                if (audioFileInput) audioFileInput.click();
            } else if (currentInputMode === "text") {
                if (textQueryInput && textQueryInput.value.trim()) {
                    sendTextQuery(textQueryInput.value.trim());
                } else if (textQueryInput) {
                    textQueryInput.focus();
                }
            }
        });
    }

    if (orbWrapper) {
        orbWrapper.addEventListener("click", () => {
            if (mainRecordBtn) mainRecordBtn.click();
        });
    }

    if (sendTextBtn && textQueryInput) {
        sendTextBtn.addEventListener("click", () => {
            sendTextQuery(textQueryInput.value);
        });
        textQueryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") sendTextQuery(textQueryInput.value);
        });
    }

    console.log("⚡ Amirlan Voice AI Agent UI initialized.");
});
