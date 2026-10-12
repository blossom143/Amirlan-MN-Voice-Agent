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
    const tabVoiceCall = document.getElementById("tabVoiceCall");
    const tabDesktopMic = document.getElementById("tabDesktopMic");
    const tabUpload = document.getElementById("tabUpload");
    const tabText = document.getElementById("tabText");
    const audioFileInput = document.getElementById("audioFileInput");
    const textInputWrapper = document.getElementById("textInputWrapper");
    const textQueryInput = document.getElementById("textQueryInput");
    const sendTextBtn = document.getElementById("sendTextBtn");

    // Live Voice Call HUD Elements
    const callHudBanner = document.getElementById("callHudBanner");
    const callHudTitle = document.getElementById("callHudTitle");
    const callDurationTimer = document.getElementById("callDurationTimer");
    const callVadStatus = document.getElementById("callVadStatus");
    const bargeInTag = document.getElementById("bargeInTag");
    const vadStatusPill = document.getElementById("vadStatusPill");
    const mainRecordIcon = document.getElementById("mainRecordIcon");
    const callHudEndBtn = document.getElementById("callHudEndBtn");

    // Stepper & Live Timers
    const totalTimeBadge = document.getElementById("totalTimeBadge");

    // Dialogue & Escalation
    const chatFeed = document.getElementById("chatFeed");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const escalationCard = document.getElementById("escalationCard");

    // Intelligence Drawer & Navbar Tab
    const intelCard = document.getElementById("intelCard");
    const intelNavTabBtn = document.getElementById("intelNavTabBtn");
    const intelCloseBtn = document.getElementById("intelCloseBtn");
    const intelBadge = document.getElementById("intelBadge");
    const intelToggleBtn = document.getElementById("intelToggleBtn");
    const intelBody = document.getElementById("intelBody");
    const intelIntentVal = document.getElementById("intelIntentVal");
    const intelSnippetVal = document.getElementById("intelSnippetVal");
    const intelUriLink = document.getElementById("intelUriLink");
    const intelUriBadge = document.getElementById("intelUriBadge");

    // Voice Selector (Ankhmaa default, Ganbaatar supported)
    const voiceSelect = document.getElementById("voiceSelect");

    function getSelectedVoice() {
        return voiceSelect ? voiceSelect.value : "Ankhmaa";
    }

    // --- State Variables ---
    let currentInputMode = "call"; // 'call' | 'desktop' | 'upload' | 'text'

    // Continuous Real-Time Voice Call State Engine (VAD + Barge-In)
    let isCallActive = false;
    let callStartTime = 0;
    let callTimerInterval = null;
    let vadStream = null;
    let vadAudioCtx = null;
    let vadAnalyser = null;
    let vadRecorder = null;
    let vadChunks = [];
    let vadAnimFrame = null;
    let isUserSpeaking = false;
    let speechStartTimestamp = 0;
    let lastSoundTimestamp = 0;
    let consecutiveSpeechFrames = 0;
    let isTurnInFlight = false;

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
    let stepSimulationInterval = null;

    // Global Active Audio Player tracker (only 1 plays at a time)
    let currentlyPlayingAudio = null;

    // =========================================================================
    // 1. INPUT MODE SELECTION & ICON CONTROLLER
    // =========================================================================
    const modeTabs = [tabVoiceCall, tabDesktopMic, tabUpload, tabText];

    function updateCallIcon(isInCall) {
        if (!mainRecordIcon) return;
        if (isInCall) {
            mainRecordIcon.innerHTML = `<path d="M12 9c-1.6 0-3.15.25-4.6.72v3.1c0 .39-.23.74-.56.9-.98.49-1.87 1.12-2.66 1.85-.18.18-.43.28-.7.28-.28 0-.53-.11-.71-.29L.29 13.08c-.18-.17-.29-.42-.29-.7 0-.28.11-.53.29-.71C3.34 8.78 7.46 7 12 7s8.66 1.78 11.71 4.67c.18.18.29.43.29.71 0 .28-.11-.53-.29-.71l-2.48-2.48c-.18-.18-.43-.29-.71-.29-.27 0-.52.1-.7.28-.79-.74-1.69-1.36-2.67-1.85-.33-.16-.56-.5-.56-.9v-3.1C15.15 9.25 13.6 9 12 9z" fill="currentColor"/>`;
        } else {
            mainRecordIcon.innerHTML = `<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />`;
        }
    }

    function updateMicIcon() {
        if (!mainRecordIcon) return;
        mainRecordIcon.innerHTML = `<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>`;
    }

    function updateUploadIcon() {
        if (!mainRecordIcon) return;
        mainRecordIcon.innerHTML = `<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>`;
    }

    function updateSendIcon() {
        if (!mainRecordIcon) return;
        mainRecordIcon.innerHTML = `<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>`;
    }

    function setInputMode(mode) {
        if (currentInputMode === "call" && mode !== "call" && isCallActive) {
            stopVoiceCall();
        }
        currentInputMode = mode;
        modeTabs.forEach(tab => {
            if (tab) tab.classList.toggle("active", tab.dataset.mode === mode);
        });

        if (textInputWrapper) {
            textInputWrapper.style.display = (mode === "text") ? "flex" : "none";
        }
        if (callHudBanner) {
            callHudBanner.style.display = (mode === "call" && isCallActive) ? "flex" : "none";
        }

        if (mainRecordBtn) {
            mainRecordBtn.classList.remove("btn-call-start", "btn-end-call", "recording");
        }

        if (mode === "call") {
            if (isCallActive) {
                recordBtnLabel.textContent = "End call";
                mainRecordBtn.classList.remove("btn-call-start");
                mainRecordBtn.classList.add("btn-end-call");
                updateCallIcon(true);
            } else {
                recordBtnLabel.textContent = "Start Voice Call";
                mainRecordBtn.classList.remove("btn-end-call");
                mainRecordBtn.classList.add("btn-call-start");
                updateCallIcon(false);
                setOrbState("", "Ready to Call");
            }
        } else if (mode === "desktop") {
            recordBtnLabel.textContent = "Hardware Mic (5s)";
            updateMicIcon();
        } else if (mode === "upload") {
            recordBtnLabel.textContent = "Select Audio File";
            updateUploadIcon();
        } else if (mode === "text") {
            recordBtnLabel.textContent = "Send Text Query";
            updateSendIcon();
        }
    }

    if (tabVoiceCall) tabVoiceCall.addEventListener("click", () => setInputMode("call"));
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
        if (stepSimulationInterval) {
            clearInterval(stepSimulationInterval);
            stepSimulationInterval = null;
        }
        if (liveStepInterval) {
            clearInterval(liveStepInterval);
            liveStepInterval = null;
        }
        if (totalLiveInterval) {
            clearInterval(totalLiveInterval);
            totalLiveInterval = null;
        }
        pipelineStartTime = 0;
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

        // Explicitly clear any lingering active states from all steps
        for (let i = 1; i <= 6; i++) {
            const stepEl = document.getElementById(`step${i}`);
            if (stepEl) stepEl.classList.remove("active");
        }

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
    // 4.5. CONTINUOUS REAL-TIME VOICE CALL ENGINE (VAD + BARGE-IN)
    // =========================================================================
    async function startVoiceCall() {
        primeAudioContext();
        try {
            vadStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            vadAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            vadAnalyser = vadAudioCtx.createAnalyser();
            vadAnalyser.fftSize = 256;
            const source = vadAudioCtx.createMediaStreamSource(vadStream);
            source.connect(vadAnalyser);

            let mimeType = "audio/webm";
            if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
                mimeType = "audio/webm;codecs=opus";
            } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
                mimeType = "audio/mp4";
            }

            vadRecorder = new MediaRecorder(vadStream, { mimeType });
            vadRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    vadChunks.push(e.data);
                }
            };

            vadRecorder.onstop = () => {
                if (vadChunks.length > 0 && isCallActive) {
                    const audioBlob = new Blob(vadChunks, { type: mimeType });
                    vadChunks = [];
                    dispatchCallTurn(audioBlob);
                }
            };

            isCallActive = true;
            isUserSpeaking = false;
            isTurnInFlight = false;
            vadChunks = [];

            // Start call duration stopwatch
            callStartTime = performance.now();
            if (callDurationTimer) callDurationTimer.textContent = "00:00";
            if (callTimerInterval) clearInterval(callTimerInterval);
            callTimerInterval = setInterval(() => {
                const totalSec = Math.floor((performance.now() - callStartTime) / 1000);
                const m = String(Math.floor(totalSec / 60)).padStart(2, "0");
                const s = String(totalSec % 60).padStart(2, "0");
                if (callDurationTimer) callDurationTimer.textContent = `${m}:${s}`;
            }, 1000);

            // Update UI
            if (callHudBanner) callHudBanner.style.display = "flex";
            if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
            if (bargeInTag) bargeInTag.textContent = "⚡ Barge-In Ready";
            setOrbState("listening", "Call Active • Speak now...");

            mainRecordBtn.classList.remove("btn-call-start");
            mainRecordBtn.classList.add("btn-end-call");
            recordBtnLabel.textContent = "End call";
            updateCallIcon(true);

            // Connect canvas visualizer to live vadStream
            initVisualizer(vadStream);

            // Start VAD Loop
            runVadLoop();

        } catch (err) {
            console.error("Call mic access error:", err);
            alert("Microphone access is required for the Voice Call: " + err.message);
            stopVoiceCall();
        }
    }

    function stopVoiceCall() {
        isCallActive = false;
        isUserSpeaking = false;
        isTurnInFlight = false;

        if (callTimerInterval) {
            clearInterval(callTimerInterval);
            callTimerInterval = null;
        }

        if (vadAnimFrame) {
            cancelAnimationFrame(vadAnimFrame);
            vadAnimFrame = null;
        }

        if (vadRecorder && vadRecorder.state !== "inactive") {
            try { vadRecorder.stop(); } catch (_) { }
        }
        vadRecorder = null;
        vadChunks = [];

        if (vadStream) {
            vadStream.getTracks().forEach(t => t.stop());
            vadStream = null;
        }

        if (vadAudioCtx && vadAudioCtx.state !== "closed") {
            try { vadAudioCtx.close(); } catch (_) { }
            vadAudioCtx = null;
        }
        vadAnalyser = null;

        stopVisualizer();

        if (currentlyPlayingAudio) {
            currentlyPlayingAudio.pause();
        }

        // Reset UI
        if (callHudBanner) callHudBanner.style.display = "none";
        mainRecordBtn.classList.remove("btn-end-call");
        mainRecordBtn.classList.add("btn-call-start");
        recordBtnLabel.textContent = "Start Voice Call";
        updateCallIcon(false);
        setOrbState("", "Call Ended • Ready to Call");
    }

    function runVadLoop() {
        if (!vadAnalyser) return;
        const timeData = new Uint8Array(vadAnalyser.fftSize);

        function vadTick() {
            if (!isCallActive || !vadAnalyser) return;

            vadAnalyser.getByteTimeDomainData(timeData);
            let sum = 0;
            for (let i = 0; i < timeData.length; i++) {
                const sample = (timeData[i] - 128) / 128;
                sum += sample * sample;
            }
            const rms = Math.sqrt(sum / timeData.length);
            const now = performance.now();

            const SPEECH_RMS_THRESHOLD = 0.026;
            const SILENCE_TIMEOUT_MS = 1350;
            const MIN_SPEECH_DURATION_MS = 500;

            const isAssistantSpeaking = currentlyPlayingAudio && !currentlyPlayingAudio.paused;

            if (rms > SPEECH_RMS_THRESHOLD) {
                consecutiveSpeechFrames++;

                // --- Barge-In (Interruption Handling) ---
                if (isAssistantSpeaking && consecutiveSpeechFrames >= 3) {
                    console.log("⚡ Barge-in: User interrupted assistant playback!");
                    currentlyPlayingAudio.pause();
                    currentlyPlayingAudio.currentTime = 0;
                    if (bargeInTag) {
                        bargeInTag.textContent = "⚡ Interrupted!";
                        setTimeout(() => {
                            if (bargeInTag) bargeInTag.textContent = "⚡ Barge-In Ready";
                        }, 2000);
                    }
                }

                if (consecutiveSpeechFrames >= 3 && !isUserSpeaking && !isTurnInFlight) {
                    // Speech Start
                    isUserSpeaking = true;
                    speechStartTimestamp = now;
                    lastSoundTimestamp = now;
                    vadChunks = [];

                    if (vadRecorder && vadRecorder.state === "inactive") {
                        vadRecorder.start(100);
                    }

                    setOrbState("listening", "Listening to you speak...");
                    if (callVadStatus) callVadStatus.textContent = "You are speaking...";
                } else if (isUserSpeaking) {
                    lastSoundTimestamp = now;
                }
            } else {
                consecutiveSpeechFrames = 0;
                if (isUserSpeaking) {
                    const silenceDuration = now - lastSoundTimestamp;
                    const speechDuration = now - speechStartTimestamp;

                    if (silenceDuration > SILENCE_TIMEOUT_MS) {
                        if (speechDuration >= MIN_SPEECH_DURATION_MS) {
                            // Speech End -> Finalize turn
                            isUserSpeaking = false;
                            if (callVadStatus) callVadStatus.textContent = "Amirlan thinking (Gemini)...";
                            setOrbState("analyzing", "Reasoning & Searching Knowledge...");
                            if (vadRecorder && vadRecorder.state === "recording") {
                                vadRecorder.stop();
                            }
                        } else {
                            // Discard false trigger / background tap
                            isUserSpeaking = false;
                            vadChunks = [];
                            if (vadRecorder && vadRecorder.state === "recording") {
                                vadRecorder.stop();
                            }
                            setOrbState("listening", "Call Active • Speak anytime");
                            if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
                        }
                    }
                }
            }

            vadAnimFrame = requestAnimationFrame(vadTick);
        }

        vadAnimFrame = requestAnimationFrame(vadTick);
    }

    async function dispatchCallTurn(audioBlob) {
        if (!isCallActive) return;
        isTurnInFlight = true;

        startPipelineTimers();
        simulateLiveStepTransitions();

        const formData = new FormData();
        const extension = audioBlob.type.includes("mp4") ? "mp4" : "webm";
        formData.append("audio", audioBlob, `call_input.${extension}`);
        formData.append("voice", getSelectedVoice());
        formData.append("is_call_mode", "true");

        try {
            const res = await fetch("/api/voice/process", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            isTurnInFlight = false;

            if (!isCallActive) return;

            handlePipelineResponse(data);

            // Wait for audio playback to finish, then automatically re-arm for next turn!
            const allAudios = chatFeed.querySelectorAll("audio");
            if (allAudios.length > 0) {
                const latestAudio = allAudios[allAudios.length - 1];
                const rearmCall = () => {
                    if (isCallActive) {
                        console.log("Assistant turn complete -> Call re-armed for user voice.");
                        setOrbState("listening", "Call Active • Listening (Speak anytime)");
                        if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
                        recordBtnLabel.textContent = "End call";
                        mainRecordBtn.classList.remove("btn-call-start");
                        mainRecordBtn.classList.add("btn-end-call");
                        updateCallIcon(true);
                    }
                };
                latestAudio.addEventListener("ended", rearmCall, { once: true });
                latestAudio.addEventListener("error", rearmCall, { once: true });
            } else {
                setOrbState("listening", "Call Active • Listening (Speak anytime)");
                if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
                recordBtnLabel.textContent = "End call";
                mainRecordBtn.classList.remove("btn-call-start");
                mainRecordBtn.classList.add("btn-end-call");
                updateCallIcon(true);
            }

        } catch (err) {
            console.error("Call turn network error:", err);
            isTurnInFlight = false;
            stopPipelineTimers();
            if (isCallActive) {
                appendAssistantMessage("Network error during call: " + err.message);
                setOrbState("listening", "Call Active • Listening (Speak anytime)");
                if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
                recordBtnLabel.textContent = "End call";
                mainRecordBtn.classList.remove("btn-call-start");
                mainRecordBtn.classList.add("btn-end-call");
                updateCallIcon(true);
            }
        }
    }

    // =========================================================================
    // 6. PIPELINE API EXECUTION
    // =========================================================================
    async function sendAudioBlobToPipeline(audioBlob) {
        startPipelineTimers();

        const formData = new FormData();
        const extension = audioBlob.type.includes("mp4") ? "mp4" : "webm";
        formData.append("audio", audioBlob, `speech_input.${extension}`);
        formData.append("voice", getSelectedVoice());

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
            if (isCallActive) {
                setOrbState("listening", "Call Active • Speak anytime");
                if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
                recordBtnLabel.textContent = "End call";
                mainRecordBtn.classList.remove("btn-call-start");
                mainRecordBtn.classList.add("btn-end-call");
                updateCallIcon(true);
            } else if (currentInputMode === "call") {
                recordBtnLabel.textContent = "Start Voice Call";
                mainRecordBtn.classList.remove("btn-end-call");
                mainRecordBtn.classList.add("btn-call-start");
                updateCallIcon(false);
                setOrbState("", "Ready to Call");
            } else if (currentInputMode === "desktop") {
                recordBtnLabel.textContent = "Hardware Mic (5s)";
                updateMicIcon();
                setOrbState("", "Ready to Listen");
            } else if (currentInputMode === "upload") {
                recordBtnLabel.textContent = "Select Audio File";
                updateUploadIcon();
                setOrbState("", "Ready to Listen");
            } else if (currentInputMode === "text") {
                recordBtnLabel.textContent = "Send Text Query";
                updateSendIcon();
                setOrbState("", "Ready to Listen");
            }
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
                body: JSON.stringify({ duration: 5, voice: getSelectedVoice() })
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
        setInputMode("upload");
        setOrbState("analyzing", `Processing audio: ${file.name}...`);
        startPipelineTimers();

        const formData = new FormData();
        formData.append("audio", file, file.name);
        formData.append("voice", getSelectedVoice());

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
                body: JSON.stringify({ text: text.trim(), voice: getSelectedVoice() })
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
        if (stepSimulationInterval) {
            clearInterval(stepSimulationInterval);
            stepSimulationInterval = null;
        }
        let step = startStep;
        stepSimulationInterval = setInterval(() => {
            if (!pipelineStartTime) {
                if (stepSimulationInterval) {
                    clearInterval(stepSimulationInterval);
                    stepSimulationInterval = null;
                }
                return;
            }
            if (step < 5) {
                step++;
                startStepLiveTimer(step);
            } else {
                if (stepSimulationInterval) {
                    clearInterval(stepSimulationInterval);
                    stepSimulationInterval = null;
                }
            }
        }, 2200);
    }

    // =========================================================================
    // 7. PIPELINE RESPONSE HANDLER
    // =========================================================================
    function handlePipelineResponse(data) {
        if (isCallActive) {
            recordBtnLabel.textContent = "End call";
            mainRecordBtn.classList.remove("btn-call-start");
            mainRecordBtn.classList.add("btn-end-call");
            updateCallIcon(true);
        } else if (currentInputMode === "call") {
            recordBtnLabel.textContent = "Start Voice Call";
            mainRecordBtn.classList.remove("btn-end-call");
            mainRecordBtn.classList.add("btn-call-start");
            updateCallIcon(false);
        } else if (currentInputMode === "desktop") {
            recordBtnLabel.textContent = "Hardware Mic (5s)";
            updateMicIcon();
        } else if (currentInputMode === "upload") {
            recordBtnLabel.textContent = "Select Audio File";
            updateUploadIcon();
        } else if (currentInputMode === "text") {
            recordBtnLabel.textContent = "Send Text Query";
            updateSendIcon();
        }

        if (!data.success) {
            stopPipelineTimers();
            appendAssistantMessage("Error: " + (data.error || "Unknown pipeline error"));
            if (isCallActive) {
                setOrbState("listening", "Call Active • Speak anytime");
                if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
            } else {
                setOrbState("", "Ready to Listen");
            }
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
            if (isCallActive) {
                setOrbState("listening", "Call Active • Speak anytime");
                if (callVadStatus) callVadStatus.textContent = "Listening • Speak anytime";
            } else if (!data.is_escalated) {
                setOrbState("", "Ready to Listen");
            }
        }

        // 7. Auto-refresh Developer Metrics & History in background
        if (typeof refreshDevMetrics === "function") {
            refreshDevMetrics();
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
                    <span class="grounded-tag">Verified Grounding</span>
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
                    setOrbState("speaking", `Speaking • ${getSelectedVoice()} voice`);
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
                if (isCallActive && callVadStatus) {
                    callVadStatus.textContent = "Amirlan speaking...";
                }
                const btn = latestAudio.parentElement.querySelector(".bubble-play-btn");
                if (btn) {
                    btn.querySelector(".icon-play").style.display = "none";
                    btn.querySelector(".icon-pause").style.display = "block";
                }
            }).catch(e => {
                console.warn("Autoplay deferred:", e);
                setOrbState("", "Response Ready • Click Play in Chat");
                if (isCallActive && callVadStatus) {
                    callVadStatus.textContent = "Listening • Speak anytime";
                }
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
            silent.play().catch(() => { });
        } catch (_) { }
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.innerText = text;
        return div.innerHTML;
    }

    // =========================================================================
    // 9. INTELLIGENCE DRAWER (NAVBAR TAB CONTROLLER)
    // =========================================================================
    function updateIntelligenceDrawer(data) {
        if (intelIntentVal && data.intent) {
            intelIntentVal.textContent = `${data.intent.topic} (${data.intent.category})`;
        }
        if (intelDecisionVal && data.decision) {
            intelDecisionVal.textContent = `${data.decision.decision.toUpperCase()} - ${data.decision.reason}`;
        }
        if (data.retrieval) {
            const pageUri = data.retrieval.source_url || "https://www.khanbank.com";
            if (intelUriLink) {
                intelUriLink.href = pageUri;
                intelUriLink.textContent = pageUri;
                intelUriLink.title = `Access Khan Bank Page: ${pageUri}`;
            }
            if (intelUriBadge) {
                intelUriBadge.style.display = "inline-flex";
            }
            if (intelSnippetVal) {
                intelSnippetVal.textContent = data.retrieval.search_results || data.retrieval.verified_context;
            }
        }
        // Pulse badge on navbar tab
        if (intelBadge) {
            intelBadge.style.display = "inline-block";
        }
    }

    if (intelNavTabBtn && intelCard) {
        intelNavTabBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const isClosed = intelCard.style.display === "none";
            intelCard.style.display = isClosed ? "block" : "none";
            intelNavTabBtn.classList.toggle("active", isClosed);
            if (isClosed && devCard) {
                devCard.style.display = "none";
                if (devNavTabBtn) devNavTabBtn.classList.remove("active");
            }
            if (isClosed && intelBadge) {
                intelBadge.style.display = "none";
            }
        });
    }

    if (intelCloseBtn && intelCard) {
        intelCloseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            intelCard.style.display = "none";
            if (intelNavTabBtn) intelNavTabBtn.classList.remove("active");
        });
    }

    // =========================================================================
    // 10. DEVELOPER MODE CONSOLE & TELEMETRY CONTROLLER
    // =========================================================================
    const devCard = document.getElementById("devCard");
    const devNavTabBtn = document.getElementById("devNavTabBtn");
    const devCloseBtn = document.getElementById("devCloseBtn");
    const devRefreshBtn = document.getElementById("devRefreshBtn");

    // Subtabs
    const devTabBtns = document.querySelectorAll(".dev-tab-btn");
    const devTabPanes = {
        cache: document.getElementById("paneDevCache"),
        prompt: document.getElementById("paneDevPrompt"),
        metrics: document.getElementById("paneDevMetrics"),
        monitoring: document.getElementById("paneDevMonitoring")
    };

    // Cache Controls
    const cacheToggle = document.getElementById("cacheToggle");
    const cacheToggleLabel = document.getElementById("cacheToggleLabel");
    const cacheFileCount = document.getElementById("cacheFileCount");
    const cacheDiskSize = document.getElementById("cacheDiskSize");
    const cacheHitRateBadge = document.getElementById("cacheHitRateBadge");
    const clearCacheBtn = document.getElementById("clearCacheBtn");
    const cacheStatusNote = document.getElementById("cacheStatusNote");

    // Prompt Tuning Controls
    const devPromptTextarea = document.getElementById("devPromptTextarea");
    const savePromptBtn = document.getElementById("savePromptBtn");
    const resetPromptBtn = document.getElementById("resetPromptBtn");
    const promptSaveNote = document.getElementById("promptSaveNote");

    // Metrics & History Controls
    const metricTotalReqs = document.getElementById("metricTotalReqs");
    const metricAvgTotal = document.getElementById("metricAvgTotal");
    const metricAvgTTS = document.getElementById("metricAvgTTS");
    const metricCacheHit = document.getElementById("metricCacheHit");
    const metricEscRate = document.getElementById("metricEscRate");
    const devHistoryTableBody = document.getElementById("devHistoryTableBody");

    // Monitoring & Evaluation Controls
    const devLogTerminal = document.getElementById("devLogTerminal");
    const clearLogsBtn = document.getElementById("clearLogsBtn");
    const runEvaluationBtn = document.getElementById("runEvaluationBtn");
    const evalBtnText = document.getElementById("evalBtnText");
    const evalScorecard = document.getElementById("evalScorecard");
    const evalPlaceholder = document.getElementById("evalPlaceholder");
    const evalScoreOverall = document.getElementById("evalScoreOverall");
    const evalScorePassed = document.getElementById("evalScorePassed");
    const evalScoreLatency = document.getElementById("evalScoreLatency");
    const evalCasesList = document.getElementById("evalCasesList");

    // Toggle Dev Card
    if (devNavTabBtn && devCard) {
        devNavTabBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const isClosed = devCard.style.display === "none";
            devCard.style.display = isClosed ? "block" : "none";
            devNavTabBtn.classList.toggle("active", isClosed);
            if (isClosed && intelCard) {
                intelCard.style.display = "none";
                if (intelNavTabBtn) intelNavTabBtn.classList.remove("active");
            }
            if (isClosed) {
                fetchDevConfig();
                refreshDevMetrics();
            }
        });
    }

    if (devCloseBtn && devCard) {
        devCloseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            devCard.style.display = "none";
            if (devNavTabBtn) devNavTabBtn.classList.remove("active");
        });
    }

    if (devRefreshBtn) {
        devRefreshBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fetchDevConfig();
            refreshDevMetrics();
        });
    }

    // Subtab switching
    devTabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabKey = btn.dataset.devtab;
            devTabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            Object.entries(devTabPanes).forEach(([k, pane]) => {
                if (pane) pane.style.display = (k === tabKey) ? "block" : "none";
            });
        });
    });

    // 1. Fetch & Update Dev Config
    function fetchDevConfig() {
        fetch("/api/dev/config")
            .then(r => r.json())
            .then(data => {
                if (!data.success) return;
                if (cacheToggle) {
                    cacheToggle.checked = !!data.caching_enabled;
                    updateCacheToggleLabel(data.caching_enabled);
                }
                if (devPromptTextarea && (!devPromptTextarea.value || document.activeElement !== devPromptTextarea)) {
                    devPromptTextarea.value = data.system_prompt || "";
                }
                if (data.cache_stats) {
                    if (cacheFileCount) cacheFileCount.textContent = data.cache_stats.count;
                    if (cacheDiskSize) cacheDiskSize.textContent = `${data.cache_stats.size_kb} KB`;
                }
            })
            .catch(e => console.warn("Failed to load dev config:", e));
    }

    function updateCacheToggleLabel(enabled) {
        if (!cacheToggleLabel) return;
        cacheToggleLabel.innerHTML = enabled 
            ? `Audio Caching: <strong style="color:var(--color-orange)">ENABLED</strong>` 
            : `Audio Caching: <strong style="color:var(--text-muted)">DISABLED</strong>`;
    }

    // Toggle Caching
    if (cacheToggle) {
        cacheToggle.addEventListener("change", () => {
            const enabled = cacheToggle.checked;
            updateCacheToggleLabel(enabled);
            fetch("/api/dev/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ caching_enabled: enabled })
            })
                .then(r => r.json())
                .then(data => {
                    if (cacheStatusNote) {
                        cacheStatusNote.textContent = enabled ? "Cache enabled" : "Cache bypassed";
                        setTimeout(() => { if (cacheStatusNote) cacheStatusNote.textContent = ""; }, 3000);
                    }
                    refreshDevMetrics();
                })
                .catch(e => console.warn("Cache toggle error:", e));
        });
    }

    // Clear Audio Cache
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener("click", () => {
            clearCacheBtn.disabled = true;
            fetch("/api/dev/cache/clear", { method: "POST" })
                .then(r => r.json())
                .then(data => {
                    clearCacheBtn.disabled = false;
                    if (data.success) {
                        if (cacheFileCount) cacheFileCount.textContent = "0";
                        if (cacheDiskSize) cacheDiskSize.textContent = "0.0 KB";
                        if (cacheStatusNote) {
                            cacheStatusNote.textContent = `Purged ${data.cleared_count} cached audio files!`;
                            setTimeout(() => { if (cacheStatusNote) cacheStatusNote.textContent = ""; }, 3500);
                        }
                    }
                })
                .catch(e => {
                    clearCacheBtn.disabled = false;
                    console.warn("Clear cache error:", e);
                });
        });
    }

    // Save System Prompt
    if (savePromptBtn && devPromptTextarea) {
        savePromptBtn.addEventListener("click", () => {
            const promptVal = devPromptTextarea.value.trim();
            if (!promptVal) return;
            savePromptBtn.disabled = true;
            fetch("/api/dev/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ system_prompt: promptVal })
            })
                .then(r => r.json())
                .then(data => {
                    savePromptBtn.disabled = false;
                    if (promptSaveNote) {
                        promptSaveNote.textContent = "✓ Prompt saved! Active on next inquiry.";
                        setTimeout(() => { if (promptSaveNote) promptSaveNote.textContent = ""; }, 4000);
                    }
                })
                .catch(e => {
                    savePromptBtn.disabled = false;
                    console.warn("Save prompt error:", e);
                });
        });
    }

    // Reset System Prompt
    if (resetPromptBtn && devPromptTextarea) {
        resetPromptBtn.addEventListener("click", () => {
            if (!confirm("Reset the agent prompt back to default Khan Bank template?")) return;
            fetch("/api/dev/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ reset_prompt: true })
            })
                .then(r => r.json())
                .then(data => {
                    if (data.system_prompt) {
                        devPromptTextarea.value = data.system_prompt;
                    }
                    if (promptSaveNote) {
                        promptSaveNote.textContent = "Reset to default prompt.";
                        setTimeout(() => { if (promptSaveNote) promptSaveNote.textContent = ""; }, 3000);
                    }
                })
                .catch(e => console.warn("Reset prompt error:", e));
        });
    }

    // 2. Refresh Metrics, Duration History, & Logs
    function refreshDevMetrics() {
        fetch("/api/dev/metrics")
            .then(r => r.json())
            .then(data => {
                if (!data.success) return;

                // Summary Stats
                const s = data.summary || {};
                if (metricTotalReqs) metricTotalReqs.textContent = s.total_requests || 0;
                if (metricAvgTotal) metricAvgTotal.textContent = `${s.avg_total_duration || 0}s`;
                if (metricAvgTTS) metricAvgTTS.textContent = `${s.avg_tts_duration || 0}s`;
                if (metricCacheHit) metricCacheHit.textContent = `${s.cache_hit_rate || 0}%`;
                if (cacheHitRateBadge) cacheHitRateBadge.textContent = `${s.cache_hit_rate || 0}%`;
                if (metricEscRate) metricEscRate.textContent = `${s.escalation_rate || 0}%`;

                // History Table
                if (devHistoryTableBody && Array.isArray(data.history)) {
                    if (data.history.length === 0) {
                        devHistoryTableBody.innerHTML = `<tr><td colspan="9" class="text-center text-muted" style="text-align:center;padding:16px;">No requests recorded yet. Make a voice or text query to view live latency breakdowns.</td></tr>`;
                    } else {
                        devHistoryTableBody.innerHTML = data.history.map(item => {
                            const stages = item.stages || {};
                            const cachedBadge = item.is_cached 
                                ? `<span style="color:#10b981;font-weight:700;">CACHED</span>` 
                                : `<span style="color:#6b7280;">LIVE</span>`;
                            const escBadge = item.is_escalated 
                                ? `<span style="color:#ff414e;font-weight:700;">ESCALATED</span>` 
                                : `<span style="color:#3b82f6;">RESOLVED</span>`;
                            const qSnippet = (item.query || "-").slice(0, 36) + (item.query && item.query.length > 36 ? "..." : "");

                            return `
                                <tr>
                                    <td class="font-mono text-muted">${item.timestamp || "-"}</td>
                                    <td title="${item.query || ''}"><strong>${qSnippet}</strong></td>
                                    <td class="font-mono">${stages.stt || 0}s</td>
                                    <td class="font-mono">${stages.intent || 0}s</td>
                                    <td class="font-mono">${stages.search || 0}s</td>
                                    <td class="font-mono">${stages.decision || 0}s</td>
                                    <td class="font-mono" style="color:var(--color-orange);font-weight:700;">${stages.tts || 0}s</td>
                                    <td class="font-mono" style="font-weight:800;">${item.total_duration || 0}s</td>
                                    <td>${cachedBadge} / ${escBadge}</td>
                                </tr>
                            `;
                        }).join("");
                    }
                }

                // Log Terminal
                if (devLogTerminal && Array.isArray(data.logs)) {
                    devLogTerminal.innerHTML = data.logs.map(log => {
                        let levelClass = "log-info";
                        if (log.level === "WARN") levelClass = "log-warn";
                        if (log.level === "ERROR") levelClass = "log-error";
                        return `
                            <div class="log-entry ${levelClass}">
                                <span class="log-ts">[${log.timestamp}]</span>
                                <span class="log-stage">[${log.stage}]</span>
                                <span class="log-msg">${log.message}</span>
                            </div>
                        `;
                    }).join("");
                    devLogTerminal.scrollTop = devLogTerminal.scrollHeight;
                }

                // Latest Evaluation Scorecard
                if (data.latest_evaluations) {
                    renderEvaluationScorecard(data.latest_evaluations);
                }
            })
            .catch(e => console.warn("Failed to refresh metrics:", e));
    }

    if (clearLogsBtn && devLogTerminal) {
        clearLogsBtn.addEventListener("click", () => {
            devLogTerminal.innerHTML = `<div class="log-entry log-info"><span class="log-ts">[Cleared]</span><span class="log-stage">[System]</span><span class="log-msg">Log display cleared.</span></div>`;
        });
    }

    // 3. Automated Evaluation Suite Runner
    if (runEvaluationBtn) {
        runEvaluationBtn.addEventListener("click", () => {
            runEvaluationBtn.disabled = true;
            if (evalBtnText) evalBtnText.textContent = "Benchmarking 4 Test Cases...";

            fetch("/api/dev/evaluate", { method: "POST" })
                .then(r => r.json())
                .then(data => {
                    runEvaluationBtn.disabled = false;
                    if (evalBtnText) evalBtnText.textContent = "Run Evaluation Suite";
                    if (data.success && data.evaluation) {
                        renderEvaluationScorecard(data.evaluation);
                    }
                })
                .catch(e => {
                    runEvaluationBtn.disabled = false;
                    if (evalBtnText) evalBtnText.textContent = "Run Evaluation Suite";
                    console.warn("Evaluation error:", e);
                });
        });
    }

    function renderEvaluationScorecard(ev) {
        if (!evalScorecard) return;
        if (evalPlaceholder) evalPlaceholder.style.display = "none";
        evalScorecard.style.display = "block";

        if (evalScoreOverall) evalScoreOverall.textContent = `${ev.overall_score || 0}%`;
        if (evalScorePassed) evalScorePassed.textContent = `${ev.passed_cases || 0} / ${ev.total_cases || 0}`;
        if (evalScoreLatency) evalScoreLatency.textContent = `${ev.avg_latency_sec || 0}s`;

        if (evalCasesList && Array.isArray(ev.cases)) {
            evalCasesList.innerHTML = ev.cases.map(c => {
                const badgeClass = c.passed ? "badge-pass" : "badge-fail";
                const badgeText = c.passed ? "PASS" : "FAIL";
                return `
                    <div class="eval-case-card">
                        <div class="eval-case-left">
                            <span class="eval-case-title">${c.id}: ${c.name}</span>
                            <span class="eval-case-query">"${c.query}"</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span class="font-mono text-muted">${c.duration || 0}s</span>
                            <span class="eval-case-badge ${badgeClass}">${badgeText}</span>
                        </div>
                    </div>
                `;
            }).join("");
        }
    }

    // Dismiss drawer when clicking outside
    document.addEventListener("click", (e) => {
        if (intelCard && intelCard.style.display !== "none") {
            if (!intelCard.contains(e.target) && !intelNavTabBtn.contains(e.target)) {
                intelCard.style.display = "none";
                if (intelNavTabBtn) intelNavTabBtn.classList.remove("active");
            }
        }
        if (devCard && devCard.style.display !== "none") {
            if (!devCard.contains(e.target) && !devNavTabBtn.contains(e.target)) {
                devCard.style.display = "none";
                if (devNavTabBtn) devNavTabBtn.classList.remove("active");
            }
        }
    });

    // =========================================================================
    // 11. CHAT CLEANUP & SESSION DISCONNECT HANDLERS
    // =========================================================================
    if (clearChatBtn && chatFeed) {
        clearChatBtn.addEventListener("click", () => {
            chatFeed.innerHTML = "";
            appendAssistantMessage("Dialogue history cleared. Ask a new inquiry to begin.");
            // Clean up input & output audio files on server
            fetch("/api/chat/clear", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ clear_cache: true })
            })
                .then(r => r.json())
                .then(res => {
                    console.log("Audio cleaned up:", res);
                    refreshDevMetrics();
                })
                .catch(e => console.warn("Cleanup error:", e));
        });
    }

    // Clean up audio files automatically when user disconnects or navigates away
    window.addEventListener("beforeunload", () => {
        navigator.sendBeacon("/api/chat/clear", new Blob([JSON.stringify({ clear_cache: true })], { type: "application/json" }));
    });
    window.addEventListener("pagehide", () => {
        navigator.sendBeacon("/api/chat/clear", new Blob([JSON.stringify({ clear_cache: true })], { type: "application/json" }));
    });

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
            if (isCallActive) {
                stopVoiceCall();
                return;
            }
            if (currentInputMode === "call") {
                startVoiceCall();
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

    if (callHudEndBtn) {
        callHudEndBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            stopVoiceCall();
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && isCallActive) {
            stopVoiceCall();
        }
    });

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

    // Initial metrics background sync
    fetchDevConfig();

    // Default to Live Voice Call mode
    setInputMode("call");

    console.log("⚡ Amirlan Voice AI Agent UI initialized (Continuous Voice Call enabled).");
});

