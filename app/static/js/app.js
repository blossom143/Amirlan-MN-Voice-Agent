/**
 * KHAN BANK AMIRLAN LUXURY AI VOICEBOT - CLIENT CONTROLLER
 * ==========================================================
 * Manages browser MediaRecorder, real-time AudioContext visualizer,
 * 6-stage workflow stepper animations, audio playback, and escalation.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const orbContainer = document.getElementById("orbContainer");
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

    // Dialogue & Escalation
    const dialogueContainer = document.getElementById("dialogueContainer");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const escalationCard = document.getElementById("escalationCard");
    const audioPlayerCard = document.getElementById("audioPlayerCard");
    const nativeAudioElement = document.getElementById("nativeAudioElement");
    const customPlayPauseBtn = document.getElementById("customPlayPauseBtn");
    const audioSeeker = document.getElementById("audioSeeker");
    const currentTimeLabel = document.getElementById("currentTimeLabel");
    const durationLabel = document.getElementById("durationLabel");
    const downloadAudioLink = document.getElementById("downloadAudioLink");
    const audioPlayingStatus = document.getElementById("audioPlayingStatus");

    // Stepper
    const totalTimeBadge = document.getElementById("totalTimeBadge");
    const intelToggleBtn = document.getElementById("intelToggleBtn");
    const intelligenceCard = document.getElementById("intelligenceCard");
    const intelDrawer = document.getElementById("intelDrawer");

    // --- State Variables ---
    let currentInputMode = "browser"; // 'browser' | 'desktop' | 'upload' | 'text'
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let recordingSeconds = 0;

    // Web Audio API for Canvas Visualizer
    let audioCtx = null;
    let analyserNode = null;
    let micStream = null;
    let animationFrameId = null;
    const canvasCtx = waveformCanvas ? waveformCanvas.getContext("2d") : null;

    // =========================================================================
    // 1. INPUT MODE SELECTION
    // =========================================================================
    const modeTabs = [tabBrowserMic, tabDesktopMic, tabUpload, tabText];

    function setInputMode(mode) {
        currentInputMode = mode;
        modeTabs.forEach(tab => {
            if (tab) tab.classList.toggle("active", tab.dataset.mode === mode);
        });

        // Toggle text input bar visibility
        if (textInputWrapper) {
            textInputWrapper.style.display = (mode === "text") ? "flex" : "none";
        }

        // Update Record Button Label
        if (mode === "browser") {
            recordBtnLabel.textContent = isRecording ? "БИЧЛЭГ ЗОГСООХ" : "ДУУГААРАА АСУУХ";
        } else if (mode === "desktop") {
            recordBtnLabel.textContent = "КОМПЬЮТЕРЫН МИКРОФОН (5С)";
        } else if (mode === "upload") {
            recordBtnLabel.textContent = "АУДИО ФАЙЛ СОНГОХ";
        } else if (mode === "text") {
            recordBtnLabel.textContent = "АСУУЛТ БИЧИХ";
        }
    }

    if (tabBrowserMic) tabBrowserMic.addEventListener("click", () => setInputMode("browser"));
    if (tabDesktopMic) tabDesktopMic.addEventListener("click", () => setInputMode("desktop"));
    if (tabUpload) tabUpload.addEventListener("click", () => {
        setInputMode("upload");
        if (audioFileInput) audioFileInput.click();
    });
    if (tabText) tabText.addEventListener("click", () => setInputMode("text"));

    // File Upload Handler
    if (audioFileInput) {
        audioFileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                processUploadedAudioFile(file);
            }
        });
    }

    // =========================================================================
    // 2. ORB STATE CONTROLLER
    // =========================================================================
    function setOrbState(state, statusText) {
        if (!orbContainer) return;
        orbContainer.classList.remove("state-listening", "state-analyzing", "state-speaking", "state-escalated");
        if (state) {
            orbContainer.classList.add(`state-${state}`);
        }
        if (agentStateText && statusText) {
            agentStateText.textContent = statusText;
        }
    }

    // =========================================================================
    // 3. AUDIO VISUALIZER (Web Audio API -> Canvas)
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
            console.warn("AudioContext visualizer init error:", err);
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

        // Luxury Gold & Emerald Gradient Bars
        const barWidth = (width / bufferLength) * 2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * (height - 10);

            const gradient = canvasCtx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, "rgba(0, 77, 37, 0.4)");
            gradient.addColorStop(0.5, "#D4AF37");
            gradient.addColorStop(1, "#FDF2C2");

            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(x, (height - barHeight) / 2, barWidth - 2, barHeight || 3);
            x += barWidth;
        }

        animationFrameId = requestAnimationFrame(drawWaveform);
    }

    function stopVisualizer() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (canvasCtx) {
            canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        }
    }

    // =========================================================================
    // 4. BROWSER MICROPHONE RECORDING (Phone & PC)
    // =========================================================================
    async function startBrowserRecording() {
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];

            // Detect supported MIME types (Safari iOS prefers mp4, Chrome prefers webm)
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

            mediaRecorder.start(250);
            isRecording = true;

            // UI Updates
            setOrbState("listening", "Сонсож байна • Үгээ хэлнэ үү...");
            mainRecordBtn.classList.add("recording");
            recordBtnLabel.textContent = "БИЧЛЭГ ЗОГСООХ";

            // Visualizer & Timer
            initVisualizer(micStream);
            startTimer();

        } catch (err) {
            console.error("Microphone access error:", err);
            alert("Микрофон ашиглах зөвшөөрөл шаардлагатай байна.");
            isRecording = false;
        }
    }

    function stopBrowserRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            stopTimer();
            setOrbState("analyzing", "Дүн шинжилгээ хийж байна (Chirp 3 • Gemini 2.5)...");
            mainRecordBtn.classList.remove("recording");
            recordBtnLabel.textContent = "БОЛОВСРУУЛЖ БАЙНА...";
        }
    }

    function startTimer() {
        recordingSeconds = 0;
        if (recordingTimer) recordingTimer.style.display = "inline-flex";
        timerInterval = setInterval(() => {
            recordingSeconds++;
            const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
            const secs = String(recordingSeconds % 60).padStart(2, "0");
            if (timerDigits) timerDigits.textContent = `${mins}:${secs}`;
            // Max recording duration: 40 seconds
            if (recordingSeconds >= 40) {
                stopBrowserRecording();
            }
        }, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        if (recordingTimer) recordingTimer.style.display = "none";
    }

    // =========================================================================
    // 5. SERVER-SIDE DESKTOP RECORDING (sounddevice)
    // =========================================================================
    async function triggerDesktopRecording() {
        setOrbState("listening", "Компьютерын микрофон бичиж байна (5 секунд)...");
        resetStepper();
        animateStepActive(1);

        try {
            const res = await fetch("/api/voice/record-desktop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ duration: 5 })
            });
            const data = await res.json();
            handlePipelineResponse(data);
        } catch (err) {
            console.error("Desktop record error:", err);
            appendChatMessage("user", "[Компьютерын микрофон]");
            appendChatMessage("amirlan", "Алдаа гарлаа: " + err.message);
            setOrbState("", "Бэлэн байна");
        }
    }

    // =========================================================================
    // 6. PIPELINE API INVOCATION
    // =========================================================================
    async function sendAudioBlobToPipeline(audioBlob) {
        resetStepper();
        animateStepActive(1);

        const formData = new FormData();
        const extension = audioBlob.type.includes("mp4") ? "mp4" : "webm";
        formData.append("audio", audioBlob, `speech_recording.${extension}`);

        try {
            const res = await fetch("/api/voice/process", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            handlePipelineResponse(data);
        } catch (err) {
            console.error("Pipeline request error:", err);
            appendChatMessage("user", "[Дуут асуулт]");
            appendChatMessage("amirlan", "Холболтын алдаа гарлаа: " + err.message);
            setOrbState("", "Бэлэн байна");
            recordBtnLabel.textContent = "ДУУГААРАА АСУУХ";
        }
    }

    async function processUploadedAudioFile(file) {
        setInputMode("browser");
        setOrbState("analyzing", `Файл боловсруулж байна: ${file.name}...`);
        resetStepper();
        animateStepActive(1);

        const formData = new FormData();
        formData.append("audio", file, file.name);

        try {
            const res = await fetch("/api/voice/process", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            handlePipelineResponse(data);
        } catch (err) {
            console.error("Uploaded file error:", err);
            appendChatMessage("amirlan", "Алдаа гарлаа: " + err.message);
            setOrbState("", "Бэлэн байна");
        }
    }

    async function sendTextQuery(text) {
        if (!text.trim()) return;
        setOrbState("analyzing", "Шинжилж байна (Gemini 2.5 Flash)...");
        resetStepper();
        animateStepActive(2);

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
            appendChatMessage("user", text);
            appendChatMessage("amirlan", "Алдаа гарлаа: " + err.message);
            setOrbState("", "Бэлэн байна");
        }
    }

    // =========================================================================
    // 7. PIPELINE RESPONSE HANDLER (Stages 1 -> 6)
    // =========================================================================
    function handlePipelineResponse(data) {
        recordBtnLabel.textContent = "ДУУГААРАА АСУУХ";

        if (!data.success) {
            appendChatMessage("amirlan", "Уучлаарай, алдаа гарлаа: " + (data.error || "Үл мэдэгдэх алдаа"));
            setOrbState("", "Бэлэн байна");
            return;
        }

        // 1. Append User transcript and Amirlan response to Chat Feed
        const userText = data.transcript || "[Хоосон яриа]";
        appendChatMessage("user", userText);
        appendChatMessage("amirlan", data.response_text, data.intent);

        // 2. Animate and update 6-Stage Stepper
        if (data.stage_logs) {
            updateStepperLogs(data.stage_logs);
        }
        if (totalTimeBadge && data.execution_time_sec) {
            totalTimeBadge.textContent = `${data.execution_time_sec}с`;
        }

        // 3. Update Intelligence Drawer
        updateIntelligenceDrawer(data);

        // 4. Handle Stage 6: Human Escalation
        if (data.is_escalated && data.escalation) {
            setOrbState("escalated", "Мэргэжилтэнд шилжүүллээ • 1800-1917");
            if (escalationCard) {
                escalationCard.style.display = "block";
                escalationCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        } else {
            if (escalationCard) escalationCard.style.display = "none";
        }

        // 5. Handle Audio Playback (Narakeet .m4a)
        if (data.audio_url) {
            playGeneratedAudio(data.audio_url);
        } else {
            if (!data.is_escalated) {
                setOrbState("", "Бэлэн байна");
            }
        }
    }

    // =========================================================================
    // 8. STEPPER UI LOGIC
    // =========================================================================
    function resetStepper() {
        for (let i = 1; i <= 6; i++) {
            const stepEl = document.getElementById(`step${i}`);
            const metaEl = document.getElementById(`step${i}Meta`);
            if (stepEl) {
                stepEl.classList.remove("active", "completed", "escalated");
            }
            if (metaEl && i !== 6) {
                metaEl.textContent = "Хүлээгдэж байна...";
            }
        }
    }

    function animateStepActive(stepNum) {
        for (let i = 1; i < stepNum; i++) {
            const prev = document.getElementById(`step${i}`);
            if (prev) {
                prev.classList.remove("active");
                prev.classList.add("completed");
            }
        }
        const current = document.getElementById(`step${stepNum}`);
        if (current) current.classList.add("active");
    }

    function updateStepperLogs(stageLogs) {
        stageLogs.forEach(log => {
            const stepEl = document.getElementById(`step${log.stage}`);
            const metaEl = document.getElementById(`step${log.stage}Meta`);
            if (stepEl) {
                stepEl.classList.remove("active");
                if (log.status === "triggered") {
                    stepEl.classList.add("escalated");
                } else {
                    stepEl.classList.add("completed");
                }
            }
            if (metaEl) {
                metaEl.textContent = `${log.details} (${log.duration}с)`;
            }
        });
    }

    // =========================================================================
    // 9. AUDIO PLAYER (Narakeet .m4a Playback & Autoplay Unlocking)
    // =========================================================================
    function primeAudioPlayback() {
        if (!nativeAudioElement) return;
        try {
            // Priming audio element on user gesture allows subsequent async autoplay
            nativeAudioElement.play().then(() => {
                nativeAudioElement.pause();
                nativeAudioElement.currentTime = 0;
            }).catch(() => {});
        } catch (_) {}
    }

    function playGeneratedAudio(audioUrl) {
        if (!nativeAudioElement) return;

        nativeAudioElement.src = audioUrl;
        nativeAudioElement.load();
        if (downloadAudioLink) downloadAudioLink.href = audioUrl;
        if (audioPlayerCard) audioPlayerCard.style.display = "block";

        const playPromise = nativeAudioElement.play();
        if (playPromise !== undefined) {
            playPromise.then(() => {
                setOrbState("speaking", "Амирлан ярьж байна • Ганбаатар хоолой");
                if (audioPlayingStatus) audioPlayingStatus.textContent = "Тоглож байна...";
                togglePlayIcon(true);
            }).catch(e => {
                console.warn("Audio autoplay blocked by browser policy:", e);
                if (audioPlayingStatus) {
                    audioPlayingStatus.innerHTML = "<strong style='color: var(--gold-light); cursor: pointer;'>▶ СОНСОХЫН ТУЛД ДАРНА УУ</strong>";
                }
                togglePlayIcon(false);
                setOrbState("", "Хариу бэлэн боллоо • Дээрх товчийг дарж сонсоно уу");
            });
        }

        nativeAudioElement.onended = () => {
            setOrbState("", "Бэлэн байна • Үг хэлэхэд бэлэн");
            if (audioPlayingStatus) audioPlayingStatus.textContent = "Дууслаа";
            togglePlayIcon(false);
        };

        nativeAudioElement.ontimeupdate = () => {
            if (nativeAudioElement.duration) {
                const progress = (nativeAudioElement.currentTime / nativeAudioElement.duration) * 100;
                if (audioSeeker) audioSeeker.value = progress;
                if (currentTimeLabel) currentTimeLabel.textContent = formatTime(nativeAudioElement.currentTime);
                if (durationLabel) durationLabel.textContent = formatTime(nativeAudioElement.duration);
            }
        };
    }

    function togglePlayIcon(isPlaying) {
        const iconPlay = customPlayPauseBtn.querySelector(".icon-play");
        const iconPause = customPlayPauseBtn.querySelector(".icon-pause");
        if (iconPlay && iconPause) {
            iconPlay.style.display = isPlaying ? "none" : "block";
            iconPause.style.display = isPlaying ? "block" : "none";
        }
    }

    if (customPlayPauseBtn && nativeAudioElement) {
        customPlayPauseBtn.addEventListener("click", () => {
            if (nativeAudioElement.paused) {
                nativeAudioElement.play();
                setOrbState("speaking", "Амирлан ярьж байна");
                togglePlayIcon(true);
            } else {
                nativeAudioElement.pause();
                setOrbState("", "Түр зогссон");
                togglePlayIcon(false);
            }
        });
    }

    if (audioSeeker && nativeAudioElement) {
        audioSeeker.addEventListener("input", () => {
            if (nativeAudioElement.duration) {
                nativeAudioElement.currentTime = (audioSeeker.value / 100) * nativeAudioElement.duration;
            }
        });
    }

    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s < 10 ? "0" : ""}${s}`;
    }

    // =========================================================================
    // 10. CHAT FEED & INTELLIGENCE DRAWER
    // =========================================================================
    function appendChatMessage(sender, text, intentData = null) {
        if (!dialogueContainer) return;

        const bubble = document.createElement("div");
        bubble.className = `chat-bubble bubble-${sender}`;

        const isUser = (sender === "user");
        const author = isUser ? "Харилцагч" : "Амирлан • Хаан Банк";
        const timeNow = new Date().toLocaleTimeString("mn-MN", { hour: "2-digit", minute: "2-digit" });

        let footerHtml = "";
        if (!isUser) {
            const intentLabel = intentData ? `🏷️ ${intentData.topic || intentData.category}` : "✓ Албан ёсны эх сурвалж";
            footerHtml = `
                <div class="bubble-footer">
                    <span class="verified-tag">✓ Албан ёсны</span>
                    <span class="intent-tag">${intentLabel}</span>
                </div>
            `;
        }

        bubble.innerHTML = `
            <div class="bubble-header">
                <span class="bubble-author">${author}</span>
                <span class="bubble-time">${timeNow}</span>
            </div>
            <div class="bubble-content">${escapeHtml(text)}</div>
            ${footerHtml}
        `;

        dialogueContainer.appendChild(bubble);
        dialogueContainer.scrollTop = dialogueContainer.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.innerText = text;
        return div.innerHTML;
    }

    if (clearChatBtn && dialogueContainer) {
        clearChatBtn.addEventListener("click", () => {
            dialogueContainer.innerHTML = "";
            appendChatMessage("amirlan", "Түүх цэвэрлэгдлээ. Та шинэ асуултаа асууна уу.");
        });
    }

    function updateIntelligenceDrawer(data) {
        if (!intelligenceCard) return;
        intelligenceCard.style.display = "block";

        const intentVal = document.getElementById("intelIntentVal");
        const decisionVal = document.getElementById("intelDecisionVal");
        const sourceVal = document.getElementById("intelSourceVal");

        if (intentVal && data.intent) {
            intentVal.textContent = `${data.intent.topic} (${data.intent.category})`;
        }
        if (decisionVal && data.decision) {
            decisionVal.textContent = `${data.decision.decision.toUpperCase()} - ${data.decision.reason}`;
        }
        if (sourceVal && data.retrieval) {
            sourceVal.textContent = data.retrieval.search_results || data.retrieval.verified_context;
        }
    }

    if (intelToggleBtn && intelDrawer) {
        intelToggleBtn.addEventListener("click", () => {
            const isClosed = intelDrawer.style.display === "none";
            intelDrawer.style.display = isClosed ? "block" : "none";
            const arrow = intelToggleBtn.querySelector(".intel-arrow");
            if (arrow) arrow.textContent = isClosed ? "▲" : "▼";
        });
    }

    // =========================================================================
    // 11. BUTTON LISTENERS & PRESET CHIPS
    // =========================================================================
    if (mainRecordBtn) {
        mainRecordBtn.addEventListener("click", () => {
            primeAudioPlayback();
            if (currentInputMode === "browser") {
                if (isRecording) {
                    stopBrowserRecording();
                } else {
                    startBrowserRecording();
                }
            } else if (currentInputMode === "desktop") {
                triggerDesktopRecording();
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

    if (orbContainer) {
        orbContainer.addEventListener("click", () => {
            if (mainRecordBtn) mainRecordBtn.click();
        });
    }

    if (sendTextBtn && textQueryInput) {
        sendTextBtn.addEventListener("click", () => {
            primeAudioPlayback();
            sendTextQuery(textQueryInput.value);
        });
        textQueryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                primeAudioPlayback();
                sendTextQuery(textQueryInput.value);
            }
        });
    }

    // Preset Prompt Chips
    document.querySelectorAll(".preset-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            primeAudioPlayback();
            const query = chip.dataset.query;
            if (query) {
                sendTextQuery(query);
            }
        });
    });

    console.log("🏦 Khan Bank Amirlan App client initialized successfully.");
});
