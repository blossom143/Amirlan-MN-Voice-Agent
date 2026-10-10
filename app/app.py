"""
Khan Bank - Amirlan Luxury AI Voicebot Web Application
======================================================
Flask server providing REST APIs and serving the dynamic luxury web UI.
"""

import os
import mimetypes
import uuid
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from voicebot import AmirlanVoicebot, SOUNDDEVICE_AVAILABLE

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB max upload

# Initialize Voicebot Engine
voicebot = AmirlanVoicebot()


@app.route("/")
def index():
    """Renders the Khan Bank themed Amirlan App UI."""
    return render_template(
        "index.html",
        has_sounddevice=SOUNDDEVICE_AVAILABLE,
        project_id=voicebot.project_id
    )


@app.route("/api/voice/process", methods=["POST"])
def process_voice():
    """
    Stage 1 Intake: Receives live audio captured by browser MediaRecorder or uploaded file.
    Saves the input audio into input-audio/ directory so it is preserved until chat clear or disconnect.
    Executes Stages 1-6 end-to-end and returns structured stage logs and response.
    """
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "Аудио файл илгээгдээгүй байна."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"success": False, "error": "Хоосон аудио файл байна."}), 400

    try:
        audio_bytes = audio_file.read()
        if not audio_bytes:
            return jsonify({"success": False, "error": "Аудио өгөгдөл хоосон байна."}), 400

        # Save input audio file to input-audio/ directory
        ext = ".wav"
        if audio_file.filename and "." in audio_file.filename:
            ext = "." + audio_file.filename.rsplit(".", 1)[1]
        input_filename = f"user_input_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        input_filepath = os.path.join(voicebot.input_dir, input_filename)
        try:
            with open(input_filepath, "wb") as f_in:
                f_in.write(audio_bytes)
            voicebot.log_event("Input", f"Saved incoming voice audio: {input_filename} ({len(audio_bytes)} bytes)")
        except Exception as save_err:
            print(f"Warning: Failed to save input audio: {save_err}")

        voice = request.form.get("voice", "Ankhmaa").strip()
        result = voicebot.run_full_pipeline(audio_bytes=audio_bytes, voice=voice)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/voice/record-desktop", methods=["POST"])
def record_desktop():
    """
    Triggers local hardware microphone recording via sounddevice (5 seconds)
    and processes it through Stages 1-6.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Компьютерын sounddevice сан идэвхгүй байна. Хөтчийн микрофон ашиглана уу."
        }), 400

    try:
        data = request.get_json(silent=True) or {}
        duration = int(data.get("duration", 5))
        voice = data.get("voice", "Ankhmaa").strip()

        recorded_filepath = voicebot.record_microphone_desktop(duration_secs=duration)
        with open(recorded_filepath, "rb") as f:
            audio_bytes = f.read()

        result = voicebot.run_full_pipeline(audio_bytes=audio_bytes, voice=voice)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/text/process", methods=["POST"])
def process_text():
    """Processes text inquiries directly through Stages 2-6."""
    data = request.get_json(silent=True) or {}
    text_query = data.get("text", "").strip()
    voice = data.get("voice", "Ankhmaa").strip()

    if not text_query:
        return jsonify({"success": False, "error": "Асуулт хоосон байна."}), 400

    try:
        result = voicebot.run_full_pipeline(text_query=text_query, voice=voice)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    """
    Cleans up input audio recordings and synthesized output audio files
    when the conversation is deleted or when the user disconnects.
    """
    try:
        data = request.get_json(silent=True) or {}
        clear_cache = data.get("clear_cache", True)
        cleaned_count = voicebot.cleanup_audio_files(clear_cache=clear_cache)
        return jsonify({"success": True, "cleaned_files": cleaned_count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """Streams synthesized .m4a audio files with appropriate audio/mp4 headers."""
    mime_type, _ = mimetypes.guess_type(filename)
    if filename.endswith(".m4a"):
        mime_type = "audio/mp4"
    return send_from_directory(
        voicebot.output_dir,
        filename,
        mimetype=mime_type or "audio/mp4",
        as_attachment=False
    )


@app.route("/api/status")
def system_status():
    """Health check endpoint displaying system telemetry."""
    return jsonify({
        "status": "online",
        "agent_name": "Amirlan (Khan Bank Voicebot)",
        "project_id": voicebot.project_id,
        "gcp_location": voicebot.gcp_location,
        "stt_model": "chirp_3",
        "stt_endpoint": f"{voicebot.stt_location}-speech.googleapis.com",
        "llm_model": "gemini-2.5-flash",
        "tts_default_voice": "Ankhmaa (.m4a)",
        "available_voices": ["Ankhmaa", "Ganbaatar"],
        "sounddevice_enabled": SOUNDDEVICE_AVAILABLE,
        "has_narakeet_key": bool(voicebot.narakeet_api_key),
        "caching_enabled": voicebot.caching_enabled
    })


# =============================================================================
# DEVELOPER MODE REST APIs
# =============================================================================

@app.route("/api/dev/config", methods=["GET", "POST"])
def dev_config():
    """Developer Mode: Get or update system configuration (caching, system prompt)."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "caching_enabled" in data:
            voicebot.caching_enabled = bool(data["caching_enabled"])
            voicebot.log_event("Config", f"Caching toggled to {voicebot.caching_enabled}")
        if "system_prompt" in data:
            new_prompt = str(data["system_prompt"]).strip()
            if new_prompt:
                voicebot.system_prompt_template = new_prompt
                voicebot.log_event("Config", "System prompt updated")
        if data.get("reset_prompt"):
            voicebot.system_prompt_template = voicebot.DEFAULT_SYSTEM_PROMPT
            voicebot.log_event("Config", "System prompt reset to default")

    return jsonify({
        "success": True,
        "caching_enabled": voicebot.caching_enabled,
        "system_prompt": voicebot.system_prompt_template,
        "default_prompt": voicebot.DEFAULT_SYSTEM_PROMPT,
        "cache_stats": voicebot.get_cache_stats()
    })


@app.route("/api/dev/cache/clear", methods=["POST"])
def dev_cache_clear():
    """Developer Mode: Purge all memoized TTS cached audio files."""
    try:
        cleared = voicebot.clear_tts_cache()
        return jsonify({
            "success": True,
            "cleared_count": cleared,
            "cache_stats": voicebot.get_cache_stats()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dev/metrics", methods=["GET"])
def dev_metrics():
    """Developer Mode: Retrieve telemetry history, metrics summary, and system logs."""
    try:
        metrics = voicebot.get_metrics_summary()
        return jsonify({
            "success": True,
            "summary": metrics,
            "history": voicebot.request_history,
            "logs": voicebot.system_logs[-80:],
            "latest_evaluations": voicebot.latest_evaluations
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dev/evaluate", methods=["POST"])
def dev_evaluate():
    """Developer Mode: Run automated evaluation benchmark suite across test cases."""
    try:
        eval_result = voicebot.run_evaluation_suite()
        return jsonify({"success": True, "evaluation": eval_result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    print(f"\n* Starting Khan Bank Amirlan App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

