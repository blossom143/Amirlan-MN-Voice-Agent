"""
Khan Bank - Amirlan Luxury AI Voicebot Web Application
======================================================
Flask server providing REST APIs and serving the dynamic luxury web UI.
"""

import os
import mimetypes
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

        result = voicebot.run_full_pipeline(audio_bytes=audio_bytes)
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
        duration = int(request.json.get("duration", 5)) if request.is_json else 5
        recorded_filepath = voicebot.record_microphone_desktop(duration_secs=duration)
        with open(recorded_filepath, "rb") as f:
            audio_bytes = f.read()

        result = voicebot.run_full_pipeline(audio_bytes=audio_bytes)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/text/process", methods=["POST"])
def process_text():
    """Processes text inquiries directly through Stages 2-6."""
    data = request.get_json(silent=True) or {}
    text_query = data.get("text", "").strip()

    if not text_query:
        return jsonify({"success": False, "error": "Асуулт хоосон байна."}), 400

    try:
        result = voicebot.run_full_pipeline(text_query=text_query)
        return jsonify({"success": True, **result})
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
        "tts_voice": "Ganbaatar (.m4a)",
        "sounddevice_enabled": SOUNDDEVICE_AVAILABLE,
        "has_narakeet_key": bool(voicebot.narakeet_api_key)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    print(f"\n* Starting Khan Bank Amirlan App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
