# ==============================================================================
# Khan Bank Amirlan AI Voicebot - Production Dockerfile for Render.com
# ==============================================================================

FROM python:3.11-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Python runtime configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=5005

# Set working directory
WORKDIR /app

# ------------------------------------------------------------------------------
# 1. Install Low-Level Linux Audio Drivers & System Dependencies
# ------------------------------------------------------------------------------
# In containerized/cloud environments like Render.com, Python audio libraries
# (sounddevice, scipy, soundfile) require PortAudio, ALSA, and libsndfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compilers & build tools for C extensions
    build-essential \
    gcc \
    g++ \
    make \
    libffi-dev \
    # Low-level PortAudio C libraries for sounddevice
    libportaudio2 \
    portaudio19-dev \
    # ALSA audio subsystem & plugins
    libasound2 \
    libasound2-plugins \
    alsa-utils \
    libpulse0 \
    # Soundfile & media format processing
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    # Networking & Security
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 2. Configure Headless Dummy ALSA Device
# ------------------------------------------------------------------------------
# Cloud containers do not have physical soundcards (/dev/snd). This configuration
# provides a null PCM sink so PortAudio and sounddevice initialize cleanly
# without raising "cannot find card 0" errors or hanging on startup.
RUN mkdir -p /etc/alsa && \
    echo 'pcm.!default { type plug; slave.pcm "null"; }' > /etc/asound.conf && \
    echo 'ctl.!default { type null; }' >> /etc/asound.conf

# ------------------------------------------------------------------------------
# 3. Install Python Dependencies
# ------------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# 4. Copy Application Source Code & Setup Audio Storage Directories
# ------------------------------------------------------------------------------
COPY . .

# Ensure input and output audio storage directories exist with write access
RUN mkdir -p input-audio output-audio && \
    chmod -R 777 input-audio output-audio

# Expose default port (Render will override via $PORT environment variable)
EXPOSE 5005

# ------------------------------------------------------------------------------
# 5. Production Entrypoint
# ------------------------------------------------------------------------------
# Render dynamically passes the $PORT environment variable (e.g. 10000).
# Using 'exec' allows Gunicorn to receive SIGTERM directly for graceful shutdowns.
# --timeout 120 accommodates multi-stage LLM reasoning and TTS audio synthesis.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5005} --workers 1 --threads 4 --timeout 120 run:app"]
