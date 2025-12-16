# Deployment Guide

## Two Virtual Environments Setup

This project requires **two separate Python virtual environments** due to incompatible NumPy version requirements:

### venv-host (NumPy 1.x)
Main application environment for video processing, face detection, and API server.

**Dependencies:**
- NumPy < 2.0 (required by onnxruntime-gpu, insightface)
- onnxruntime-gpu
- insightface (face detection)
- scenedetect (scene change detection)
- opencv-contrib-python
- FastAPI, WhisperX, etc.

**Setup:**
```bash
python3 -m venv venv-host
source venv-host/bin/activate
pip install -r requirements.txt
```

---

### venv-diar (NumPy 2.x)
Separate environment for speaker diarization only.

**Dependencies:**
- NumPy >= 2.0 (required by pyannote.audio)
- pyannote.audio
- torch, torchaudio

**Setup:**
```bash
python3 -m venv venv-diar
source venv-diar/bin/activate
pip install -r requirements-diar.txt
```

---

## Environment Variables

Add to `.env`:
```bash
# Diarization configuration
EXTERNAL_DIARIZATION_ENABLED=1
EXTERNAL_DIAR_PY=/opt/youtube-shorts-generator/venv-diar/bin/python
EXTERNAL_DIAR_SCRIPT=/opt/youtube-shorts-generator/backend/tools/diarize.py

# HuggingFace token (for model downloads)
HUGGINGFACE_TOKEN=your_token_here

# CUDA libraries path (if needed)
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
```

---

## Running the Application

**Start backend (in venv-host):**
```bash
cd /opt/youtube-shorts-generator
source venv-host/bin/activate
set -a && source .env && set +a
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The backend will automatically call venv-diar for diarization tasks via subprocess.

---

## Quick Reference

**Rule of thumb for dependency placement:**

- `venv-host` (requirements.txt):
  - ✅ Face detection (insightface, onnxruntime-gpu)
  - ✅ Video processing (opencv, ffmpeg, moviepy)
  - ✅ Scene detection (scenedetect)
  - ✅ API server (FastAPI, uvicorn)
  - ✅ Transcription (whisperx)
  - ❌ Speaker diarization (pyannote.audio)

- `venv-diar` (requirements-diar.txt):
  - ✅ Speaker diarization (pyannote.audio)
  - ❌ Everything else

**If in doubt:**
- `onnxruntime` / `insightface` → venv-host
- `pyannote` → venv-diar
