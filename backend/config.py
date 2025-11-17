"""Configuration settings for the application."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Processing Configuration
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", 7200))  # 2 hours
MIN_SEGMENT_DURATION = 20  # seconds
MAX_SEGMENT_DURATION = 180  # 3 minutes

# Directories
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# GPU Configuration
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Whisper Configuration
WHISPER_MODEL = "large-v3"  # or "large-v2", "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"  # or "int8" for less VRAM

# Translation Configuration
NLLB_MODEL = "facebook/nllb-200-distilled-600M"  # or "facebook/nllb-200-3.3B"

# TTS Configuration
SILERO_LANGUAGE = "ru"
SILERO_SPEAKER = "v3_1_ru"  # Russian voice

# Video Processing Configuration
VERTICAL_CONVERSION_METHOD = "blur_background"  # blur_background, center_crop, smart_crop
TARGET_WIDTH = 1080  # Width for vertical video (9:16)
TARGET_HEIGHT = 1920  # Height for vertical video (9:16)

