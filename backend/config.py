"""Configuration settings for the application."""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# DeepSeek Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", 60))
DEEPSEEK_TRANSLATION_MODEL = os.getenv("DEEPSEEK_TRANSLATION_MODEL", "deepseek-chat")
DEEPSEEK_TRANSLATION_TIMEOUT = float(os.getenv("DEEPSEEK_TRANSLATION_TIMEOUT", DEEPSEEK_TIMEOUT))
DEEPSEEK_TRANSLATION_CHUNK_SIZE = int(os.getenv("DEEPSEEK_TRANSLATION_CHUNK_SIZE", 12))
DEEPSEEK_TRANSLATION_MAX_GROUP_CHARS = int(os.getenv("DEEPSEEK_TRANSLATION_MAX_GROUP_CHARS", 4000))
DEEPSEEK_TRANSLATION_TEMPERATURE = float(os.getenv("DEEPSEEK_TRANSLATION_TEMPERATURE", 0.1))
DEEPSEEK_TRANSLATION_CONCURRENCY = int(os.getenv("DEEPSEEK_TRANSLATION_CONCURRENCY", 2))
DEEPSEEK_MARKUP_TEMPERATURE = float(os.getenv("DEEPSEEK_MARKUP_TEMPERATURE", 0.15))

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
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", 16))
WHISPERX_ENABLE_DIARIZATION = os.getenv("WHISPERX_ENABLE_DIARIZATION", "true").lower() in ("1", "true", "yes")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

HIGHLIGHT_CONCURRENT_REQUESTS = int(os.getenv("HIGHLIGHT_CONCURRENT_REQUESTS", 3))
HIGHLIGHT_SEGMENTS_PER_CHUNK = int(os.getenv("HIGHLIGHT_SEGMENTS_PER_CHUNK", 10))

# Translation
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
TRANSLATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSLATION_MAX_LENGTH = 1024

# TTS Configuration (local Silero defaults)
SILERO_LANGUAGE = "ru"
SILERO_SPEAKER = "eugene"  # Russian voice (aidar, baya, kseniya, xenia, eugene, random)
SILERO_MODEL_VERSION = "v3_1_ru"  # Model version for torch.hub.load
TTS_ENABLE_MARKUP = os.getenv("TTS_ENABLE_MARKUP", "true").lower() in ("1", "true", "yes")
TTS_MARKUP_MODEL = os.getenv("TTS_MARKUP_MODEL", DEEPSEEK_MODEL)
TTS_MARKUP_MAX_TOKENS = int(os.getenv("TTS_MARKUP_MAX_TOKENS", 160))

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io/v1")
ELEVENLABS_STABILITY = float(os.getenv("ELEVENLABS_STABILITY", 0.35))
ELEVENLABS_SIMILARITY = float(os.getenv("ELEVENLABS_SIMILARITY", 0.75))
ELEVENLABS_STYLE = float(os.getenv("ELEVENLABS_STYLE", 0.0))
ELEVENLABS_SPEAKER_BOOST = os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() in ("1", "true", "yes")
ELEVENLABS_MAX_CHARS = int(os.getenv("ELEVENLABS_MAX_CHARS", 500))
ELEVENLABS_REQUEST_TIMEOUT = float(os.getenv("ELEVENLABS_REQUEST_TIMEOUT", 45))
ELEVENLABS_PROXY = os.getenv("ELEVENLABS_PROXY")
ELEVENLABS_VOICE_IDS_MALE = _split_env_list(os.getenv("ELEVENLABS_VOICE_IDS_MALE"))
ELEVENLABS_VOICE_IDS_FEMALE = _split_env_list(os.getenv("ELEVENLABS_VOICE_IDS_FEMALE"))

# Video Processing Configuration
VERTICAL_CONVERSION_METHOD = "letterbox"  # letterbox, center_crop
TARGET_WIDTH = 1080  # Width for vertical video (9:16)
TARGET_HEIGHT = 1920  # Height for vertical video (9:16)

