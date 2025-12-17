import sys

print(f"Python: {sys.version.split()[0]}")

try:
    import numpy
    print(f"NumPy: {numpy.__version__} (Should be 1.26.4)")
except ImportError:
    print("NumPy: NOT INSTALLED")

try:
    import torch
    print(f"Torch: {torch.__version__}")
except ImportError:
    print("Torch: NOT INSTALLED")

try:
    import torchaudio
    print(f"Torchaudio: {torchaudio.__version__}")
except ImportError:
    print("Torchaudio: NOT INSTALLED")

try:
    import faster_whisper
    print(f"faster-whisper: {faster_whisper.__version__} (CRITICAL: Should be 1.0.3)")
except ImportError:
    print("faster-whisper: NOT INSTALLED")

try:
    import whisperx
    print(f"WhisperX: {whisperx.__version__} (Should be 3.1.1)")
except ImportError:
    print("WhisperX: NOT INSTALLED")

try:
    import pyannote.audio as pa
    print(f"pyannote.audio: {pa.__version__}")
except ImportError:
    print("pyannote.audio: NOT INSTALLED")

