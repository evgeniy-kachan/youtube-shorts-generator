"""
CLI for speaker diarization in a separate virtualenv (NumPy >=2, pyannote).

Usage (run from venv-diar):
    python backend/tools/diarize.py --input path/to/audio_or_video --output diarization.json --hf_token <token>

Output JSON format:
{
  "segments": [
    {"start": 0.00, "end": 3.12, "speaker": "SPEAKER_00"},
    {"start": 3.12, "end": 7.45, "speaker": "SPEAKER_01"},
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_wav(input_path: str, dst_path: str, sample_rate: int = 16000) -> None:
    """Extract mono wav with desired sample rate."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        dst_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_diarization(wav_path: str, hf_token: str, model: str = "pyannote/speaker-diarization-3.1") -> List[Dict]:
    """Run pyannote diarization and return list of segments."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    diarization = pipeline(wav_path)

    segments: List[Dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    # sort by time
    segments.sort(key=lambda x: x["start"])
    logger.info("Diarization produced %d segments", len(segments))
    return segments


def main():
    parser = argparse.ArgumentParser(description="Run diarization and save JSON.")
    parser.add_argument("--input", required=True, help="Input video/audio file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--hf_token", required=False, help="HuggingFace token (or set HUGGINGFACE_TOKEN env)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("HuggingFace token is required. Pass --hf_token or set HUGGINGFACE_TOKEN.")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        extract_wav(str(input_path), str(wav_path), sample_rate=16000)
        segments = run_diarization(str(wav_path), hf_token=hf_token)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    logger.info("Saved diarization to %s", output_path)


if __name__ == "__main__":
    main()

