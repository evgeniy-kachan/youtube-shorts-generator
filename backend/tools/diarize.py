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

import torch

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


def is_wav_file(path: str) -> bool:
    """Check if file is already a WAV file (16kHz mono)."""
    try:
        import wave
        with wave.open(path, 'rb') as wf:
            return wf.getnchannels() == 1 and wf.getframerate() == 16000
    except Exception:
        return False


def run_diarization(
    wav_path: str,
    hf_token: str,
    device: str = "cuda",
    model: str = "pyannote/speaker-diarization-3.1",
    num_speakers: int = 2,
) -> List[Dict]:
    """Run pyannote diarization and return list of segments."""
    from pyannote.audio import Pipeline

    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    logger.info("Loading Pyannote pipeline on device: %s", device)
    pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    
    # Move pipeline to GPU for faster inference
    if device == "cuda":
        pipeline.to(torch.device("cuda"))
        logger.info("Pyannote pipeline moved to GPU")
    
    # Fine-tune hyperparameters for MORE AGGRESSIVE speaker separation
    # These settings help detect multiple speakers even when voices are similar
    pipeline.instantiate({
        "segmentation": {
            "min_duration_off": 0.05,  # Reduced from 0.1 - detect shorter pauses (50ms)
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 3,     # Reduced from 6 - allow smaller speaker clusters
            "threshold": 0.35,         # Reduced from 0.5 - more aggressive speaker separation
        },
    })
    
    logger.info("Running diarization with num_speakers=%d", num_speakers)
    diarization = pipeline(wav_path, num_speakers=num_speakers)

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
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_speakers", type=int, default=2, help="Expected number of speakers (default: 2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("HuggingFace token is required. Pass --hf_token or set HUGGINGFACE_TOKEN.")

    # Check if input is already a valid WAV file (skip extraction)
    if input_path.suffix.lower() == ".wav" and is_wav_file(str(input_path)):
        logger.info("Input is already a valid WAV file, skipping extraction")
        segments = run_diarization(
            str(input_path), 
            hf_token=hf_token, 
            device=args.device,
            num_speakers=args.num_speakers,
        )
    else:
        # Extract audio from video/other format
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            extract_wav(str(input_path), str(wav_path), sample_rate=16000)
            segments = run_diarization(
                str(wav_path), 
                hf_token=hf_token, 
                device=args.device,
                num_speakers=args.num_speakers,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    logger.info("Saved diarization to %s", output_path)


if __name__ == "__main__":
    main()

