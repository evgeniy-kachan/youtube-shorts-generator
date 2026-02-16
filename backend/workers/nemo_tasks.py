"""
NeMo diarization tasks for the NeMo worker.

These tasks run in a separate worker process with venv-nemo environment.
This ensures NeMo has its own CUDA context without conflicts with Pyannote.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Path to NeMo venv Python
NEMO_PYTHON = "/opt/youtube-shorts-generator/venv-nemo/bin/python"
NEMO_SCRIPT = "/opt/youtube-shorts-generator/backend/tools/diarize_nemo.py"


def diarize_nemo(
    audio_path: str,
    num_speakers: int = 0,
    max_speakers: int = 8,
    device: str = "cuda",
) -> List[Dict]:
    """
    Run NeMo MSDD speaker diarization via subprocess.
    
    IMPORTANT: This runs diarize_nemo.py as a subprocess with its own
    Python interpreter (venv-nemo). We must NOT import torch here,
    as it would initialize CUDA and corrupt the subprocess's context.
    
    Args:
        audio_path: Path to audio/video file
        num_speakers: Number of speakers (0 = auto-detect)
        max_speakers: Maximum speakers for auto-detection
        device: Device to use (cuda or cpu)
    
    Returns:
        List of diarization segments
    """
    # NOTE: Do NOT import torch here! It would initialize CUDA and
    # cause CUBLAS_STATUS_NOT_INITIALIZED in the subprocess.
    
    logger.info("NeMo Task: diarize_nemo started, file=%s, device=%s", 
                Path(audio_path).name, device)
    
    # Create temp file for output
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name
    
    try:
        # Build command
        cmd = [
            NEMO_PYTHON,
            NEMO_SCRIPT,
            "--input", audio_path,
            "--output", output_path,
            "--device", device,
            "--max_speakers", str(max_speakers),
        ]
        
        if num_speakers > 0:
            cmd.extend(["--num_speakers", str(num_speakers)])
        
        speakers_str = f"{num_speakers} speakers" if num_speakers > 0 else "auto"
        logger.info("Running NeMo: %s", " ".join(cmd[:6]) + " ...")
        
        # Run NeMo subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        
        # Log output
        if result.stdout:
            logger.info("NeMo stdout:\n%s", result.stdout[-2000:])
        if result.stderr:
            # Log as debug since NeMo outputs a lot to stderr
            logger.debug("NeMo stderr:\n%s", result.stderr[-3000:])
        
        if result.returncode != 0:
            logger.error("NeMo failed with code %d: %s", result.returncode, result.stderr[-1000:])
            raise RuntimeError(f"NeMo diarization failed: {result.stderr[-500:]}")
        
        # Check output file
        if not os.path.exists(output_path):
            logger.error("NeMo output file not created")
            raise RuntimeError("NeMo did not create output file")
        
        # Read results
        with open(output_path) as f:
            nemo_result = json.load(f)
        
        segments = nemo_result.get("segments", [])
        
        if not segments:
            logger.warning("NeMo returned 0 segments!")
            # Log full stderr for debugging
            if result.stderr:
                logger.warning("Full NeMo stderr:\n%s", result.stderr)
        
        segments.sort(key=lambda x: x["start"])
        
        speakers = set(s["speaker"] for s in segments)
        logger.info("NeMo Task: diarize_nemo complete, %d segments, %d speakers",
                   len(segments), len(speakers))
        
        return segments
        
    finally:
        # Clean up temp file
        if os.path.exists(output_path):
            os.unlink(output_path)


def nemo_diarize_task(
    audio_path: str,
    num_speakers: int = 0,
    max_speakers: int = 8,
) -> Dict[str, Any]:
    """
    NeMo diarization task for RQ queue.
    
    This is the main entry point called by the NeMo worker.
    
    IMPORTANT: Do NOT import torch here! The subprocess (diarize_nemo.py)
    needs a clean CUDA context. Any torch import here would corrupt it.
    
    Args:
        audio_path: Path to audio/video file
        num_speakers: Number of speakers (0 = auto)
        max_speakers: Maximum speakers for auto-detection
    
    Returns:
        {
            "segments": [...],
            "num_speakers": 2,
            "speaker_stats": {...},
            "total_speech_duration": 123.4,
        }
    """
    logger.info("NeMo Task: nemo_diarize_task started, file=%s", Path(audio_path).name)
    
    # Run diarization (subprocess will handle CUDA)
    segments = diarize_nemo(
        audio_path=audio_path,
        num_speakers=num_speakers,
        max_speakers=max_speakers,
        device="cuda",  # Subprocess will fallback to CPU if needed
    )
    
    # Calculate speaker stats
    speaker_stats = {}
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        duration = seg.get("end", 0) - seg.get("start", 0)
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"count": 0, "duration": 0.0}
        speaker_stats[speaker]["count"] += 1
        speaker_stats[speaker]["duration"] += duration
    
    num_speakers_detected = len(speaker_stats)
    total_speech = sum(s["duration"] for s in speaker_stats.values())
    
    logger.info("NeMo Task: complete, %d speakers, %.1fs speech",
               num_speakers_detected, total_speech)
    
    return {
        "segments": segments,
        "num_speakers": num_speakers_detected,
        "speaker_stats": speaker_stats,
        "total_speech_duration": total_speech,
    }
