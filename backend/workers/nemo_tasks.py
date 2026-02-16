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
    Run NeMo MSDD speaker diarization.
    
    This function runs directly in the NeMo worker process,
    which has its own CUDA context.
    
    Args:
        audio_path: Path to audio/video file
        num_speakers: Number of speakers (0 = auto-detect)
        max_speakers: Maximum speakers for auto-detection
        device: Device to use (cuda or cpu)
    
    Returns:
        List of diarization segments
    """
    import torch
    
    logger.info("NeMo Task: diarize_nemo started, file=%s", Path(audio_path).name)
    
    # Log GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info("GPU: %s (%.1f GB total, %.2f GB allocated)", gpu_name, gpu_mem, allocated)
    else:
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
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
    import torch
    
    logger.info("NeMo Task: nemo_diarize_task started, file=%s", Path(audio_path).name)
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU available: %s (%.1f GB)", gpu_name, gpu_mem)
    else:
        logger.warning("CUDA not available!")
    
    # Run diarization
    segments = diarize_nemo(
        audio_path=audio_path,
        num_speakers=num_speakers,
        max_speakers=max_speakers,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
