"""
Service to run Pyannote speaker diarization via venv-diar subprocess.

This is the PRIMARY diarization method. WhisperX only does transcription,
Pyannote handles all speaker separation.

Pyannote runs on GPU for speed.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class PyAnnoteDiarizationRunner:
    """
    Runs Pyannote diarization in a separate venv (venv-diar) via subprocess.
    
    This is the main diarization service. WhisperX only handles transcription,
    Pyannote handles all speaker separation with full control over parameters.
    """

    def __init__(
        self,
        num_speakers: int = 0,  # 0 = auto-detect
        device: str = "cuda",   # GPU for speed
    ):
        self.python_path = os.getenv(
            "PYANNOTE_PYTHON",
            "/opt/youtube-shorts-generator/venv-diar/bin/python"
        )
        self.script_path = os.getenv(
            "PYANNOTE_SCRIPT",
            "/opt/youtube-shorts-generator/backend/tools/diarize.py"
        )
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.num_speakers = num_speakers
        self.device = device

    def is_available(self) -> bool:
        """Check if Pyannote diarization is available."""
        if not Path(self.python_path).exists():
            logger.warning("Pyannote python not found at %s", self.python_path)
            return False
        
        if not Path(self.script_path).exists():
            logger.warning("Pyannote script not found at %s", self.script_path)
            return False
        
        if not self.hf_token:
            logger.warning("HUGGINGFACE_TOKEN not set, Pyannote requires it")
            return False
        
        return True

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Run Pyannote diarization on audio/video file.

        Args:
            audio_path: Path to the media file.

        Returns:
            List of segments with speaker info: 
            [{'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'}, ...]
        """
        if not self.is_available():
            logger.error("Pyannote diarization is not available")
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "pyannote_diar.json"
            
            cmd = [
                self.python_path,
                self.script_path,
                "--input", audio_path,
                "--output", str(out_json),
                "--device", self.device,
                "--num_speakers", str(self.num_speakers),
                "--hf_token", self.hf_token,
            ]
            
            mode_str = f"fixed={self.num_speakers}" if self.num_speakers > 0 else "auto-detect"
            logger.info(
                "Running Pyannote diarization on %s (speakers=%s, device=%s)",
                Path(audio_path).name,
                mode_str,
                self.device.upper(),
            )
            
            try:
                # Timeout: 60 min for long videos (2+ hours)
                # GPU diarization: ~10-15 min for 2-hour video
                # CPU diarization: can take 30+ min
                import time
                start_time = time.time()
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=3600,  # 60 min max (increased from 30)
                )
                
                elapsed = time.time() - start_time
                logger.info("Pyannote diarization completed in %.1f seconds", elapsed)
                
                if result.stdout:
                    logger.debug("Pyannote stdout: %s", result.stdout.decode(errors="ignore"))
                if result.stderr:
                    # Pyannote logs to stderr
                    stderr_text = result.stderr.decode(errors="ignore")
                    # Log important lines (device info, GPU, speakers, errors)
                    for line in stderr_text.split("\n"):
                        line_lower = line.lower()
                        if any(kw in line_lower for kw in ["speaker", "segment", "error", "cuda", "gpu", "device", "pipeline", "diarization"]):
                            logger.info("Pyannote: %s", line.strip())
                    
            except subprocess.TimeoutExpired:
                logger.error("Pyannote diarization timed out (60 min)")
                return []
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "Pyannote diarization failed (rc=%s).\nStdout: %s\nStderr: %s",
                    exc.returncode,
                    (exc.stdout or b"").decode(errors="ignore"),
                    (exc.stderr or b"").decode(errors="ignore"),
                )
                return []

            if not out_json.exists():
                logger.warning("Pyannote output file not created: %s", out_json)
                return []

            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                segments = data.get("segments") or []
                
                # Log quality summary
                self._log_quality_summary(segments, audio_path)
                
                return segments
            except Exception as parse_exc:
                logger.warning("Failed to parse Pyannote JSON: %s", parse_exc)
                return []

    def _log_quality_summary(self, segments: List[Dict], input_path: str) -> None:
        """Log a brief quality summary of diarization results."""
        if not segments:
            logger.warning("[Pyannote] No segments returned")
            return
        
        # Count speakers
        speakers: Dict[str, Dict] = {}
        for seg in segments:
            speaker = seg.get("speaker", "unknown")
            duration = seg.get("end", 0) - seg.get("start", 0)
            if speaker not in speakers:
                speakers[speaker] = {"count": 0, "duration": 0.0}
            speakers[speaker]["count"] += 1
            speakers[speaker]["duration"] += duration
        
        total_speech = sum(s["duration"] for s in speakers.values())
        
        logger.info("[Pyannote] ========================================")
        logger.info("[Pyannote] Input: %s", Path(input_path).name)
        logger.info("[Pyannote] Speakers: %d, Segments: %d, Speech: %.1fs", 
                   len(speakers), len(segments), total_speech)
        
        for spk in sorted(speakers.keys()):
            stats = speakers[spk]
            pct = (stats["duration"] / total_speech * 100) if total_speech > 0 else 0
            logger.info("[Pyannote]   %s: %d segs, %.1fs (%.1f%%)", 
                       spk, stats["count"], stats["duration"], pct)
        
        # Quality warnings
        if len(speakers) == 1 and self.num_speakers != 1:
            logger.warning("[Pyannote] ⚠ Only 1 speaker detected - voices may be too similar")
        elif len(speakers) > 5:
            logger.warning("[Pyannote] ⚠ %d speakers detected - unusually high", len(speakers))
        else:
            logger.info("[Pyannote] ✓ Results look reasonable")
        
        logger.info("[Pyannote] ========================================")


# Singleton instance
_runner: PyAnnoteDiarizationRunner | None = None


def get_pyannote_diarization_runner(
    num_speakers: int = 0,
    device: str = "cuda",
) -> PyAnnoteDiarizationRunner:
    """Get or create the PyAnnoteDiarizationRunner instance."""
    global _runner
    if _runner is None or _runner.num_speakers != num_speakers:
        _runner = PyAnnoteDiarizationRunner(
            num_speakers=num_speakers,
            device=device,
        )
    return _runner


def is_pyannote_available() -> bool:
    """Check if Pyannote diarization is available on this system."""
    runner = PyAnnoteDiarizationRunner()
    return runner.is_available()
