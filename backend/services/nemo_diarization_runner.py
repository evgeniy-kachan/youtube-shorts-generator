"""
Service to run NVIDIA NeMo MSDD diarization via venv-nemo subprocess.

NeMo MSDD provides state-of-the-art speaker diarization for:
- Similar voices
- Overlapping speech
- Long recordings
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class NemoDiarizationRunner:
    """
    Manages external execution of NeMo MSDD diarization in venv-nemo.
    """

    def __init__(
        self,
        nemo_python: str | None = None,
        nemo_script: str | None = None,
        device: str = "cuda",  # GPU for speed - subprocess has isolated CUDA context
        num_speakers: int = 0,
        max_speakers: int = 8,
    ):
        self.nemo_python = nemo_python or os.getenv(
            "NEMO_PYTHON",
            "/opt/youtube-shorts-generator/venv-nemo/bin/python"
        )
        self.nemo_script = nemo_script or str(
            Path(__file__).resolve().parents[1] / "tools" / "diarize_nemo.py"
        )
        self.device = device
        self.num_speakers = num_speakers  # 0 = auto-detect
        self.max_speakers = max_speakers

    def is_available(self) -> bool:
        """Check if NeMo diarization is available."""
        if not Path(self.nemo_python).exists():
            logger.warning("NeMo python not found at %s", self.nemo_python)
            return False
        
        if not Path(self.nemo_script).exists():
            logger.warning("NeMo script not found at %s", self.nemo_script)
            return False
        
        return True

    def _release_gpu_memory(self) -> None:
        """Aggressively release GPU memory before NeMo subprocess."""
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                # Clear all cached memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # Force garbage collection again
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info(
                    "GPU memory released. Current allocated: %.1f MB",
                    torch.cuda.memory_allocated() / 1024 / 1024
                )
        except Exception as e:
            logger.warning("Could not release GPU memory: %s", e)

    def run(self, input_path: str) -> List[Dict]:
        """
        Run NeMo MSDD diarization on the input audio/video file.

        Args:
            input_path: Path to the media file.

        Returns:
            List of segments with speaker info: 
            [{'start': 0.0, 'end': 1.0, 'speaker': 'speaker_0'}, ...]
        """
        if not self.is_available():
            logger.error("NeMo diarization is not available")
            return []

        # Set spawn method to avoid CUDA context inheritance issues
        # This ensures NeMo subprocess gets a clean CUDA context
        try:
            current_method = mp.get_start_method(allow_none=True)
            if current_method != "spawn":
                mp.set_start_method("spawn", force=True)
                logger.info("Set multiprocessing start method to 'spawn' for clean CUDA context")
        except RuntimeError as e:
            logger.debug("Could not set spawn method (may already be set): %s", e)
        
        # Release GPU memory before spawning NeMo
        if self.device == "cuda":
            logger.info("Releasing GPU memory before NeMo...")
            self._release_gpu_memory()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "nemo_diar.json"
            
            cmd = [
                self.nemo_python,
                self.nemo_script,
                "--input", input_path,
                "--output", str(out_json),
                "--device", self.device,
                "--num_speakers", str(self.num_speakers),
                "--max_speakers", str(self.max_speakers),
            ]
            
            logger.info("Running NeMo diarization on %s: %s", self.device.upper(), " ".join(cmd[:6]) + " ...")
            
            # Set environment variables for clean CUDA context in subprocess
            env = os.environ.copy()
            env["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA execution
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Better memory management
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure subprocess sees GPU
            # Force subprocess to create new CUDA context
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=1800,  # 30 min max
                    env=env,
                )
                
                if result.stdout:
                    logger.debug("NeMo stdout: %s", result.stdout.decode(errors="ignore"))
                if result.stderr:
                    logger.info("NeMo stderr: %s", result.stderr.decode(errors="ignore"))
                    
            except subprocess.TimeoutExpired:
                logger.error("NeMo diarization timed out (30 min)")
                return []
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "NeMo diarization failed (rc=%s).\nStdout: %s\nStderr: %s",
                    exc.returncode,
                    (exc.stdout or b"").decode(errors="ignore"),
                    (exc.stderr or b"").decode(errors="ignore"),
                )
                return []

            if not out_json.exists():
                logger.warning("NeMo output file not created: %s", out_json)
                return []

            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                segments = data.get("segments") or []
                
                # Log quality summary at service level
                self._log_quality_summary(segments, input_path)
                
                return segments
            except Exception as parse_exc:
                logger.warning("Failed to parse NeMo JSON: %s", parse_exc)
                return []

    def _log_quality_summary(self, segments: List[Dict], input_path: str) -> None:
        """Log a brief quality summary of diarization results."""
        if not segments:
            logger.warning("[NeMo Quality] No segments returned")
            return
        
        # Count speakers
        speakers = {}
        for seg in segments:
            speaker = seg.get("speaker", "unknown")
            duration = seg.get("end", 0) - seg.get("start", 0)
            if speaker not in speakers:
                speakers[speaker] = {"count": 0, "duration": 0.0}
            speakers[speaker]["count"] += 1
            speakers[speaker]["duration"] += duration
        
        total_speech = sum(s["duration"] for s in speakers.values())
        
        logger.info("[NeMo Quality] ========================================")
        logger.info("[NeMo Quality] Input: %s", Path(input_path).name)
        logger.info("[NeMo Quality] Speakers: %d, Segments: %d, Speech: %.1fs", 
                   len(speakers), len(segments), total_speech)
        
        for spk in sorted(speakers.keys()):
            stats = speakers[spk]
            pct = (stats["duration"] / total_speech * 100) if total_speech > 0 else 0
            logger.info("[NeMo Quality]   %s: %d segs, %.1fs (%.1f%%)", 
                       spk, stats["count"], stats["duration"], pct)
        
        # Quick quality check
        if len(speakers) == 1 and self.num_speakers != 1:
            logger.warning("[NeMo Quality] ⚠ Only 1 speaker detected - may need adjustment")
        elif len(speakers) > 5:
            logger.warning("[NeMo Quality] ⚠ %d speakers detected - unusually high", len(speakers))
        else:
            logger.info("[NeMo Quality] ✓ Results look reasonable")
        
        logger.info("[NeMo Quality] ========================================")


# Singleton instance
_runner: NemoDiarizationRunner | None = None


def get_nemo_diarization_runner() -> NemoDiarizationRunner:
    """Get or create the singleton NemoDiarizationRunner instance."""
    global _runner
    if _runner is None:
        _runner = NemoDiarizationRunner()
    return _runner


def is_nemo_available() -> bool:
    """Check if NeMo diarization is available on this system."""
    runner = get_nemo_diarization_runner()
    return runner.is_available()
