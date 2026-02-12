"""
Service to run external WhisperX transcription via venv-asr subprocess.

NOTE: Diarization is now handled separately by Pyannote (pyannote_diarization_runner.py).
WhisperX only does transcription + word alignment.
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TranscriptionRunner:
    """
    Runs WhisperX transcription in a separate venv (venv-asr) to avoid NumPy conflicts.
    
    NOTE: This class now only handles transcription + alignment.
    Diarization is handled separately by PyAnnoteDiarizationRunner.
    """

    def __init__(self):
        self.enabled = os.getenv("EXTERNAL_TRANSCRIPTION_ENABLED", "0") == "1"
        self.python_path = os.getenv(
            "EXTERNAL_ASR_PY",
            "/opt/youtube-shorts-generator/venv-asr/bin/python"
        )
        self.script_path = os.getenv(
            "EXTERNAL_ASR_SCRIPT",
            "/opt/youtube-shorts-generator/backend/tools/transcribe.py"
        )

        if self.enabled:
            logger.info(
                "External transcription enabled: python=%s, script=%s (diarization handled separately by Pyannote)",
                self.python_path,
                self.script_path,
            )
        else:
            logger.warning(
                "External transcription DISABLED. Set EXTERNAL_TRANSCRIPTION_ENABLED=1 to enable."
            )

    def transcribe(
        self,
        audio_path: str,
        model: str = "large-v2",
        language: str = "ru",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> dict[str, Any]:
        """
        Run WhisperX transcription + alignment via subprocess (NO diarization).

        Returns:
            {
                "segments": [{"start": float, "end": float, "text": str, "words": [...]}, ...],
                "language": str,
                "diarization_enabled": False
            }
        
        NOTE: Diarization is handled separately by PyAnnoteDiarizationRunner.
        """
        if not self.enabled:
            raise RuntimeError(
                "External transcription is disabled. Set EXTERNAL_TRANSCRIPTION_ENABLED=1."
            )

        # Create temp file for output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp_out:
            output_path = tmp_out.name

        try:
            # NOTE: No --diarize flag! Diarization is handled by Pyannote separately.
            cmd = [
                self.python_path,
                self.script_path,
                "--audio", audio_path,
                "--model", model,
                "--language", language,
                "--device", device,
                "--compute_type", compute_type,
                "--output", output_path,
            ]

            logger.info("Running WhisperX transcription (no diarization): %s", " ".join(cmd[:8]) + " ...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min max
                check=True,
            )

            logger.info("Transcription subprocess stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Transcription subprocess stderr:\n%s", result.stderr)

            # Read output JSON
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(
                "WhisperX transcription completed: %d segments (diarization will be done by Pyannote)",
                len(data.get("segments", []))
            )
            return data

        except subprocess.TimeoutExpired:
            logger.error("Transcription subprocess timed out (30 min)")
            raise RuntimeError("Transcription timed out")
        except subprocess.CalledProcessError as e:
            logger.error(
                "Transcription subprocess failed (exit %d):\nstdout=%s\nstderr=%s",
                e.returncode,
                e.stdout,
                e.stderr,
            )
            raise RuntimeError(f"Transcription failed: {e.stderr}")
        except Exception as e:
            logger.error("Transcription error: %s", e, exc_info=True)
            raise
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)


# Singleton instance
_runner: TranscriptionRunner | None = None


def get_transcription_runner() -> TranscriptionRunner:
    """Get or create the singleton TranscriptionRunner instance."""
    global _runner
    if _runner is None:
        _runner = TranscriptionRunner()
    return _runner


def merge_transcription_with_diarization(
    transcription: dict[str, Any],
    diarization_segments: list[dict],
) -> dict[str, Any]:
    """
    Merge WhisperX transcription with Pyannote diarization results.
    
    Assigns speaker labels to transcription segments and words based on
    time overlap with diarization segments.
    
    Args:
        transcription: WhisperX output with segments and words
        diarization_segments: Pyannote output [{"start": float, "end": float, "speaker": str}, ...]
    
    Returns:
        Transcription with speaker labels added to segments and words
    """
    if not diarization_segments:
        logger.warning("No diarization segments provided, returning transcription without speakers")
        return transcription
    
    def find_speaker_at_time(time: float) -> str | None:
        """Find which speaker is talking at a given time."""
        for diar_seg in diarization_segments:
            if diar_seg["start"] <= time <= diar_seg["end"]:
                return diar_seg["speaker"]
        return None
    
    def find_dominant_speaker(start: float, end: float) -> str | None:
        """Find the speaker with most overlap in a time range."""
        speaker_durations: dict[str, float] = {}
        
        for diar_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(start, diar_seg["start"])
            overlap_end = min(end, diar_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                speaker = diar_seg["speaker"]
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap
        
        if not speaker_durations:
            return None
        
        # Return speaker with most overlap
        return max(speaker_durations, key=speaker_durations.get)
    
    # Process each segment
    segments = transcription.get("segments", [])
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        # Assign speaker to segment based on dominant speaker
        seg["speaker"] = find_dominant_speaker(seg_start, seg_end) or "SPEAKER_UNKNOWN"
        
        # Assign speaker to each word
        for word in seg.get("words", []):
            word_start = word.get("start", 0)
            word_end = word.get("end", word_start)
            word_mid = (word_start + word_end) / 2
            
            # Use midpoint to determine speaker
            word["speaker"] = find_speaker_at_time(word_mid) or seg["speaker"]
    
    # Log merge statistics
    speakers_found = set()
    for seg in segments:
        if "speaker" in seg:
            speakers_found.add(seg["speaker"])
    
    logger.info(
        "Merged transcription with diarization: %d segments, %d speakers (%s)",
        len(segments),
        len(speakers_found),
        ", ".join(sorted(speakers_found))
    )
    
    transcription["diarization_enabled"] = True
    return transcription

