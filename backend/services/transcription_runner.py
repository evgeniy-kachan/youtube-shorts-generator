"""
Service to run external WhisperX transcription via venv-asr subprocess.
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
    """Runs WhisperX transcription in a separate venv (venv-asr) to avoid NumPy conflicts."""

    def __init__(
        self,
        enable_diarization: bool = True,
        num_speakers: int = 2,
        min_speakers: int = 1,
        max_speakers: int = 4,
    ):
        self.enabled = os.getenv("EXTERNAL_TRANSCRIPTION_ENABLED", "0") == "1"
        self.python_path = os.getenv(
            "EXTERNAL_ASR_PY",
            "/opt/youtube-shorts-generator/venv-asr/bin/python"
        )
        self.script_path = os.getenv(
            "EXTERNAL_ASR_SCRIPT",
            "/opt/youtube-shorts-generator/backend/tools/transcribe.py"
        )
        self.enable_diarization = enable_diarization
        self.num_speakers = num_speakers  # 0 = auto-detect
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        if self.enabled:
            mode = f"fixed={num_speakers}" if num_speakers > 0 else f"auto [{min_speakers}-{max_speakers}]"
            logger.info(
                "External transcription enabled: python=%s, script=%s, diarize=%s, speakers=%s",
                self.python_path,
                self.script_path,
                self.enable_diarization,
                mode,
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
        Run WhisperX transcription with built-in diarization via subprocess.

        Returns:
            {
                "segments": [{"start": float, "end": float, "text": str, "speaker": str, "words": [...]}, ...],
                "language": str,
                "diarization_enabled": bool
            }
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
            
            # Add diarization flags
            if self.enable_diarization and self.hf_token:
                cmd.extend([
                    "--diarize",
                    "--num_speakers", str(self.num_speakers),
                    "--min_speakers", str(self.min_speakers),
                    "--max_speakers", str(self.max_speakers),
                    "--hf_token", self.hf_token,
                ])

            logger.info("Running external transcription: %s", " ".join(cmd[:10]) + " ...")
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
                "External transcription completed: %d segments",
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

