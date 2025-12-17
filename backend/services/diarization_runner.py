from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DiarizationRunner:
    """
    Manages external execution of speaker diarization in venv-diar.
    """

    def __init__(
        self,
        diar_python: str | None = None,
        diar_script: str | None = None,
        hf_token: str | None = None,
    ):
        self.diar_python = diar_python or "/opt/youtube-shorts-generator/venv-diar/bin/python"
        self.diar_script = diar_script or str(Path(__file__).resolve().parents[1] / "tools" / "diarize.py")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

    def run(self, input_path: str) -> List[Dict]:
        """
        Run diarization on the input audio/video file.

        Args:
            input_path: Path to the media file.

        Returns:
            List of segments with speaker info: [{'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'}, ...]
        """
        if not self.hf_token:
            logger.warning("Diarization skipped: HUGGINGFACE_TOKEN is not set.")
            return []

        if not Path(self.diar_python).exists():
            logger.warning("Diarization skipped: python interpreter not found at %s", self.diar_python)
            return []

        if not Path(self.diar_script).exists():
            logger.warning("Diarization skipped: script not found at %s", self.diar_script)
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "diar.json"
            
            cmd = [
                self.diar_python,
                self.diar_script,
                "--input",
                input_path,
                "--output",
                str(out_json),
                "--hf_token",
                self.hf_token,
            ]
            
            logger.info("Running external diarization: %s", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "External diarization failed (rc=%s).\nStdout: %s\nStderr: %s",
                    exc.returncode,
                    (exc.stdout or b"").decode(errors="ignore"),
                    (exc.stderr or b"").decode(errors="ignore"),
                )
                return []

            if not out_json.exists():
                logger.warning("Diarization output file not created: %s", out_json)
                return []

            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                segments = data.get("segments") or []
                logger.info("External diarization produced %d segments", len(segments))
                return segments
            except Exception as parse_exc:
                logger.warning("Failed to parse diarization JSON: %s", parse_exc)
                return []


# Singleton instance
_runner: DiarizationRunner | None = None


def get_diarization_runner() -> DiarizationRunner:
    """Get or create the singleton DiarizationRunner instance."""
    global _runner
    if _runner is None:
        _runner = DiarizationRunner()
    return _runner
