from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def run_external_diarization(
    input_path: str,
    diar_python: str | None = None,
    diar_script: str | None = None,
    hf_token: str | None = None,
) -> List[Dict]:
    """
    Call external diarization (venv-diar) and return list of segments.

    Args:
        input_path: path to audio/video segment
        diar_python: python binary of diarization venv
                     default: /opt/youtube-shorts-generator/venv-diar/bin/python
        diar_script: diarization CLI script
                     default: backend/tools/diarize.py
        hf_token: HuggingFace token (fallback to env HUGGINGFACE_TOKEN)
    """
    diar_python = diar_python or "/opt/youtube-shorts-generator/venv-diar/bin/python"
    diar_script = diar_script or str(Path(__file__).resolve().parents[1] / "tools" / "diarize.py")
    hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        logger.warning("Diarization skipped: HUGGINGFACE_TOKEN is not set.")
        return []

    if not Path(diar_python).exists():
        logger.warning("Diarization skipped: diarization python not found at %s", diar_python)
        return []

    if not Path(diar_script).exists():
        logger.warning("Diarization skipped: diarization script not found at %s", diar_script)
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "diar.json"
        cmd = [
            diar_python,
            diar_script,
            "--input",
            input_path,
            "--output",
            str(out_json),
            "--hf_token",
            hf_token,
        ]
        logger.info("Running external diarization: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "External diarization failed (rc=%s). stdout=%s stderr=%s",
                exc.returncode,
                (exc.stdout or b"").decode(errors="ignore"),
                (exc.stderr or b"").decode(errors="ignore"),
            )
            return []

        if not out_json.exists():
            logger.warning("Diarization output not found: %s", out_json)
            return []

        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
            segments = data.get("segments") or []
            logger.info("External diarization produced %d segments", len(segments))
            return segments
        except Exception as parse_exc:
            logger.warning("Failed to parse diarization output: %s", parse_exc)
            return []

