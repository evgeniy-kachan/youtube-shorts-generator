"""Video transcription service using external WhisperX runner (venv-asr)."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import ffmpeg
import torch

from backend.config import (
    HUGGINGFACE_TOKEN,
    TEMP_DIR,
    WHISPERX_BATCH_SIZE,
    WHISPERX_ENABLE_DIARIZATION,
)
from backend.services.transcription_runner import get_transcription_runner
from backend.services.diarization_runner import get_diarization_runner

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribe audio using external WhisperX runner (venv-asr) with optional diarization (venv-diar)."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = WHISPERX_BATCH_SIZE,
        enable_diarization: bool = WHISPERX_ENABLE_DIARIZATION,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize TranscriptionService using external runners.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Preferred device (cuda, cpu)
            compute_type: Computation type (float16, int8, float32)
            batch_size: Transcription batch size (unused in external runner)
            enable_diarization: Enable speaker diarization via Pyannote (external venv-diar)
            hf_token: Hugging Face token for diarization model access
        """
        requested_device = device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Will use CPU.")
            device = "cpu"
        self.device = device

        # float16 is not supported on CPU, fall back to int8 for stability
        if self.device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        self.model_name = model_name
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token or HUGGINGFACE_TOKEN

        self.transcription_runner = get_transcription_runner()
        self.diarization_runner = get_diarization_runner() if enable_diarization else None

        logger.info(
            "TranscriptionService initialized: model=%s, device=%s (requested=%s), compute=%s, diarization=%s",
            model_name,
            self.device,
            requested_device,
            compute_type,
            enable_diarization,
        )

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict]:
        """
        Transcribe audio file using external WhisperX runner and (optionally) diarize speakers.
        """
        try:
            logger.info("Transcribing audio via external runner: %s", audio_path)
            
            # Run external WhisperX transcription
            result = self.transcription_runner.transcribe(
                audio_path=audio_path,
                model=self.model_name,
                language=language,
                device=self.device,
                compute_type=self.compute_type,
            )
            
            segments = result.get("segments", [])
            detected_language = result.get("language", language)

            # Optional diarization via external venv-diar
            speaker_map = {}
            if self.enable_diarization and self.diarization_runner:
                try:
                    logger.info("Running external diarization for %s", audio_path)
                    diar_result = self.diarization_runner.run(input_path=audio_path)
                    
                    # Build speaker map: {(start, end): speaker}
                    for seg in diar_result:
                        speaker_map[(seg["start"], seg["end"])] = seg["speaker"]
                    logger.info("Diarization completed: %d speaker segments", len(diar_result))
                except Exception as exc:
                    logger.warning("Diarization failed (%s). Continuing without speaker tags.", exc)

            # Format segments and assign speakers
            formatted_segments = []
            for idx, segment in enumerate(segments):
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", start))
                text = (segment.get("text") or "").strip()
                
                # Find matching speaker (simple overlap matching)
                speaker = None
                for (spk_start, spk_end), spk_label in speaker_map.items():
                    if spk_start <= start < spk_end or spk_start < end <= spk_end:
                        speaker = spk_label
                        break

                preview = text[:80]
                logger.debug(
                    "Transcription segment %s: start=%.2f end=%.2f speaker=%s len=%s text=%r",
                    idx,
                    start,
                    end,
                    speaker,
                    len(text),
                    preview,
                )

                # Get word-level timestamps from WhisperX alignment
                words = segment.get("words", [])
                
                formatted_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text,
                        "speaker": speaker,
                        "words": words,  # Word-level timestamps from WhisperX!
                    }
                )

            logger.info("Transcription completed: %s segments (language=%s)", len(formatted_segments), detected_language)
            return formatted_segments

        except Exception as e:
            logger.error("Error during external transcription: %s", e, exc_info=True)
            raise

    def get_full_text(self, segments: List[Dict]) -> str:
        """Get full text from segments."""
        return " ".join(seg.get("text", "") for seg in segments).strip()

    def transcribe_audio_from_video(self, video_path: str, language: str = "en") -> Dict:
        """
        Extract audio from a video file and transcribe it.

        Returns:
            Dict with segments list, speaker info, and full text.
        """
        temp_dir = Path(TEMP_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"{Path(video_path).stem}_whisper.wav"
        try:
            (
                ffmpeg.input(video_path)
                .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            segments = self.transcribe(str(audio_path), language=language)
            if os.getenv("DEBUG_SAVE_TRANSCRIPT", "0") == "1":
                full_text = self.get_full_text(segments)
                debug_dir = TEMP_DIR / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"{Path(video_path).stem}_raw.txt"
                debug_path.write_text(full_text, encoding="utf-8")
            return {
                "segments": segments,
                "text": self.get_full_text(segments),
            }
        except Exception as e:
            logger.error("Error transcribing video %s: %s", video_path, e, exc_info=True)
            raise
        finally:
            audio_path.unlink(missing_ok=True)
