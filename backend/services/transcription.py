"""Video transcription service using external WhisperX runner (venv-asr) with BUILT-IN diarization."""
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
from backend.services.transcription_runner import TranscriptionRunner

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribe audio using WhisperX with BUILT-IN diarization (much more accurate than separate Pyannote)."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = WHISPERX_BATCH_SIZE,
        enable_diarization: bool = WHISPERX_ENABLE_DIARIZATION,
        num_speakers: int = 2,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize TranscriptionService using external WhisperX runner with built-in diarization.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Preferred device (cuda, cpu)
            compute_type: Computation type (float16, int8, float32)
            batch_size: Transcription batch size (unused in external runner)
            enable_diarization: Enable speaker diarization (built into WhisperX)
            num_speakers: Expected number of speakers for diarization
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
        self.num_speakers = num_speakers
        self.hf_token = hf_token or HUGGINGFACE_TOKEN

        # Use WhisperX built-in diarization (assigns speakers at word level!)
        self.transcription_runner = TranscriptionRunner(
            enable_diarization=enable_diarization,
            num_speakers=num_speakers,
        )

        logger.info(
            "TranscriptionService initialized: model=%s, device=%s (requested=%s), compute=%s, "
            "diarization=%s (BUILT-IN WhisperX), num_speakers=%d",
            model_name,
            self.device,
            requested_device,
            compute_type,
            enable_diarization,
            num_speakers,
        )

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict]:
        """
        Transcribe audio file using WhisperX with BUILT-IN diarization.
        
        WhisperX now does diarization internally, assigning speakers at the WORD level,
        which is much more accurate than running Pyannote separately.
        """
        try:
            logger.info("Transcribing audio via WhisperX (built-in diarization): %s", audio_path)
            
            # WhisperX now handles both transcription AND diarization in one pass
            transcription_result = self.transcription_runner.transcribe(
                audio_path=audio_path,
                model=self.model_name,
                language=language,
                device=self.device,
                compute_type=self.compute_type,
            )
            
            segments = transcription_result.get("segments", [])
            detected_language = transcription_result.get("language", language)
            diarization_enabled = transcription_result.get("diarization_enabled", False)
            
            logger.info(
                "WhisperX returned %d segments (diarization=%s)",
                len(segments),
                diarization_enabled,
            )

            # Format segments (speakers already assigned by WhisperX!)
            formatted_segments = []
            for idx, segment in enumerate(segments):
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", start))
                text = (segment.get("text") or "").strip()
                
                # Speaker is now assigned by WhisperX at segment and word level
                speaker = segment.get("speaker")
                words = segment.get("words", [])

                preview = text[:80]
                logger.debug(
                    "Segment %d: %.2f-%.2f speaker=%s words=%d text=%r",
                    idx, start, end, speaker, len(words), preview,
                )
                
                formatted_segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker,
                    "words": words,  # Words now have speaker info too!
                })

            logger.info(
                "Transcription completed: %d segments (language=%s)",
                len(formatted_segments),
                detected_language,
            )
            return formatted_segments

        except Exception as e:
            logger.error("Error during transcription: %s", e, exc_info=True)
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
