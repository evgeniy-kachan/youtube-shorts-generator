"""Video transcription service using WhisperX with diarization support."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ffmpeg
import torch
import whisperx

from backend.config import (
    HUGGINGFACE_TOKEN,
    TEMP_DIR,
    WHISPERX_BATCH_SIZE,
    WHISPERX_ENABLE_DIARIZATION,
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribe audio using WhisperX with optional diarization."""

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
        Initialize WhisperX pipeline.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Preferred device (cuda, cpu)
            compute_type: Computation type (float16, int8, float32)
            batch_size: Transcription batch size
            enable_diarization: Enable speaker diarization via Pyannote
            hf_token: Hugging Face token for diarization model access
        """
        requested_device = device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU for WhisperX.")
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
        self._align_models: Dict[str, Tuple[object, dict]] = {}
        self.diarize_model = None

        logger.info(
            "Loading WhisperX model '%s' on %s (requested=%s, compute_type=%s, batch=%s)",
            model_name,
            self.device,
            requested_device,
            compute_type,
            batch_size,
        )
        self.model = whisperx.load_model(model_name, device=self.device, compute_type=compute_type)
        logger.info("WhisperX model loaded successfully")

        if self.enable_diarization:
            self._load_diarization_model()

    def _load_diarization_model(self) -> None:
        """Load diarization pipeline if possible."""
        if not self.hf_token:
            logger.warning(
                "HUGGINGFACE_TOKEN is not set. WhisperX diarization will be disabled until the token is provided."
            )
            self.enable_diarization = False
            return

        try:
            diarization_pipeline = None

            if hasattr(whisperx, "DiarizationPipeline"):
                diarization_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device,
                )

            if diarization_pipeline is None:
                try:
                    from whisperx.diarize import DiarizationPipeline as WXDP
                except Exception:
                    WXDP = None
                if WXDP is not None:
                    diarization_pipeline = WXDP(
                        use_auth_token=self.hf_token,
                        device=self.device,
                    )

            if diarization_pipeline is None and hasattr(whisperx, "load_diarize_model"):
                diarization_pipeline = whisperx.load_diarize_model(
                    device=self.device,
                    use_auth_token=self.hf_token,
                )

            if diarization_pipeline is None:
                raise AttributeError("Current whisperx version does not expose a diarization loader")

            self.diarize_model = diarization_pipeline
            logger.info("WhisperX diarization model loaded successfully")
        except Exception as exc:
            self.diarize_model = None
            self.enable_diarization = False
            logger.warning("Failed to load diarization model: %s. Continuing without diarization.", exc)

    def _get_align_model(self, language_code: str):
        """Load (or reuse) the alignment model for a specific language."""
        language_code = (language_code or "en").lower()
        if language_code not in self._align_models:
            logger.info("Loading WhisperX alignment model for language '%s'", language_code)
            align_model, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
            self._align_models[language_code] = (align_model, metadata)
        return self._align_models[language_code]

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict]:
        """
        Transcribe audio file with WhisperX and (optionally) diarize speakers.
        """
        try:
            logger.info("Transcribing audio with WhisperX: %s", audio_path)
            audio = whisperx.load_audio(audio_path)

            result = self.model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=language,
            )
            segments = result.get("segments", [])
            detected_language = language or result.get("language") or "en"
            result_dict = {"segments": segments}

            # Word-level alignment for precise timestamps
            try:
                align_model, metadata = self._get_align_model(detected_language)
                result_dict = whisperx.align(
                    segments,
                    align_model,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                segments = result_dict.get("segments", segments)
            except Exception as exc:
                logger.warning("WhisperX alignment failed (%s). Using raw segments.", exc)

            # Optional diarization
            if self.enable_diarization and self.diarize_model:
                try:
                    diarize_segments = self.diarize_model(audio)
                    result_dict = whisperx.assign_word_speakers(diarize_segments, result_dict)
                    segments = result_dict.get("segments", segments)
                except Exception as exc:
                    logger.warning("Diarization failed (%s). Continuing without speaker tags.", exc)

            formatted_segments = []
            for idx, segment in enumerate(segments):
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", start))
                text = (segment.get("text") or "").strip()
                speaker = segment.get("speaker")
                words = segment.get("words") or []

                preview = text[:80]
                logger.debug(
                    "WhisperX segment %s: start=%.2f end=%.2f speaker=%s len=%s text=%r",
                    idx,
                    start,
                    end,
                    speaker,
                    len(text),
                    preview,
                )

                formatted_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text,
                        "speaker": speaker,
                        "words": [
                            {
                                "word": word.get("word"),
                                "start": float(word.get("start", 0.0)),
                                "end": float(word.get("end", 0.0)),
                                "speaker": word.get("speaker"),
                                "probability": word.get("probability"),
                            }
                            for word in words
                        ],
                    }
                )

            logger.info("Transcription completed: %s segments (language=%s)", len(formatted_segments), detected_language)
            return formatted_segments

        except Exception as e:
            logger.error("Error during WhisperX transcription: %s", e, exc_info=True)
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
