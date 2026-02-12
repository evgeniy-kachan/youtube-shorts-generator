"""
Video transcription service using:
- WhisperX (venv-asr) for transcription + word alignment
- Pyannote (venv-diar) for speaker diarization

Diarization is now handled SEPARATELY by Pyannote for better control and quality.
"""
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
    DIARIZATION_NUM_SPEAKERS,
    DIARIZATION_MIN_SPEAKERS,
    DIARIZATION_MAX_SPEAKERS,
)
from backend.services.transcription_runner import (
    TranscriptionRunner,
    merge_transcription_with_diarization,
)
from backend.services.pyannote_diarization_runner import (
    PyAnnoteDiarizationRunner,
    is_pyannote_available,
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Transcribe audio using WhisperX + Pyannote diarization.
    
    Pipeline:
    1. WhisperX: transcription + word alignment (NO diarization)
    2. Pyannote: speaker diarization (separate, full control)
    3. Merge: assign speakers to words/segments
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = WHISPERX_BATCH_SIZE,
        enable_diarization: bool = WHISPERX_ENABLE_DIARIZATION,
        num_speakers: int = DIARIZATION_NUM_SPEAKERS,
        min_speakers: int = DIARIZATION_MIN_SPEAKERS,
        max_speakers: int = DIARIZATION_MAX_SPEAKERS,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize TranscriptionService.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Preferred device (cuda, cpu)
            compute_type: Computation type (float16, int8, float32)
            batch_size: Transcription batch size (unused in external runner)
            enable_diarization: Enable speaker diarization via Pyannote
            num_speakers: Expected number of speakers (0 for auto-detect)
            min_speakers: Min speakers for auto-detect mode (unused, kept for compatibility)
            max_speakers: Max speakers for auto-detect mode (unused, kept for compatibility)
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
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token = hf_token or HUGGINGFACE_TOKEN

        # WhisperX for transcription only (no diarization)
        self.transcription_runner = TranscriptionRunner()

        # Pyannote for diarization (separate, full control)
        self.diarization_runner = PyAnnoteDiarizationRunner(
            num_speakers=num_speakers,
            device=device,  # GPU for speed
        )

        # Log configuration
        diar_mode = f"fixed={num_speakers}" if num_speakers > 0 else "auto-detect"
        pyannote_status = "available" if is_pyannote_available() else "NOT AVAILABLE"
        logger.info(
            "TranscriptionService initialized: model=%s, device=%s (requested=%s), compute=%s, "
            "diarization=%s via Pyannote (%s), speakers=%s",
            model_name,
            self.device,
            requested_device,
            compute_type,
            enable_diarization,
            pyannote_status,
            diar_mode,
        )

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict]:
        """
        Transcribe audio file using WhisperX + Pyannote diarization.
        
        Pipeline:
        1. WhisperX: transcription + word alignment
        2. Pyannote: speaker diarization (if enabled)
        3. Merge: assign speakers to segments/words
        """
        try:
            logger.info("=== TRANSCRIPTION PIPELINE START ===")
            logger.info("Step 1: WhisperX transcription (no diarization): %s", audio_path)
            
            # Step 1: WhisperX transcription only (no diarization)
            transcription_result = self.transcription_runner.transcribe(
                audio_path=audio_path,
                model=self.model_name,
                language=language,
                device=self.device,
                compute_type=self.compute_type,
            )
            
            segments = transcription_result.get("segments", [])
            detected_language = transcription_result.get("language", language)
            
            logger.info(
                "WhisperX returned %d segments (transcription only, no speakers yet)",
                len(segments),
            )
            
            # Step 2: Pyannote diarization (if enabled)
            diarization_segments = []
            if self.enable_diarization and is_pyannote_available():
                logger.info("Step 2: Pyannote diarization (speakers=%s)", 
                           self.num_speakers if self.num_speakers > 0 else "auto")
                
                # Update num_speakers in runner
                self.diarization_runner.num_speakers = self.num_speakers
                
                diarization_segments = self.diarization_runner.diarize(audio_path)
                
                if diarization_segments:
                    logger.info("Pyannote returned %d diarization segments", len(diarization_segments))
                else:
                    logger.warning("Pyannote returned no segments!")
            elif self.enable_diarization:
                logger.warning("Diarization enabled but Pyannote not available!")
            else:
                logger.info("Step 2: Diarization DISABLED, skipping")
            
            # Step 3: Merge transcription + diarization
            if diarization_segments:
                logger.info("Step 3: Merging transcription with diarization")
                transcription_result = merge_transcription_with_diarization(
                    transcription_result,
                    diarization_segments,
                )
                segments = transcription_result.get("segments", [])
            
            # DIARIZATION DIAGNOSTIC: Analyze speaker distribution
            speaker_stats: dict[str, dict] = {}
            for seg in segments:
                speaker = seg.get("speaker") or "UNKNOWN"
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {"segments": 0, "words": 0, "duration": 0.0}
                speaker_stats[speaker]["segments"] += 1
                speaker_stats[speaker]["words"] += len(seg.get("words", []))
                speaker_stats[speaker]["duration"] += seg.get("end", 0) - seg.get("start", 0)
            
            logger.info("DIARIZATION STATS: %d unique speakers detected", len(speaker_stats))
            for spk, stats in sorted(speaker_stats.items()):
                logger.info(
                    "  %s: %d segments, %d words, %.1fs total",
                    spk, stats["segments"], stats["words"], stats["duration"]
                )
            
            # Warn if fewer speakers detected than expected
            if self.num_speakers > 1 and len(speaker_stats) < self.num_speakers:
                # Filter out UNKNOWN
                real_speakers = [s for s in speaker_stats.keys() if s != "UNKNOWN" and s != "SPEAKER_UNKNOWN"]
                if len(real_speakers) < self.num_speakers:
                    logger.warning(
                        "DIARIZATION WARNING: Only %d speaker(s) detected (expected %d)! "
                        "This may indicate that voices are too similar or Pyannote needs tuning. "
                        "Detected speakers: %s",
                        len(real_speakers),
                        self.num_speakers,
                        real_speakers,
                    )

            # Format segments
            formatted_segments = []
            for idx, segment in enumerate(segments):
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", start))
                text = (segment.get("text") or "").strip()
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
                    "words": words,
                })

            logger.info(
                "=== TRANSCRIPTION PIPELINE COMPLETE: %d segments (language=%s) ===",
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
