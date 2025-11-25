"""Video transcription service using Whisper."""
from faster_whisper import WhisperModel
from typing import List, Dict
import logging
from pathlib import Path
import ffmpeg

from backend.config import TEMP_DIR

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribe audio using Whisper model."""
    
    def __init__(self, model_name: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        """
        Initialize Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use (cuda, cpu)
            compute_type: Computation type (float16, int8, float32)
        """
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")
        
    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Source language code (default: en)
            
        Returns:
            List of segments with text and timestamps
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=1,
                temperature=0.0,
                max_segment_length=40,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=450,
                    min_silence_duration_ms=1100
                ),
                word_timestamps=False,
            )
            
            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            # Convert segments to list of dictionaries
            result = []
            for segment in segments:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'words': []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        segment_dict['words'].append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })
                
                result.append(segment_dict)
                
            logger.info(f"Transcription completed: {len(result)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
    
    def get_full_text(self, segments: List[Dict]) -> str:
        """Get full text from segments."""
        return " ".join([seg['text'] for seg in segments])

    def transcribe_audio_from_video(self, video_path: str, language: str = "en") -> Dict:
        """
        Extract audio from a video file and transcribe it.

        Returns:
            Dict with segments list and full text.
        """
        temp_dir = Path(TEMP_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"{Path(video_path).stem}_whisper.wav"
        try:
            (
                ffmpeg
                .input(video_path)
                .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            segments = self.transcribe(str(audio_path), language=language)
            return {
                "segments": segments,
                "text": self.get_full_text(segments),
                "audio_path": str(audio_path)
            }
        except Exception as e:
            logger.error(f"Error transcribing video {video_path}: {e}")
            raise
        finally:
            audio_path.unlink(missing_ok=True)

