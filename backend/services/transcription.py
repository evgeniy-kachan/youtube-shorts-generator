"""Video transcription service using Whisper."""
from faster_whisper import WhisperModel
from typing import List, Dict
import logging
from pathlib import Path

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
                best_of=5,
                temperature=0.0,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                ),
                word_timestamps=True,  # Important for word-level subtitles
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

