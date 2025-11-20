"""Text-to-Speech service using Silero TTS."""
import torch
import logging
from pathlib import Path
import torchaudio

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech using Silero models."""
    
    def __init__(self, language: str = "ru", speaker: str = "v3_1_ru", device: str = "cuda"):
        """
        Initialize Silero TTS model.
        
        Args:
            language: Language code (ru, en, etc.)
            speaker: Speaker model ID
            device: Device to use (cuda, cpu)
        """
        logger.info(f"Loading Silero TTS model: {language}/{speaker}")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.language = language
        self.speaker = speaker
        self.sample_rate = 48000
        
        # Load Silero TTS model
        # Model will be downloaded automatically on first use
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=speaker
        )
        
        self.model = self.model.to(self.device)
        
        logger.info("Silero TTS model loaded successfully")
        
    def synthesize(self, text: str, output_path: str, speaker: str | None = None) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            speaker: Speaker name (xenia, eugene, etc.)
            
        Returns:
            Path to generated audio file
        """
        try:
            # Generate audio
            audio = self.model.apply_tts(
                text=text,
                speaker=speaker or self.speaker,
                sample_rate=self.sample_rate
            )
            
            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torchaudio.save(
                str(output_path),
                audio.unsqueeze(0),
                self.sample_rate
            )
            
            logger.info(f"Audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise
    
    def synthesize_and_save(self, text: str, output_path: str, speaker: str | None = None) -> str:
        """Compatibility wrapper used elsewhere in the codebase."""
        return self.synthesize(text, output_path, speaker=speaker)
    
    def get_available_speakers(self) -> list:
        """Get list of available speakers for current language."""
        speakers = {
            'ru': ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'],
            'en': ['lj', 'random'],
        }
        return speakers.get(self.language, ['random'])

