"""ElevenLabs Dubbing API integration."""
import logging
import time
from pathlib import Path

import httpx

from backend.config import ELEVENLABS_API_KEY

logger = logging.getLogger(__name__)


class DubbingService:
    """ElevenLabs Dubbing API - automatic video/audio translation with voice cloning."""
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    # Polling settings
    POLL_INTERVAL = 5  # seconds
    MAX_POLL_TIME = 600  # 10 minutes max wait
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ELEVENLABS_API_KEY
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is not configured")
    
    def _headers(self) -> dict:
        return {"xi-api-key": self.api_key}
    
    def create_dub(
        self,
        file_path: str,
        source_lang: str = "en",
        target_lang: str = "ru",
        num_speakers: int = 0,  # 0 = auto-detect
        watermark: bool = False,
        project_name: str | None = None,
    ) -> str:
        """
        Create a dubbing project.
        
        Args:
            file_path: Path to video/audio file
            source_lang: Source language code (e.g. 'en', 'auto')
            target_lang: Target language code (e.g. 'ru', 'es')
            num_speakers: Number of speakers (0 = auto-detect)
            watermark: Add watermark to output
            project_name: Optional project name
            
        Returns:
            dubbing_id for tracking the project
        """
        url = f"{self.BASE_URL}/dubbing"
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        project_name = project_name or file_path.stem
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "video/mp4")}
            data = {
                "name": project_name,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "num_speakers": str(num_speakers),
                "watermark": str(watermark).lower(),
            }
            
            logger.info(
                "Creating dubbing project: %s → %s (file: %s)",
                source_lang, target_lang, file_path.name
            )
            
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    url,
                    headers=self._headers(),
                    files=files,
                    data=data,
                )
                response.raise_for_status()
        
        result = response.json()
        dubbing_id = result.get("dubbing_id")
        
        if not dubbing_id:
            raise RuntimeError(f"No dubbing_id in response: {result}")
        
        logger.info("Dubbing project created: %s", dubbing_id)
        return dubbing_id
    
    def get_status(self, dubbing_id: str) -> dict:
        """
        Get dubbing project status.
        
        Returns:
            dict with 'status' ('dubbing', 'dubbed', 'failed') and other info
        """
        url = f"{self.BASE_URL}/dubbing/{dubbing_id}"
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
        
        return response.json()
    
    def wait_for_completion(
        self,
        dubbing_id: str,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        progress_callback: callable | None = None,
    ) -> dict:
        """
        Wait for dubbing to complete.
        
        Args:
            dubbing_id: Dubbing project ID
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time in seconds
            progress_callback: Optional callback(status_dict) for progress updates
            
        Returns:
            Final status dict
            
        Raises:
            TimeoutError if max_wait exceeded
            RuntimeError if dubbing failed
        """
        poll_interval = poll_interval or self.POLL_INTERVAL
        max_wait = max_wait or self.MAX_POLL_TIME
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Dubbing timed out after {max_wait}s")
            
            status = self.get_status(dubbing_id)
            status_code = status.get("status", "unknown")
            
            logger.info(
                "Dubbing %s: status=%s (%.0fs elapsed)",
                dubbing_id, status_code, elapsed
            )
            
            if progress_callback:
                progress_callback(status)
            
            if status_code == "dubbed":
                logger.info("Dubbing completed: %s", dubbing_id)
                return status
            
            if status_code == "failed":
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"Dubbing failed: {error}")
            
            time.sleep(poll_interval)
    
    def download_dubbed_audio(
        self,
        dubbing_id: str,
        output_path: str,
        language_code: str = "ru",
    ) -> str:
        """
        Download dubbed audio file.
        
        Args:
            dubbing_id: Dubbing project ID
            output_path: Path to save the audio file
            language_code: Target language code
            
        Returns:
            Path to downloaded file
        """
        url = f"{self.BASE_URL}/dubbing/{dubbing_id}/audio/{language_code}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading dubbed audio: %s → %s", dubbing_id, output_path)
        
        with httpx.Client(timeout=120.0) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        logger.info("Dubbed audio saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1024 / 1024)
        return str(output_path)
    
    def dub_video(
        self,
        video_path: str,
        output_audio_path: str,
        source_lang: str = "en",
        target_lang: str = "ru",
        num_speakers: int = 0,
        progress_callback: callable | None = None,
    ) -> str:
        """
        Full dubbing pipeline: upload → wait → download.
        
        Args:
            video_path: Path to source video
            output_audio_path: Path to save dubbed audio
            source_lang: Source language
            target_lang: Target language
            num_speakers: Number of speakers (0 = auto)
            progress_callback: Optional progress callback
            
        Returns:
            Path to dubbed audio file
        """
        # Step 1: Create dubbing project
        dubbing_id = self.create_dub(
            file_path=video_path,
            source_lang=source_lang,
            target_lang=target_lang,
            num_speakers=num_speakers,
        )
        
        # Step 2: Wait for completion
        self.wait_for_completion(
            dubbing_id=dubbing_id,
            progress_callback=progress_callback,
        )
        
        # Step 3: Download result
        return self.download_dubbed_audio(
            dubbing_id=dubbing_id,
            output_path=output_audio_path,
            language_code=target_lang,
        )


# Singleton instance
_dubbing_service: DubbingService | None = None


def get_dubbing_service() -> DubbingService:
    """Get or create DubbingService instance."""
    global _dubbing_service
    if _dubbing_service is None:
        _dubbing_service = DubbingService()
    return _dubbing_service

