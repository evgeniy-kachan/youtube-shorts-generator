"""YouTube video downloader service."""
import yt_dlp
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Download videos from YouTube."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def download(self, youtube_url: str) -> Dict[str, any]:
        """
        Download video from YouTube.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Dictionary with video info and file paths
        """
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info without downloading first
                info = ydl.extract_info(youtube_url, download=False)
                
                # Check duration (max 2 hours)
                duration = info.get('duration', 0)
                if duration > 7200:
                    raise ValueError(f"Video is too long: {duration/60:.1f} minutes (max 120 minutes)")
                
                # Download video
                logger.info(f"Downloading video: {info['title']}")
                info = ydl.extract_info(youtube_url, download=True)
                
                video_id = info['id']
                video_path = self.output_dir / f"{video_id}.mp4"
                
                # Download audio separately for better quality
                audio_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio',
                    'outtmpl': str(self.output_dir / f'{video_id}_audio.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(audio_opts) as ydl_audio:
                    ydl_audio.download([youtube_url])
                
                # Find audio file (extension might vary)
                audio_path = None
                for ext in ['m4a', 'webm', 'mp3']:
                    potential_path = self.output_dir / f"{video_id}_audio.{ext}"
                    if potential_path.exists():
                        audio_path = potential_path
                        break
                
                return {
                    'video_id': video_id,
                    'title': info['title'],
                    'duration': duration,
                    'video_path': str(video_path),
                    'audio_path': str(audio_path) if audio_path else None,
                    'thumbnail': info.get('thumbnail'),
                    'description': info.get('description', ''),
                }
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise

