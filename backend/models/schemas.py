"""Pydantic models for API requests and responses."""
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field


class VideoAnalysisRequest(BaseModel):
    """Request to analyze a YouTube video."""
    youtube_url: str = Field(..., description="YouTube video URL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            }
        }


class Segment(BaseModel):
    """A video segment with timestamps and metadata."""
    id: str
    start_time: float  # seconds
    end_time: float  # seconds
    duration: float  # seconds
    text_en: str  # Original English text
    text_ru: str  # Translated Russian text
    highlight_score: float  # 0.0 to 1.0
    criteria_scores: dict  # Individual criterion scores
    

class VideoAnalysisResponse(BaseModel):
    """Response from video analysis."""
    video_id: str
    title: str
    duration: float
    segments: List[Segment]
    

class ProcessRequest(BaseModel):
    """Request to process selected segments."""
    video_id: str
    segment_ids: List[str] = Field(..., description="List of segment IDs to process")
    tts_provider: str = Field(
        default="local",
        description="TTS backend to use: local, elevenlabs"
    )
    voice_mix: str = Field(
        default="male_duo",
        description="Voice combination preset (male_duo, mixed_duo, female_duo)"
    )
    vertical_method: str = Field(
        default="letterbox",
        description="Method for vertical conversion: letterbox, center_crop"
    )
    subtitle_animation: str = Field(
        default="fade",
        description="Subtitle animation preset: fade, bounce, slide, spark"
    )
    subtitle_position: str = Field(
        default="mid_low",
        description="Subtitle position preset identifier"
    )
    subtitle_font: str = Field(
        default="Montserrat Light",
        description="Font family name for subtitles"
    )
    subtitle_font_size: int = Field(
        default=86,
        description="Base subtitle font size"
    )
    subtitle_background: bool = Field(
        default=False,
        description="Enable background blur/box behind subtitles"
    )
    preserve_background_audio: bool = Field(
        default=True,
        description="Mix original track quietly under synthesized voice"
    )
    crop_focus: str = Field(
        default="center",
        description="Horizontal crop focus for center_crop method: center, left, right"
    )
    

class ProcessedSegment(BaseModel):
    """A processed video segment ready for download."""
    segment_id: str
    download_url: str
    filename: str
    duration: float
    

class ProcessResponse(BaseModel):
    """Response from processing segments."""
    video_id: str
    processed_segments: List[ProcessedSegment]
    

class TaskStatus(BaseModel):
    """Status of a processing task."""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: Optional[str] = None
    result: Optional[dict] = None


class DubbingRequest(BaseModel):
    """Request for AI dubbing using ElevenLabs Dubbing API."""
    video_id: str
    segment_id: str = Field(..., description="Segment ID to dub")
    source_lang: str = Field(default="en", description="Source language code")
    target_lang: str = Field(default="ru", description="Target language code")
    vertical_method: str = Field(
        default="center_crop",
        description="Method for vertical conversion: letterbox, center_crop"
    )
    crop_focus: str = Field(
        default="face_auto",
        description="Horizontal crop focus for center_crop method"
    )
    subtitle_animation: str = Field(default="fade", description="Subtitle animation")
    subtitle_position: str = Field(default="mid_low", description="Subtitle position")
    subtitle_font: str = Field(default="Montserrat Light", description="Font family")
    subtitle_font_size: int = Field(default=86, description="Font size")
    subtitle_background: bool = Field(default=False, description="Subtitle background")

