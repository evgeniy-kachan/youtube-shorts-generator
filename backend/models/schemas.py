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
    vertical_method: str = Field(
        default="blur_background",
        description="Method for vertical conversion: blur_background, center_crop, smart_crop"
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

