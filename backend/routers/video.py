"""Video processing API routes."""
import asyncio
import uuid
from pathlib import Path
from typing import Dict
import logging
import torch

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from backend.models.schemas import (
    VideoAnalysisRequest,
    VideoAnalysisResponse,
    ProcessRequest,
    ProcessResponse,
    TaskStatus,
    Segment,
    ProcessedSegment
)
from backend.services.youtube_downloader import YouTubeDownloader
from backend.services.transcription import TranscriptionService
from backend.services.highlight_analyzer import HighlightAnalyzer
from backend.services.translation import Translator
from backend.services.tts import TTSService
from backend.services.video_processor import VideoProcessor
from backend.config import TEMP_DIR, OUTPUT_DIR
import backend.config as config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])

# Store task statuses in memory (in production, use Redis or database)
tasks: Dict[str, TaskStatus] = {}

# Store analysis results
analysis_cache: Dict[str, dict] = {}

# Initialize services (lazy loading)
_services = {}


def get_service(name: str):
    """Get or initialize a service."""
    if name not in _services:
        if name == "downloader":
            _services[name] = YouTubeDownloader(TEMP_DIR)
        elif name == "transcription":
            _services[name] = TranscriptionService(
                model_name=config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE
            )
        elif name == "analyzer":
            _services[name] = HighlightAnalyzer()
        elif name == "translation":
            _services[name] = Translator(
                model_name=config.TRANSLATION_MODEL_NAME,
                device="cuda"
            )
        elif name == "tts":
            _services[name] = TTSService(
                language=config.SILERO_LANGUAGE,
                speaker=config.SILERO_SPEAKER
            )
        elif name == "video_processor":
            _services[name] = VideoProcessor(OUTPUT_DIR)
    
    return _services[name]


@router.post("/analyze", response_model=dict)
async def analyze_video(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze YouTube video and find interesting segments.
    This is a long-running task, returns task_id for status checking.
    """
    task_id = str(uuid.uuid4())
    
    # Create task status
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Task created"
    )
    
    # Run analysis in background
    background_tasks.add_task(
        _analyze_video_task,
        task_id,
        request.youtube_url
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Analysis started"
    }


@router.post("/analyze-local", response_model=dict)
async def analyze_local_video(background_tasks: BackgroundTasks, filename: str):
    """
    Analyze local video file (for regions where YouTube is blocked).
    
    Usage:
    1. Upload video via scp: scp video.mp4 root@SERVER:/opt/.../temp/my_video.mp4
    2. Call this endpoint with filename=my_video.mp4
    """
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Task created"
    )
    
    # Run analysis in background
    background_tasks.add_task(
        _analyze_local_video_task,
        task_id,
        filename
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Local file analysis started"
    }


async def _analyze_video_task(task_id: str, youtube_url: str):
    """Background task for video analysis."""
    try:
        # Update status
        tasks[task_id].status = "processing"
        tasks[task_id].progress = 0.1
        tasks[task_id].message = "Downloading video..."
        
        # 1. Download video
        downloader = get_service("downloader")
        video_info = downloader.download(youtube_url)
        
        tasks[task_id].progress = 0.2
        tasks[task_id].message = "Transcribing audio..."
        
        # 2. Transcribe
        transcription_service = get_service("transcription")
        segments = transcription_service.transcribe(video_info['audio_path'])
        
        tasks[task_id].progress = 0.5
        tasks[task_id].message = "Analyzing content..."
        
        # 3. Analyze for highlights
        analyzer = get_service("analyzer")
        highlights = analyzer.analyze_segments(
            segments,
            min_duration=config.MIN_SEGMENT_DURATION,
            max_duration=config.MAX_SEGMENT_DURATION
        )
        
        tasks[task_id].progress = 0.7
        tasks[task_id].message = "Translating to Russian..."
        
        # 4. Translate highlights (batch processing for speed)
        # Free GPU memory: unload Ollama model and clear CUDA cache
        del analyzer
        _services.pop("analyzer", None)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache before translation")
        
        translator = get_service("translation")
        if highlights:
            texts = [h['text'] for h in highlights]
            translations = translator.translate_batch(texts)
            for highlight, translation in zip(highlights, translations):
                highlight['text_ru'] = translation
        
        tasks[task_id].progress = 0.9
        tasks[task_id].message = "Finalizing..."
        
        # Store results
        analysis_cache[video_info['video_id']] = {
            'video_info': video_info,
            'highlights': highlights,
            'segments': segments
        }
        
        # Prepare response
        response = VideoAnalysisResponse(
            video_id=video_info['video_id'],
            title=video_info['title'],
            duration=video_info['duration'],
            segments=[
                Segment(
                    id=h['id'],
                    start_time=h['start_time'],
                    end_time=h['end_time'],
                    duration=h['duration'],
                    text_en=h['text'],
                    text_ru=h['text_ru'],
                    highlight_score=h['highlight_score'],
                    criteria_scores=h['criteria_scores']
                )
                for h in highlights
            ]
        )
        
        # Update task status
        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].message = "Analysis completed"
        tasks[task_id].result = response.dict()
        
    except Exception as e:
        logger.error(f"Error in analysis task: {e}", exc_info=True)
        tasks[task_id].status = "failed"
        tasks[task_id].message = str(e)


async def _analyze_local_video_task(task_id: str, filename: str):
    """Background task for local video analysis."""
    try:
        tasks[task_id].status = "processing"
        tasks[task_id].progress = 0.1
        tasks[task_id].message = "Processing local file..."
        
        video_path = TEMP_DIR / filename
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {filename}. Upload to temp/ first.")
        
        # Get video duration
        import ffmpeg
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe['format']['duration'])
        
        if duration > 7200: # 120 minutes limit
            raise ValueError(f"Video too long: {duration/60:.1f} min (max 120 min)")
        
        # Extract audio
        audio_path = TEMP_DIR / f"{video_path.stem}_audio.wav"
        # Check if audio already exists
        if not audio_path.exists():
            video_processor = get_service("video_processor")
            video_processor.extract_audio(str(video_path), str(audio_path))
        
        video_info = {
            'video_id': video_path.stem,
            'title': filename,
            'duration': duration,
            'video_path': str(video_path),
            'audio_path': str(audio_path)
        }
        
        tasks[task_id].progress = 0.3
        tasks[task_id].message = "Transcribing audio..."
        
        transcription_service = get_service("transcription")
        segments = transcription_service.transcribe(video_info['audio_path'])
        
        tasks[task_id].progress = 0.5
        tasks[task_id].message = "Analyzing content..."
        
        analyzer = get_service("analyzer")
        highlights = analyzer.analyze_segments(
            segments,
            min_duration=config.MIN_SEGMENT_DURATION,
            max_duration=config.MAX_SEGMENT_DURATION
        )
        
        tasks[task_id].progress = 0.7
        tasks[task_id].message = "Translating to Russian..."
        
        # Free GPU memory: unload Ollama model and clear CUDA cache
        del analyzer
        _services.pop("analyzer", None)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache before translation")
        
        # Batch translation for speed
        translator = get_service("translation")
        if highlights:
            texts = [h['text'] for h in highlights]
            logger.info("Starting batch translation...")
            translations = translator.translate_batch(texts)
            logger.info("Batch translation finished. Assigning results...")
            for highlight, translation in zip(highlights, translations):
                highlight['text_ru'] = translation
            logger.info("Translation results assigned.")
        
        tasks[task_id].progress = 0.9
        tasks[task_id].message = "Finalizing..."
        
        analysis_cache[video_info['video_id']] = {
            'video_info': video_info,
            'highlights': highlights,
            'segments': segments
        }
        
        logger.info("Analysis results cached.")

        response = VideoAnalysisResponse(
            video_id=video_info['video_id'],
            title=video_info['title'],
            duration=video_info['duration'],
            segments=[
                Segment(
                    id=h['id'],
                    start_time=h['start_time'],
                    end_time=h['end_time'],
                    duration=h['duration'],
                    text_en=h['text'],
                    text_ru=h['text_ru'],
                    highlight_score=h['highlight_score'],
                    criteria_scores=h['criteria_scores']
                )
                for h in highlights
            ]
        )
        
        logger.info("Response object created. Updating task to completed.")

        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].message = "Analysis completed"
        tasks[task_id].result = response.dict()
        
    except Exception as e:
        logger.error(f"Error in local analysis: {e}", exc_info=True)
        tasks[task_id].status = "failed"
        tasks[task_id].message = str(e)


@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]


@router.post("/process", response_model=dict)
async def process_segments(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process selected segments: cut video, add Russian TTS and subtitles.
    Returns task_id for status checking.
    """
    if request.video_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Video not found. Please analyze first.")
    
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Processing started"
    )
    
    # Run processing in background
    background_tasks.add_task(
        _process_segments_task,
        task_id,
        request.video_id,
        request.segment_ids,
        request.vertical_method
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Processing started"
    }


async def _process_segments_task(task_id: str, video_id: str, segment_ids: list, vertical_method: str = "blur_background"):
    """Background task for processing segments."""
    try:
        tasks[task_id].status = "processing"
        tasks[task_id].message = "Loading data..."
        
        # Get cached data
        cache = analysis_cache[video_id]
        video_info = cache['video_info']
        highlights = cache['highlights']
        
        # Filter selected segments
        selected_segments = [h for h in highlights if h['id'] in segment_ids]
        
        if not selected_segments:
            raise ValueError("No valid segments selected")
        
        # Initialize services
        tts_service = get_service("tts")
        video_processor = get_service("video_processor")
        
        processed_segments = []
        total = len(selected_segments)
        
        for i, segment in enumerate(selected_segments):
            tasks[task_id].progress = i / total
            tasks[task_id].message = f"Processing segment {i+1}/{total}..."
            
            segment_id = segment['id']
            start_time = segment['start_time']
            end_time = segment['end_time']
            text_ru = segment['text_ru']
            
            # Create output filenames
            video_output = OUTPUT_DIR / f"{video_id}_{segment_id}.mp4"
            audio_output = TEMP_DIR / f"{video_id}_{segment_id}_tts.wav"
            
            # 1. Generate TTS audio
            logger.info(f"Generating TTS for segment {segment_id}")
            tts_service.synthesize(text_ru, str(audio_output), speaker="xenia")
            
            # 2. Cut video segment
            logger.info(f"Cutting video segment {segment_id}")
            temp_video = TEMP_DIR / f"{video_id}_{segment_id}_temp.mp4"
            video_processor.cut_segment(
                video_info['video_path'],
                start_time,
                end_time,
                str(temp_video)
            )
            
            # 3. Create subtitles from transcription
            # Find matching transcription segments
            transcription_segments = cache['segments']
            subtitle_data = []
            
            for ts in transcription_segments:
                if ts['start'] >= start_time and ts['end'] <= end_time:
                    # Adjust timestamps relative to segment start
                    adjusted_segment = {
                        'start': ts['start'] - start_time,
                        'end': ts['end'] - start_time,
                        'text': text_ru,  # Use translated text
                        'words': []
                    }
                    
                    # Adjust word timestamps if available
                    if 'words' in ts and ts['words']:
                        for word in ts['words']:
                            adjusted_segment['words'].append({
                                'word': word['word'],
                                'start': word['start'] - start_time,
                                'end': word['end'] - start_time
                            })
                    
                    subtitle_data.append(adjusted_segment)
            
            # 4. Add audio and subtitles (with vertical conversion)
            logger.info(f"Adding audio and subtitles to segment {segment_id}")
            video_processor.add_audio_and_subtitles(
                str(temp_video),
                str(audio_output),
                subtitle_data,
                str(video_output),
                style="tiktok",
                convert_to_vertical=True,
                vertical_method=vertical_method
            )
            
            # Clean up temp files
            temp_video.unlink(missing_ok=True)
            audio_output.unlink(missing_ok=True)
            
            processed_segments.append(
                ProcessedSegment(
                    segment_id=segment_id,
                    download_url=f"/api/video/download/{video_id}/{segment_id}",
                    filename=video_output.name,
                    duration=end_time - start_time
                )
            )
        
        # Prepare response
        response = ProcessResponse(
            video_id=video_id,
            processed_segments=processed_segments
        )
        
        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].message = "Processing completed"
        tasks[task_id].result = response.dict()
        
    except Exception as e:
        logger.error(f"Error in processing task: {e}", exc_info=True)
        tasks[task_id].status = "failed"
        tasks[task_id].message = str(e)


@router.get("/download/{video_id}/{segment_id}")
async def download_segment(video_id: str, segment_id: str):
    """Download a processed video segment."""
    video_file = OUTPUT_DIR / f"{video_id}_{segment_id}.mp4"
    
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Video segment not found")
    
    return FileResponse(
        path=str(video_file),
        media_type="video/mp4",
        filename=f"{video_id}_{segment_id}.mp4"
    )


@router.delete("/cleanup/{video_id}")
async def cleanup_video(video_id: str):
    """Clean up temporary and output files for a video."""
    try:
        # Clean up temp files
        for file in TEMP_DIR.glob(f"{video_id}*"):
            file.unlink()
        
        # Clean up output files
        for file in OUTPUT_DIR.glob(f"{video_id}*"):
            file.unlink()
        
        # Remove from cache
        if video_id in analysis_cache:
            del analysis_cache[video_id]
        
        return {"message": "Cleanup completed"}
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))