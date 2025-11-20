import os
import uuid
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

import torch

from backend.services.youtube_downloader import YouTubeDownloader
from backend.services.transcription import TranscriptionService
from backend.services.highlight_analyzer import HighlightAnalyzer
from backend.services.translation import Translator
from backend.services.tts import TTSService
from backend.services.video_processor import VideoProcessor
from backend.utils.file_utils import get_temp_dir, get_output_dir, clear_temp_dir
from backend import config

router = APIRouter(prefix="/api/video", tags=["video"])

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for tasks and results
tasks = {}
analysis_results_cache = {}

# Lazy-loaded services
_services = {}


class AnalyzeRequest(BaseModel):
    youtube_url: str

class ProcessRequest(BaseModel):
    video_id: str
    segment_ids: List[str]
    vertical_method: str = "blur_background"

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    result: Optional[dict] = None

def get_service(name: str):
    """Lazy-load and cache services to save resources."""
    if name not in _services:
        logger.info(f"Initializing service: {name}")
        if name == "transcription":
            _services[name] = TranscriptionService(model_name=config.WHISPER_MODEL)
        elif name == "highlight_analyzer":
            _services[name] = HighlightAnalyzer()
        elif name == "translation":
            # Check if CUDA is available and use it for translation
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Translation service will use device: {device}")
            _services[name] = Translator(
                model_name=config.TRANSLATION_MODEL_NAME,
                device=device
            )
        elif name == "tts":
            _services[name] = TTSService(
                language=config.SILERO_LANGUAGE,
                speaker=config.SILERO_SPEAKER,
                model_version=config.SILERO_MODEL_VERSION
            )
        elif name == "renderer":
            _services[name] = VideoProcessor()
        else:
            raise ValueError(f"Unknown service: {name}")
    return _services[name]

def _calculate_iou(segment1: dict, segment2: dict) -> float:
    """Calculate Intersection over Union for two time segments."""
    start1, end1 = segment1['start_time'], segment1['end_time']
    start2, end2 = segment2['start_time'], segment2['end_time']

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    intersection = max(0, intersection_end - intersection_start)
    if intersection == 0:
        return 0.0

    union = (end1 - start1) + (end2 - start2) - intersection
    if union == 0:
        # This case happens if both segments are identical
        return 1.0
        
    return intersection / union

def _is_segment_fully_contained(inner: dict, outer: dict) -> bool:
    """Check if inner segment is fully contained within outer segment."""
    return (inner['start_time'] >= outer['start_time'] and 
            inner['end_time'] <= outer['end_time'])

def _filter_overlapping_segments(segments: list, iou_threshold: float = 0.7) -> list:
    """
    Filter out segments that have a high IoU with higher-scoring segments.
    Uses stricter filtering: IoU threshold 0.7, checks for full containment,
    and ensures minimum 10s gap between segment starts.
    """
    if not segments:
        return []

    # Ensure segments are sorted by highlight_score, descending
    segments.sort(key=lambda x: x.get('highlight_score', 0), reverse=True)
    
    kept_segments = []
    for segment in segments:
        should_keep = True
        
        for kept_segment in kept_segments:
            # Check 1: Full containment - if new segment is fully inside kept one, discard it
            if _is_segment_fully_contained(segment, kept_segment):
                logger.info(f"Segment {segment['id']} (score: {segment['highlight_score']:.2f}, {segment['start_time']:.1f}-{segment['end_time']:.1f}s) is fully contained in {kept_segment['id']} (score: {kept_segment['highlight_score']:.2f}, {kept_segment['start_time']:.1f}-{kept_segment['end_time']:.1f}s). Discarding.")
                should_keep = False
                break
            
            # Check 2: If kept segment is fully inside new one, remove kept and keep new (if new has better score)
            if _is_segment_fully_contained(kept_segment, segment):
                logger.info(f"Kept segment {kept_segment['id']} (score: {kept_segment['highlight_score']:.2f}) is fully contained in new {segment['id']} (score: {segment['highlight_score']:.2f}). Replacing.")
                kept_segments.remove(kept_segment)
                break
            
            # Check 3: IoU overlap threshold (stricter: 0.7)
            iou = _calculate_iou(segment, kept_segment)
            if iou >= iou_threshold:
                logger.info(f"Segment {segment['id']} (score: {segment['highlight_score']:.2f}) overlaps with {kept_segment['id']} (score: {kept_segment['highlight_score']:.2f}) with IoU {iou:.2f}. Discarding.")
                should_keep = False
                break
            
            # Check 4: Minimum gap between starts (10 seconds)
            if abs(segment['start_time'] - kept_segment['start_time']) < 10.0:
                logger.info(f"Segment {segment['id']} starts too close ({abs(segment['start_time'] - kept_segment['start_time']):.1f}s) to {kept_segment['id']}. Discarding.")
                should_keep = False
                break
        
        if should_keep:
            kept_segments.append(segment)
            
    # Re-sort by start time for chronological order in the UI
    kept_segments.sort(key=lambda x: x['start_time'])
    
    logger.info(f"Filtered {len(segments)} segments down to {len(kept_segments)} non-overlapping segments.")
    return kept_segments

def _analyze_video_task(task_id: str, youtube_url: str):
    try:
        tasks[task_id] = {"status": "processing", "progress": 0.1, "message": "Downloading video..."}
        
        # 1. Download video
        downloader = YouTubeDownloader(output_dir=get_temp_dir())
        video_info = downloader.download(youtube_url)
        video_id = video_info['video_id']
        video_path = video_info['video_path']
        
        if not video_path:
            raise Exception("Failed to download video.")

        # This is a blocking call that performs the full analysis
        _run_analysis_pipeline(task_id, video_id, video_path)

    except Exception as e:
        logger.error(f"Error in analysis task {task_id}: {e}", exc_info=True)
        tasks[task_id] = {"status": "failed", "progress": tasks[task_id]['progress'], "message": str(e)}

def _analyze_local_video_task(task_id: str, filename: str):
    try:
        tasks[task_id] = {"status": "processing", "progress": 0.1, "message": "Processing local video..."}
        
        video_id = os.path.splitext(filename)[0]
        video_path = os.path.join(get_temp_dir(), filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found in temp directory: {filename}")

        _run_analysis_pipeline(task_id, video_id, video_path)

    except Exception as e:
        logger.error(f"Error in local analysis task {task_id}: {e}", exc_info=True)
        tasks[task_id] = {"status": "failed", "progress": tasks[task_id].get('progress', 0.1), "message": str(e)}

def _run_analysis_pipeline(task_id: str, video_id: str, video_path: str):
    """Core analysis logic, shared by YouTube and local video."""
    
    # 2. Transcribe audio
    tasks[task_id] = {"status": "processing", "progress": 0.2, "message": "Transcribing audio..."}
    transcriber = get_service("transcription")
    transcription_result = transcriber.transcribe_audio_from_video(video_path)
    segments = transcription_result['segments']

    tasks[task_id] = {"status": "processing", "progress": 0.5, "message": "Analyzing content..."}
    
    # 3. Analyze for highlights
    analyzer = get_service("highlight_analyzer")
    highlights = analyzer.analyze_segments(segments)

    # 4. Filter out highly overlapping segments
    logger.info(f"Filtering {len(highlights)} segments for overlaps...")
    filtered_highlights = _filter_overlapping_segments(highlights, iou_threshold=0.5)
    logger.info(f"Found {len(filtered_highlights)} non-overlapping segments.")
    
    tasks[task_id] = {"status": "processing", "progress": 0.7, "message": "Translating segments..."}

    # 5. Translate text to Russian
    translator = get_service("translation")
    
    # Prepare texts for translation
    texts_to_translate = [seg['text'] for seg in filtered_highlights]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache before translation")

    logger.info("Starting batch translation...")
    translations = translator.translate_batch(texts_to_translate)
    logger.info("Batch translation finished. Assigning results...")

    # Assign translations back to segments
    for i, segment in enumerate(filtered_highlights):
        segment['text_en'] = segment.pop('text')  # Rename original text
        segment['text_ru'] = translations[i]
    logger.info("Translation results assigned.")

    # Cache the result
    analysis_result = {
        'video_id': video_id,
        'video_path': video_path,
        'segments': filtered_highlights
    }
    analysis_results_cache[video_id] = analysis_result
    logger.info("Analysis results cached.")

    # Prepare response object
    response_result = {
        'video_id': video_id,
        'segments': filtered_highlights
    }
    
    logger.info("Response object created. Updating task to completed.")
    tasks[task_id] = {
        "status": "completed",
        "progress": 1.0,
        "message": "Analysis complete",
        "result": response_result
    }

def _process_segments_task(task_id: str, video_id: str, segment_ids: list, vertical_method: str = "blur_background"):
    try:
        tasks[task_id] = {"status": "processing", "progress": 0.1, "message": "Preparing to render..."}
        
        # 1. Retrieve cached analysis
        if video_id not in analysis_results_cache:
            raise ValueError("Analysis results not found. Please analyze the video first.")
        
        cached_data = analysis_results_cache[video_id]
        video_path = cached_data['video_path']
        all_segments = cached_data['segments']
        
        # Filter segments to be processed
        segments_to_process = [seg for seg in all_segments if seg['id'] in segment_ids]
        
        if not segments_to_process:
            raise ValueError("No valid segments selected for processing.")
            
        # 2. Generate audio for each segment
        tasks[task_id] = {"status": "processing", "progress": 0.3, "message": "Generating audio..."}
        tts_service = get_service("tts")
        output_dir = get_output_dir(video_id)
        
        for segment in segments_to_process:
            audio_path = os.path.join(output_dir, f"{segment['id']}.wav")
            tts_service.synthesize_and_save(segment['text_ru'], audio_path)
            segment['audio_path'] = audio_path
            
        # 3. Render videos
        tasks[task_id] = {"status": "processing", "progress": 0.6, "message": "Rendering videos..."}
        renderer = get_service("renderer")
        
        output_files = []
        for segment in segments_to_process:
            output_path = os.path.join(output_dir, f"{segment['id']}.mp4")
            final_clip = renderer.create_vertical_video(
                video_path=video_path,
                audio_path=segment['audio_path'],
                text=segment['text_ru'],
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                method=vertical_method
            )
            renderer.save_video(final_clip, output_path)
            
            # Relative path for API response
            relative_path = os.path.join(video_id, f"{segment['id']}.mp4")
            output_files.append(relative_path)
        
        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Processing complete",
            "result": {"output_videos": output_files}
        }
        
    except Exception as e:
        logger.error(f"Error in processing task {task_id}: {e}", exc_info=True)
        tasks[task_id] = {"status": "failed", "progress": tasks[task_id]['progress'], "message": str(e)}


@router.post("/analyze", response_model=TaskStatus)
async def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Task queued"}
    background_tasks.add_task(_analyze_video_task, task_id, request.youtube_url)
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="Task queued")

@router.post("/analyze-local", response_model=TaskStatus)
async def analyze_local_video(filename: str, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Task queued"}
    background_tasks.add_task(_analyze_local_video_task, task_id, filename)
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="Task queued")

@router.post("/upload-video", response_model=TaskStatus)
async def upload_video(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    temp_dir = get_temp_dir()
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        return TaskStatus(
            task_id=task_id,
            status="completed",
            progress=1.0,
            message=f"File '{file.filename}' uploaded successfully.",
            result={"filename": file.filename}
        )
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")

@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # If the task is completed, we can add the result to the response
    if task['status'] == 'completed':
        return TaskStatus(task_id=task_id, **task)
    
    return TaskStatus(task_id=task_id, status=task['status'], progress=task['progress'], message=task['message'])

@router.post("/process", response_model=TaskStatus)
async def process_segments(request: ProcessRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Processing task queued"}
    background_tasks.add_task(
        _process_segments_task,
        task_id,
        request.video_id,
        request.segment_ids,
        request.vertical_method
    )
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="Processing task queued")

@router.on_event("startup")
async def startup_event():
    # Clean up temp and output directories on startup
    logger.info("Clearing temp and output directories...")
    # clear_temp_dir() # Optional: decide if you want to clear on startup
    if not os.path.exists(get_temp_dir()):
        os.makedirs(get_temp_dir())
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    logger.info("Directories are ready.")
