import os
import re
import uuid
import logging
import subprocess
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import List, Optional

import ffmpeg
import httpx

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydub import AudioSegment

from backend.services.youtube_downloader import YouTubeDownloader
from backend.services.transcription import TranscriptionService
from backend.services.highlight_analyzer import HighlightAnalyzer, get_min_highlights
from backend.services.translation import Translator
from backend.services.tts import TTSService, ElevenLabsTTSService, ElevenLabsTTDService
from backend.services.video_processor import VideoProcessor
from backend.services.text_markup import TextMarkupService
from backend.services.dubbing import get_dubbing_service
from backend.utils.file_utils import get_temp_dir, get_output_dir, clear_temp_dir
from backend import config
from backend.models.schemas import ProcessRequest, TaskStatus, DubbingRequest

router = APIRouter(prefix="/api/video", tags=["video"])

logger = logging.getLogger(__name__)

# In-memory storage for tasks and results
tasks = {}
analysis_results_cache = {}

# Cache for uploaded videos (for development - can be easily removed)
uploaded_videos_cache = {}  # {video_id: {"filename": str, "path": str, "uploaded_at": str}}

# Lazy-loaded services
_services = {}


class AnalyzeRequest(BaseModel):
    youtube_url: str
    diarizer: str = Field(
        default="pyannote",
        description="Speaker diarization system: pyannote (fast) or nemo (accurate)"
    )

def get_service(name: str):
    """Lazy-load and cache services to save resources."""
    if name not in _services:
        logger.info(f"Initializing service: {name}")
        if name == "transcription":
            logger.info(
                "Creating TranscriptionService with diarization: num_speakers=%d, min=%d, max=%d",
                config.DIARIZATION_NUM_SPEAKERS,
                config.DIARIZATION_MIN_SPEAKERS,
                config.DIARIZATION_MAX_SPEAKERS,
            )
            _services[name] = TranscriptionService(
                model_name=config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
                num_speakers=config.DIARIZATION_NUM_SPEAKERS,
                min_speakers=config.DIARIZATION_MIN_SPEAKERS,
                max_speakers=config.DIARIZATION_MAX_SPEAKERS,
            )
        elif name == "highlight_analyzer":
            _services[name] = HighlightAnalyzer()
        elif name == "translation":
            _services[name] = Translator()
        elif name == "tts":
            _services[name] = TTSService(
                language=config.SILERO_LANGUAGE,
                speaker=config.SILERO_SPEAKER,
                model_version=config.SILERO_MODEL_VERSION
            )
        elif name == "text_markup":
            if not config.TTS_ENABLE_MARKUP:
                raise ValueError("Text markup service requested but disabled in config")
            _services[name] = TextMarkupService()
        elif name == "renderer":
            _services[name] = VideoProcessor(output_dir=get_output_dir())
        else:
            raise ValueError(f"Unknown service: {name}")
    return _services[name]


def get_tts_service(provider: str):
    """Return configured TTS service instance (local Silero or ElevenLabs TTD)."""
    normalized = (provider or "local").lower()
    if normalized == "elevenlabs":
        # Use TTD (Text-to-Dialogue) with eleven_v3 model for better multi-speaker support
        cache_key = "tts_elevenlabs_ttd"
        if cache_key not in _services:
            if not config.ELEVENLABS_API_KEY:
                raise ValueError("ElevenLabs API key is not configured on the server.")
            _services[cache_key] = ElevenLabsTTDService(
                api_key=config.ELEVENLABS_API_KEY,
                voice_id=config.ELEVENLABS_VOICE_ID,
                language=config.SILERO_LANGUAGE,
                base_url=config.ELEVENLABS_BASE_URL,
                request_timeout=120.0,  # TTD may take longer
                proxy_url=config.ELEVENLABS_PROXY,
            )
        return _services[cache_key]

    cache_key = "tts_local"
    if cache_key not in _services:
        _services[cache_key] = TTSService(
            language=config.SILERO_LANGUAGE,
            speaker=config.SILERO_SPEAKER,
            model_version=config.SILERO_MODEL_VERSION,
        )
    return _services[cache_key]


VOICE_MIX_PRESETS = {
    "male_duo": ["male", "male", "male"],
    "mixed_duo": ["male", "female", "male"],
    "female_duo": ["female", "female", "female"],
}


def _extract_unique_speakers(segments: list[dict]) -> list[str]:
    order: list[str] = []
    for segment in segments:
        for speaker_id in segment.get("speakers") or []:
            if speaker_id and speaker_id not in order:
                order.append(speaker_id)
    if not order:
        order = ["speaker_0"]
    return order


def _build_voice_plan(segments: list[dict], mix: str, num_speakers: int = 0) -> dict[str, str]:
    pattern = VOICE_MIX_PRESETS.get(mix, VOICE_MIX_PRESETS["male_duo"])
    male_pool = config.ELEVENLABS_VOICE_IDS_MALE or [config.ELEVENLABS_VOICE_ID]
    female_pool = config.ELEVENLABS_VOICE_IDS_FEMALE or [config.ELEVENLABS_VOICE_ID]
    male_iter = cycle(male_pool)
    female_iter = cycle(female_pool)

    plan: dict[str, str] = {}
    speakers = _extract_unique_speakers(segments)
    
    # If user specified num_speakers, ensure we have voice mappings for all potential speakers
    if num_speakers >= 2:
        # Add speaker names that might be assigned during processing
        for i in range(num_speakers):
            speaker_name = f"SPEAKER_{i:02d}"
            if speaker_name not in speakers:
                speakers.append(speaker_name)

    for idx, speaker_id in enumerate(speakers):
        desired_gender = pattern[min(idx, len(pattern) - 1)]
        if desired_gender == "female":
            plan[speaker_id] = next(female_iter)
        else:
            plan[speaker_id] = next(male_iter)

    plan["__default__"] = plan.get(speakers[0], config.ELEVENLABS_VOICE_ID) if speakers else config.ELEVENLABS_VOICE_ID
    return plan

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

def _filter_overlapping_segments(
    segments: list,
    iou_threshold: float = 0.7,
    min_start_gap: float = 10.0,
    desired_count: Optional[int] = None,
) -> list:
    """
    Filter out segments that have a high IoU with higher-scoring segments.
    Uses stricter filtering: IoU threshold 0.7, checks for full containment,
    and ensures minimum gap between segment starts.
    """
    if not segments:
        return []

    # Ensure segments are sorted by highlight_score, descending
    segments.sort(key=lambda x: x.get('highlight_score', 0), reverse=True)
    original_count = len(segments)
    
    kept_segments = []
    kept_ids = set()
    removed_segments = []
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
            if abs(segment['start_time'] - kept_segment['start_time']) < min_start_gap:
                logger.info(
                    f"Segment {segment['id']} starts too close ({abs(segment['start_time'] - kept_segment['start_time']):.1f}s) "
                    f"to {kept_segment['id']} (min gap {min_start_gap}s). Discarding."
                )
                should_keep = False
                break
        
        if should_keep:
            kept_segments.append(segment)
            kept_ids.add(segment['id'])
        else:
            removed_segments.append(segment)

    if desired_count and len(kept_segments) < desired_count and removed_segments:
        deficit = desired_count - len(kept_segments)
        logger.info(
            "Relaxing overlap filter to reach %d segments (currently %d, deficit %d).",
            desired_count,
            len(kept_segments),
            deficit,
        )
        for segment in removed_segments:
            if segment['id'] in kept_ids:
                continue
            kept_segments.append(segment)
            kept_ids.add(segment['id'])
            deficit -= 1
            if deficit <= 0:
                break
            
    # Re-sort by start time for chronological order in the UI
    kept_segments.sort(key=lambda x: x['start_time'])
    
    logger.info(
        "Filtered %d segments down to %d non-overlapping segments%s.",
        original_count,
        len(kept_segments),
        f" (target {desired_count})" if desired_count else "",
    )
    return kept_segments


def _scale_dialogue_offsets(dialogue: list[dict] | None, scale: float) -> None:
    """Scale cached TTS offsets/durations when we retime synthesized dialogue audio."""
    if not dialogue or not scale or abs(scale - 1.0) < 1e-3:
        return
    for idx, turn in enumerate(dialogue):
        old_start = turn.get("tts_start_offset")
        old_end = turn.get("tts_end_offset")
        
        # Scale turn-level timestamps
        for key in ("tts_start_offset", "tts_end_offset", "tts_duration"):
            value = turn.get(key)
            if isinstance(value, (int, float)):
                turn[key] = value * scale
        
        # Scale word-level timestamps if present
        tts_words = turn.get("tts_words")
        if tts_words:
            for word in tts_words:
                if "start" in word:
                    word["start"] *= scale
                if "end" in word:
                    word["end"] *= scale
        
        new_start = turn.get("tts_start_offset")
        new_end = turn.get("tts_end_offset")
        logger.info(
            "Scaled turn %d: %.2f-%.2fs -> %.2f-%.2fs (scale=%.3f, %d words)",
            idx,
            old_start or 0, old_end or 0,
            new_start or 0, new_end or 0,
            scale,
            len(tts_words) if tts_words else 0,
        )


def _split_words_by_pauses(
    words: list[dict], 
    pause_threshold: float = 0.4,
    require_sentence_end: bool = True,
) -> list[list[dict]]:
    """
    Split word-level timestamps into phrases based on pauses.
    
    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        pause_threshold: Minimum pause duration (seconds) to split on
        require_sentence_end: If True, only split when pause follows sentence-ending
                              punctuation (.!?;:) - prevents mid-sentence splits
        
    Returns:
        List of phrase groups (each phrase is a list of words)
    """
    if not words:
        return []
    
    # Sentence-ending punctuation (word must end with one of these to allow split)
    SENTENCE_END_CHARS = '.!?;:'
    
    phrases = []
    current_phrase = []
    
    for i, word in enumerate(words):
        current_phrase.append(word)
        
        # Check if there's a pause after this word
        if i < len(words) - 1:
            current_end = word.get("end", 0)
            next_start = words[i + 1].get("start", 0)
            pause_duration = next_start - current_end
            
            if pause_duration >= pause_threshold:
                # Check if we should split here
                should_split = True
                
                if require_sentence_end:
                    # Only split if word ends with sentence-ending punctuation
                    word_text = word.get("word", "").strip()
                    ends_with_punctuation = any(
                        word_text.endswith(char) for char in SENTENCE_END_CHARS
                    )
                    should_split = ends_with_punctuation
                
                if should_split:
                    # Significant pause at sentence boundary - end current phrase
                    phrases.append(current_phrase)
                    current_phrase = []
    
    # Add final phrase if not empty
    if current_phrase:
        phrases.append(current_phrase)
    
    return phrases


def _create_pseudo_dialogue_from_words(
    words: list[dict],
    speaker: str,
    segment_start: float,
    pause_threshold: float = 0.4,
) -> list[dict]:
    """
    Create pseudo-dialogue structure from word timestamps by detecting pauses.
    
    This allows single-speaker segments to be processed like multi-speaker dialogue,
    preserving natural pauses in speech.
    
    IMPORTANT: Only splits at pauses that follow sentence-ending punctuation (.!?;:)
    to prevent mid-sentence breaks that confuse translation.
    
    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        speaker: Speaker ID for all turns
        segment_start: Start time of the segment (for absolute timing)
        pause_threshold: Minimum pause duration to create new turn
        
    Returns:
        List of dialogue turns with 'speaker', 'text', 'start', 'end', 'words'
    """
    # Use smart splitting: pause + sentence-end punctuation
    phrases = _split_words_by_pauses(words, pause_threshold, require_sentence_end=True)
    
    if not phrases:
        return []
    
    dialogue = []
    for phrase_words in phrases:
        if not phrase_words:
            continue
        
        # Extract text and timing from phrase
        text = " ".join(w.get("word", "").strip() for w in phrase_words).strip()
        start = phrase_words[0].get("start", 0)
        end = phrase_words[-1].get("end", 0)
        
        dialogue.append({
            "speaker": speaker,
            "text": text,
            "start": start,
            "end": end,
            "words": phrase_words,
        })
    
    return dialogue


def _find_phrase_in_segment(segment: dict, phrase: str) -> float | None:
    """
    Find a phrase in segment transcription and return its start time (relative to segment start).
    
    Args:
        segment: Segment dict with 'words', 'text_ru', 'dialogue', etc.
        phrase: Phrase to search for (case-insensitive, partial match)
        
    Returns:
        Start time in seconds (relative to segment start) or None if not found
    """
    if not phrase or not phrase.strip():
        return None
    
    phrase_lower = phrase.strip().lower()
    segment_start = float(segment.get('start_time', 0))
    
    # Try to find in dialogue turns first (most accurate)
    # Prefer text_ru (translated) as user will input Russian phrases
    dialogue = segment.get('dialogue') or []
    for turn in dialogue:
        # Check both Russian and English text
        turn_text_ru = (turn.get('text_ru') or '').lower()
        turn_text_en = (turn.get('text') or '').lower()
        if phrase_lower in turn_text_ru or phrase_lower in turn_text_en:
            turn_start = float(turn.get('start', 0)) - segment_start
            logger.info(
                "Found phrase '%s' in dialogue turn at %.2fs (segment %s)",
                phrase, turn_start, segment.get('id')
            )
            return max(0.0, turn_start)
    
    # Fallback: search in words
    words = segment.get('words') or []
    if words:
        # Build text from words and find position
        full_text = ' '.join(w.get('word', '') for w in words).lower()
        if phrase_lower in full_text:
            # Find the word index where phrase starts
            words_text = ''
            for idx, word in enumerate(words):
                words_text += word.get('word', '') + ' '
                if phrase_lower in words_text.lower():
                    word_start = float(word.get('start', 0)) - segment_start
                    logger.info(
                        "Found phrase '%s' in words at %.2fs (segment %s)",
                        phrase, word_start, segment.get('id')
                    )
                    return max(0.0, word_start)
    
    # Last resort: search in text_ru (full segment text)
    text_ru = (segment.get('text_ru') or segment.get('text_en') or '').lower()
    if phrase_lower in text_ru:
        # Approximate: use segment start (not very accurate)
        logger.warning(
            "Found phrase '%s' in segment text but no word timestamps (segment %s). Using segment start.",
            phrase, segment.get('id')
        )
        return 0.0
    
    logger.warning("Phrase '%s' not found in segment %s", phrase, segment.get('id'))
    return None


def _rediarize_segment(
    video_path: str,
    segment: dict,
    num_speakers: int = 2,
    hf_token: str | None = None,
) -> list[dict] | None:
    """
    Run diarization on a specific segment (more accurate for short fragments).
    
    Args:
        video_path: Path to source video
        segment: Segment dict with 'start_time', 'end_time'
        num_speakers: Expected number of speakers
        hf_token: HuggingFace token (if None, uses env var)
        
    Returns:
        List of diarization segments or None if failed
    """
    try:
        from backend.services.diarization_runner import get_diarization_runner
        import tempfile
        import subprocess
        from pathlib import Path
        
        segment_start = float(segment.get('start_time', 0))
        segment_end = float(segment.get('end_time', 0))
        segment_duration = segment_end - segment_start
        
        if segment_duration < 1.0:
            logger.warning("Segment too short for rediarization: %.2fs", segment_duration)
            return None
        
        logger.info(
            "REDIARIZATION [%s]: Running diarization on segment %.2f-%.2fs (%.2fs duration, %d speakers)",
            segment.get('id'), segment_start, segment_end, segment_duration, num_speakers
        )
        
        # Extract audio segment
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_audio = Path(tmpdir) / "segment_audio.wav"
            
            # Extract audio segment using ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(segment_start),
                "-t", str(segment_duration),
                "-ac", "1",
                "-ar", "16000",
                "-vn",
                str(segment_audio),
            ]
            
            logger.debug("Extracting segment audio: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            if not segment_audio.exists():
                logger.error("Failed to extract segment audio")
                return None
            
            # Run diarization on extracted segment
            runner = get_diarization_runner()
            if hf_token:
                runner.hf_token = hf_token
            runner.num_speakers = num_speakers
            
            diar_segments = runner.run(str(segment_audio))
            
            if not diar_segments:
                logger.warning("Rediarization returned no segments")
                return None
            
            # Adjust timestamps to be relative to segment start
            adjusted_segments = []
            for seg in diar_segments:
                adjusted_segments.append({
                    "start": float(seg.get("start", 0)) + segment_start,
                    "end": float(seg.get("end", 0)) + segment_start,
                    "speaker": seg.get("speaker", "SPEAKER_00"),
                })
            
            logger.info(
                "REDIARIZATION [%s]: Successfully rediarized, found %d segments",
                segment.get('id'), len(adjusted_segments)
            )
            return adjusted_segments
            
    except Exception as e:
        logger.error("Rediarization failed: %s", e, exc_info=True)
        return None


def _refine_turn_boundaries(dialogue: list[dict] | None) -> None:
    """
    Refine turn start/end times using word-level timestamps from WhisperX.
    
    Pyannote diarization can have inaccurate boundaries, but WhisperX word
    timestamps are more precise. This function updates turn boundaries to
    match actual speech.
    """
    if not dialogue:
        return
    
    for turn in dialogue:
        words = turn.get("words") or []
        if not words:
            continue
        
        # Find actual first and last word timestamps
        first_word_start = None
        last_word_end = None
        
        for w in words:
            w_start = w.get("start")
            w_end = w.get("end")
            if w_start is not None and (first_word_start is None or w_start < first_word_start):
                first_word_start = w_start
            if w_end is not None and (last_word_end is None or w_end > last_word_end):
                last_word_end = w_end
        
        # Update turn boundaries if word timestamps are available
        old_start, old_end = turn.get("start", 0), turn.get("end", 0)
        old_duration = old_end - old_start
        
        if first_word_start is not None:
            turn["start"] = first_word_start
        if last_word_end is not None:
            turn["end"] = last_word_end
        
        new_duration = turn["end"] - turn["start"]
        
        # Log significant corrections
        if abs(new_duration - old_duration) > 0.5:
            logger.info(
                "Refined turn [%s]: %.1f-%.1fs → %.1f-%.1fs (%.1fs → %.1fs)",
                turn.get("speaker", "?"),
                old_start, old_end,
                turn["start"], turn["end"],
                old_duration, new_duration,
            )


def _speed_match_audio_duration(
    audio_path: str,
    current_duration: float,
    target_duration: float,
    max_tempo: float = 1.25,
    min_tempo: float = 0.7,
) -> bool:
    """
    Use ffmpeg atempo filters to adjust audio duration to match target.
    
    Args:
        audio_path: Path to audio file to modify
        current_duration: Current audio duration in seconds
        target_duration: Target duration to match
        max_tempo: Maximum tempo (speed up limit, default 1.35 - balance between timing and quality)
        min_tempo: Minimum tempo (slow down limit, default 0.7)
    """
    if not audio_path or current_duration <= 0 or target_duration <= 0:
        return False

    tempo = current_duration / target_duration
    
    # Skip if difference is negligible (within 2%)
    if 0.98 <= tempo <= 1.02:
        return False
    
    # Clamp tempo to allowed range (for voice quality)
    original_tempo = tempo
    tempo = max(min_tempo, min(max_tempo, tempo))
    
    if abs(tempo - original_tempo) > 0.01:
        logger.info(
            "Clamping tempo from %.2f to %.2f for %s (voice-safe range: %.1f-%.1fx)",
            original_tempo,
            tempo,
            audio_path,
            min_tempo,
            max_tempo,
        )
    
    # Hard limit safety check
    if tempo > 4.0 or tempo < 0.5:
        logger.warning(
            "Skipping tempo adjustment for %s (tempo=%.2f outside hard limits 0.5-4.0).",
            audio_path,
            tempo,
        )
        return False

    temp_path = Path(audio_path).with_suffix(".tempo.wav")
    try:
        audio_stream = ffmpeg.input(audio_path).audio
        filters: list[float] = []
        remaining = tempo
        
        # FFmpeg atempo filter range: 0.5 - 2.0
        # For values outside this range, chain multiple filters
        if remaining > 1.0:
            # Speed up: chain 2.0x filters until remaining < 2.0
            while remaining > 2.0:
                filters.append(2.0)
                remaining /= 2.0
            filters.append(remaining)
        else:
            # Slow down: chain 0.5x filters until remaining > 0.5
            while remaining < 0.5:
                filters.append(0.5)
                remaining /= 0.5
            filters.append(remaining)

        for factor in filters:
            audio_stream = audio_stream.filter("atempo", factor)

        (
            ffmpeg
            .output(
                audio_stream,
                str(temp_path),
                acodec="pcm_s16le",
                ac=1,
                ar="44100",
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        os.replace(temp_path, audio_path)
        action = "sped up" if tempo > 1.0 else "slowed down"
        logger.info(
            "Audio %s %s by tempo %.2f to match target duration %.2fs",
            audio_path,
            action,
            tempo,
            target_duration,
        )
        return True
    except ffmpeg.Error as exc:
        stderr = (exc.stderr or b"").decode(errors="ignore")
        logger.warning("Tempo adjustment failed for %s: %s", audio_path, stderr.strip())
        temp_path.unlink(missing_ok=True)
        return False
def _generate_thumbnail(video_path: str, video_id: str, timestamp: float = 5.0) -> Optional[str]:
    """Generate a single thumbnail frame for preview."""
    thumbnails_dir = Path(config.TEMP_DIR) / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = thumbnails_dir / f"{video_id}.jpg"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(timestamp, 0.0)),
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(thumbnail_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            logger.warning(
                "Thumbnail generation failed for %s: %s",
                video_id,
                result.stderr.decode(errors="ignore"),
            )
            return None
        return str(thumbnail_path)
    except Exception as exc:
        logger.warning("Thumbnail generation error for %s: %s", video_id, exc)
        return None

def _analyze_video_task(task_id: str, youtube_url: str, diarizer: str = "pyannote"):
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
        _run_analysis_pipeline(task_id, video_id, video_path, diarizer=diarizer)

    except Exception as e:
        logger.error(f"Error in analysis task {task_id}: {e}", exc_info=True)
        tasks[task_id] = {"status": "failed", "progress": tasks[task_id]['progress'], "message": str(e)}

def _analyze_local_video_task(task_id: str, filename: str, analysis_mode: str = "fast", diarizer: str = "pyannote"):
    try:
        tasks[task_id] = {"status": "processing", "progress": 0.1, "message": "Processing local video..."}
        
        video_id = os.path.splitext(filename)[0]
        video_path = os.path.join(get_temp_dir(), filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found in temp directory: {filename}")

        _run_analysis_pipeline(task_id, video_id, video_path, analysis_mode, diarizer=diarizer)

    except Exception as e:
        logger.error(f"Error in local analysis task {task_id}: {e}", exc_info=True)
        tasks[task_id] = {"status": "failed", "progress": tasks[task_id].get('progress', 0.1), "message": str(e)}

def _run_analysis_pipeline(task_id: str, video_id: str, video_path: str, analysis_mode: str = "fast", diarizer: str = "pyannote"):
    """Core analysis logic, shared by YouTube and local video."""
    
    thumbnail_path = _generate_thumbnail(video_path, video_id)
    thumbnail_url = f"/api/video/thumbnail/{video_id}" if thumbnail_path else None

    # 2. Transcribe audio + diarization
    # Check if Redis Queue is available for GPU tasks
    from backend.services.task_queue import is_queue_available, get_task_queue
    
    diarizer_label = "NeMo MSDD" if diarizer == "nemo" else "Pyannote"
    tasks[task_id] = {"status": "processing", "progress": 0.2, "message": f"Транскрипция + диаризация ({diarizer_label})..."}
    
    if is_queue_available():
        # Use GPU Worker via Redis Queue (production mode)
        logger.info("Using GPU Worker for transcription (diarizer=%s)", diarizer)
        queue = get_task_queue()
        
        # Extract audio for transcription
        import tempfile
        import ffmpeg as ffmpeg_lib
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_path = tmp_audio.name
        
        try:
            # Extract audio
            (
                ffmpeg_lib.input(video_path)
                .output(audio_path, acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            
            # Enqueue transcription + diarization task
            job = queue.enqueue_transcription(
                audio_path=audio_path,
                diarizer=diarizer,
                model=config.WHISPER_MODEL,
                language="en",
                num_speakers=config.DIARIZATION_NUM_SPEAKERS,
            )
            
            # Wait for result (blocking)
            logger.info("Waiting for GPU worker result (job_id=%s)...", job.id)
            transcription_result = job.wait_result(timeout=1800)  # 30 min max
            
            segments = transcription_result['segments']
            logger.info(
                "GPU Worker completed: %d segments, %d speakers, diarizer=%s",
                len(segments),
                transcription_result.get('num_speakers', 0),
                transcription_result.get('diarizer_used', diarizer),
            )
        finally:
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    else:
        # Fallback: direct call (development mode or Redis unavailable)
        logger.warning("Redis Queue not available, using direct transcription (slower)")
        transcriber = get_service("transcription")
        transcription_result = transcriber.transcribe_audio_from_video(video_path)
        segments = transcription_result['segments']
    
    video_duration = segments[-1]['end'] if segments else 0.0
    min_highlight_count = get_min_highlights(video_duration)

    # 3. Analyze for highlights
    # Select model based on analysis_mode: 'fast' = deepseek-chat, 'deep' = deepseek-reasoner
    analysis_model = "deepseek-reasoner" if analysis_mode == "deep" else "deepseek-chat"
    mode_label = "глубокий (R1)" if analysis_mode == "deep" else "быстрый"
    tasks[task_id] = {"status": "processing", "progress": 0.5, "message": f"Анализ контента ({mode_label})..."}
    
    analyzer = HighlightAnalyzer(model_name=analysis_model)
    highlights = analyzer.analyze_segments(segments)

    # 4. Filter out highly overlapping segments
    logger.info(
        "Filtering %d segments for overlaps (min count %d)...",
        len(highlights),
        min_highlight_count,
    )
    filtered_highlights = _filter_overlapping_segments(
        highlights,
        iou_threshold=0.5,
        desired_count=min_highlight_count,
    )
    logger.info(f"Found {len(filtered_highlights)} non-overlapping segments.")
    
    tasks[task_id] = {"status": "processing", "progress": 0.7, "message": "Translating segments..."}

    # 5. Translate text to Russian
    translator = get_service("translation")
    markup_service = get_service("text_markup") if config.TTS_ENABLE_MARKUP else None
    
    # Prepare texts for translation
    # We collect ALL dialogue turns from ALL segments to batch translate them
    all_turns = []
    for seg in filtered_highlights:
        # If dialogue structure is present, use it. Otherwise fallback to main text.
        if seg.get('dialogue'):
            for turn in seg['dialogue']:
                all_turns.append(turn)
        else:
            # Fake turn for segments without dialogue structure
            all_turns.append({'text': seg['text'], 'parent_seg': seg})

    texts_to_translate = [turn['text'] for turn in all_turns]
    
    logger.info(f"Starting batch translation for {len(texts_to_translate)} items...")
    translations = translator.translate_batch(texts_to_translate)
    
    if os.getenv("DEBUG_SAVE_TRANSLATIONS", "0") == "1":
        debug_dir = Path(config.TEMP_DIR) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{video_id}_translations.txt"
        with open(debug_path, "w", encoding="utf-8") as debug_file:
            for original, translated in zip(texts_to_translate, translations):
                debug_file.write("=== ORIGINAL ===\n")
                debug_file.write(original.strip() + "\n")
                debug_file.write("--- TRANSLATED ---\n")
                debug_file.write((translated or "").strip() + "\n\n")
    
    logger.info("Batch translation finished. Assigning results...")

    # Assign translations back to dialogue turns
    for i, turn in enumerate(all_turns):
        turn['text_ru'] = translations[i]
        
        # Markup (only if needed, though for dialogue we might want to do it later or now)
        # For now, let's just store raw translated text in the turn
    
    # Reconstruct full Russian text for each segment
    for segment in filtered_highlights:
        segment['text_en'] = segment.pop('text')  # Rename original text
        
        if segment.get('dialogue'):
            # Join translated turns to form the full segment text
            ru_parts = [turn['text_ru'] for turn in segment['dialogue']]
            full_ru = " ".join(ru_parts)
            segment['text_ru'] = full_ru
        else:
            # Should correspond to the logic above where we appended a fake turn
            # But wait, 'all_turns' has references. The loop above updated 'turn' dicts.
            # If we used 'parent_seg', we need to retrieve it.
            # Actually, simpler: if no dialogue, we just used the text directly.
            # Let's handle the 'fake turn' case properly.
            # Since 'all_turns' contains mutable dicts from 'dialogue', they are updated in place.
            # For the fallback case, we need to manually update the segment.
            pass # Handled by reconstruction below if dialogue exists.
                 # If NO dialogue, we need to find which turn belonged to this segment.
    
    # Fix for segments without dialogue (fallback)
    # Since we flattened everything into 'all_turns', we need to know which turns belong to which segment
    # Re-iterating is safer
    
    current_turn_idx = 0
    for segment in filtered_highlights:
        if segment.get('dialogue'):
            num_turns = len(segment['dialogue'])
            # These turns were updated in place in 'all_turns' list
            ru_parts = [t.get('text_ru', '') for t in segment['dialogue']]
            segment['text_ru'] = " ".join(ru_parts)
            current_turn_idx += num_turns
        else:
            # Single text item
            segment['text_ru'] = translations[current_turn_idx]
            current_turn_idx += 1

        # Markup for the FULL text (for subtitles)
        if markup_service:
            # logger.info("Calling TextMarkup for segment %s", segment['id'])
            segment['text_ru_tts'] = markup_service.mark_text(segment['text_ru'])
        else:
            # logger.info("Markup service unavailable, using raw text for segment %s", segment['id'])
            segment['text_ru_tts'] = segment['text_ru']

    logger.info("Translation results assigned.")

    # Cache the result
    analysis_result = {
        'video_id': video_id,
        'video_path': video_path,
        'segments': filtered_highlights,
        'transcript_segments': transcription_result['segments'],
        'thumbnail_path': thumbnail_path,
    }
    analysis_results_cache[video_id] = analysis_result
    logger.info("Analysis results cached.")

    # Prepare response object
    response_result = {
        'video_id': video_id,
        'segments': filtered_highlights,
        'thumbnail_url': thumbnail_url,
    }
    
    logger.info("Response object created. Updating task to completed.")
    tasks[task_id] = {
        "status": "completed",
        "progress": 1.0,
        "message": "Analysis complete",
        "result": response_result
    }

def _save_transcription_json(segments: list, output_dir: Path, video_id: str) -> Path:
    """
    Collect word-level timestamps from all segments and save in Yandex transcription format.
    
    Args:
        segments: List of processed segments with dialogue and tts_words
        output_dir: Directory where to save the JSON file (Path object)
        video_id: Video ID for filename
    
    Returns:
        Path to saved JSON file
    """
    import json
    from datetime import datetime
    
    transcription_data = {
        "video_id": video_id,
        "timestamp": datetime.now().isoformat(),
        "segments": []
    }
    
    for segment in segments:
        segment_id = segment.get('id', 'unknown')
        dialogue = segment.get('dialogue', [])
        
        # Collect all words from all turns in this segment
        all_words = []
        full_text_parts_ru = []
        full_text_parts_en = []
        
        for turn in dialogue:
            # Russian text
            turn_text_ru = turn.get('text_ru') or ''
            # Remove emotion tags
            clean_text_ru = re.sub(r'\[[\w]+\]\s*', '', turn_text_ru)
            if clean_text_ru:
                full_text_parts_ru.append(clean_text_ru)
            
            # English original text
            turn_text_en = turn.get('text') or ''
            clean_text_en = re.sub(r'\[[\w]+\]\s*', '', turn_text_en)
            if clean_text_en:
                full_text_parts_en.append(clean_text_en)
            
            tts_words = turn.get('tts_words', [])
            if tts_words:
                all_words.extend(tts_words)
        
        if all_words:
            full_text_ru = " ".join(full_text_parts_ru)
            full_text_en = " ".join(full_text_parts_en)
            
            # Fallback to segment-level text if dialogue didn't have English
            if not full_text_en:
                full_text_en = segment.get('text_en') or segment.get('text') or ''
            
            segment_duration = all_words[-1]['end'] - all_words[0]['start'] if all_words else 0
            original_duration = segment.get('end_time', 0) - segment.get('start_time', 0)
            
            segment_data = {
                "segment_id": segment_id,
                "text": full_text_ru,
                "text_en": full_text_en,  # English original for comparison
                "words": all_words,
                "duration": segment_duration,
                "original_duration": original_duration,  # Original video segment duration
                "language": "ru",
                "start_time": segment.get('start_time', 0),
                "end_time": segment.get('end_time', 0),
                "word_count_ru": len(full_text_ru.split()),
                "word_count_en": len(full_text_en.split()) if full_text_en else 0,
            }
            transcription_data["segments"].append(segment_data)
    
    # Save to JSON file
    json_filename = f"transcription_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = output_dir / json_filename
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_data, f, ensure_ascii=False, indent=2)
    
    logger.info(
        "Saved transcription JSON: %s (%d segments, %d total words)",
        json_path,
        len(transcription_data["segments"]),
        sum(len(seg.get('words', [])) for seg in transcription_data["segments"])
    )
    
    return json_path


def _process_segments_task(
    task_id: str,
    video_id: str,
    segment_ids: list,
    tts_provider: str = "local",
    voice_mix: str = "male_duo",
    vertical_method: str = "letterbox",
    subtitle_animation: str = "fade",
    subtitle_position: str = "mid_low",
    subtitle_font: str = "Montserrat Light",
    subtitle_font_size: int = 86,
    subtitle_background: bool = False,
    subtitle_glow: bool = True,
    subtitle_gradient: bool = False,
    preserve_background_audio: bool = True,
    crop_focus: str = "center",
    speaker_color_mode: str = "colored",
    num_speakers: int = 0,
    speaker_change_times: str = "",
    speaker_change_phrases: str = "",
    rediarize_segments: bool = False,
):
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
        tts_service = get_tts_service(tts_provider)
        voice_plan = None
        if (tts_provider or "").lower() == "elevenlabs":
            voice_plan = _build_voice_plan(segments_to_process, voice_mix, num_speakers)
        output_dir = get_output_dir(video_id)
        
        for segment in segments_to_process:
            audio_path = os.path.join(output_dir, f"{segment['id']}.wav")
            
            # Optional: Rediarize segment for better accuracy (only in AUTO mode)
            # In manual mode (num_speakers >= 2), user controls speaker assignment via phrases/times
            if rediarize_segments and num_speakers == 0:
                logger.info(
                    "REDIARIZATION [%s]: Running in AUTO mode - pyannote will detect speakers",
                    segment['id']
                )
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                rediar_segments = _rediarize_segment(
                    video_path=video_path,
                    segment=segment,
                    num_speakers=0,  # Let pyannote auto-detect
                    hf_token=hf_token,
                )
                if rediar_segments:
                    # Rebuild dialogue from rediarization results
                    # Match rediarization segments with existing dialogue turns
                    segment_start = float(segment.get('start_time', 0))
                    dialogue = segment.get('dialogue') or []
                    
                    # Create mapping: time -> speaker from rediarization
                    time_speaker_map = {}
                    for rediar_seg in rediar_segments:
                        rediar_start = float(rediar_seg.get('start', 0)) - segment_start
                        rediar_end = float(rediar_seg.get('end', 0)) - segment_start
                        rediar_speaker = rediar_seg.get('speaker', 'SPEAKER_00')
                        # Map all times in this range to this speaker
                        for turn in dialogue:
                            turn_start = float(turn.get('start', 0)) - segment_start
                            if rediar_start <= turn_start < rediar_end:
                                turn['speaker'] = rediar_speaker
                    
                    logger.info(
                        "REDIARIZATION [%s]: Updated %d turns with new speaker assignments",
                        segment['id'], len([t for t in dialogue if 'speaker' in t])
                    )
            
            # DIARIZATION DIAGNOSTIC: Log speaker info for this segment
            dialogue = segment.get('dialogue') or []
            speakers_in_segment = set()
            for turn in dialogue:
                spk = turn.get('speaker')
                if spk:
                    speakers_in_segment.add(spk)
            
            # Always log speaker info (was only logging multi-speaker before)
            logger.info(
                "DIARIZATION [%s]: %d turns, %d speaker(s): %s | TTS Provider: %s | User num_speakers: %s",
                segment['id'], len(dialogue), len(speakers_in_segment), 
                list(speakers_in_segment) if speakers_in_segment else ['NO_SPEAKERS'],
                tts_provider,
                num_speakers if num_speakers > 0 else 'auto'
            )
            
            # If user specified num_speakers >= 2 but diarization found only 1, force multi-speaker
            detected_speakers = len(speakers_in_segment)
            if num_speakers >= 2 and detected_speakers < num_speakers and dialogue:
                # Parse speaker change times/phrases (relative to segment start)
                change_times = []
                
                # Priority: phrases > times
                if speaker_change_phrases:
                    # Find phrases in segment and get their start times
                    phrases = [p.strip() for p in speaker_change_phrases.split(',') if p.strip()]
                    for phrase in phrases:
                        phrase_time = _find_phrase_in_segment(segment, phrase)
                        if phrase_time is not None:
                            change_times.append(phrase_time)
                        else:
                            logger.warning(
                                "Phrase '%s' not found in segment %s, skipping",
                                phrase, segment.get('id')
                            )
                    change_times.sort()
                    logger.info(
                        "SPEAKER CHANGE [%s]: Found %d phrases, times: %s",
                        segment['id'], len(change_times), change_times
                    )
                elif speaker_change_times:
                    # Fallback to time-based
                    try:
                        change_times = [float(t.strip()) for t in speaker_change_times.split(',') if t.strip()]
                        change_times.sort()
                    except ValueError:
                        logger.warning("Invalid speaker_change_times format: %s", speaker_change_times)
                
                speaker_names = [f"SPEAKER_{i:02d}" for i in range(num_speakers)]
                segment_start = float(segment.get('start_time', 0))
                
                if change_times:
                    # Use time-based speaker assignment
                    logger.info(
                        "SPEAKER OVERRIDE [%s]: Using time-based assignment. Change times: %s (relative to segment start)",
                        segment['id'], change_times
                    )
                    for turn in dialogue:
                        turn_start = float(turn.get('start', 0)) - segment_start
                        # Find which speaker based on change times
                        speaker_idx = 0
                        for i, change_time in enumerate(change_times):
                            if turn_start >= change_time:
                                speaker_idx = min(i + 1, num_speakers - 1)
                        turn['speaker'] = speaker_names[speaker_idx]
                        logger.debug(
                            "Turn at %.1fs -> %s (change_times=%s)",
                            turn_start, turn['speaker'], change_times
                        )
                else:
                    # Fallback to alternating assignment
                    logger.info(
                        "SPEAKER OVERRIDE [%s]: User requested %d speakers but only %d detected. Assigning alternating speakers.",
                        segment['id'], num_speakers, detected_speakers
                    )
                    for idx, turn in enumerate(dialogue):
                        turn['speaker'] = speaker_names[idx % num_speakers]
                
                speakers_in_segment = set(speaker_names)
            
            # Check if we have a dialogue structure for multi-speaker synthesis
            has_dialogue = bool(segment.get('dialogue') and len(segment['dialogue']) > 1)
            
            if has_dialogue and voice_plan:
                logger.info(
                    "TTS METHOD [%s]: Using TTD (Text-to-Dialogue) synthesis with %d turns",
                    segment['id'], len(segment['dialogue'])
                )
                # Refine turn boundaries using word-level timestamps
                _refine_turn_boundaries(segment['dialogue'])
                
                # Isochronic translation with precise timings
                try:
                    translator = Translator()
                    segment['dialogue'] = translator.translate_with_timings(
                        dialogue_turns=segment['dialogue'],
                        segment_context=segment.get('title', ''),
                    )
                except Exception as trans_exc:
                    logger.warning("Isochronic translation failed: %s", trans_exc)
                tts_service.synthesize_dialogue(
                    dialogue_turns=segment['dialogue'],
                    output_path=audio_path,
                    voice_map=voice_plan,
                    base_start=segment.get('start_time'),
                )
            else:
                # Single speaker - split by pauses to preserve natural rhythm
                segment_words = segment.get('words') or []
                speaker_chain = segment.get('speakers') or []
                primary_speaker = speaker_chain[0] if speaker_chain else "SPEAKER_0"
                segment_start = float(segment.get('start_time', 0))
                
                if segment_words:
                    # Create pseudo-dialogue by detecting pauses
                    pseudo_dialogue = _create_pseudo_dialogue_from_words(
                        words=segment_words,
                        speaker=primary_speaker,
                        segment_start=segment_start,
                        pause_threshold=0.4,
                    )
                    
                    if pseudo_dialogue:
                        segment['dialogue'] = pseudo_dialogue
                        has_dialogue = True
                        
                        logger.info(
                            "TTS METHOD [%s]: Using TTD (pseudo-dialogue from pauses) with %d turns",
                            segment['id'], len(pseudo_dialogue)
                        )
                        
                        _refine_turn_boundaries(segment['dialogue'])
                        
                        try:
                            translator = Translator()
                            segment['dialogue'] = translator.translate_with_timings(
                                dialogue_turns=segment['dialogue'],
                                segment_context=segment.get('title', ''),
                            )
                        except Exception as trans_exc:
                            logger.warning("Isochronic translation failed: %s", trans_exc)
                        
                        tts_service.synthesize_dialogue(
                            dialogue_turns=segment['dialogue'],
                            output_path=audio_path,
                            voice_map=voice_plan or {primary_speaker: tts_service.voice_id},
                            base_start=segment.get('start_time'),
                        )
                    else:
                        logger.warning("Failed to create pseudo-dialogue for %s", segment['id'])
                        has_dialogue = False
                
                # Fallback: if no words available, use old single-block approach
                if not has_dialogue:
                    logger.info(
                        "TTS METHOD [%s]: Using standard TTS (single-block, no dialogue structure)",
                        segment['id']
                    )
                    tts_text = segment.get('text_ru_tts') or segment.get('text_ru') or segment.get('text', '')
                    
                    voice_override = None
                    if voice_plan:
                        voice_override = voice_plan.get(primary_speaker) or voice_plan.get("__default__")
                    
                    segment_end = float(segment.get('end_time', 0))
                    target_duration = max(0.5, segment_end - segment_start)
                    
                    _, tts_words = tts_service.synthesize_and_save(
                        tts_text, audio_path,
                        speaker=voice_override,
                        target_duration=target_duration,
                    )
                    
                    # Create minimal pseudo-dialogue for compatibility
                    if tts_words:
                        try:
                            temp_audio = AudioSegment.from_file(audio_path)
                            single_audio_duration = temp_audio.duration_seconds or 0.0
                        except Exception:
                            single_audio_duration = tts_words[-1]["end"] if tts_words else 0.0
                        
                        segment['dialogue'] = [{
                            "speaker": primary_speaker,
                            "text": tts_text,
                            "text_ru": tts_text,
                            "tts_start_offset": 0.0,
                            "tts_end_offset": single_audio_duration,
                            "tts_duration": single_audio_duration,
                            "tts_words": tts_words,
                        }]
            
            segment['audio_path'] = audio_path
            try:
                audio_segment = AudioSegment.from_file(audio_path)
                audio_duration = audio_segment.duration_seconds or 0.0
            except Exception as audio_exc:
                logger.warning("Failed to read synthesized audio for %s: %s", segment['id'], audio_exc)
                audio_duration = 0.0
                audio_segment = None

            original_duration = max(0.1, float(segment.get('end_time', 0)) - float(segment.get('start_time', 0)))

            # Tempo adjustment: keep audio within 0.7x-1.35x of original duration
            # Increased from 1.25x to 1.35x to better fit longer translations
            duration_diff = abs(audio_duration - original_duration)
            if duration_diff > 0.2:  # More than 200ms difference
                before_duration = audio_duration
                
                logger.info(
                    "Applying tempo adjustment for %s: %.2fs -> %.2fs (range: 0.7x-1.35x)%s",
                    segment['id'],
                    audio_duration,
                    original_duration,
                    " [multi-speaker]" if has_dialogue else "",
                )
                
                if _speed_match_audio_duration(audio_path, audio_duration, original_duration, max_tempo=1.35, min_tempo=0.7):
                    try:
                        audio_segment = AudioSegment.from_file(audio_path)
                        audio_duration = audio_segment.duration_seconds or original_duration
                    except Exception as reload_exc:
                        logger.warning("Failed to reload sped-up audio for %s: %s", segment['id'], reload_exc)
                        audio_segment = None
                        audio_duration = original_duration

                    scale = audio_duration / before_duration if before_duration else 1.0
                    if scale and segment.get('dialogue'):
                        _scale_dialogue_offsets(segment['dialogue'], scale)
                    
                    # Only log significant tempo changes
                    if abs(scale - 1.0) > 0.1:
                        logger.info(
                            "TEMPO: %s %.2fs -> %.2fs (%.2fx)",
                            segment['id'], before_duration, audio_duration, scale
                        )

            target_duration = max(audio_duration, original_duration)

            if audio_segment and audio_duration + 0.01 < target_duration:
                pad_ms = int(round((target_duration - audio_duration) * 1000))
                audio_segment = audio_segment + AudioSegment.silent(duration=pad_ms)
                audio_segment.export(audio_path, format="wav")
                audio_duration = target_duration

            segment['tts_duration'] = audio_duration
            segment['target_duration'] = target_duration
            
        # 3. Render videos and generate descriptions in parallel
        tasks[task_id] = {"status": "processing", "progress": 0.6, "message": "Rendering videos..."}
        renderer = get_service("renderer")
        
        # Import DeepSeek client for description generation
        from backend.services.deepseek_client import DeepSeekClient
        
        output_files = []
        for idx, segment in enumerate(segments_to_process):
            # Update progress
            segment_progress = 0.6 + (0.35 * idx / len(segments_to_process))
            tasks[task_id] = {
                "status": "processing",
                "progress": segment_progress,
                "message": f"Rendering video {idx + 1}/{len(segments_to_process)}..."
            }
            
            output_path = os.path.join(output_dir, f"{segment['id']}.mp4")
            final_clip = renderer.create_vertical_video(
                video_path=video_path,
                audio_path=segment['audio_path'],
                text=segment['text_ru'],
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                target_duration=segment.get('target_duration'),
                method=vertical_method,
                subtitle_animation=subtitle_animation,
                subtitle_position=subtitle_position,
                subtitle_font=subtitle_font,
                subtitle_font_size=subtitle_font_size,
                subtitle_background=subtitle_background,
                subtitle_glow=subtitle_glow,
                subtitle_gradient=subtitle_gradient,
                dialogue=segment.get('dialogue'),
                preserve_background_audio=preserve_background_audio,
                crop_focus=crop_focus,
                speaker_color_mode=speaker_color_mode,
            )
            renderer.save_video(final_clip, output_path)
            
            # Generate description for this segment
            try:
                # Get English text - fallback to dialogue if main text is empty
                text_en = segment.get('text', '')
                if not text_en and segment.get('dialogue'):
                    text_en = ' '.join(turn.get('text', '') for turn in segment['dialogue'])
                
                deepseek_client = DeepSeekClient()
                description_data = deepseek_client.generate_shorts_description(
                    text_en=text_en,
                    text_ru=segment.get('text_ru', ''),
                    duration=segment.get('duration', 60),
                    highlight_score=segment.get('highlight_score', 0),
                )
                deepseek_client.close()
            except httpx.HTTPStatusError as http_exc:
                logger.warning(
                    "Description generation HTTP error for %s: status=%s, response=%s",
                    segment['id'], http_exc.response.status_code, http_exc.response.text[:200]
                )
                description_data = {
                    "category": "другое",
                    "title": "Интересный момент",
                    "description": "Смотрите до конца! 🔥",
                    "hashtags": ["#shorts", "#viral", "#рекомендации"]
                }
            except Exception as desc_exc:
                logger.warning("Failed to generate description for %s: %s", segment['id'], desc_exc)
                description_data = {
                    "category": "другое",
                    "title": "Интересный момент",
                    "description": "Смотрите до конца! 🔥",
                    "hashtags": ["#shorts", "#viral", "#рекомендации"]
                }
            
            # Relative path for API response
            relative_path = os.path.join(video_id, f"{segment['id']}.mp4")
            output_files.append({
                "path": relative_path,
                "segment_id": segment['id'],
                "description": description_data
            })
            
            # Clean up memory after each segment to prevent OOM
            # Especially important when processing multiple segments
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            
            logger.debug("Memory cleaned after segment %s", segment['id'])
        
        # Save transcription JSON with word timestamps (Yandex format)
        json_relative_path = None
        try:
            json_path = _save_transcription_json(segments_to_process, output_dir, video_id)
            logger.info("Transcription JSON saved: %s", json_path)
            # Save relative path for API response
            json_relative_path = f"{video_id}/{json_path.name}"
        except Exception as json_exc:
            logger.warning("Failed to save transcription JSON: %s", json_exc)
        
        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Processing complete",
            "result": {
                "output_videos": output_files,
                "transcription_json": json_relative_path  # Path to JSON file for download
            }
        }
        
    except Exception as e:
        logger.error(f"Error in processing task {task_id}: {e}", exc_info=True)
        tasks[task_id] = {"status": "failed", "progress": tasks[task_id]['progress'], "message": str(e)}


@router.post("/analyze", response_model=TaskStatus)
async def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Task queued"}
    background_tasks.add_task(_analyze_video_task, task_id, request.youtube_url, request.diarizer)
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="Task queued")

@router.post("/analyze-local", response_model=TaskStatus)
async def analyze_local_video(
    filename: str, 
    background_tasks: BackgroundTasks,
    analysis_mode: str = "fast",  # 'fast' (deepseek-chat) or 'deep' (deepseek-reasoner)
    diarizer: str = "pyannote",  # 'pyannote' (fast) or 'nemo' (accurate)
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Task queued"}
    background_tasks.add_task(_analyze_local_video_task, task_id, filename, analysis_mode, diarizer)
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="Task queued")

@router.post("/upload-video", response_model=TaskStatus)
async def upload_video(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    temp_dir = get_temp_dir()
    # Use temporary filename during upload to avoid leaving incomplete files
    temp_file_path = os.path.join(temp_dir, f"{file.filename}.tmp.{task_id}")
    final_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Stream file in chunks for better memory efficiency
        # Progress is tracked on client side via axios onUploadProgress
        total_size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(8192):  # 8KB chunks
                buffer.write(chunk)
                total_size += len(chunk)
        
        # Only rename to final name after successful upload
        if os.path.exists(temp_file_path):
            # If final file exists, remove it first (overwrite)
            if os.path.exists(final_file_path):
                os.remove(final_file_path)
            os.rename(temp_file_path, final_file_path)
        
        file_size_mb = total_size / (1024 * 1024)
        logger.info(f"File '{file.filename}' uploaded successfully. Size: {file_size_mb:.2f} MB")
        
        # Cache uploaded video metadata (for development - can be easily removed)
        video_id = str(uuid.uuid4())
        uploaded_videos_cache[video_id] = {
            "filename": file.filename,
            "path": final_file_path,
            "uploaded_at": datetime.now().isoformat(),
            "size_mb": file_size_mb
        }
        
        return TaskStatus(
            task_id=task_id,
            status="completed",
            progress=1.0,
            message=f"File '{file.filename}' uploaded successfully ({file_size_mb:.2f} MB).",
            result={"filename": file.filename, "video_id": video_id}
        )
    except Exception as e:
        # Clean up incomplete file on error
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Removed incomplete upload: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove incomplete file {temp_file_path}: {cleanup_error}")
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")

@router.get("/uploaded-videos")
async def get_uploaded_videos():
    """Get list of cached uploaded videos (for development - can be easily removed)."""
    videos = []
    for video_id, metadata in uploaded_videos_cache.items():
        # Check if file still exists
        if os.path.exists(metadata["path"]):
            videos.append({
                "video_id": video_id,
                "filename": metadata["filename"],
                "uploaded_at": metadata["uploaded_at"],
                "size_mb": metadata["size_mb"]
            })
        else:
            # Remove from cache if file doesn't exist
            del uploaded_videos_cache[video_id]
    return {"videos": videos}

@router.post("/use-cached-video/{video_id}")
async def use_cached_video(video_id: str):
    """Use cached video file instead of uploading (for development - can be easily removed)."""
    if video_id not in uploaded_videos_cache:
        raise HTTPException(status_code=404, detail="Cached video not found")
    
    metadata = uploaded_videos_cache[video_id]
    if not os.path.exists(metadata["path"]):
        del uploaded_videos_cache[video_id]
        raise HTTPException(status_code=404, detail="Cached video file not found")
    
    return {
        "filename": metadata["filename"],
        "path": metadata["path"],
        "video_id": video_id
    }

@router.get("/download/{video_id}/{segment_id}")
async def download_segment(video_id: str, segment_id: str):
    """Download processed video clip."""
    output_dir = get_output_dir(video_id)
    file_path = output_dir / f"{segment_id}.mp4"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed segment not found")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=file_path.name
    )

@router.get("/download-transcription/{video_id}")
async def download_transcription(video_id: str):
    """Download transcription JSON file with word timestamps (Yandex format)."""
    output_dir = get_output_dir(video_id)
    
    # Find the most recent transcription JSON file for this video
    json_files = list(output_dir.glob("transcription_*.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Transcription JSON not found")
    
    # Get the most recent file (by modification time)
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    return FileResponse(
        path=json_file,
        media_type="application/json",
        filename=json_file.name
    )

@router.get("/thumbnail/{video_id}")
async def get_thumbnail(video_id: str):
    """Return saved thumbnail for preview."""
    thumbnail_path = Path(config.TEMP_DIR) / "thumbnails" / f"{video_id}.jpg"
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(
        path=thumbnail_path,
        media_type="image/jpeg",
        filename=thumbnail_path.name,
    )

@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # If the task is completed, we can add the result to the response
    if task['status'] == 'completed':
        return TaskStatus(task_id=task_id, **task)
    
    return TaskStatus(task_id=task_id, status=task['status'], progress=task['progress'], message=task['message'])


@router.get("/session/restore")
async def restore_session():
    """
    Return the most recent active or completed task for session recovery.
    Used when browser wakes from sleep and needs to restore polling.
    """
    # Find the most recent task that is either processing or completed
    recent_task = None
    recent_task_id = None
    
    for task_id, task in tasks.items():
        if task.get('status') in ('processing', 'completed', 'pending'):
            # Prefer completed tasks, then processing, then pending
            if recent_task is None:
                recent_task = task
                recent_task_id = task_id
            elif task.get('status') == 'completed' and recent_task.get('status') != 'completed':
                recent_task = task
                recent_task_id = task_id
            elif task.get('status') == 'processing' and recent_task.get('status') == 'pending':
                recent_task = task
                recent_task_id = task_id
    
    if not recent_task:
        return {"has_session": False, "task": None}
    
    return {
        "has_session": True,
        "task": {
            "task_id": recent_task_id,
            "status": recent_task.get('status'),
            "progress": recent_task.get('progress', 0),
            "message": recent_task.get('message', ''),
            "result": recent_task.get('result') if recent_task.get('status') == 'completed' else None
        }
    }

@router.post("/process", response_model=TaskStatus)
async def process_segments(request: ProcessRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Processing task queued"}
    background_tasks.add_task(
        _process_segments_task,
        task_id,
        request.video_id,
        request.segment_ids,
        request.tts_provider,
        request.voice_mix,
        request.vertical_method,
        request.subtitle_animation,
        request.subtitle_position,
        request.subtitle_font,
        request.subtitle_font_size,
        request.subtitle_background,
        request.subtitle_glow,
        request.subtitle_gradient,
        request.preserve_background_audio,
        request.crop_focus,
        request.speaker_color_mode,
        request.num_speakers,
        request.speaker_change_times,
        request.speaker_change_phrases,
        request.rediarize_segments,
    )
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="Processing task queued")


@router.post("/dubbing", response_model=TaskStatus)
async def dub_segment(request: DubbingRequest, background_tasks: BackgroundTasks):
    """
    AI Dubbing using ElevenLabs Dubbing API.
    Automatically translates and dubs video with voice cloning.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Dubbing task queued"}
    background_tasks.add_task(
        _dubbing_task,
        task_id,
        request.video_id,
        request.segment_id,
        request.source_lang,
        request.target_lang,
        request.vertical_method,
        request.crop_focus,
        request.subtitle_animation,
        request.subtitle_position,
        request.subtitle_font,
        request.subtitle_font_size,
        request.subtitle_background,
        request.subtitle_glow,
        request.subtitle_gradient,
    )
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="Dubbing task queued")


def _dubbing_task(
    task_id: str,
    video_id: str,
    segment_id: str,
    source_lang: str,
    target_lang: str,
    vertical_method: str,
    crop_focus: str,
    subtitle_animation: str,
    subtitle_position: str,
    subtitle_font: str,
    subtitle_font_size: int,
    subtitle_background: bool,
    subtitle_glow: bool,
    subtitle_gradient: bool,
):
    """Background task for AI dubbing."""
    try:
        tasks[task_id] = {"status": "processing", "progress": 0.1, "message": "Loading segment data..."}
        
        # Get cached analysis results
        if video_id not in analysis_results_cache:
            raise ValueError(f"Analysis results not found for video {video_id}")
        
        cached = analysis_results_cache[video_id]
        segments = cached.get("segments", [])
        
        # Find the segment
        segment = None
        for s in segments:
            if s.get("id") == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValueError(f"Segment {segment_id} not found")
        
        # Get video path
        video_path = cached.get("video_path")
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Source video not found: {video_path}")
        
        output_dir = get_output_dir(video_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Cut segment from original video
        tasks[task_id] = {"status": "processing", "progress": 0.15, "message": "Cutting video segment..."}
        
        start_time = segment.get("start_time", 0)
        end_time = segment.get("end_time", 0)
        duration = end_time - start_time
        
        cut_segment_path = os.path.join(output_dir, f"{segment_id}_source.mp4")
        
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output(cut_segment_path, vcodec='libx264', crf=18, avoid_negative_ts='make_zero')
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        
        # Step 2: Send to ElevenLabs Dubbing API
        tasks[task_id] = {"status": "processing", "progress": 0.2, "message": "Uploading to ElevenLabs Dubbing..."}
        
        dubbed_audio_path = os.path.join(output_dir, f"{segment_id}_dubbed.mp3")
        
        def progress_callback(status):
            status_code = status.get("status", "unknown")
            tasks[task_id] = {
                "status": "processing",
                "progress": 0.3 + 0.4 * (0.5 if status_code == "dubbing" else 1.0),
                "message": f"ElevenLabs: {status_code}..."
            }
        
        dubbing_service = get_dubbing_service()
        dubbing_service.dub_video(
            video_path=cut_segment_path,
            output_audio_path=dubbed_audio_path,
            source_lang=source_lang,
            target_lang=target_lang,
            progress_callback=progress_callback,
        )
        
        # Step 3: Apply video processing (crop, no subtitles for AI dubbing)
        tasks[task_id] = {"status": "processing", "progress": 0.75, "message": "Applying video effects..."}
        
        renderer = get_service("renderer")
        final_output_path = os.path.join(output_dir, f"{segment_id}.mp4")
        
        # Process video with dubbed audio (no subtitles - ElevenLabs handles lip-sync)
        processed_path = renderer.create_vertical_video(
            video_path=cached.get("video_path"),  # Original source video
            audio_path=dubbed_audio_path,
            text="",  # No subtitles for AI dubbing
            start_time=start_time,
            end_time=end_time,
            method=vertical_method,
            subtitle_style="none",  # Disable subtitles
            subtitle_animation=subtitle_animation,
            subtitle_position=subtitle_position,
            subtitle_font=subtitle_font,
            subtitle_font_size=subtitle_font_size,
            subtitle_background=subtitle_background,
            subtitle_glow=subtitle_glow,
            subtitle_gradient=subtitle_gradient,
            crop_focus=crop_focus,
        )
        
        # Move to final path
        renderer.save_video(processed_path, final_output_path)
        
        # Clean up temporary files
        if os.path.exists(cut_segment_path):
            os.remove(cut_segment_path)
        if os.path.exists(dubbed_audio_path):
            os.remove(dubbed_audio_path)
        if os.path.exists(processed_path) and processed_path != final_output_path:
            os.remove(processed_path)
        
        # Use same format as regular processing for UI compatibility
        relative_path = os.path.join(video_id, f"{segment_id}.mp4")
        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "AI Dubbing completed!",
            "result": {"output_videos": [relative_path]}
        }
        
        logger.info("AI Dubbing completed for %s/%s", video_id, segment_id)
        
    except Exception as e:
        logger.error("AI Dubbing failed for task %s: %s", task_id, e, exc_info=True)
        tasks[task_id] = {
            "status": "failed",
            "progress": tasks[task_id].get("progress", 0.1),
            "message": str(e)
        }


class GenerateDescriptionRequest(BaseModel):
    """Request for generating shorts description."""
    text_en: str
    text_ru: str
    duration: float
    highlight_score: float = 0.0


class GenerateDescriptionResponse(BaseModel):
    """Response with generated description."""
    title: str
    description: str
    hashtags: List[str]


@router.post("/generate-description", response_model=GenerateDescriptionResponse)
async def generate_description(request: GenerateDescriptionRequest):
    """
    Generate a catchy title, description, and hashtags for a short-form video.
    Uses DeepSeek to create viral-optimized content.
    """
    try:
        from backend.services.deepseek_client import DeepSeekClient
        
        client = DeepSeekClient()
        result = client.generate_shorts_description(
            text_en=request.text_en,
            text_ru=request.text_ru,
            duration=request.duration,
            highlight_score=request.highlight_score,
        )
        client.close()
        
        return GenerateDescriptionResponse(
            title=result.get("title", ""),
            description=result.get("description", ""),
            hashtags=result.get("hashtags", ["#shorts"]),
        )
    except Exception as e:
        logger.error("Error generating description: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AUDIO TRANSCRIPTION ENDPOINT (for analyzing reference translations)
# ============================================================================

class TranscribeResponse(BaseModel):
    """Response model for audio transcription."""
    text: str
    words: list
    duration: float
    language: str


@router.post("/transcribe-audio", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "ru",
):
    """
    Transcribe audio file with word-level timestamps.
    
    Useful for analyzing reference translations (e.g., Yandex sync).
    Returns text and word timings for comparison with DeepSeek translations.
    """
    import tempfile
    import json
    
    # Validate file type
    allowed_types = [".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4"]
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".mp3"
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_types}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info("Transcribing uploaded audio: %s (%d bytes)", file.filename, len(content))
        
        # Convert to WAV if needed (WhisperX works best with WAV)
        wav_path = tmp_path
        if file_ext != ".wav":
            wav_path = tmp_path.replace(file_ext, ".wav")
            try:
                (
                    ffmpeg
                    .input(tmp_path)
                    .output(wav_path, ac=1, ar=16000)
                    .overwrite_output()
                    .run(quiet=True)
                )
            except Exception as e:
                logger.warning("FFmpeg conversion failed, using original: %s", e)
                wav_path = tmp_path
        
        # Run WhisperX transcription with word timestamps
        from backend.services.transcription_runner import TranscriptionRunner
        
        runner = TranscriptionRunner(
            enable_diarization=False,  # Don't need diarization for reference
        )
        
        result = runner.transcribe(
            audio_path=wav_path,
            model="large-v3",
            language=language,
            device="cuda" if config.WHISPER_DEVICE == "cuda" else "cpu",
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        
        # Extract words with timestamps
        words = []
        full_text_parts = []
        
        for segment in result.get("segments", []):
            segment_words = segment.get("words", [])
            for word_info in segment_words:
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": round(word_info.get("start", 0), 3),
                    "end": round(word_info.get("end", 0), 3),
                })
            # Also collect segment text
            if segment.get("text"):
                full_text_parts.append(segment["text"].strip())
        
        full_text = " ".join(full_text_parts)
        
        # Calculate duration from last word
        duration = words[-1]["end"] if words else 0.0
        
        logger.info(
            "Transcription complete: %d words, %.2fs duration, language=%s",
            len(words), duration, language
        )
        
        # Log word-level details for analysis
        logger.info("TRANSCRIPTION RESULT: '%s'", full_text)
        for i, w in enumerate(words[:10]):  # Log first 10 words
            logger.debug("  Word %d: %.2f-%.2f '%s'", i, w["start"], w["end"], w["word"])
        if len(words) > 10:
            logger.debug("  ... and %d more words", len(words) - 10)
        
        # Cleanup temp files
        try:
            os.unlink(tmp_path)
            if wav_path != tmp_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass
        
        return TranscribeResponse(
            text=full_text,
            words=words,
            duration=duration,
            language=language,
        )
        
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ============================================================================
# NeMo MSDD Diarization Endpoint
# ============================================================================

class NemoDiarizationRequest(BaseModel):
    """Request for NeMo MSDD diarization with optional auto-render."""
    video_id: str
    num_speakers: int = 0  # 0 = auto-detect
    max_speakers: int = 8
    # Auto-render after diarization
    auto_render: bool = False
    segment_ids: list[str] = []  # Segments to render (empty = all strict/extended)
    # Render settings (used if auto_render=True)
    tts_provider: str = "elevenlabs"
    voice_mix: str = "male_duo"
    vertical_method: str = "center_crop"
    subtitle_animation: str = "highlight"
    subtitle_position: str = "mid_low"
    subtitle_font: str = "Montserrat Light"
    subtitle_font_size: int = 86
    subtitle_background: bool = False
    subtitle_glow: bool = True
    subtitle_gradient: bool = False
    speaker_color_mode: str = "colored"
    preserve_background_audio: bool = True
    crop_focus: str = "center"


class NemoDiarizationResponse(BaseModel):
    """Response from NeMo diarization."""
    success: bool
    segments: list = []
    num_speakers_detected: int = 0
    message: str = ""


@router.get("/nemo/status")
async def get_nemo_status():
    """Check if NeMo MSDD diarization is available."""
    try:
        from backend.services.nemo_diarization_runner import is_nemo_available
        available = is_nemo_available()
        return {
            "available": available,
            "message": "NeMo MSDD is ready" if available else "NeMo MSDD not installed (venv-nemo not found)"
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"Error checking NeMo status: {str(e)}"
        }


@router.post("/nemo/diarize", response_model=TaskStatus)
async def run_nemo_diarization(request: NemoDiarizationRequest, background_tasks: BackgroundTasks):
    """
    Run NeMo MSDD diarization on the entire video.
    
    NeMo MSDD provides state-of-the-art speaker diarization for:
    - Similar voices
    - Overlapping speech
    - Long recordings
    """
    # Check if video exists
    if request.video_id not in analysis_results_cache:
        raise HTTPException(status_code=404, detail=f"Video {request.video_id} not found in cache")
    
    cached = analysis_results_cache[request.video_id]
    video_path = cached.get("video_path")
    
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Source video file not found")
    
    # Check if NeMo is available
    try:
        from backend.services.nemo_diarization_runner import is_nemo_available
        if not is_nemo_available():
            raise HTTPException(
                status_code=503, 
                detail="NeMo MSDD not available. Please install venv-nemo with nemo_toolkit[asr]"
            )
    except ImportError:
        raise HTTPException(status_code=503, detail="NeMo diarization runner not found")
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "NeMo diarization queued"}
    
    background_tasks.add_task(
        _nemo_diarization_task,
        task_id,
        request.video_id,
        video_path,
        request.num_speakers,
        request.max_speakers,
        request.auto_render,
        request.segment_ids,
        request.tts_provider,
        request.voice_mix,
        request.vertical_method,
        request.subtitle_animation,
        request.subtitle_position,
        request.subtitle_font,
        request.subtitle_font_size,
        request.subtitle_background,
        request.subtitle_glow,
        request.subtitle_gradient,
        request.speaker_color_mode,
        request.preserve_background_audio,
        request.crop_focus,
    )
    
    return TaskStatus(task_id=task_id, status="pending", progress=0.0, message="NeMo diarization started")


def _apply_nemo_diarization_to_segments(segments: list, nemo_segments: list) -> None:
    """
    Apply NeMo diarization results to segment dialogues.
    
    NeMo provides speaker labels for the entire video. This function maps
    those labels to individual segment dialogue turns based on time overlap.
    """
    if not nemo_segments:
        return
    
    for segment in segments:
        dialogue = segment.get('dialogue') or []
        if not dialogue:
            continue
        
        segment_start = float(segment.get('start_time', 0))
        segment_end = float(segment.get('end_time', 0))
        
        # Find NeMo segments that overlap with this segment
        relevant_nemo = [
            ns for ns in nemo_segments
            if float(ns.get('start', 0)) < segment_end and float(ns.get('end', 0)) > segment_start
        ]
        
        if not relevant_nemo:
            continue
        
        # Update dialogue turns with NeMo speaker labels
        for turn in dialogue:
            turn_start = float(turn.get('start', 0))
            turn_end = float(turn.get('end', turn_start + 0.1))
            turn_mid = (turn_start + turn_end) / 2
            
            # Find best matching NeMo segment (by midpoint overlap)
            best_match = None
            best_overlap = 0
            
            for ns in relevant_nemo:
                ns_start = float(ns.get('start', 0))
                ns_end = float(ns.get('end', 0))
                
                # Check if turn midpoint falls within NeMo segment
                if ns_start <= turn_mid <= ns_end:
                    overlap = min(turn_end, ns_end) - max(turn_start, ns_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = ns
            
            if best_match:
                turn['speaker'] = best_match.get('speaker', 'SPEAKER_00')
        
        # Log update
        speakers_in_segment = set(t.get('speaker') for t in dialogue if t.get('speaker'))
        logger.debug(
            "Applied NeMo diarization to segment %s: %d turns, speakers: %s",
            segment.get('id'), len(dialogue), list(speakers_in_segment)
        )


def _nemo_diarization_task(
    task_id: str,
    video_id: str,
    video_path: str,
    num_speakers: int,
    max_speakers: int,
    auto_render: bool = False,
    segment_ids: list[str] = None,
    tts_provider: str = "elevenlabs",
    voice_mix: str = "male_duo",
    vertical_method: str = "center_crop",
    subtitle_animation: str = "highlight",
    subtitle_position: str = "mid_low",
    subtitle_font: str = "Montserrat Light",
    subtitle_font_size: int = 86,
    subtitle_background: bool = False,
    subtitle_glow: bool = True,
    subtitle_gradient: bool = False,
    speaker_color_mode: str = "colored",
    preserve_background_audio: bool = True,
    crop_focus: str = "center",
):
    """Background task for NeMo MSDD diarization with optional auto-render."""
    try:
        from backend.services.nemo_diarization_runner import get_nemo_diarization_runner
        
        tasks[task_id] = {"status": "processing", "progress": 0.1, "message": "Freeing GPU for NeMo..."}
        
        # CRITICAL: Free GPU memory before NeMo subprocess to avoid CUBLAS conflicts
        # NeMo runs in a separate process and needs exclusive GPU access
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Unload any cached models that might hold GPU memory
                global _services
                _services.clear()
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared for NeMo diarization")
        except Exception as e:
            logger.warning("Could not clear GPU memory: %s", e)
        
        tasks[task_id] = {"status": "processing", "progress": 0.15, "message": "Initializing NeMo MSDD..."}
        
        runner = get_nemo_diarization_runner()
        runner.num_speakers = num_speakers
        runner.max_speakers = max_speakers
        
        tasks[task_id] = {"status": "processing", "progress": 0.2, "message": "Running NeMo diarization (this may take a while)..."}
        
        # Run diarization
        diar_segments = runner.run(video_path)
        
        if not diar_segments:
            tasks[task_id] = {
                "status": "failed",
                "progress": 1.0,
                "message": "NeMo diarization returned no segments",
                "result": {"success": False, "segments": [], "num_speakers_detected": 0}
            }
            return
        
        tasks[task_id] = {"status": "processing", "progress": 0.8, "message": "Processing diarization results..."}
        
        # Count unique speakers and calculate stats
        speaker_stats = {}
        for seg in diar_segments:
            speaker = seg.get("speaker", "unknown")
            duration = seg.get("end", 0) - seg.get("start", 0)
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"count": 0, "duration": 0.0}
            speaker_stats[speaker]["count"] += 1
            speaker_stats[speaker]["duration"] += duration
        
        num_speakers_detected = len(speaker_stats)
        total_speech = sum(s["duration"] for s in speaker_stats.values())
        
        # Log detailed results
        logger.info("=" * 60)
        logger.info("NEMO DIARIZATION RESULTS for %s", video_id)
        logger.info("=" * 60)
        logger.info("Speakers detected: %d", num_speakers_detected)
        logger.info("Total segments: %d", len(diar_segments))
        logger.info("Total speech duration: %.1f seconds", total_speech)
        logger.info("-" * 60)
        
        for speaker in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker]
            pct = (stats["duration"] / total_speech * 100) if total_speech > 0 else 0
            avg_seg = stats["duration"] / stats["count"] if stats["count"] > 0 else 0
            logger.info(
                "  %s: %d segments, %.1fs total (%.1f%%), avg segment: %.2fs",
                speaker, stats["count"], stats["duration"], pct, avg_seg
            )
        
        logger.info("=" * 60)
        
        # Update analysis cache with new diarization
        if video_id in analysis_results_cache:
            cached = analysis_results_cache[video_id]
            cached["nemo_diarization"] = {
                "segments": diar_segments,
                "num_speakers": num_speakers_detected,
                "speaker_stats": speaker_stats,
                "total_speech_duration": total_speech,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Apply NeMo diarization to segment dialogues
            if "segments" in cached:
                _apply_nemo_diarization_to_segments(cached["segments"], diar_segments)
                logger.info("Applied NeMo diarization to %d segments", len(cached["segments"]))
        
        # Auto-render if requested
        if auto_render and video_id in analysis_results_cache:
            tasks[task_id] = {
                "status": "processing",
                "progress": 0.85,
                "message": "NeMo diarization done. Starting render...",
            }
            
            # Get segments to render
            cached = analysis_results_cache[video_id]
            all_segments = cached.get("segments", [])
            
            # Filter segments by IDs or use strict/extended
            if segment_ids:
                ids_to_render = segment_ids
            else:
                # Use strict and extended selection by default
                ids_to_render = [
                    s.get("id") for s in all_segments 
                    if s.get("highlight_score", 0) >= 0.5  # strict + extended
                ]
            
            if ids_to_render:
                logger.info("Auto-rendering %d segments after NeMo diarization", len(ids_to_render))
                
                # Call the existing _process_segments_task directly
                # This reuses all the complex rendering logic
                try:
                    _process_segments_task(
                        task_id=task_id,
                        video_id=video_id,
                        segment_ids=ids_to_render,
                        tts_provider=tts_provider,
                        voice_mix=voice_mix,
                        vertical_method=vertical_method,
                        subtitle_animation=subtitle_animation,
                        subtitle_position=subtitle_position,
                        subtitle_font=subtitle_font,
                        subtitle_font_size=subtitle_font_size,
                        subtitle_background=subtitle_background,
                        subtitle_glow=subtitle_glow,
                        subtitle_gradient=subtitle_gradient,
                        preserve_background_audio=preserve_background_audio,
                        crop_focus=crop_focus,
                        speaker_color_mode=speaker_color_mode,
                        num_speakers=num_speakers_detected,  # Use detected speakers
                        speaker_change_times="",
                        speaker_change_phrases="",
                        rediarize_segments=False,  # Already diarized by NeMo
                    )
                    # _process_segments_task updates tasks[task_id] itself
                    return
                    
                except Exception as render_error:
                    logger.error("Auto-render failed: %s", render_error, exc_info=True)
                    tasks[task_id] = {
                        "status": "completed",
                        "progress": 1.0,
                        "message": f"NeMo done but render failed: {str(render_error)}",
                        "result": {
                            "success": True,
                            "nemo_speakers": num_speakers_detected,
                            "nemo_segments": len(diar_segments),
                            "render_error": str(render_error),
                        }
                    }
                    return
            else:
                logger.warning("No segments to render after NeMo diarization")
        
        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": f"NeMo diarization completed: {num_speakers_detected} speakers, {len(diar_segments)} segments",
            "result": {
                "success": True,
                "segments": diar_segments,
                "num_speakers_detected": num_speakers_detected,
            }
        }
        
    except Exception as e:
        logger.error("NeMo diarization failed: %s", e, exc_info=True)
        tasks[task_id] = {
            "status": "failed",
            "progress": 1.0,
            "message": f"NeMo diarization failed: {str(e)}",
            "result": {"success": False, "error": str(e)}
        }


@router.on_event("startup")
async def startup_event():
    # Clean up temp and output directories on startup
    logger.info("Clearing temp and output directories...")
    
    temp_dir = get_temp_dir()
    
    # Clean up incomplete uploads (files with .tmp.* extension)
    if os.path.exists(temp_dir):
        cleaned_count = 0
        for file_path in Path(temp_dir).glob("*.tmp.*"):
            try:
                file_path.unlink()
                cleaned_count += 1
                logger.debug(f"Removed incomplete upload: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove incomplete file {file_path}: {e}")
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} incomplete upload(s) on startup")
    
    # Restore uploaded videos cache from existing files in temp_dir
    if os.path.exists(temp_dir):
        restored_count = 0
        video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}
        
        for file_path in Path(temp_dir).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                # Skip incomplete uploads
                if '.tmp.' in file_path.name:
                    continue
                
                # Generate video_id from filename or use existing
                video_id = str(uuid.uuid4())
                file_stat = file_path.stat()
                file_size_mb = file_stat.st_size / (1024 * 1024)
                
                uploaded_videos_cache[video_id] = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "uploaded_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "size_mb": round(file_size_mb, 2)
                }
                restored_count += 1
                logger.debug(f"Restored cached video: {file_path.name} ({file_size_mb:.2f} MB)")
        
        if restored_count > 0:
            logger.info(f"Restored {restored_count} cached video(s) from temp directory")
    
    # Optional: uncomment to clear all temp files on startup
    # clear_temp_dir()
    
    if not os.path.exists(get_temp_dir()):
        os.makedirs(get_temp_dir())
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    logger.info("Directories are ready.")
