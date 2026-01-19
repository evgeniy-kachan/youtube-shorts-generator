import os
import uuid
import logging
import subprocess
from itertools import cycle
from pathlib import Path
from typing import List, Optional

import ffmpeg

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydub import AudioSegment

from backend.services.youtube_downloader import YouTubeDownloader
from backend.services.transcription import TranscriptionService
from backend.services.highlight_analyzer import HighlightAnalyzer, determine_target_highlights
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

# Lazy-loaded services
_services = {}


class AnalyzeRequest(BaseModel):
    youtube_url: str

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


def _build_voice_plan(segments: list[dict], mix: str) -> dict[str, str]:
    pattern = VOICE_MIX_PRESETS.get(mix, VOICE_MIX_PRESETS["male_duo"])
    male_pool = config.ELEVENLABS_VOICE_IDS_MALE or [config.ELEVENLABS_VOICE_ID]
    female_pool = config.ELEVENLABS_VOICE_IDS_FEMALE or [config.ELEVENLABS_VOICE_ID]
    male_iter = cycle(male_pool)
    female_iter = cycle(female_pool)

    plan: dict[str, str] = {}
    speakers = _extract_unique_speakers(segments)

    for idx, speaker_id in enumerate(speakers):
        desired_gender = pattern[min(idx, len(pattern) - 1)]
        if desired_gender == "female":
            plan[speaker_id] = next(female_iter)
        else:
            plan[speaker_id] = next(male_iter)

    plan["__default__"] = plan.get(speakers[0], config.ELEVENLABS_VOICE_ID)
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
        max_tempo: Maximum tempo (speed up limit, default 1.25 - reduced from 1.4 to minimize artifacts)
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

def _analyze_local_video_task(task_id: str, filename: str, analysis_mode: str = "fast"):
    try:
        tasks[task_id] = {"status": "processing", "progress": 0.1, "message": "Processing local video..."}
        
        video_id = os.path.splitext(filename)[0]
        video_path = os.path.join(get_temp_dir(), filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found in temp directory: {filename}")

        _run_analysis_pipeline(task_id, video_id, video_path, analysis_mode)

    except Exception as e:
        logger.error(f"Error in local analysis task {task_id}: {e}", exc_info=True)
        tasks[task_id] = {"status": "failed", "progress": tasks[task_id].get('progress', 0.1), "message": str(e)}

def _run_analysis_pipeline(task_id: str, video_id: str, video_path: str, analysis_mode: str = "fast"):
    """Core analysis logic, shared by YouTube and local video."""
    
    thumbnail_path = _generate_thumbnail(video_path, video_id)
    thumbnail_url = f"/api/video/thumbnail/{video_id}" if thumbnail_path else None

    # 2. Transcribe audio
    tasks[task_id] = {"status": "processing", "progress": 0.2, "message": "Transcribing audio..."}
    transcriber = get_service("transcription")
    transcription_result = transcriber.transcribe_audio_from_video(video_path)
    segments = transcription_result['segments']
    video_duration = segments[-1]['end'] if segments else 0.0
    target_highlight_count = determine_target_highlights(video_duration)

    # 3. Analyze for highlights
    # Select model based on analysis_mode: 'fast' = deepseek-chat, 'deep' = deepseek-reasoner
    analysis_model = "deepseek-reasoner" if analysis_mode == "deep" else "deepseek-chat"
    mode_label = "глубокий (R1)" if analysis_mode == "deep" else "быстрый"
    tasks[task_id] = {"status": "processing", "progress": 0.5, "message": f"Анализ контента ({mode_label})..."}
    
    analyzer = HighlightAnalyzer(model_name=analysis_model)
    highlights = analyzer.analyze_segments(segments)

    # 4. Filter out highly overlapping segments
    logger.info(
        "Filtering %d segments for overlaps (target count %d)...",
        len(highlights),
        target_highlight_count,
    )
    filtered_highlights = _filter_overlapping_segments(
        highlights,
        iou_threshold=0.5,
        desired_count=target_highlight_count,
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
    preserve_background_audio: bool = True,
    crop_focus: str = "center",
    speaker_color_mode: str = "colored",
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
            voice_plan = _build_voice_plan(segments_to_process, voice_mix)
        output_dir = get_output_dir(video_id)
        
        for segment in segments_to_process:
            audio_path = os.path.join(output_dir, f"{segment['id']}.wav")
            
            # DIARIZATION DIAGNOSTIC: Log speaker info for this segment
            dialogue = segment.get('dialogue') or []
            speakers_in_segment = set()
            for turn in dialogue:
                spk = turn.get('speaker')
                if spk:
                    speakers_in_segment.add(spk)
            
            logger.info(
                "DIARIZATION CHECK [%s]: %d turns, %d unique speakers: %s",
                segment['id'],
                len(dialogue),
                len(speakers_in_segment),
                list(speakers_in_segment)
            )
            
            # Log first few words of each turn to help identify speaker changes
            for i, turn in enumerate(dialogue[:5]):  # First 5 turns
                turn_text = (turn.get('text') or turn.get('text_ru') or '')[:50]
                logger.info(
                    "  Turn %d [%s]: '%s...'",
                    i, turn.get('speaker', '?'), turn_text
                )
            if len(dialogue) > 5:
                logger.info("  ... and %d more turns", len(dialogue) - 5)
            
            # Check if we have a dialogue structure for multi-speaker synthesis
            has_dialogue = bool(segment.get('dialogue') and len(segment['dialogue']) > 1)
            
            if has_dialogue and voice_plan:
                # Debug: check if words are present
                for i, turn in enumerate(segment['dialogue']):
                    words = turn.get('words') or []
                    logger.info(
                        "Turn %d [%s]: %d words, first=%s, last=%s",
                        i,
                        turn.get('speaker', '?'),
                        len(words),
                        words[0] if words else None,
                        words[-1] if words else None,
                    )
                
                # Refine turn boundaries using word-level timestamps (more accurate than Pyannote)
                _refine_turn_boundaries(segment['dialogue'])
                
                # SECOND PASS: Isochronic translation with precise timings
                # This improves translation quality by considering exact durations
                try:
                    translator = Translator()
                    logger.info(f"Running isochronic translation for {segment['id']}")
                    segment['dialogue'] = translator.translate_with_timings(
                        dialogue_turns=segment['dialogue'],
                        segment_context=segment.get('title', ''),
                    )
                except Exception as trans_exc:
                    logger.warning("Isochronic translation failed, using original: %s", trans_exc)
                
                logger.info(f"Synthesizing multi-speaker dialogue for {segment['id']}")
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
                    # Create pseudo-dialogue by detecting pauses (like multi-speaker)
                    logger.info(
                        "Splitting single-speaker %s into pseudo-turns by pauses (%d words)",
                        segment['id'], len(segment_words)
                    )
                    pseudo_dialogue = _create_pseudo_dialogue_from_words(
                        words=segment_words,
                        speaker=primary_speaker,
                        segment_start=segment_start,
                        pause_threshold=0.4,  # 400ms pause = new turn
                    )
                    
                    if pseudo_dialogue:
                        segment['dialogue'] = pseudo_dialogue
                        # Log sample of turn boundaries for debugging
                        turn_previews = [
                            f"T{i}: '{t['text'][:25]}...' ({len(t['text'].split())}w)"
                            for i, t in enumerate(pseudo_dialogue[:3])
                        ]
                        logger.info(
                            "Created %d pseudo-turns for single-speaker %s (smart split: pause + sentence-end)",
                            len(pseudo_dialogue), segment['id']
                        )
                        logger.info("  First turns: %s", turn_previews)
                        
                        # Now process exactly like multi-speaker
                        has_dialogue = True
                        
                        # Refine boundaries
                        _refine_turn_boundaries(segment['dialogue'])
                        
                        # Apply isochronic translation per turn
                        try:
                            translator = Translator()
                            logger.info(f"Running isochronic translation for single-speaker {segment['id']} ({len(pseudo_dialogue)} turns)")
                            segment['dialogue'] = translator.translate_with_timings(
                                dialogue_turns=segment['dialogue'],
                                segment_context=segment.get('title', ''),
                            )
                        except Exception as trans_exc:
                            logger.warning("Isochronic translation failed for single-speaker: %s", trans_exc)
                        
                        # Synthesize with TTD (preserves pauses)
                        logger.info(f"Synthesizing single-speaker dialogue for {segment['id']} ({len(pseudo_dialogue)} turns)")
                        tts_service.synthesize_dialogue(
                            dialogue_turns=segment['dialogue'],
                            output_path=audio_path,
                            voice_map=voice_plan or {primary_speaker: tts_service.voice_id},
                            base_start=segment.get('start_time'),
                        )
                        
                        # Log post-synthesis timing coverage
                        if segment.get('dialogue'):
                            first_t = segment['dialogue'][0]
                            last_t = segment['dialogue'][-1]
                            tts_start = first_t.get('tts_start_offset', 0)
                            tts_end = last_t.get('tts_end_offset', 0)
                            logger.info(
                                "POST-TTD %s: %d turns, tts_coverage=%.2f-%.2fs (%.2fs)",
                                segment['id'], len(segment['dialogue']),
                                tts_start, tts_end, tts_end - tts_start
                            )
                    else:
                        logger.warning("Failed to create pseudo-dialogue for %s, falling back to simple synthesis", segment['id'])
                        has_dialogue = False
                
                # Fallback: if no words available, use old single-block approach
                if not has_dialogue:
                    logger.info("Using fallback single-block synthesis for %s", segment['id'])
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

            # Tempo adjustment: keep audio within 0.7x-1.25x of original duration
            # Reduced from 1.4x to 1.25x to minimize audio artifacts (stuttering/cutting)
            duration_diff = abs(audio_duration - original_duration)
            if duration_diff > 0.2:  # More than 200ms difference
                before_duration = audio_duration
                
                logger.info(
                    "Applying tempo adjustment for %s: %.2fs -> %.2fs (range: 0.7x-1.25x)%s",
                    segment['id'],
                    audio_duration,
                    original_duration,
                    " [multi-speaker]" if has_dialogue else "",
                )
                
                if _speed_match_audio_duration(audio_path, audio_duration, original_duration, max_tempo=1.25, min_tempo=0.7):
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
                    
                    logger.info(
                        "Tempo adjusted %s: %.2fs -> %.2fs (scale=%.2f)",
                        segment['id'],
                        before_duration,
                        audio_duration,
                        scale,
                    )
                else:
                    logger.info(
                        "Tempo adjustment skipped for %s (negligible diff or limit reached)",
                        segment['id'],
                    )

            target_duration = max(audio_duration, original_duration)

            if audio_segment and audio_duration + 0.01 < target_duration:
                pad_ms = int(round((target_duration - audio_duration) * 1000))
                audio_segment = audio_segment + AudioSegment.silent(duration=pad_ms)
                audio_segment.export(audio_path, format="wav")
                audio_duration = target_duration
                logger.info(
                    "Padded audio for %s with %.2fs silence to reach target duration %.2fs",
                    segment['id'],
                    pad_ms / 1000,
                    target_duration,
                )

            segment['tts_duration'] = audio_duration
            segment['target_duration'] = target_duration
            
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
                target_duration=segment.get('target_duration'),
                method=vertical_method,
                subtitle_animation=subtitle_animation,
                subtitle_position=subtitle_position,
                subtitle_font=subtitle_font,
                subtitle_font_size=subtitle_font_size,
                subtitle_background=subtitle_background,
                dialogue=segment.get('dialogue'),
                preserve_background_audio=preserve_background_audio,
                crop_focus=crop_focus,
                speaker_color_mode=speaker_color_mode,
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
async def analyze_local_video(
    filename: str, 
    background_tasks: BackgroundTasks,
    analysis_mode: str = "fast"  # 'fast' (deepseek-chat) or 'deep' (deepseek-reasoner)
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0.0, "message": "Task queued"}
    background_tasks.add_task(_analyze_local_video_task, task_id, filename, analysis_mode)
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
        request.preserve_background_audio,
        request.crop_focus,
        request.speaker_color_mode,
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
