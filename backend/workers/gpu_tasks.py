"""
GPU Tasks for RQ Worker.

These tasks run in a separate process with exclusive GPU access.
No CUDA context conflicts because only one worker owns the GPU.

Tasks:
- transcribe_audio: WhisperX transcription + alignment
- diarize_pyannote: Pyannote speaker diarization
- diarize_nemo: NeMo MSDD speaker diarization
- detect_faces: InsightFace face detection for focus timeline
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# NOTE: Don't call torch.cuda functions at module level!
# RQ workers fork processes and CUDA can't be re-initialized in forked subprocess.
# GPU info will be logged when tasks actually run.


def transcribe_audio(
    audio_path: str,
    model: str = "large-v3",
    language: str = "en",
    device: str = "cuda",
    compute_type: str = "float16",
) -> Dict[str, Any]:
    """
    Transcribe audio using WhisperX.
    
    Returns:
        {
            "segments": [...],  # List of transcribed segments with word timings
            "language": "en",
        }
    """
    logger.info("GPU Task: transcribe_audio started, file=%s, model=%s, device=%s", 
                Path(audio_path).name, model, device)
    
    try:
        import whisperx
        
        # Adjust for CPU if needed
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            compute_type = "int8"
            logger.warning("CUDA not available, falling back to CPU")
        
        # Load model
        logger.info("Loading WhisperX model: %s on %s", model, device)
        whisper_model = whisperx.load_model(
            model, 
            device=device, 
            compute_type=compute_type,
            language=language,
        )
        
        # Transcribe
        logger.info("Transcribing audio...")
        audio = whisperx.load_audio(audio_path)
        result = whisper_model.transcribe(audio, batch_size=16)
        
        detected_language = result.get("language", language)
        logger.info("Transcription complete, detected language: %s", detected_language)
        
        # Align for word-level timestamps
        logger.info("Aligning transcription...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language, 
            device=device
        )
        result = whisperx.align(
            result["segments"], 
            align_model, 
            align_metadata, 
            audio, 
            device,
            return_char_alignments=False,
        )
        
        # Clean up GPU memory
        del whisper_model, align_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        segments = result.get("segments", [])
        logger.info("GPU Task: transcribe_audio complete, %d segments", len(segments))
        
        return {
            "segments": segments,
            "language": detected_language,
        }
        
    except Exception as e:
        logger.error("GPU Task: transcribe_audio failed: %s", e, exc_info=True)
        raise


def diarize_pyannote(
    audio_path: str,
    num_speakers: int = 0,  # 0 = auto-detect
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> List[Dict]:
    """
    Run Pyannote speaker diarization.
    
    Returns:
        List of segments: [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}, ...]
    """
    logger.info("GPU Task: diarize_pyannote started, file=%s, speakers=%s",
                Path(audio_path).name, num_speakers if num_speakers > 0 else "auto")
    
    try:
        from pyannote.audio import Pipeline
        
        # Get HF token
        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN required for Pyannote")
        
        # Adjust for CPU if needed
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")
        
        # Load pipeline
        logger.info("Loading Pyannote pipeline on %s", device)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        
        if device == "cuda":
            pipeline.to(torch.device("cuda"))
            logger.info("Pyannote pipeline moved to GPU")
        
        # Fine-tune parameters for better speaker separation
        pipeline.instantiate({
            "segmentation": {
                "min_duration_off": 0.05,
            },
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 3,
                "threshold": 0.35,
            },
        })
        
        # Run diarization
        actual_num_speakers = num_speakers if num_speakers > 0 else None
        logger.info("Running diarization with num_speakers=%s", 
                   actual_num_speakers if actual_num_speakers else "auto")
        
        diarization = pipeline(audio_path, num_speakers=actual_num_speakers)
        
        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
        
        segments.sort(key=lambda x: x["start"])
        
        # Clean up
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log summary
        speakers = set(s["speaker"] for s in segments)
        logger.info("GPU Task: diarize_pyannote complete, %d segments, %d speakers",
                   len(segments), len(speakers))
        
        return segments
        
    except Exception as e:
        logger.error("GPU Task: diarize_pyannote failed: %s", e, exc_info=True)
        raise


def diarize_nemo(
    audio_path: str,
    num_speakers: int = 0,  # 0 = auto-detect
    max_speakers: int = 8,
    device: str = "cuda",  # NeMo runs in separate venv with its own CUDA context
) -> List[Dict]:
    """
    Run NeMo MSDD speaker diarization via subprocess.
    
    NeMo is installed in a separate venv (venv-nemo), so we run it
    as a subprocess using the NemoDiarizationRunner.
    
    Returns:
        List of segments: [{"start": 0.0, "end": 1.5, "speaker": "speaker_0"}, ...]
    """
    logger.info("GPU Task: diarize_nemo started, file=%s, speakers=%s",
                Path(audio_path).name, num_speakers if num_speakers > 0 else "auto")
    
    try:
        import subprocess
        
        # Path to NeMo venv and script
        project_root = Path(__file__).parent.parent.parent
        nemo_python = project_root / "venv-nemo" / "bin" / "python"
        nemo_script = project_root / "backend" / "tools" / "diarize_nemo.py"
        
        if not nemo_python.exists():
            raise FileNotFoundError(f"NeMo venv not found: {nemo_python}")
        if not nemo_script.exists():
            raise FileNotFoundError(f"NeMo script not found: {nemo_script}")
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            # Build command
            cmd = [
                str(nemo_python),
                str(nemo_script),
                "--input", audio_path,
                "--output", output_path,
                "--device", device,
                "--max_speakers", str(max_speakers),
            ]
            if num_speakers > 0:
                cmd.extend(["--num_speakers", str(num_speakers)])
            
            logger.info("Running NeMo subprocess: %s", " ".join(cmd))
            
            # Run NeMo in subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )
            
            # Always log output for debugging
            if result.stdout:
                logger.info("NeMo stdout: %s", result.stdout[:2000])
            if result.stderr:
                # Log first 3000 chars of stderr (NeMo is verbose)
                logger.warning("NeMo stderr: %s", result.stderr[:3000])
            
            if result.returncode != 0:
                logger.error("NeMo subprocess failed with code %d", result.returncode)
                raise RuntimeError(f"NeMo diarization failed: {result.stderr[-1000:]}")
            
            # Check if output file exists
            if not os.path.exists(output_path):
                logger.error("NeMo output file not created: %s", output_path)
                raise RuntimeError("NeMo did not create output file")
            
            # Read results
            with open(output_path) as f:
                nemo_result = json.load(f)
            
            # Log if empty result
            if not nemo_result.get("segments"):
                logger.warning("NeMo returned empty segments! Full stderr: %s", result.stderr[-2000:])
            
            segments = nemo_result.get("segments", [])
            segments.sort(key=lambda x: x["start"])
            
            speakers = set(s["speaker"] for s in segments)
            logger.info("GPU Task: diarize_nemo complete, %d segments, %d speakers",
                       len(segments), len(speakers))
            
            return segments
            
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
            
    except Exception as e:
        logger.error("GPU Task: diarize_nemo failed: %s", e, exc_info=True)
        raise


def transcribe_and_diarize(
    audio_path: str,
    diarizer: str = "pyannote",  # "pyannote" or "nemo"
    model: str = "large-v3",
    language: str = "en",
    num_speakers: int = 0,
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combined task: transcribe + diarize + merge results.
    
    This is the main entry point for video analysis.
    Runs WhisperX transcription, then selected diarizer, then merges.
    
    Returns:
        {
            "segments": [...],  # Transcribed segments with speaker labels
            "language": "en",
            "diarizer_used": "pyannote",
            "num_speakers": 2,
        }
    """
    logger.info("GPU Task: transcribe_and_diarize started, diarizer=%s", diarizer)
    
    # Step 1: Transcribe
    transcription = transcribe_audio(
        audio_path=audio_path,
        model=model,
        language=language,
        device=device,
    )
    
    whisper_segments = transcription["segments"]
    detected_language = transcription["language"]
    
    # Step 2: Diarize
    if diarizer == "nemo":
        diar_segments = diarize_nemo(
            audio_path=audio_path,
            num_speakers=num_speakers,
            device=device,
        )
    else:  # default to pyannote
        diar_segments = diarize_pyannote(
            audio_path=audio_path,
            num_speakers=num_speakers,
            device=device,
            hf_token=hf_token,
        )
    
    # Step 3: Merge transcription with diarization
    merged_segments = _merge_transcription_with_diarization(
        whisper_segments, 
        diar_segments
    )
    
    # Count speakers
    speakers = set()
    for seg in merged_segments:
        speakers.add(seg.get("speaker", "UNKNOWN"))
    
    logger.info("GPU Task: transcribe_and_diarize complete, %d segments, %d speakers",
               len(merged_segments), len(speakers))
    
    return {
        "segments": merged_segments,
        "language": detected_language,
        "diarizer_used": diarizer,
        "num_speakers": len(speakers),
    }


def nemo_diarize_only(
    audio_path: str,
    num_speakers: int = 0,
    max_speakers: int = 8,
) -> Dict[str, Any]:
    """
    Standalone NeMo diarization task for RQ queue.
    
    This runs NeMo MSDD on GPU in the worker process.
    Called from video.py for re-diarization requests.
    
    Returns:
        {
            "segments": [...],  # Diarization segments
            "num_speakers": 2,
            "speaker_stats": {...},
        }
    """
    logger.info("GPU Task: nemo_diarize_only started, file=%s", Path(audio_path).name)
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU available: %s (%.1f GB)", gpu_name, gpu_mem)
    else:
        logger.warning("CUDA not available!")
    
    # Run NeMo diarization
    segments = diarize_nemo(
        audio_path=audio_path,
        num_speakers=num_speakers,
        max_speakers=max_speakers,
        device="cuda",  # Always GPU in worker
    )
    
    # Calculate speaker stats
    speaker_stats = {}
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        duration = seg.get("end", 0) - seg.get("start", 0)
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"count": 0, "duration": 0.0}
        speaker_stats[speaker]["count"] += 1
        speaker_stats[speaker]["duration"] += duration
    
    num_speakers_detected = len(speaker_stats)
    total_speech = sum(s["duration"] for s in speaker_stats.values())
    
    logger.info("GPU Task: nemo_diarize_only complete, %d speakers, %.1fs speech",
               num_speakers_detected, total_speech)
    
    return {
        "segments": segments,
        "num_speakers": num_speakers_detected,
        "speaker_stats": speaker_stats,
        "total_speech_duration": total_speech,
    }


def _merge_transcription_with_diarization(
    whisper_segments: List[Dict],
    diar_segments: List[Dict],
) -> List[Dict]:
    """
    Merge WhisperX transcription with diarization results.
    
    For each whisper segment/word, find the overlapping diarization segment
    and assign the speaker label.
    """
    if not diar_segments:
        logger.warning("No diarization segments, returning transcription without speakers")
        for seg in whisper_segments:
            seg["speaker"] = "UNKNOWN"
        return whisper_segments
    
    def find_speaker(start: float, end: float) -> str:
        """Find speaker with maximum overlap."""
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        
        for diar in diar_segments:
            # Calculate overlap
            overlap_start = max(start, diar["start"])
            overlap_end = min(end, diar["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar["speaker"]
        
        return best_speaker
    
    # Assign speakers to segments
    for seg in whisper_segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg["speaker"] = find_speaker(seg_start, seg_end)
        
        # Also assign speakers to words if present
        if "words" in seg:
            for word in seg["words"]:
                word_start = word.get("start", seg_start)
                word_end = word.get("end", seg_end)
                word["speaker"] = find_speaker(word_start, word_end)
    
    return whisper_segments
