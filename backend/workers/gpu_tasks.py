"""
GPU Tasks for RQ Worker (gpu-worker).

These tasks run in a separate process with exclusive GPU access.
No CUDA context conflicts because only one worker owns the GPU.

Tasks:
- transcribe_audio: WhisperX transcription + alignment
- diarize_pyannote: Pyannote speaker diarization
- transcribe_and_diarize: Combined transcription + diarization

NOTE: Standalone NeMo diarization now runs in a SEPARATE worker (nemo-worker)
via nemo_tasks.py. This prevents CUDA context conflicts between Pyannote and NeMo.
The diarize_nemo function here is only used when diarizer="nemo" is passed to
transcribe_and_diarize (legacy support).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gc
import torch

logger = logging.getLogger(__name__)


def _clear_gpu_memory(force_release: bool = False):
    """
    Aggressively clear GPU memory before/after running a task.
    This prevents CUDA context conflicts between different models.
    
    Args:
        force_release: If True, attempt to release ALL GPU memory (for NeMo compatibility)
    """
    if torch.cuda.is_available():
        # Synchronize all CUDA streams
        torch.cuda.synchronize()
        # Clear cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        # Clear cache again after GC
        torch.cuda.empty_cache()
        
        if force_release:
            # Additional aggressive cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        # Log memory status
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info("GPU memory after cleanup: %.1f MB allocated, %.1f MB reserved", 
                   allocated, reserved)

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
        
        segments = result.get("segments", [])
        logger.info("GPU Task: transcribe_audio complete, %d segments", len(segments))
        
        result_data = {
            "segments": segments,
            "language": detected_language,
        }
        
        # Clean up GPU memory AFTER preparing result
        del whisper_model, align_model
        _clear_gpu_memory(force_release=True)
        
        return result_data
        
    except Exception as e:
        logger.error("GPU Task: transcribe_audio failed: %s", e, exc_info=True)
        # Try to clean up even on error
        _clear_gpu_memory(force_release=True)
        raise


def diarize_pyannote(
    audio_path: str,
    num_speakers: int = 0,  # 0 = auto-detect
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> List[Dict]:
    """
    Run Pyannote speaker diarization via subprocess (venv-diar).
    
    Uses a separate process with clean CUDA context for better performance.
    This avoids memory fragmentation and CUDA conflicts with WhisperX.
    
    Returns:
        List of segments: [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}, ...]
    """
    import subprocess
    import time
    
    logger.info("GPU Task: diarize_pyannote started, file=%s, speakers=%s",
                Path(audio_path).name, num_speakers if num_speakers > 0 else "auto")
    
    try:
        # Clear GPU memory before spawning subprocess
        _clear_gpu_memory(force_release=True)
        
        # Get HF token
        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN required for Pyannote")
        
        # Path to Pyannote venv and script
        pyannote_python = Path("/opt/youtube-shorts-generator/venv-diar/bin/python")
        pyannote_script = Path("/opt/youtube-shorts-generator/backend/tools/diarize.py")
        
        if not pyannote_python.exists():
            # Fallback to local path
            project_root = Path(__file__).parent.parent.parent
            pyannote_python = project_root / "venv-diar" / "bin" / "python"
            pyannote_script = project_root / "backend" / "tools" / "diarize.py"
        
        if not pyannote_python.exists():
            raise FileNotFoundError(f"Pyannote venv not found: {pyannote_python}")
        if not pyannote_script.exists():
            raise FileNotFoundError(f"Pyannote script not found: {pyannote_script}")
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            # Build command
            cmd = [
                str(pyannote_python),
                str(pyannote_script),
                "--input", audio_path,
                "--output", output_path,
                "--device", device,
                "--num_speakers", str(num_speakers),
                "--hf_token", hf_token,
            ]
            
            logger.info("Running Pyannote subprocess: %s %s --input %s ...",
                       pyannote_python.name, pyannote_script.name, Path(audio_path).name)
            
            start_time = time.time()
            
            # Run in subprocess with clean environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                env=env,
            )
            
            elapsed = time.time() - start_time
            logger.info("Pyannote subprocess completed in %.1f seconds", elapsed)
            
            # Log stderr (pyannote logs there)
            if result.stderr:
                stderr_text = result.stderr
                for line in stderr_text.split("\n"):
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in ["speaker", "segment", "error", "cuda", "gpu", "device", "pipeline", "diarization", "complete"]):
                        logger.info("Pyannote: %s", line.strip())
            
            if result.returncode != 0:
                logger.error("Pyannote subprocess failed with code %d", result.returncode)
                logger.error("Pyannote stderr: %s", result.stderr[-2000:])
                raise RuntimeError(f"Pyannote diarization failed: {result.stderr[-1000:]}")
            
            # Check if output file exists
            if not os.path.exists(output_path):
                logger.error("Pyannote output file not created: %s", output_path)
                raise RuntimeError("Pyannote did not create output file")
            
            # Read results
            with open(output_path) as f:
                pyannote_result = json.load(f)
            
            segments = pyannote_result.get("segments", [])
            segments.sort(key=lambda x: x["start"])
            
            # Log summary
            speakers = set(s["speaker"] for s in segments)
            logger.info("GPU Task: diarize_pyannote complete, %d segments, %d speakers, %.1fs",
                       len(segments), len(speakers), elapsed)
            
            return segments
            
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
        
    except subprocess.TimeoutExpired:
        logger.error("GPU Task: diarize_pyannote timed out (60 min)")
        raise RuntimeError("Pyannote diarization timed out after 60 minutes")
    except Exception as e:
        logger.error("GPU Task: diarize_pyannote failed: %s", e, exc_info=True)
        raise


def diarize_nemo(
    audio_path: str,
    num_speakers: int = 0,  # 0 = auto-detect
    max_speakers: int = 8,
    device: str = "cuda",
) -> List[Dict]:
    """
    Run NeMo MSDD speaker diarization via subprocess.
    
    NeMo is installed in a separate venv (venv-nemo), so we run it
    as a subprocess. This ensures NeMo gets a clean CUDA context.
    
    IMPORTANT: We clear GPU memory before running NeMo to prevent
    CUBLAS_STATUS_NOT_INITIALIZED errors from stale CUDA state.
    
    Returns:
        List of segments: [{"start": 0.0, "end": 1.5, "speaker": "speaker_0"}, ...]
    """
    logger.info("GPU Task: diarize_nemo started, file=%s, speakers=%s",
                Path(audio_path).name, num_speakers if num_speakers > 0 else "auto")
    
    try:
        import subprocess
        
        # CRITICAL: Clear GPU memory before NeMo subprocess
        # This prevents CUDA context conflicts with Pyannote/WhisperX
        logger.info("Clearing GPU memory before NeMo...")
        _clear_gpu_memory()
        
        # Path to NeMo venv and script
        # Try production path first, then local
        nemo_python = Path("/opt/youtube-shorts-generator/venv-nemo/bin/python")
        nemo_script = Path("/opt/youtube-shorts-generator/backend/tools/diarize_nemo.py")
        
        if not nemo_python.exists():
            # Fallback to local path
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
            
            logger.info("Running NeMo subprocess: %s", " ".join(cmd[:6]) + " ...")
            
            # Run NeMo in subprocess with clean environment
            env = os.environ.copy()
            # Ensure CUDA is visible
            env["CUDA_VISIBLE_DEVICES"] = "0"
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                env=env,
            )
            
            # Always log output for debugging
            if result.stdout:
                logger.info("NeMo stdout: %s", result.stdout[:2000])
            if result.stderr:
                # Log first 3000 chars of stderr (NeMo is verbose)
                logger.debug("NeMo stderr (truncated): %s", result.stderr[:3000])
            
            if result.returncode != 0:
                logger.error("NeMo subprocess failed with code %d", result.returncode)
                logger.error("NeMo stderr: %s", result.stderr[-2000:])
                raise RuntimeError(f"NeMo diarization failed: {result.stderr[-1000:]}")
            
            # Check if output file exists
            if not os.path.exists(output_path):
                logger.error("NeMo output file not created: %s", output_path)
                raise RuntimeError("NeMo did not create output file")
            
            # Read results
            with open(output_path) as f:
                nemo_result = json.load(f)
            
            segments = nemo_result.get("segments", [])
            
            # Log if empty result
            if not segments:
                logger.warning("NeMo returned 0 segments!")
                logger.warning("Full NeMo stderr: %s", result.stderr[-3000:])
            
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
    
    For Pyannote: uses WhisperX BUILT-IN diarization (same process, same CUDA context).
      This is 3x faster than separate subprocess because pyannote reuses WhisperX's
      loaded audio and runs in the same GPU context.
    For NeMo: uses separate subprocess (different venv).
    
    Returns:
        {
            "segments": [...],  # Transcribed segments with speaker labels
            "language": "en",
            "diarizer_used": "pyannote",
            "num_speakers": 2,
        }
    """
    logger.info("GPU Task: transcribe_and_diarize started, diarizer=%s", diarizer)
    
    if diarizer == "nemo":
        # NeMo: transcribe first, then diarize separately
        return _transcribe_and_diarize_nemo(
            audio_path=audio_path,
            model=model,
            language=language,
            num_speakers=num_speakers,
            device=device,
        )
    else:
        # Pyannote: use WhisperX BUILT-IN diarization (fast, single process)
        return _transcribe_and_diarize_whisperx_builtin(
            audio_path=audio_path,
            model=model,
            language=language,
            num_speakers=num_speakers,
            device=device,
            hf_token=hf_token,
        )


def _transcribe_and_diarize_whisperx_builtin(
    audio_path: str,
    model: str = "large-v3",
    language: str = "en",
    num_speakers: int = 0,
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    WhisperX transcription + BUILT-IN Pyannote diarization in ONE process.
    
    This is the FAST path (~10 min for 2-hour file vs 30+ min with separate subprocess).
    WhisperX loads pyannote internally and passes audio data directly — no file I/O overhead.
    """
    import time
    
    logger.info("GPU Task: WhisperX built-in diarization, file=%s", Path(audio_path).name)
    start_time = time.time()
    
    try:
        import whisperx
        
        # Adjust for CPU if needed
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")
        
        compute_type = "float16" if device == "cuda" else "int8"
        
        # Step 1: Load model and transcribe
        logger.info("Step 1: Loading WhisperX model: %s on %s", model, device)
        whisper_model = whisperx.load_model(
            model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        
        logger.info("Step 1: Transcribing audio...")
        audio = whisperx.load_audio(audio_path)
        result = whisper_model.transcribe(audio, batch_size=16, language=language)
        
        detected_language = result.get("language", language)
        logger.info("Step 1: Transcription complete, language=%s", detected_language)
        
        # Step 2: Align for word-level timestamps
        logger.info("Step 2: Aligning transcription...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        
        logger.info("Step 2: Alignment complete, %d segments", len(result.get("segments", [])))
        
        # Free whisper model to make room for diarization
        del whisper_model, align_model
        _clear_gpu_memory()
        
        # Step 3: Built-in WhisperX diarization (uses pyannote internally)
        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN required for Pyannote diarization")
        
        # Determine speaker count mode
        if num_speakers > 0:
            min_spk = num_speakers
            max_spk = num_speakers
            mode_str = f"fixed={num_speakers}"
        else:
            min_spk = 1
            max_spk = 4
            mode_str = f"auto [{min_spk}-{max_spk}]"
        
        logger.info("Step 3: WhisperX built-in diarization (mode=%s)...", mode_str)
        
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        
        diarize_model = DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )
        
        # Apply fine-tuned hyperparameters for better speaker separation
        try:
            diarize_model.model.instantiate({
                "segmentation": {
                    "min_duration_off": 0.05,
                },
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 3,
                    "threshold": 0.35,
                },
            })
            logger.info("Applied fine-tuned diarization parameters")
        except Exception as tune_err:
            logger.warning("Could not apply fine-tuned parameters: %s", tune_err)
        
        diarize_segments = diarize_model(
            audio,
            min_speakers=min_spk,
            max_speakers=max_spk,
        )
        
        # Assign speakers to words
        result = assign_word_speakers(diarize_segments, result)
        
        # Free diarization model
        del diarize_model
        _clear_gpu_memory(force_release=True)
        
        elapsed = time.time() - start_time
        
        # Extract segments with speaker info
        segments = []
        speaker_set = set()
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                word_data = {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                }
                if "speaker" in w:
                    word_data["speaker"] = w["speaker"]
                words.append(word_data)
            
            seg_data = {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
                "words": words,
            }
            if "speaker" in seg:
                seg_data["speaker"] = seg["speaker"]
                speaker_set.add(seg["speaker"])
            
            segments.append(seg_data)
        
        logger.info(
            "GPU Task: transcribe_and_diarize complete in %.1fs, %d segments, %d speakers",
            elapsed, len(segments), len(speaker_set),
        )
        
        return {
            "segments": segments,
            "language": detected_language,
            "diarizer_used": "pyannote",
            "num_speakers": len(speaker_set),
        }
        
    except Exception as e:
        logger.error("GPU Task: transcribe_and_diarize failed: %s", e, exc_info=True)
        _clear_gpu_memory(force_release=True)
        raise


def _transcribe_and_diarize_nemo(
    audio_path: str,
    model: str = "large-v3",
    language: str = "en",
    num_speakers: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """NeMo path: transcribe with WhisperX, then diarize with NeMo subprocess."""
    logger.info("GPU Task: NeMo diarization path")
    
    # Step 1: Transcribe
    transcription = transcribe_audio(
        audio_path=audio_path,
        model=model,
        language=language,
        device=device,
    )
    
    whisper_segments = transcription["segments"]
    detected_language = transcription["language"]
    
    # Step 2: Diarize with NeMo
    diar_segments = diarize_nemo(
        audio_path=audio_path,
        num_speakers=num_speakers,
        device=device,
    )
    
    # Step 3: Merge
    merged_segments = _merge_transcription_with_diarization(
        whisper_segments,
        diar_segments,
    )
    
    speakers = set()
    for seg in merged_segments:
        speakers.add(seg.get("speaker", "UNKNOWN"))
    
    logger.info("GPU Task: transcribe_and_diarize (NeMo) complete, %d segments, %d speakers",
               len(merged_segments), len(speakers))
    
    return {
        "segments": merged_segments,
        "language": detected_language,
        "diarizer_used": "nemo",
        "num_speakers": len(speakers),
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
