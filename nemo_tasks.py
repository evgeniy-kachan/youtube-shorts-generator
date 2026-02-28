# -*- coding: utf-8 -*-
"""
NeMo diarization task - runs on Selectel server.
Downloads audio from main server, processes with NeMo, returns results.
"""
import os
import json
import logging
import tempfile
import subprocess
import requests
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
from redis import Redis

logger = logging.getLogger(__name__)

# Configuration
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "http://178.72.132.235:8000")
REDIS_HOST = os.getenv("REDIS_HOST", "178.72.132.235")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "NeMo2026SecurePass!")
NEMO_INTERNAL_TOKEN = os.getenv("NEMO_INTERNAL_TOKEN", "NeMo2026InternalToken!")
STATUS_KEY = "nemo_server:status"
HEARTBEAT_KEY = "nemo_server:heartbeat"

# Chunking configuration
MAX_CHUNK_DURATION_SEC = 90 * 60  # 90 minutes per chunk


def set_status(status: str):
    """Update NeMo server status in Redis."""
    import time
    try:
        redis_conn = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD
        )
        redis_conn.setex(STATUS_KEY, 120, status)
        redis_conn.setex(HEARTBEAT_KEY, 120, str(time.time()))
    except Exception as e:
        logger.warning(f"Failed to set status: {e}")


def download_audio_by_path(audio_path: str, dest_path: str) -> bool:
    """
    Download audio file from main server.
    
    If audio_path is a full path on the main server, use the project file endpoint.
    Otherwise, try to download by filename from temp directory.
    """
    filename = Path(audio_path).name
    
    # If audio_path is a full path starting with /opt/youtube-shorts-generator/
    # use the project file endpoint with relative path
    project_prefix = "/opt/youtube-shorts-generator/"
    if audio_path.startswith(project_prefix):
        relative_path = audio_path[len(project_prefix):]
        url = f"{MAIN_SERVER_URL}/api/video/files/project/{relative_path}"
    else:
        # Fallback to temp file endpoint
        url = f"{MAIN_SERVER_URL}/api/video/files/temp/{filename}"
    
    logger.info(f"Downloading audio from: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(dest_path)
        logger.info(f"Downloaded {file_size} bytes to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def _extract_wav(input_path: str, output_path: str, sample_rate: int = 16000):
    """Convert audio to WAV format."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", str(sample_rate), "-vn", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _split_audio_chunk(input_wav: str, output_wav: str, start_sec: float, end_sec: float):
    """
    Extract a chunk from audio file using ffmpeg.
    Uses stream copy for speed (no re-encoding).
    """
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", input_wav,
        "-t", str(duration),
        "-c", "copy",
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fallback: re-encode if copy fails
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", input_wav,
            "-t", str(duration),
            "-ac", "1", "-ar", "16000",
            output_wav
        ]
        subprocess.run(cmd, check=True, capture_output=True)


def _get_audio_chunks(total_duration: float) -> List[Tuple[float, float]]:
    """
    Calculate chunk boundaries for audio splitting.
    
    Returns list of (start, end) tuples in seconds.
    """
    if total_duration <= MAX_CHUNK_DURATION_SEC:
        return [(0, total_duration)]
    
    num_chunks = math.ceil(total_duration / MAX_CHUNK_DURATION_SEC)
    chunk_size = total_duration / num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_duration)
        chunks.append((start, end))
    
    return chunks


def _parse_rttm(rttm_path: str) -> List[Dict]:
    """Parse RTTM file to segments."""
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                segments.append({
                    "start": float(parts[3]),
                    "end": float(parts[3]) + float(parts[4]),
                    "speaker": parts[7]
                })
    return sorted(segments, key=lambda x: x["start"])


def _detect_speaker_genders(wav_path: str, segments: List[Dict]) -> Dict[str, str]:
    """
    Detect gender for each speaker using fundamental frequency (F0) analysis.

    Male voices:   median F0 ~85–165 Hz
    Female voices: median F0 ~165–255 Hz
    Threshold: 165 Hz

    Returns dict: {speaker_id: "male"/"female"/"unknown"}
    """
    try:
        import librosa
        import numpy as np

        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        total_dur = len(audio) / sr

        # Collect F0 samples per speaker
        speaker_f0: Dict[str, list] = {}
        for seg in segments:
            spk = seg["speaker"]
            start = seg["start"]
            end = min(seg["end"], total_dur)
            dur = end - start
            if dur < 0.3:
                continue

            # Extract audio chunk for this segment
            s = int(start * sr)
            e = int(end * sr)
            chunk = audio[s:e]

            # Compute F0 with pyin (robust for speech)
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    chunk,
                    fmin=60.0,
                    fmax=400.0,
                    sr=sr,
                    frame_length=2048,
                )
                voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
                if len(voiced_f0) > 0:
                    speaker_f0.setdefault(spk, []).extend(voiced_f0.tolist())
            except Exception:
                pass

        # Classify each speaker
        genders: Dict[str, str] = {}
        THRESHOLD_HZ = 165.0
        for spk, f0_vals in speaker_f0.items():
            if not f0_vals:
                genders[spk] = "unknown"
                continue
            median_f0 = float(np.median(f0_vals))
            gender = "female" if median_f0 >= THRESHOLD_HZ else "male"
            genders[spk] = gender
            logger.info(
                "Gender detection: %s → median F0=%.1f Hz → %s",
                spk, median_f0, gender
            )

        # Speakers with no voiced frames → unknown
        all_speakers = set(s["speaker"] for s in segments)
        for spk in all_speakers:
            genders.setdefault(spk, "unknown")

        return genders

    except Exception as e:
        logger.warning("Gender detection failed: %s", e)
        return {s["speaker"]: "unknown" for s in segments}


def _get_multiscale_params(duration: float, gpu_memory_gb: float = 6.0):
    """
    Choose multi-scale parameters based on audio duration and GPU memory.
    
    The affinity matrix is N×N where N ~ segments count (duration / min_window).
    For long audio this grows quadratically, causing OOM even on T4 (16GB).
    
    Strategy:
    - Short  (<30min):            5 scales (best quality, any GPU)
    - Medium (30-90min):          5 scales for >=12GB, 3 scales for smaller
    - Long   (90-120min):         3 scales for any GPU
    - Very long (>120min):        2 scales for any GPU
    """
    duration_min = duration / 60.0
    
    if duration_min <= 30:
        # Full 5-scale for short audio on any GPU
        return {
            "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
            "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
            "multiscale_weights": [1, 1, 1, 1, 1],
        }
    elif duration_min <= 90:
        if gpu_memory_gb >= 12:
            # 5 scales for medium audio on T4/large GPU
            return {
                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                "multiscale_weights": [1, 1, 1, 1, 1],
            }
        else:
            # 3 scales for medium audio on small GPU
            return {
                "window_length_in_sec": [1.5, 1.0, 0.5],
                "shift_length_in_sec": [0.75, 0.5, 0.25],
                "multiscale_weights": [1, 1, 1],
            }
    elif duration_min <= 120:
        # 3 scales for long audio (90-120min) on any GPU
        return {
            "window_length_in_sec": [1.5, 1.0, 0.5],
            "shift_length_in_sec": [0.75, 0.5, 0.25],
            "multiscale_weights": [1, 1, 1],
        }
    else:
        # 2 scales for very long audio (>120 min) on any GPU
        return {
            "window_length_in_sec": [1.5, 0.75],
            "shift_length_in_sec": [0.75, 0.375],
            "multiscale_weights": [1, 1],
        }


def _diarize_single_chunk(
    wav_path: str,
    tmpdir: str,
    chunk_idx: int,
    duration: float,
    num_speakers: int,
    max_speakers: int,
    gpu_memory_gb: float,
    device: str,
) -> List[Dict]:
    """
    Run NeMo diarization on a single audio chunk.
    
    Returns list of segments with start/end times relative to chunk start.
    """
    import gc
    import torch
    from nemo.collections.asr.models import ClusteringDiarizer
    from omegaconf import OmegaConf
    
    logger.info(f"Diarizing chunk {chunk_idx + 1}: {duration:.1f}s ({duration/60:.1f} min)")
    
    # Choose multi-scale params based on chunk duration
    ms_params = _get_multiscale_params(duration, gpu_memory_gb)
    num_scales = len(ms_params["window_length_in_sec"])
    logger.info(f"Chunk {chunk_idx + 1}: using {num_scales} scales")
    
    # Adjust batch size
    if gpu_memory_gb >= 12:
        batch_size = 128
    elif duration > 1800 and gpu_memory_gb < 8:
        batch_size = 32
    else:
        batch_size = 64
    
    # Create manifest for this chunk
    manifest_path = os.path.join(tmpdir, f"manifest_chunk{chunk_idx}.json")
    output_dir = os.path.join(tmpdir, f"output_chunk{chunk_idx}")
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_entry = {
        "audio_filepath": wav_path,
        "offset": 0,
        "duration": duration,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers if num_speakers > 0 else None,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_entry, f)
        f.write("\n")
    
    # NeMo configuration
    config = OmegaConf.create({
        "device": device,
        "verbose": True,
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": batch_size,
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": output_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    **ms_params,
                    "save_embeddings": False,
                }
            },
            
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers > 0,
                    "max_num_speakers": max_speakers,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                }
            },
            
            "msdd_model": {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    "infer_batch_size": 50 if gpu_memory_gb >= 12 else 25,
                    "sigmoid_threshold": [0.7],
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "diar_window_length": 50,
                    "overlap_infer_spk_limit": 5,
                }
            },
            
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.5,
                    "onset": 0.4,
                    "offset": 0.3,
                    "pad_onset": 0.05,
                    "pad_offset": -0.1,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                }
            },
        }
    })
    
    diarizer = ClusteringDiarizer(cfg=config)
    diarizer.diarize()
    
    # Find results
    rttm_files = list(Path(output_dir).glob("pred_rttms/*.rttm"))
    if not rttm_files:
        logger.warning(f"Chunk {chunk_idx + 1}: no RTTM output")
        return []
    
    segments = _parse_rttm(str(rttm_files[0]))
    logger.info(f"Chunk {chunk_idx + 1}: found {len(segments)} segments")
    
    # Cleanup
    del diarizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return segments


def nemo_diarize_task(
    audio_path: str,
    num_speakers: int = 0,
    max_speakers: int = 8,
    voice_mix: str = "male_duo",
) -> Dict[str, Any]:
    """
    NeMo MSDD diarization task with automatic chunking for long audio.
    
    Args:
        audio_path: Path or filename of audio on main server
        num_speakers: Fixed number of speakers (0 for auto-detect)
        max_speakers: Maximum speakers for auto-detection
        voice_mix: Voice mix mode. Gender detection only runs when "mixed_duo".
    
    Returns:
        Dict with segments, num_speakers, etc.
    
    Raises:
        RuntimeError: on CUDA OOM or other diarization errors
    """
    import time as _time
    task_start = _time.time()
    filename = Path(audio_path).name
    logger.info(f"NeMo Task started: {filename}")
    
    # Optimize CUDA memory allocation for large affinity matrices
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:512"
    )
    
    # Set status to busy
    set_status("busy")
    
    duration = 0
    gpu_memory_gb = 0.0
    
    try:
        # Lazy imports for clean CUDA context
        import gc
        import torch
        import soundfile as sf
        
        # Initialize CUDA
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            
            total_mem = torch.cuda.get_device_properties(0).total_memory
            free_mem, _ = torch.cuda.mem_get_info(0)
            gpu_memory_gb = total_mem / (1024**3)
            free_gb = free_mem / (1024**3)
            
            logger.info(
                f"CUDA device: {torch.cuda.get_device_name(0)}, "
                f"VRAM total: {gpu_memory_gb:.1f} GB, free: {free_gb:.1f} GB"
            )
            
            if free_gb < 2.0:
                logger.warning(
                    f"Low GPU memory before start ({free_gb:.1f} GB free). "
                    f"Forcing full cache clear..."
                )
                for _ in range(3):
                    gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free_mem, _ = torch.cuda.mem_get_info(0)
                free_gb = free_mem / (1024**3)
                logger.info(f"After aggressive cleanup: {free_gb:.1f} GB free")
            
            # Warm up cuBLAS
            a = torch.randn(64, 64, device='cuda')
            b = torch.randn(64, 64, device='cuda')
            _ = torch.mm(a, b)
            del a, b, _
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            logger.info(f"CUDA ready: {torch.cuda.get_device_name(0)}, VRAM: {gpu_memory_gb:.1f} GB")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download audio from main server
            local_audio = os.path.join(tmpdir, filename)
            if not download_audio_by_path(audio_path, local_audio):
                raise RuntimeError(f"Failed to download audio: {filename}")
            
            # Convert to WAV if needed
            if not local_audio.lower().endswith('.wav'):
                wav_path = os.path.join(tmpdir, "audio.wav")
                _extract_wav(local_audio, wav_path)
            else:
                wav_path = local_audio
            
            # Get duration
            audio_info = sf.info(wav_path)
            duration = audio_info.duration
            logger.info(f"Audio duration: {duration:.2f}s ({duration/60:.1f} min)")
            
            # Calculate chunks
            chunks = _get_audio_chunks(duration)
            num_chunks = len(chunks)
            
            if num_chunks == 1:
                logger.info("Audio fits in single chunk, processing directly...")
            else:
                logger.info(f"Audio too long, splitting into {num_chunks} chunks "
                           f"(max {MAX_CHUNK_DURATION_SEC/60:.0f} min each)")
            
            all_segments = []
            
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                chunk_duration = chunk_end - chunk_start
                
                if num_chunks > 1:
                    # Split audio for this chunk
                    chunk_wav = os.path.join(tmpdir, f"chunk_{chunk_idx}.wav")
                    logger.info(f"Extracting chunk {chunk_idx + 1}/{num_chunks}: "
                               f"{chunk_start/60:.1f}-{chunk_end/60:.1f} min")
                    _split_audio_chunk(wav_path, chunk_wav, chunk_start, chunk_end)
                    process_wav = chunk_wav
                else:
                    process_wav = wav_path
                
                # Diarize this chunk
                chunk_segments = _diarize_single_chunk(
                    wav_path=process_wav,
                    tmpdir=tmpdir,
                    chunk_idx=chunk_idx,
                    duration=chunk_duration,
                    num_speakers=num_speakers,
                    max_speakers=max_speakers,
                    gpu_memory_gb=gpu_memory_gb,
                    device=device,
                )
                
                # Adjust timestamps and speaker IDs for multi-chunk
                for seg in chunk_segments:
                    # Shift timestamps by chunk start
                    seg["start"] += chunk_start
                    seg["end"] += chunk_start
                    
                    # Rename speakers to avoid collisions between chunks
                    # SPEAKER_00 in chunk 0 stays SPEAKER_00
                    # SPEAKER_00 in chunk 1 becomes SPEAKER_10
                    # SPEAKER_00 in chunk 2 becomes SPEAKER_20
                    if num_chunks > 1:
                        old_id = seg["speaker"]
                        # Extract speaker number (e.g., "SPEAKER_00" -> 0)
                        try:
                            spk_num = int(old_id.split("_")[-1])
                            new_num = chunk_idx * 10 + spk_num
                            seg["speaker"] = f"SPEAKER_{new_num:02d}"
                        except (ValueError, IndexError):
                            # Keep original if parsing fails
                            seg["speaker"] = f"{old_id}_chunk{chunk_idx}"
                
                all_segments.extend(chunk_segments)
                
                # Clear GPU memory between chunks
                if num_chunks > 1:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Sort all segments by start time
            all_segments.sort(key=lambda x: x["start"])
            
            if not all_segments:
                raise RuntimeError("NeMo diarization produced no speech segments")
            
            # Detect speaker genders via F0 analysis — only for mixed_duo mode
            if voice_mix == "mixed_duo":
                logger.info("Running gender detection via F0 analysis (voice_mix=mixed_duo)...")
                speaker_genders = _detect_speaker_genders(wav_path, all_segments)
                logger.info(f"Gender results: {speaker_genders}")
            else:
                logger.info(f"Skipping gender detection (voice_mix={voice_mix})")
                speaker_genders = {}
            
            # Statistics
            speakers = set(s["speaker"] for s in all_segments)
            total_speech = sum(s["end"] - s["start"] for s in all_segments)
            elapsed = _time.time() - task_start
            
            logger.info(f"Diarization complete in {elapsed:.1f}s: "
                        f"{len(speakers)} speakers, {len(all_segments)} segments, "
                        f"{total_speech:.1f}s speech, {num_chunks} chunk(s)")
            
            return {
                "segments": all_segments,
                "num_speakers": len(speakers),
                "total_speech_duration": total_speech,
                "speaker_genders": speaker_genders,
            }
    
    except torch.cuda.OutOfMemoryError as oom:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        logger.error(f"CUDA OOM during NeMo diarization on {gpu_name}: {oom}")
        raise RuntimeError(
            f"GPU memory exhausted ({gpu_name}, {gpu_memory_gb:.0f}GB). "
            f"Audio too long ({duration/60:.0f} min) for this GPU. "
            f"Try a shorter file or use Pyannote diarizer."
        ) from oom
    
    except Exception as e:
        logger.error(f"NeMo diarization failed: {e}", exc_info=True)
        raise RuntimeError(f"NeMo diarization error: {e}") from e
    
    finally:
        # Cleanup GPU
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Set status back to ready
        set_status("ready")
