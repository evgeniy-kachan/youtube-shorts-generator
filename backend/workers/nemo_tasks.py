"""
NeMo diarization tasks for the NeMo worker.

These tasks run in a SEPARATE worker process (nemo-worker) with its own CUDA context.
This ensures NeMo has isolated GPU access without conflicts with Pyannote/WhisperX.

CRITICAL: This module does NOT import torch at module level!
All torch/NeMo imports happen INSIDE task functions (lazy import).
This prevents CUDA context initialization until the task actually runs.

Architecture:
    Backend → Redis (nemo_tasks queue) → nemo-worker → nemo_tasks.py
    
    nemo-worker uses venv-nemo with NeMo installed.
    CUDA context is initialized only when task runs, not at worker startup.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_wav(input_path: str, dst_path: str, sample_rate: int = 16000) -> None:
    """Extract mono wav with desired sample rate using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", str(sample_rate), "-vn", dst_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd[:6]) + " ...")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _run_nemo_diarization(
    wav_path: str,
    device: str = "cuda",
    num_speakers: Optional[int] = None,
    max_speakers: int = 8,
) -> List[Dict]:
    """
    Run NeMo MSDD diarization directly (no subprocess).
    
    LAZY IMPORT: torch and NeMo are imported here, not at module level.
    This ensures CUDA context is only initialized when task runs.
    
    Args:
        wav_path: Path to WAV file (16kHz mono)
        device: Device to use (cuda/cpu)
        num_speakers: Expected number of speakers (None = auto-detect)
        max_speakers: Maximum number of speakers for auto-detection
        
    Returns:
        List of diarization segments
    """
    # LAZY IMPORT - critical for CUDA isolation!
    import gc
    import torch
    
    logger.info("NeMo: Initializing CUDA context...")
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Initialize CUDA context properly
    if device == "cuda":
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            
            # Initialize fresh CUDA context with simple tensor
            init_tensor = torch.zeros(1, device="cuda")
            del init_tensor
            torch.cuda.synchronize()
            
            # CRITICAL: Initialize cuBLAS by doing a matrix multiplication
            # This forces cuBLAS handle creation BEFORE NeMo loads models
            logger.info("Initializing cuBLAS...")
            a = torch.randn(64, 64, device="cuda")
            b = torch.randn(64, 64, device="cuda")
            c = torch.mm(a, b)  # This initializes cuBLAS
            del a, b, c
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("cuBLAS initialized successfully")
            
            # Enable TF32 for faster computation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("CUDA initialized: %s (%.1f GB)", gpu_name, gpu_memory)
            
        except Exception as e:
            logger.warning("CUDA initialization failed: %s. Falling back to CPU.", e)
            device = "cpu"
    
    logger.info("Loading NeMo diarization model on device: %s", device)
    
    # LAZY IMPORT NeMo
    try:
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf
        import soundfile as sf
    except ImportError as e:
        logger.error("NeMo not installed. Install with: pip install nemo_toolkit[asr]")
        raise RuntimeError(f"NeMo import failed: {e}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create manifest file (NeMo requires this format)
        manifest_path = Path(tmpdir) / "manifest.json"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Get audio duration
        audio_info = sf.info(wav_path)
        duration = audio_info.duration
        
        # Write manifest
        manifest_entry = {
            "audio_filepath": wav_path,
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers if num_speakers and num_speakers > 0 else None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_entry, f)
            f.write("\n")
        
        logger.info("Created manifest for %.2fs audio", duration)
        
        # NeMo diarization config
        # CRITICAL: num_workers=0 to avoid pickle errors with spawn multiprocessing
        config = OmegaConf.create({
            "device": device,
            "verbose": True,
            "num_workers": 0,  # Must be 0 to avoid pickle errors!
            "sample_rate": 16000,
            "batch_size": 64,
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(output_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    }
                },
                
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": num_speakers is not None and num_speakers > 0,
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
                        "infer_batch_size": 25,
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
        
        if num_speakers and num_speakers > 0:
            config.diarizer.clustering.parameters.oracle_num_speakers = True
            logger.info("Using fixed number of speakers: %d", num_speakers)
        else:
            config.diarizer.clustering.parameters.oracle_num_speakers = False
            logger.info("Auto-detecting number of speakers (max: %d)", max_speakers)
        
        logger.info("Initializing NeMo ClusteringDiarizer...")
        
        try:
            diarizer = ClusteringDiarizer(cfg=config)
            
            logger.info("Running NeMo diarization...")
            diarizer.diarize()
            
            # Parse RTTM output
            rttm_files = list(output_dir.glob("pred_rttms/*.rttm"))
            if not rttm_files:
                logger.warning("No RTTM output found")
                return []
            
            segments = _parse_rttm(str(rttm_files[0]))
            
            # Log quality metrics
            _log_diarization_quality(segments, duration)
            
            return segments
            
        except Exception as e:
            logger.error("NeMo diarization failed: %s", e, exc_info=True)
            
            # Fallback: try without MSDD
            logger.info("Trying fallback without MSDD...")
            config.diarizer.msdd_model.model_path = None
            
            try:
                diarizer = ClusteringDiarizer(cfg=config)
                diarizer.diarize()
                
                rttm_files = list(output_dir.glob("pred_rttms/*.rttm"))
                if rttm_files:
                    return _parse_rttm(str(rttm_files[0]))
            except Exception as fallback_e:
                logger.error("Fallback also failed: %s", fallback_e)
            
            return []


def _parse_rttm(rttm_path: str) -> List[Dict]:
    """Parse RTTM file to list of segments."""
    segments = []
    
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                
                segments.append({
                    "start": start,
                    "end": start + duration,
                    "speaker": speaker,
                })
    
    segments.sort(key=lambda x: x["start"])
    return segments


def _log_diarization_quality(segments: List[Dict], total_duration: float) -> None:
    """Log detailed quality metrics for diarization results."""
    if not segments:
        logger.warning("QUALITY: No segments to analyze")
        return
    
    speaker_stats = {}
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        duration = seg.get("end", 0) - seg.get("start", 0)
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"segments": 0, "total_duration": 0.0}
        
        speaker_stats[speaker]["segments"] += 1
        speaker_stats[speaker]["total_duration"] += duration
    
    num_speakers = len(speaker_stats)
    total_speech = sum(s["total_duration"] for s in speaker_stats.values())
    speech_ratio = (total_speech / total_duration * 100) if total_duration > 0 else 0
    
    logger.info("=" * 60)
    logger.info("NEMO DIARIZATION QUALITY REPORT")
    logger.info("=" * 60)
    logger.info("Audio duration: %.2f sec", total_duration)
    logger.info("Total speech: %.2f sec (%.1f%% of audio)", total_speech, speech_ratio)
    logger.info("Speakers detected: %d", num_speakers)
    logger.info("Total segments: %d", len(segments))
    
    for speaker in sorted(speaker_stats.keys()):
        stats = speaker_stats[speaker]
        speaker_ratio = (stats["total_duration"] / total_speech * 100) if total_speech > 0 else 0
        logger.info("  %s: %d segments, %.1fs (%.1f%%)",
                   speaker, stats["segments"], stats["total_duration"], speaker_ratio)
    
    logger.info("=" * 60)


def nemo_diarize_task(
    audio_path: str,
    num_speakers: int = 0,
    max_speakers: int = 8,
) -> Dict[str, Any]:
    """
    NeMo diarization task for RQ queue.
    
    This is the main entry point called by nemo-worker.
    Runs NeMo DIRECTLY (no subprocess) with lazy imports.
    
    CRITICAL: torch/NeMo are imported inside _run_nemo_diarization(),
    not at module level. This ensures clean CUDA context.
    
    Args:
        audio_path: Path to audio/video file
        num_speakers: Number of speakers (0 = auto)
        max_speakers: Maximum speakers for auto-detection
    
    Returns:
        {
            "segments": [...],
            "num_speakers": 2,
            "speaker_stats": {...},
            "total_speech_duration": 123.4,
        }
    """
    logger.info("NeMo Task: nemo_diarize_task started, file=%s", Path(audio_path).name)
    
    input_path = Path(audio_path)
    num_speakers_param = num_speakers if num_speakers > 0 else None
    
    # Check if we need to extract audio
    if input_path.suffix.lower() == ".wav":
        # Check if it's already 16kHz mono
        try:
            # Lazy import soundfile
            import soundfile as sf
            info = sf.info(str(input_path))
            
            if info.samplerate == 16000 and info.channels == 1:
                logger.info("Input is already 16kHz mono WAV")
                wav_path = str(input_path)
                segments = _run_nemo_diarization(
                    wav_path,
                    device="cuda",
                    num_speakers=num_speakers_param,
                    max_speakers=max_speakers,
                )
            else:
                # Need to convert
                with tempfile.TemporaryDirectory() as tmpdir:
                    wav_path = str(Path(tmpdir) / "audio.wav")
                    _extract_wav(str(input_path), wav_path)
                    segments = _run_nemo_diarization(
                        wav_path,
                        device="cuda",
                        num_speakers=num_speakers_param,
                        max_speakers=max_speakers,
                    )
        except Exception:
            # If soundfile fails, just extract
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = str(Path(tmpdir) / "audio.wav")
                _extract_wav(str(input_path), wav_path)
                segments = _run_nemo_diarization(
                    wav_path,
                    device="cuda",
                    num_speakers=num_speakers_param,
                    max_speakers=max_speakers,
                )
    else:
        # Extract audio from video/other format
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = str(Path(tmpdir) / "audio.wav")
            _extract_wav(str(input_path), wav_path)
            segments = _run_nemo_diarization(
                wav_path,
                device="cuda",
                num_speakers=num_speakers_param,
                max_speakers=max_speakers,
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
    
    logger.info("NeMo Task: complete, %d speakers, %.1fs speech",
               num_speakers_detected, total_speech)
    
    return {
        "segments": segments,
        "num_speakers": num_speakers_detected,
        "speaker_stats": speaker_stats,
        "total_speech_duration": total_speech,
    }
