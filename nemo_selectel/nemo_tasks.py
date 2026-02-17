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
from pathlib import Path
from typing import Dict, Any, List
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
    
    If audio_path looks like a video_id (UUID), use the internal API.
    Otherwise, try to download by filename from temp directory.
    """
    import re
    
    # Check if it's a UUID (video_id)
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    filename = Path(audio_path).name
    
    # Try internal API with video_id first (if path contains video_id)
    # The audio_path from main server is the full path like /tmp/youtube-shorts/video.mp4
    # We need to extract video_id from the task context or use filename
    
    # For now, use the temp file endpoint
    url = f"{MAIN_SERVER_URL}/api/files/temp/{filename}"
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


def nemo_diarize_task(
    audio_path: str,
    num_speakers: int = 0,
    max_speakers: int = 8,
) -> Dict[str, Any]:
    """
    NeMo MSDD diarization task.
    
    Args:
        audio_path: Path or filename of audio on main server
        num_speakers: Fixed number of speakers (0 for auto-detect)
        max_speakers: Maximum speakers for auto-detection
    
    Returns:
        Dict with segments, num_speakers, etc.
    """
    filename = Path(audio_path).name
    logger.info(f"NeMo Task started: {filename}")
    
    # Set status to busy
    set_status("busy")
    
    try:
        # Lazy imports for clean CUDA context
        import gc
        import torch
        import soundfile as sf
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf
        
        # Initialize CUDA
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            
            # Warm up cuBLAS
            _ = torch.mm(torch.randn(1, 1, device='cuda'), torch.randn(1, 1, device='cuda'))
            torch.cuda.synchronize()
            
            logger.info(f"CUDA ready: {torch.cuda.get_device_name(0)}")
        
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
            logger.info(f"Audio duration: {duration:.2f}s")
            
            # Create manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            output_dir = os.path.join(tmpdir, "output")
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
                "batch_size": 64,
                "diarizer": {
                    "manifest_filepath": manifest_path,
                    "out_dir": output_dir,
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
            
            try:
                logger.info("Initializing NeMo ClusteringDiarizer...")
                diarizer = ClusteringDiarizer(cfg=config)
                
                logger.info("Running diarization...")
                diarizer.diarize()
                
                # Find results
                rttm_files = list(Path(output_dir).glob("pred_rttms/*.rttm"))
                if not rttm_files:
                    logger.warning("No RTTM output found")
                    return {"segments": [], "num_speakers": 0}
                
                segments = _parse_rttm(str(rttm_files[0]))
                
                # Statistics
                speakers = set(s["speaker"] for s in segments)
                total_speech = sum(s["end"] - s["start"] for s in segments)
                
                logger.info(f"Diarization complete: {len(speakers)} speakers, {total_speech:.1f}s speech")
                
                return {
                    "segments": segments,
                    "num_speakers": len(speakers),
                    "total_speech_duration": total_speech,
                }
                
            except Exception as e:
                logger.error(f"NeMo diarization failed: {e}", exc_info=True)
                return {"segments": [], "num_speakers": 0, "error": str(e)}
            
            finally:
                # Cleanup GPU
                if torch.cuda.is_available():
                    del diarizer
                    gc.collect()
                    torch.cuda.empty_cache()
    
    finally:
        # Set status back to ready
        set_status("ready")
