"""
CLI for speaker diarization using NVIDIA NeMo MSDD.

NeMo MSDD (Multi-Scale Diarization Decoder) provides state-of-the-art
speaker diarization, especially for:
- Similar voices
- Overlapping speech
- Long recordings

Usage (run from venv-nemo):
    python backend/tools/diarize_nemo.py --input path/to/audio --output diarization.json

Output JSON format:
{
  "segments": [
    {"start": 0.00, "end": 3.12, "speaker": "speaker_0"},
    {"start": 3.12, "end": 7.45, "speaker": "speaker_1"},
    ...
  ]
}

Requirements (venv-nemo):
    pip install nemo_toolkit[asr]
    # or for full installation:
    pip install nemo_toolkit[all]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_wav(input_path: str, dst_path: str, sample_rate: int = 16000) -> None:
    """Extract mono wav with desired sample rate."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        dst_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def is_wav_file(path: str) -> bool:
    """Check if file is already a WAV file."""
    try:
        return Path(path).suffix.lower() == ".wav"
    except Exception:
        return False


def run_nemo_diarization(
    wav_path: str,
    device: str = "cuda",
    num_speakers: Optional[int] = None,
    max_speakers: int = 8,
) -> List[Dict]:
    """
    Run NeMo MSDD diarization and return list of segments.
    
    Args:
        wav_path: Path to WAV file (16kHz mono)
        device: Device to use (cuda/cpu)
        num_speakers: Expected number of speakers (None = auto-detect)
        max_speakers: Maximum number of speakers for auto-detection
        
    Returns:
        List of diarization segments
    """
    import torch
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    logger.info("Loading NeMo diarization model on device: %s", device)
    
    try:
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf
    except ImportError as e:
        logger.error("NeMo not installed. Install with: pip install nemo_toolkit[asr]")
        raise RuntimeError(f"NeMo import failed: {e}")
    
    # Create config for NeMo diarization
    # Based on: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create manifest file (NeMo requires this format)
        manifest_path = Path(tmpdir) / "manifest.json"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Get audio duration
        import soundfile as sf
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
        # Using neural diarizer with MSDD (Multi-Scale Diarization Decoder)
        config = OmegaConf.create({
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(output_dir),
                "oracle_vad": False,  # Use neural VAD
                "collar": 0.25,  # Collar for evaluation (seconds)
                "ignore_overlap": True,  # Ignore overlapping speech regions
                
                # Speaker embeddings (TitaNet)
                "speaker_embeddings": {
                    "model_path": "titanet_large",  # Pre-trained TitaNet
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    }
                },
                
                # Clustering parameters
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
                
                # MSDD (Multi-Scale Diarization Decoder) - neural refinement
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",  # Pre-trained MSDD
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
                
                # VAD (Voice Activity Detection)
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
        
        # Set number of speakers if specified
        if num_speakers and num_speakers > 0:
            config.diarizer.clustering.parameters.oracle_num_speakers = True
            logger.info("Using fixed number of speakers: %d", num_speakers)
        else:
            config.diarizer.clustering.parameters.oracle_num_speakers = False
            logger.info("Auto-detecting number of speakers (max: %d)", max_speakers)
        
        # Initialize and run diarizer
        logger.info("Initializing NeMo ClusteringDiarizer...")
        
        try:
            # Create diarizer
            diarizer = ClusteringDiarizer(cfg=config)
            
            # Run diarization
            logger.info("Running NeMo diarization...")
            diarizer.diarize()
            
            # Parse RTTM output
            rttm_files = list(output_dir.glob("pred_rttms/*.rttm"))
            if not rttm_files:
                logger.warning("No RTTM output found")
                return []
            
            rttm_path = rttm_files[0]
            segments = parse_rttm(str(rttm_path))
            
            logger.info("NeMo diarization produced %d segments", len(segments))
            return segments
            
        except Exception as e:
            logger.error("NeMo diarization failed: %s", e, exc_info=True)
            
            # Fallback: try simpler clustering without MSDD
            logger.info("Trying fallback without MSDD...")
            config.diarizer.msdd_model.model_path = None
            
            try:
                diarizer = ClusteringDiarizer(cfg=config)
                diarizer.diarize()
                
                rttm_files = list(output_dir.glob("pred_rttms/*.rttm"))
                if rttm_files:
                    return parse_rttm(str(rttm_files[0]))
            except Exception as fallback_e:
                logger.error("Fallback also failed: %s", fallback_e)
            
            return []


def parse_rttm(rttm_path: str) -> List[Dict]:
    """
    Parse RTTM file to list of segments.
    
    RTTM format:
    SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
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
    
    # Sort by start time
    segments.sort(key=lambda x: x["start"])
    
    return segments


def main():
    parser = argparse.ArgumentParser(description="Run NeMo MSDD diarization and save JSON.")
    parser.add_argument("--input", required=True, help="Input video/audio file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_speakers", type=int, default=0, 
                       help="Expected number of speakers (0 = auto-detect)")
    parser.add_argument("--max_speakers", type=int, default=8,
                       help="Maximum speakers for auto-detection (default: 8)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # num_speakers=0 means auto-detect
    num_speakers = args.num_speakers if args.num_speakers > 0 else None

    # Check if input is already a valid WAV file
    if input_path.suffix.lower() == ".wav":
        logger.info("Input is a WAV file, checking format...")
        import soundfile as sf
        info = sf.info(str(input_path))
        
        if info.samplerate == 16000 and info.channels == 1:
            logger.info("Input is already 16kHz mono WAV, using directly")
            segments = run_nemo_diarization(
                str(input_path),
                device=args.device,
                num_speakers=num_speakers,
                max_speakers=args.max_speakers,
            )
        else:
            # Need to convert
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = Path(tmpdir) / "audio.wav"
                extract_wav(str(input_path), str(wav_path), sample_rate=16000)
                segments = run_nemo_diarization(
                    str(wav_path),
                    device=args.device,
                    num_speakers=num_speakers,
                    max_speakers=args.max_speakers,
                )
    else:
        # Extract audio from video/other format
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            extract_wav(str(input_path), str(wav_path), sample_rate=16000)
            segments = run_nemo_diarization(
                str(wav_path),
                device=args.device,
                num_speakers=num_speakers,
                max_speakers=args.max_speakers,
            )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    logger.info("Saved NeMo diarization to %s (%d segments)", output_path, len(segments))


if __name__ == "__main__":
    main()
