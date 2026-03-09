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
  ],
  "speaker_genders": {"speaker_0": "male", "speaker_1": "female"}
}

Requirements (venv-nemo):
    pip install nemo_toolkit[asr]
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
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", str(sample_rate), "-vn", dst_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def is_wav_file(path: str) -> bool:
    """Check if file is already a WAV file."""
    try:
        return Path(path).suffix.lower() == ".wav"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Adaptive multi-scale parameters (from nemo_selectel/nemo_tasks.py)
# ---------------------------------------------------------------------------

def _get_multiscale_params(duration: float, gpu_memory_gb: float = 16.0) -> dict:
    """
    Choose multi-scale parameters based on audio duration and GPU memory.

    KEY INSIGHT: The OOM bottleneck is the COMBINED affinity matrix + argsort,
    not the number of scales. The matrix size is N×N where:
        N ≈ speech_duration / min_shift × ~1.28 (NeMo overhead factor)

    Memory at argsort time (the OOM point):
        active_tensors ≈ k × N²  (affinity + intermediates)
        argsort_alloc  = N² × 8  (int64 indices)

    Empirical data from T4 (89 min audio, shift=0.25s, N≈27,400):
        active = 12.25 GiB, argsort needs 5.61 GiB → total 17.86 GiB → OOM on 15.56 GiB

    FIX: For long audio on small GPUs, increase min_shift to reduce N:
        shift=0.25s → N≈27,400 → argsort 5.6 GiB → OOM on T4
        shift=0.50s → N≈13,700 → argsort 1.5 GiB → fits easily (~6 GiB total)

    Quality impact of shift=0.5s vs 0.25s:
        Speaker boundary precision: ±0.5s instead of ±0.25s
        For podcasts (2 speakers, turns 2-10s): negligible DER difference (<1%)

    Strategy (T4 16 GB):
        ≤30 min:    5 scales, shift 0.25s (best quality)
        30-60 min:  5 scales, shift 0.25s (still fits)
        60-90 min:  3 scales, shift 0.5s  (N halved → fits with margin)
        90-120 min: 3 scales, shift 0.5s
        >120 min:   2 scales, shift 0.75s (or use chunking)
    """
    duration_min = duration / 60.0

    if duration_min <= 30:
        # Short audio — full 5 scales, finest shift (best quality)
        return {
            "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
            "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
            "multiscale_weights": [1, 1, 1, 1, 1],
        }
    elif duration_min <= 60:
        if gpu_memory_gb >= 40:
            # A100: full 5 scales
            return {
                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                "multiscale_weights": [1, 1, 1, 1, 1],
            }
        elif gpu_memory_gb >= 16:
            # T4: 5 scales still fits for ≤60 min
            return {
                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                "multiscale_weights": [1, 1, 1, 1, 1],
            }
        else:
            # Small GPU: 3 scales with fine shift
            return {
                "window_length_in_sec": [1.5, 1.0, 0.5],
                "shift_length_in_sec": [0.75, 0.5, 0.25],
                "multiscale_weights": [1, 1, 1],
            }
    elif duration_min <= 90:
        if gpu_memory_gb >= 40:
            # A100: full 5 scales
            return {
                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                "multiscale_weights": [1, 1, 1, 1, 1],
            }
        else:
            # T4 (16 GB): 3 scales + coarser shift to halve N → prevents OOM
            # N drops from ~27K to ~13K → memory drops ~4×
            return {
                "window_length_in_sec": [1.5, 1.0, 0.5],
                "shift_length_in_sec": [0.75, 0.5, 0.5],
                "multiscale_weights": [1, 1, 1],
            }
    elif duration_min <= 120:
        if gpu_memory_gb >= 40:
            return {
                "window_length_in_sec": [1.5, 1.0, 0.5],
                "shift_length_in_sec": [0.75, 0.5, 0.25],
                "multiscale_weights": [1, 1, 1],
            }
        else:
            # T4: 3 scales, coarser shift
            return {
                "window_length_in_sec": [1.5, 1.0, 0.5],
                "shift_length_in_sec": [0.75, 0.5, 0.5],
                "multiscale_weights": [1, 1, 1],
            }
    else:
        # Very long (>120 min): 2 scales with coarse shift (or use chunking)
        return {
            "window_length_in_sec": [1.5, 0.75],
            "shift_length_in_sec": [0.75, 0.75],
            "multiscale_weights": [1, 1],
        }


def _get_batch_sizes(duration: float, gpu_memory_gb: float = 16.0) -> tuple:
    """Choose batch sizes based on duration and GPU memory."""
    duration_min = duration / 60.0

    if duration_min > 130:
        # Very long audio: minimal batches
        batch_size = 18
        infer_batch_size = 10
        logger.info("Very long video (%.0f min): batch=%d, infer=%d",
                     duration_min, batch_size, infer_batch_size)
    elif duration_min > 60 and gpu_memory_gb <= 24:
        # 60-130 min on T4/A10: reduce batch to leave more room for clustering
        batch_size = 64
        infer_batch_size = 25
        logger.info("Long video on ≤24GB GPU (%.0f min): batch=%d, infer=%d",
                     duration_min, batch_size, infer_batch_size)
    elif gpu_memory_gb >= 12:
        batch_size = 128
        infer_batch_size = 50
    elif duration > 1800 and gpu_memory_gb < 8:
        batch_size = 32
        infer_batch_size = 15
    else:
        batch_size = 64
        infer_batch_size = 25

    return batch_size, infer_batch_size


# ---------------------------------------------------------------------------
# Gender detection via F0 analysis (from nemo_selectel/nemo_tasks.py)
# ---------------------------------------------------------------------------

def _detect_speaker_genders(wav_path: str, segments: List[Dict]) -> Dict[str, str]:
    """
    Detect gender for each speaker using fundamental frequency (F0) analysis.

    Male voices:   median F0 ~85-165 Hz
    Female voices: median F0 ~165-255 Hz
    Threshold: 165 Hz

    Returns dict: {speaker_id: "male"/"female"/"unknown"}
    """
    try:
        import librosa
        import numpy as np

        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        total_dur = len(audio) / sr

        speaker_f0: Dict[str, list] = {}
        for seg in segments:
            spk = seg["speaker"]
            start = seg["start"]
            end = min(seg["end"], total_dur)
            dur = end - start
            if dur < 0.3:
                continue

            s = int(start * sr)
            e = int(end * sr)
            chunk = audio[s:e]

            try:
                f0, voiced_flag, _ = librosa.pyin(
                    chunk, fmin=60.0, fmax=400.0, sr=sr, frame_length=2048,
                )
                voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
                if len(voiced_f0) > 0:
                    speaker_f0.setdefault(spk, []).extend(voiced_f0.tolist())
            except Exception:
                pass

        THRESHOLD_HZ = 165.0
        genders: Dict[str, str] = {}
        for spk, f0_vals in speaker_f0.items():
            if not f0_vals:
                genders[spk] = "unknown"
                continue
            import numpy as np
            median_f0 = float(np.median(f0_vals))
            gender = "female" if median_f0 >= THRESHOLD_HZ else "male"
            genders[spk] = gender
            logger.info("Gender: %s → median F0=%.1f Hz → %s", spk, median_f0, gender)

        all_speakers = set(s["speaker"] for s in segments)
        for spk in all_speakers:
            genders.setdefault(spk, "unknown")

        return genders

    except Exception as e:
        logger.warning("Gender detection failed: %s", e)
        return {s["speaker"]: "unknown" for s in segments}


# ---------------------------------------------------------------------------
# Main diarization function
# ---------------------------------------------------------------------------

def run_nemo_diarization(
    wav_path: str,
    device: str = "cuda",
    num_speakers: Optional[int] = None,
    max_speakers: int = 8,
    detect_gender: bool = False,
) -> Dict:
    """
    Run NeMo MSDD diarization and return segments + metadata.

    Args:
        wav_path: Path to WAV file (16kHz mono)
        device: Device to use (cuda/cpu)
        num_speakers: Expected number of speakers (None = auto-detect)
        max_speakers: Maximum number of speakers for auto-detection
        detect_gender: Whether to run F0-based gender detection

    Returns:
        Dict with segments, speaker_genders, etc.
    """
    import torch
    import gc

    # Optimize CUDA memory allocation for large affinity matrices
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:512"
    )

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    gpu_memory_gb = 0.0
    if device == "cuda":
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)

            total_mem = torch.cuda.get_device_properties(0).total_memory
            free_mem, _ = torch.cuda.mem_get_info(0)
            gpu_memory_gb = total_mem / (1024 ** 3)
            free_gb = free_mem / (1024 ** 3)

            logger.info("CUDA device: %s, VRAM total: %.1f GB, free: %.1f GB",
                        torch.cuda.get_device_name(0), gpu_memory_gb, free_gb)

            if free_gb < 2.0:
                logger.warning("Low GPU memory (%.1f GB free). Aggressive cleanup...", free_gb)
                for _ in range(3):
                    gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free_mem, _ = torch.cuda.mem_get_info(0)
                logger.info("After cleanup: %.1f GB free", free_mem / (1024 ** 3))

            # Warm up cuBLAS
            a = torch.randn(64, 64, device="cuda")
            b = torch.randn(64, 64, device="cuda")
            _ = torch.mm(a, b)
            del a, b, _
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Enable TF32 for faster computation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            logger.info("CUDA ready: %s, VRAM: %.1f GB, TF32 enabled",
                        torch.cuda.get_device_name(0), gpu_memory_gb)

        except Exception as e:
            logger.warning("CUDA initialization failed: %s. Falling back to CPU.", e)
            device = "cpu"

    logger.info("Loading NeMo diarization model on device: %s", device)

    try:
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf
        import soundfile as sf
    except ImportError as e:
        logger.error("NeMo not installed. Install with: pip install nemo_toolkit[asr]")
        raise RuntimeError(f"NeMo import failed: {e}")

    diarizer = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Get audio duration
            audio_info = sf.info(wav_path)
            duration = audio_info.duration
            logger.info("Audio duration: %.2fs (%.1f min)", duration, duration / 60)

            # Adaptive parameters based on duration and GPU
            ms_params = _get_multiscale_params(duration, gpu_memory_gb)
            batch_size, infer_batch_size = _get_batch_sizes(duration, gpu_memory_gb)
            num_scales = len(ms_params["window_length_in_sec"])
            min_shift = min(ms_params["shift_length_in_sec"])
            est_N = int(duration / min_shift * 1.28)  # empirical NeMo overhead
            est_argsort_gb = est_N * est_N * 8 / (1024**3)
            logger.info(
                "NeMo params: %d scales, min_shift=%.2fs, est_N≈%d, "
                "est_argsort≈%.1f GiB, batch=%d, infer=%d, GPU=%.1f GB",
                num_scales, min_shift, est_N, est_argsort_gb,
                batch_size, infer_batch_size, gpu_memory_gb
            )
            logger.info("Windows: %s, Shifts: %s",
                        ms_params["window_length_in_sec"],
                        ms_params["shift_length_in_sec"])

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
                "num_speakers": num_speakers if num_speakers and num_speakers > 0 else None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest_entry, f)
                f.write("\n")

            # NeMo diarization config
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
                            "infer_batch_size": infer_batch_size,
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
                logger.info("Auto-detecting speakers (max: %d)", max_speakers)

            logger.info("Initializing NeMo ClusteringDiarizer...")
            diarizer = ClusteringDiarizer(cfg=config)

            logger.info("Running NeMo diarization...")
            diarizer.diarize()

            # Parse RTTM output
            rttm_files = list(Path(output_dir).glob("pred_rttms/*.rttm"))
            if not rttm_files:
                raise RuntimeError("NeMo produced no RTTM output")

            segments = parse_rttm(str(rttm_files[0]))

            if not segments:
                raise RuntimeError("NeMo RTTM file is empty — no speech segments found")

            # Log quality metrics
            log_diarization_quality(segments, duration)

            # Gender detection
            speaker_genders = {}
            if detect_gender:
                logger.info("Running F0-based gender detection...")
                speaker_genders = _detect_speaker_genders(wav_path, segments)
                logger.info("Genders: %s", speaker_genders)

            speakers = set(s["speaker"] for s in segments)
            total_speech = sum(s["end"] - s["start"] for s in segments)

            return {
                "segments": segments,
                "num_speakers": len(speakers),
                "total_speech_duration": total_speech,
                "speaker_genders": speaker_genders,
            }

    except Exception as e:
        # Check for OOM
        if "OutOfMemoryError" in str(type(e).__name__) or "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during NeMo diarization: %s", e)
            raise RuntimeError(
                f"GPU memory exhausted ({gpu_memory_gb:.0f}GB). "
                f"Try shorter audio or use Pyannote fallback."
            ) from e

        logger.error("NeMo diarization failed: %s", e, exc_info=True)

        # Fallback: try without MSDD
        logger.info("Trying fallback without MSDD...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir2:
                audio_info = sf.info(wav_path)
                duration = audio_info.duration
                manifest_path = os.path.join(tmpdir2, "manifest.json")
                output_dir = os.path.join(tmpdir2, "output")
                os.makedirs(output_dir, exist_ok=True)

                manifest_entry = {
                    "audio_filepath": wav_path, "offset": 0, "duration": duration,
                    "label": "infer", "text": "-",
                    "num_speakers": num_speakers if num_speakers and num_speakers > 0 else None,
                    "rttm_filepath": None, "uem_filepath": None,
                }
                with open(manifest_path, "w") as f:
                    json.dump(manifest_entry, f)
                    f.write("\n")

                from nemo.collections.asr.models import ClusteringDiarizer
                from omegaconf import OmegaConf

                ms_params = _get_multiscale_params(duration, gpu_memory_gb)
                config = OmegaConf.create({
                    "device": device, "verbose": True, "num_workers": 0,
                    "sample_rate": 16000, "batch_size": 64,
                    "diarizer": {
                        "manifest_filepath": manifest_path,
                        "out_dir": output_dir,
                        "oracle_vad": False, "collar": 0.25, "ignore_overlap": True,
                        "speaker_embeddings": {
                            "model_path": "titanet_large",
                            "parameters": {**ms_params, "save_embeddings": False},
                        },
                        "clustering": {
                            "parameters": {
                                "oracle_num_speakers": num_speakers is not None and num_speakers > 0,
                                "max_num_speakers": max_speakers,
                                "enhanced_count_thres": 80, "max_rp_threshold": 0.25,
                                "sparse_search_volume": 30, "maj_vote_spk_count": False,
                            }
                        },
                        "msdd_model": {"model_path": None, "parameters": {}},
                        "vad": {
                            "model_path": "vad_multilingual_marblenet",
                            "parameters": {
                                "window_length_in_sec": 0.15, "shift_length_in_sec": 0.01,
                                "smoothing": "median", "overlap": 0.5, "onset": 0.4,
                                "offset": 0.3, "pad_onset": 0.05, "pad_offset": -0.1,
                                "min_duration_on": 0.2, "min_duration_off": 0.2,
                                "filter_speech_first": True,
                            }
                        },
                    }
                })
                fb_diarizer = ClusteringDiarizer(cfg=config)
                fb_diarizer.diarize()
                rttm_files = list(Path(output_dir).glob("pred_rttms/*.rttm"))
                if rttm_files:
                    segments = parse_rttm(str(rttm_files[0]))
                    if segments:
                        speakers = set(s["speaker"] for s in segments)
                        return {
                            "segments": segments,
                            "num_speakers": len(speakers),
                            "total_speech_duration": sum(s["end"] - s["start"] for s in segments),
                            "speaker_genders": {},
                        }
        except Exception as fb_e:
            logger.error("NeMo fallback (no MSDD) also failed: %s", fb_e)

        raise

    finally:
        # Cleanup GPU
        try:
            import gc as _gc
            import torch as _torch
            if diarizer is not None:
                del diarizer
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass


def log_diarization_quality(segments: List[Dict], total_duration: float) -> None:
    """Log detailed quality metrics for diarization results."""
    if not segments:
        logger.warning("QUALITY: No segments to analyze")
        return

    speaker_stats = {}
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        dur = seg.get("end", 0) - seg.get("start", 0)
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"segments": 0, "total_duration": 0.0,
                                       "min_duration": float("inf"), "max_duration": 0.0}
        speaker_stats[speaker]["segments"] += 1
        speaker_stats[speaker]["total_duration"] += dur
        speaker_stats[speaker]["min_duration"] = min(speaker_stats[speaker]["min_duration"], dur)
        speaker_stats[speaker]["max_duration"] = max(speaker_stats[speaker]["max_duration"], dur)

    num_speakers = len(speaker_stats)
    total_speech = sum(s["total_duration"] for s in speaker_stats.values())
    speech_ratio = (total_speech / total_duration * 100) if total_duration > 0 else 0

    logger.info("=" * 60)
    logger.info("NEMO DIARIZATION QUALITY REPORT")
    logger.info("=" * 60)
    logger.info("Audio: %.2fs | Speech: %.2fs (%.1f%%) | Speakers: %d | Segments: %d",
                total_duration, total_speech, speech_ratio, num_speakers, len(segments))

    for speaker in sorted(speaker_stats.keys()):
        s = speaker_stats[speaker]
        avg = s["total_duration"] / s["segments"] if s["segments"] else 0
        pct = (s["total_duration"] / total_speech * 100) if total_speech > 0 else 0
        logger.info("  %s: %d segs, %.1fs (%.1f%%), avg=%.2fs",
                     speaker, s["segments"], s["total_duration"], pct, avg)

    # Warnings
    short_segs = sum(1 for s in segments if (s["end"] - s["start"]) < 0.3)
    if short_segs:
        logger.warning("⚠ %d very short segments (<0.3s)", short_segs)
    for spk, st in speaker_stats.items():
        if total_speech > 0 and st["total_duration"] / total_speech > 0.9 and num_speakers > 1:
            logger.warning("⚠ %s dominates (%.1f%%)", spk, st["total_duration"] / total_speech * 100)
    if num_speakers > 5:
        logger.warning("⚠ %d speakers — unusually high", num_speakers)
    logger.info("=" * 60)


def parse_rttm(rttm_path: str) -> List[Dict]:
    """Parse RTTM file to list of segments."""
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                dur = float(parts[4])
                segments.append({
                    "start": start,
                    "end": start + dur,
                    "speaker": parts[7],
                })
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
                        help="Maximum speakers for auto-detection")
    parser.add_argument("--detect_gender", action="store_true",
                        help="Run F0-based gender detection")
    args = parser.parse_args()

    input_path = Path(args.input)
    num_speakers = args.num_speakers if args.num_speakers > 0 else None

    # Convert to WAV if needed
    if input_path.suffix.lower() == ".wav":
        import soundfile as sf
        info = sf.info(str(input_path))
        if info.samplerate == 16000 and info.channels == 1:
            wav_path = str(input_path)
        else:
            tmpdir = tempfile.mkdtemp()
            wav_path = os.path.join(tmpdir, "audio.wav")
            extract_wav(str(input_path), wav_path)
    else:
        tmpdir = tempfile.mkdtemp()
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_wav(str(input_path), wav_path)

    result = run_nemo_diarization(
        wav_path,
        device=args.device,
        num_speakers=num_speakers,
        max_speakers=args.max_speakers,
        detect_gender=args.detect_gender,
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Saved NeMo diarization to %s (%d segments)", output_path, len(result["segments"]))


if __name__ == "__main__":
    main()
