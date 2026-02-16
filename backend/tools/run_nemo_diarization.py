#!/usr/bin/env python3
"""
Standalone NeMo diarization script.

This script runs as a SEPARATE PROCESS to avoid CUDA context issues with RQ worker.
It receives parameters via command line and outputs JSON result to stdout.

Usage:
    python run_nemo_diarization.py --audio /path/to/audio.wav --output /path/to/result.json [--num-speakers 2] [--max-speakers 8]
"""
import argparse
import json
import sys
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="NeMo Speaker Diarization")
    parser.add_argument("--audio", required=True, help="Path to audio file (WAV 16kHz mono)")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--num-speakers", type=int, default=0, help="Number of speakers (0=auto)")
    parser.add_argument("--max-speakers", type=int, default=8, help="Max speakers for auto-detection")
    args = parser.parse_args()
    
    # Initialize CUDA
    import torch
    torch.cuda.init()
    
    # cuBLAS warmup
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    c = torch.mm(a, b)
    del a, b, c
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Import NeMo
    from nemo.collections.asr.models import ClusteringDiarizer
    from omegaconf import OmegaConf
    import soundfile as sf
    
    wav_path = args.audio
    num_speakers = args.num_speakers if args.num_speakers > 0 else None
    max_speakers = args.max_speakers
    
    # Get audio duration
    audio_info = sf.info(wav_path)
    duration = audio_info.duration
    
    with tempfile.TemporaryDirectory() as tmpdir:
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
            "num_speakers": num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_entry, f)
            f.write("\n")
        
        # NeMo config
        config = OmegaConf.create({
            "device": "cuda",
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
                        "oracle_num_speakers": num_speakers is not None,
                        "max_num_speakers": max_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
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
        
        # Run diarization
        diarizer = ClusteringDiarizer(cfg=config)
        diarizer.diarize()
        
        # Parse RTTM output
        rttm_dir = Path(output_dir) / "pred_rttms"
        rttm_files = list(rttm_dir.glob("*.rttm")) if rttm_dir.exists() else []
        
        segments = []
        if rttm_files:
            with open(rttm_files[0], "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == "SPEAKER":
                        start = float(parts[3])
                        seg_duration = float(parts[4])
                        speaker = parts[7]
                        segments.append({
                            "start": start,
                            "end": start + seg_duration,
                            "speaker": speaker,
                        })
        
        segments.sort(key=lambda x: x["start"])
        
        # Calculate stats
        speaker_stats = {}
        for seg in segments:
            speaker = seg.get("speaker", "unknown")
            seg_dur = seg.get("end", 0) - seg.get("start", 0)
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"count": 0, "duration": 0.0}
            speaker_stats[speaker]["count"] += 1
            speaker_stats[speaker]["duration"] += seg_dur
        
        result = {
            "segments": segments,
            "num_speakers": len(speaker_stats),
            "speaker_stats": speaker_stats,
            "total_speech_duration": sum(s["duration"] for s in speaker_stats.values()),
            "audio_duration": duration,
        }
        
        # Write output
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Diarization complete: {len(segments)} segments, {len(speaker_stats)} speakers")


if __name__ == "__main__":
    main()
