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

# Ensure we're using the GPU
if torch.cuda.is_available():
    logger.info("GPU Worker: CUDA available, device=%s", torch.cuda.get_device_name(0))
else:
    logger.warning("GPU Worker: CUDA not available, running on CPU")


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
    logger.info("GPU Task: transcribe_audio started, file=%s, model=%s", 
                Path(audio_path).name, model)
    
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
    device: str = "cuda",
) -> List[Dict]:
    """
    Run NeMo MSDD speaker diarization.
    
    Returns:
        List of segments: [{"start": 0.0, "end": 1.5, "speaker": "speaker_0"}, ...]
    """
    logger.info("GPU Task: diarize_nemo started, file=%s, speakers=%s",
                Path(audio_path).name, num_speakers if num_speakers > 0 else "auto")
    
    try:
        import gc
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import ClusteringDiarizer
        
        # Adjust for CPU if needed  
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")
        
        # Create temp directory for NeMo outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest file
            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump({
                    "audio_filepath": audio_path,
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": num_speakers if num_speakers > 0 else None,
                    "rttm_filepath": None,
                    "uem_filepath": None,
                }, f)
                f.write("\n")
            
            # NeMo config
            config = OmegaConf.create({
                "device": device,
                "verbose": True,
                "num_workers": 1,
                "sample_rate": 16000,
                "batch_size": 64,
                "diarizer": {
                    "manifest_filepath": str(manifest_path),
                    "out_dir": tmpdir,
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": True,
                    "vad": {
                        "model_path": "vad_multilingual_marblenet",
                        "parameters": {
                            "onset": 0.8,
                            "offset": 0.6,
                            "min_duration_on": 0.1,
                            "min_duration_off": 0.1,
                            "pad_onset": 0.05,
                            "pad_offset": -0.1,
                        },
                    },
                    "speaker_embeddings": {
                        "model_path": "titanet_large",
                        "parameters": {
                            "window_length_in_sec": 1.5,
                            "shift_length_in_sec": 0.75,
                            "multiscale_weights": [1, 1, 1],
                            "save_embeddings": False,
                        },
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": num_speakers > 0,
                            "max_num_speakers": max_speakers,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 30,
                            "maj_vote_spk_count": False,
                        },
                    },
                },
            })
            
            # Run diarization
            logger.info("Initializing NeMo ClusteringDiarizer on %s", device)
            diarizer = ClusteringDiarizer(cfg=config)
            diarizer.diarize()
            
            # Parse RTTM output
            rttm_path = Path(tmpdir) / "pred_rttms" / f"{Path(audio_path).stem}.rttm"
            segments = []
            
            if rttm_path.exists():
                with open(rttm_path) as f:
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
            
            # Clean up
            del diarizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            speakers = set(s["speaker"] for s in segments)
            logger.info("GPU Task: diarize_nemo complete, %d segments, %d speakers",
                       len(segments), len(speakers))
            
            return segments
            
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
