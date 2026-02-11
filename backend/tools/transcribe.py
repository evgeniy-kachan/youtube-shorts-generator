#!/usr/bin/env python3
"""
Standalone transcription script for venv-asr (NumPy 2.x + whisperx).
Receives input JSON, transcribes audio, outputs results as JSON.

Now includes BUILT-IN WhisperX diarization for better speaker assignment.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run WhisperX transcription with diarization")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", default="large-v2", help="Whisper model name")
    parser.add_argument("--language", default="ru", help="Language code")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--compute_type", default="float16", help="Compute type")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--num_speakers", type=int, default=2, help="Expected number of speakers (0 for auto-detect)")
    parser.add_argument("--min_speakers", type=int, default=1, help="Min speakers for auto-detect mode")
    parser.add_argument("--max_speakers", type=int, default=4, help="Max speakers for auto-detect mode")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for diarization")
    args = parser.parse_args()

    # Get HF token from env if not provided
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")

    try:
        import whisperx
        import torch

        print(f"[transcribe.py] Loading model={args.model}, device={args.device}", file=sys.stderr)
        
        # Load model
        model = whisperx.load_model(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )

        # Load audio
        print(f"[transcribe.py] Loading audio: {args.audio}", file=sys.stderr)
        audio = whisperx.load_audio(args.audio)

        # Transcribe
        print("[transcribe.py] Transcribing...", file=sys.stderr)
        result = model.transcribe(audio, batch_size=16, language=args.language)

        # Align for word-level timestamps
        print("[transcribe.py] Aligning...", file=sys.stderr)
        model_a, metadata = whisperx.load_align_model(
            language_code=args.language,
            device=args.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=args.device,
            return_char_alignments=False,
        )

        # Run WhisperX BUILT-IN diarization (much better than separate Pyannote!)
        diarize_segments = None
        if args.diarize and hf_token:
            # Determine speaker count mode
            if args.num_speakers > 0:
                # Fixed number of speakers (more accurate if known)
                # Use min_speakers=num_speakers to force detection of at least that many speakers
                # This prevents diarization from merging speakers when voices are similar
                min_spk = args.num_speakers
                max_spk = args.num_speakers
                mode_str = f"fixed={args.num_speakers} (min={min_spk}, max={max_spk})"
            else:
                # Auto-detect mode (less accurate but flexible)
                min_spk = args.min_speakers
                max_spk = args.max_speakers
                mode_str = f"auto [{min_spk}-{max_spk}]"
            
            print(f"[transcribe.py] Running WhisperX diarization (mode={mode_str})...", file=sys.stderr)
            try:
                from whisperx.diarize import DiarizationPipeline, assign_word_speakers
                
                diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=args.device,
                )
                
                # Apply fine-tuned hyperparameters for better speaker separation
                # These match the settings in diarize.py for consistency
                try:
                    diarize_model.model.instantiate({
                        "segmentation": {
                            "min_duration_off": 0.05,  # Detect shorter pauses (50ms vs default 0.1s)
                        },
                        "clustering": {
                            "method": "centroid",
                            "min_cluster_size": 3,     # Allow smaller speaker clusters (vs default 6)
                            "threshold": 0.35,         # More aggressive speaker separation (vs default 0.5)
                        },
                    })
                    print("[transcribe.py] Applied fine-tuned diarization parameters (threshold=0.35, min_cluster=3)", file=sys.stderr)
                except Exception as tune_err:
                    print(f"[transcribe.py] Warning: Could not apply fine-tuned parameters: {tune_err}", file=sys.stderr)
                
                print(f"[transcribe.py] Calling DiarizationPipeline with min_speakers={min_spk}, max_speakers={max_spk}", file=sys.stderr)
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_spk,
                    max_speakers=max_spk,
                )
                
                # Log detected speakers
                # WhisperX returns a DataFrame with columns: start, end, speaker
                detected_speakers = set()
                speaker_segments = {}  # Track segments per speaker
                if hasattr(diarize_segments, 'itertracks'):
                    # pyannote Annotation object
                    for seg in diarize_segments.itertracks(yield_label=True):
                        speaker = seg[2]
                        detected_speakers.add(speaker)
                        if speaker not in speaker_segments:
                            speaker_segments[speaker] = []
                        speaker_segments[speaker].append((seg[0].start, seg[0].end))
                elif hasattr(diarize_segments, 'iterrows'):
                    # pandas DataFrame
                    for _, row in diarize_segments.iterrows():
                        if 'speaker' in row:
                            speaker = row['speaker']
                            detected_speakers.add(speaker)
                            if speaker not in speaker_segments:
                                speaker_segments[speaker] = []
                            speaker_segments[speaker].append((row.get('start', 0), row.get('end', 0)))
                else:
                    print(f"[transcribe.py] Unknown diarization result type: {type(diarize_segments)}", file=sys.stderr)
                
                print(f"[transcribe.py] Diarization detected {len(detected_speakers)} speakers: {sorted(detected_speakers)}", file=sys.stderr)
                
                # Log segment counts per speaker
                for spk in sorted(detected_speakers):
                    segs = speaker_segments.get(spk, [])
                    total_duration = sum(end - start for start, end in segs)
                    print(f"[transcribe.py]   {spk}: {len(segs)} segments, {total_duration:.1f}s total", file=sys.stderr)
                
                # Warn if fewer speakers detected than expected
                if args.num_speakers > 0 and len(detected_speakers) < args.num_speakers:
                    print(
                        f"[transcribe.py] WARNING: Expected {args.num_speakers} speakers but only detected {len(detected_speakers)}! "
                        f"This may indicate that voices are too similar or the segment is too short.",
                        file=sys.stderr
                    )
                
                # Assign speakers to words
                result = assign_word_speakers(diarize_segments, result)
                print(f"[transcribe.py] Diarization complete, speakers assigned to words", file=sys.stderr)
            except Exception as diar_err:
                print(f"[transcribe.py] Diarization failed: {diar_err}, continuing without speakers", file=sys.stderr)

        # Extract segments WITH word-level timestamps AND speakers
        segments = []
        for seg in result.get("segments", []):
            # Extract word timestamps from WhisperX alignment
            words = []
            for w in seg.get("words", []):
                word_data = {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                }
                # Include speaker if available (from diarization)
                if "speaker" in w:
                    word_data["speaker"] = w["speaker"]
                words.append(word_data)
            
            seg_data = {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
                "words": words,
            }
            # Include segment-level speaker if available
            if "speaker" in seg:
                seg_data["speaker"] = seg["speaker"]
            
            segments.append(seg_data)

        output_data = {
            "segments": segments,
            "language": args.language,
            "diarization_enabled": args.diarize and hf_token is not None,
        }

        # Write output
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[transcribe.py] Success! Wrote {len(segments)} segments to {args.output}", file=sys.stderr)
        return 0

    except Exception as e:
        print(f"[transcribe.py] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

