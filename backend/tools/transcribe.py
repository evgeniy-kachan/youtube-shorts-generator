#!/usr/bin/env python3
"""
Standalone transcription script for venv-asr (NumPy 2.x + whisperx).
Receives input JSON, transcribes audio, outputs results as JSON.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run WhisperX transcription")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", default="large-v2", help="Whisper model name")
    parser.add_argument("--language", default="ru", help="Language code")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--compute_type", default="float16", help="Compute type")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

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

        # Align (optional, for word-level timestamps)
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

        # Extract segments WITH word-level timestamps
        segments = []
        for seg in result.get("segments", []):
            # Extract word timestamps from WhisperX alignment
            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                })
            
            segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
                "words": words,  # Now we keep word-level timestamps!
            })

        output_data = {
            "segments": segments,
            "language": args.language,
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

