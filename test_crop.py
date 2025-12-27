#!/usr/bin/env python3
"""
Quick test script for video cropping without TTS/DeepSeek.
Usage: python test_crop.py /path/to/video.mp4 [start_sec] [end_sec]

Example:
  python test_crop.py input.mp4 52.94 73.51
"""
import sys
import os
import tempfile
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_crop.py /path/to/video.mp4 [start_sec] [end_sec]")
        print("Example: python test_crop.py input.mp4 52.94 73.51")
        sys.exit(1)
    
    video_path = sys.argv[1]
    start_time = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    end_time = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"TEST CROP: {video_path}")
    print(f"Time range: {start_time}s - {end_time or 'end'}s")
    print(f"{'='*60}\n")
    
    # Import after path setup
    import ffmpeg
    from backend.services.face_detector import FaceDetector
    
    # Step 1: Cut segment if needed
    if start_time > 0 or end_time:
        print("Step 1: Cutting segment...")
        cut_path = tempfile.mktemp(suffix="_cut.mp4")
        
        input_opts = {"ss": start_time}
        if end_time:
            input_opts["t"] = end_time - start_time
        
        (
            ffmpeg
            .input(video_path, **input_opts)
            .output(cut_path, vcodec='libx264', acodec='aac', crf=18, preset='fast')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"  Cut saved to: {cut_path}")
    else:
        cut_path = video_path
        print("Step 1: Using original video (no cut needed)")
    
    # Step 2: Build focus timeline
    print("\nStep 2: Building focus timeline...")
    detector = FaceDetector()
    
    # No dialogue for test (scene-based fallback)
    focus_timeline = detector.build_focus_timeline(
        video_path=cut_path,
        dialogue=[],
        segment_start=start_time,
        segment_end=end_time,
    )
    
    print(f"\nFocus timeline ({len(focus_timeline)} segments):")
    for i, seg in enumerate(focus_timeline):
        print(f"  Segment {i}: [{seg['start']:.2f}s - {seg['end']:.2f}s] focus={seg['focus']:.3f}")
    
    # Step 3: Apply crop
    print("\nStep 3: Applying crop...")
    output_path = video_path.replace(".mp4", "_cropped_test.mp4")
    
    # Get video info
    probe = ffmpeg.probe(cut_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    
    # Calculate crop for first segment (or use average)
    if focus_timeline:
        avg_focus = sum(s['focus'] for s in focus_timeline) / len(focus_timeline)
    else:
        avg_focus = 0.5
    
    # Target: 1080x1920 (9:16)
    target_w, target_h = 1080, 1920
    
    # Scale to fit height
    scale_factor = target_h / height
    scaled_w = int(width * scale_factor)
    scaled_h = target_h
    
    # Crop X offset
    crop_x = int((scaled_w - target_w) * avg_focus)
    crop_x = max(0, min(crop_x, scaled_w - target_w))
    
    print(f"  Original: {width}x{height}")
    print(f"  Scaled: {scaled_w}x{scaled_h}")
    print(f"  Focus: {avg_focus:.3f} → offset_x={crop_x}")
    print(f"  Output: {output_path}")
    
    # Apply
    (
        ffmpeg
        .input(cut_path)
        .filter('scale', scaled_w, scaled_h)
        .filter('crop', target_w, target_h, crop_x, 0)
        .output(output_path, vcodec='libx264', acodec='aac', crf=18, preset='fast')
        .overwrite_output()
        .run(quiet=True)
    )
    
    print(f"\n{'='*60}")
    print(f"DONE! Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Cleanup temp file
    if cut_path != video_path and os.path.exists(cut_path):
        os.remove(cut_path)

if __name__ == "__main__":
    main()

