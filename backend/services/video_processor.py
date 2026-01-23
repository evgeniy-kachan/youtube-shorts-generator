"""Video processing service for cutting, adding audio and subtitles."""
import os
import re
import ffmpeg
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Literal
import subprocess
import uuid

from PIL import ImageFont

from backend.services.face_detector import FaceDetector
from backend.services.diarization_runner import get_diarization_runner

logger = logging.getLogger(__name__)

# =============================================================================
# FONT METRICS CACHE - for dynamic subtitle line splitting
# =============================================================================
# Cache stores: font_filename -> average character width per 1 unit of fontsize
# Measured once per font, then used for any fontsize
_font_width_cache: dict[str, float] = {}

# Reference string for measuring (Russian alphabet + spaces for realistic average)
_REFERENCE_STRING = "Абвгдеёжзийклмн опрстуфхцчшщъ ыьэюя абвгдеёжз"

# Available width for subtitles (PlayResX - MarginL - MarginR)
SUBTITLE_AVAILABLE_WIDTH = 870  # 1080 - 105 - 105

# Fallback width ratio if font measurement fails
FALLBACK_CHAR_WIDTH_RATIO = 0.55


def _measure_font_width(font_path: str) -> float:
    """
    Measure average character width per fontsize unit for a font.
    Returns width ratio: actual_char_width / fontsize
    """
    TEST_SIZE = 100  # Measure at size 100 for accuracy
    try:
        font = ImageFont.truetype(font_path, TEST_SIZE)
        width = font.getlength(_REFERENCE_STRING)
        avg_char_width = width / len(_REFERENCE_STRING)
        ratio = avg_char_width / TEST_SIZE
        logger.debug("Font '%s': width ratio = %.3f", font_path, ratio)
        return ratio
    except Exception as e:
        logger.warning("Failed to measure font '%s': %s, using fallback", font_path, e)
        return FALLBACK_CHAR_WIDTH_RATIO


def get_max_chars_for_line(font_name: str, fontsize: int, fonts_dir: str = "fonts") -> int:
    """
    Calculate maximum characters that fit in subtitle line for given font and size.
    Uses cached font metrics for speed.
    
    Args:
        font_name: Font name without extension (e.g., "Montserrat-Medium" or "Montserrat Medium")
        fontsize: Font size in pixels
        fonts_dir: Directory containing .ttf files
    
    Returns:
        Maximum number of characters that fit in SUBTITLE_AVAILABLE_WIDTH
    """
    cache_key = font_name
    
    if cache_key not in _font_width_cache:
        # Normalize font name: "Montserrat Medium" -> "Montserrat-Medium"
        normalized_name = font_name.replace(" ", "-")
        
        # Find the font file - try multiple variations
        font_path = None
        candidates = [
            f"{normalized_name}.ttf",           # Montserrat-Medium.ttf
            f"{font_name}.ttf",                 # Montserrat Medium.ttf (original)
            f"{normalized_name}-Regular.ttf",   # Montserrat-Medium-Regular.ttf
            f"{normalized_name}_Regular.ttf",   # Montserrat-Medium_Regular.ttf
        ]
        
        for candidate in candidates:
            test_path = os.path.join(fonts_dir, candidate)
            if os.path.exists(test_path):
                font_path = test_path
                break
        
        if font_path is None:
            font_path = os.path.join(fonts_dir, f"{normalized_name}.ttf")  # Default for error message
        
        _font_width_cache[cache_key] = _measure_font_width(font_path)
    
    width_ratio = _font_width_cache[cache_key]
    avg_char_width = width_ratio * fontsize
    max_chars = int(SUBTITLE_AVAILABLE_WIDTH / avg_char_width)
    
    # Safety bounds: at least 15 chars, at most 60 chars
    return max(15, min(60, max_chars))

# Regex to remove emotion tags like [curiously], [excitedly], etc.
EMOTION_TAG_PATTERN = re.compile(r'^\[[\w]+\]$')

# Temporarily enable INFO for debugging subtitle timing
logger.setLevel(logging.INFO)


class VideoProcessor:
    """Process video segments: cut, add TTS audio, add stylized subtitles."""
    
    # Target dimensions for Reels/Shorts (9:16 aspect ratio)
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    SUBTITLE_POSITIONS = {
        "mid_low": {"x": 540, "y": 1050, "an": 8, "marginv": 480},
        "lower_center": {"x": 540, "y": 1350, "an": 8, "marginv": 320},  # Updated: y=1350, marginv=320
        "lower_left": {"x": 360, "y": 1350, "an": 7, "marginv": 320},
        "lower_right": {"x": 720, "y": 1350, "an": 9, "marginv": 320},
        "bottom_center": {"x": 540, "y": 1520, "an": 2, "marginv": 260},
    }

    @staticmethod
    def _resolve_crop_offset(margin: int, focus: str) -> int:
        """Return horizontal offset in pixels for crop window based on focus hint."""
        if margin <= 0:
            return 0
        normalized = (focus or "center").lower()
        center = margin / 2
        shift = margin / 3  # move roughly a third of the available leeway

        if normalized == "left":
            desired = center - shift
        elif normalized == "right":
            desired = center + shift
        else:
            desired = center

        desired = max(0.0, min(margin, desired))
        return int(round(desired))

    def _get_face_detector(self) -> FaceDetector:
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        return self._face_detector

    def _estimate_face_focus(
        self,
        video_path: str,
        max_samples: int = 6,
        dialogue: list[dict] | None = None,
        segment_start: float = 0.0,
        segment_end: float | None = None,
    ) -> tuple[float | None, tuple[float, float] | None]:
        """
        Estimate face focus and detect two-speaker setup.
        
        Returns:
            Tuple of (focus_ratio, two_speaker_positions)
            - focus_ratio: 0..1 horizontal focus position
            - two_speaker_positions: (left_pos, right_pos) if two speakers detected, else None
        """
        try:
            detector = self._get_face_detector()
        except Exception as exc:
            logger.warning("Unable to initialize face detector: %s", exc)
            return None, None

        try:
            focus, two_speaker_positions = detector.estimate_horizontal_focus(
                video_path,
                max_samples=max_samples,
                dialogue=dialogue,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            return focus, two_speaker_positions
        except Exception as exc:
            logger.warning("Face focus estimation failed for %s: %s", video_path, exc)
            return None, None

    def _build_focus_timeline(
        self,
        video_path: str,
        dialogue: list[dict] | None,
        segment_start: float,
        segment_end: float | None,
        sample_period: float = 0.5,
    ) -> list[dict]:
        """
        Build a per-time focus timeline (no smoothing). Returns list of dicts:
        [{"start": float, "end": float, "focus": float}, ...]
        """
        try:
            detector = self._get_face_detector()
        except Exception as exc:
            logger.warning("Unable to initialize face detector: %s", exc)
            return []
        try:
            return detector.build_focus_timeline(
                video_path=video_path,
                dialogue=dialogue,
                segment_start=segment_start,
                segment_end=segment_end,
                sample_period=sample_period,
            )
        except Exception as exc:
            logger.warning("Focus timeline build failed for %s: %s", video_path, exc)
            return []

    def _build_vertical_focus_timeline(
        self,
        video_path: str,
        segment_start: float,
        segment_end: float | None,
        sample_period: float = 0.5,
    ) -> list[dict]:
        """
        Build vertical (y-axis) focus timeline (0..1 top to bottom).
        """
        try:
            detector = self._get_face_detector()
        except Exception as exc:
            logger.warning("Unable to initialize face detector: %s", exc)
            return []
        try:
            return detector.build_vertical_focus_timeline(
                video_path=video_path,
                segment_start=segment_start,
                segment_end=segment_end,
                sample_period=sample_period,
            )
        except Exception as exc:
            logger.warning("Vertical focus timeline build failed for %s: %s", video_path, exc)
            return []
    
    def __init__(self, output_dir: Path, fonts_dir: Path | None = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        default_fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        self.fonts_dir = fonts_dir or default_fonts_dir
        if self.fonts_dir.exists():
            logger.info("VideoProcessor fonts directory set to %s", self.fonts_dir)
        else:
            logger.warning(
                "Fonts directory %s was not found. FFmpeg may fall back to system fonts.",
                self.fonts_dir,
            )
        self._face_detector: FaceDetector | None = None
        
    def cut_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str
    ) -> str:
        """
        Cut a segment from video.
        
        Args:
            video_path: Path to source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path for output video
            
        Returns:
            Path to cut video
        """
        try:
            logger.info(f"Cutting segment: {start_time:.2f}s - {end_time:.2f}s")
            
            duration = end_time - start_time
            
            # Use ffmpeg to cut video precisely
            # Re-encode for frame-accurate cuts (slower but cleaner transitions)
            (
                ffmpeg
                .input(video_path, ss=start_time, t=duration)
                .output(
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    crf=18,  # High quality
                    preset='fast',
                    avoid_negative_ts='make_zero'
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f"Segment saved to: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
    
    def convert_to_vertical(
        self,
        video_path: str,
        output_path: str,
        method: Literal["letterbox", "center_crop"] = "letterbox",
        crop_focus: str = "center",
        auto_center_ratio: float | None = None,
        focus_timeline: list[dict] | None = None,
        focus_timeline_y: list[dict] | None = None,
    ) -> str:
        """
        Convert horizontal video to vertical format (9:16) for Reels/Shorts.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            method: Conversion method:
                - "letterbox": Video вписан без обрезки с чёрными полями
                - "center_crop": Simple center crop
                
        Returns:
            Path to converted video
        """
        try:
            logger.info(f"Converting to vertical format using method: {method}")
            
            # Get video dimensions
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            input_width = int(video_info['width'])
            input_height = int(video_info['height'])
            
            # Check if already vertical
            if input_height > input_width:
                logger.info("Video is already vertical, skipping conversion")
                # Just resize to target dimensions
                (
                    ffmpeg
                    .input(video_path)
                    .filter('scale', self.TARGET_WIDTH, self.TARGET_HEIGHT)
                    .output(output_path, **{'c:v': 'libx264', 'preset': 'medium'})
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                return output_path
            
            # If we have a focus timeline (multiple segments), use multi-crop flow
            if method == "center_crop" and focus_timeline and len(focus_timeline) > 1:
                return self._convert_with_focus_timeline(
                    video_path=video_path,
                    output_path=output_path,
                    focus_timeline=focus_timeline,
                    focus_timeline_y=focus_timeline_y,
                )

            if method == "letterbox":
                # Fit video into frame without cropping, add black bars where needed
                (
                    ffmpeg
                    .input(video_path)
                    .filter('scale', self.TARGET_WIDTH, self.TARGET_HEIGHT, force_original_aspect_ratio='decrease')
                    .filter('pad', self.TARGET_WIDTH, self.TARGET_HEIGHT, '(ow-iw)/2', '(oh-ih)/2', color='black')
                    .output(output_path, **{'c:v': 'libx264', 'preset': 'medium', 'crf': 23})
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                
            elif method == "center_crop":
                target_ratio = self.TARGET_WIDTH / self.TARGET_HEIGHT
                input_ratio = input_width / input_height if input_height else target_ratio
                focus = (crop_focus or "center").lower()

                if input_ratio >= target_ratio:
                    # Видео шире 9:16 → вписываем по высоте и обрезаем по бокам
                    scaled_width = max(self.TARGET_WIDTH, int(round(self.TARGET_HEIGHT * input_ratio)))
                    if scaled_width % 2 != 0:
                        scaled_width += 1
                    margin = max(scaled_width - self.TARGET_WIDTH, 0)
                    
                    if auto_center_ratio is not None and 0.0 <= auto_center_ratio <= 1.0:
                        desired_center = scaled_width * auto_center_ratio
                        offset_x = int(round(desired_center - (self.TARGET_WIDTH / 2)))
                        offset_x = max(0, min(margin, offset_x))
                        
                        # Calculate where faces will appear in final 1080px wide crop
                        crop_start_x = offset_x
                        crop_end_x = offset_x + self.TARGET_WIDTH
                        face_center_in_scaled = scaled_width * auto_center_ratio
                        face_center_in_crop = face_center_in_scaled - crop_start_x
                        face_center_ratio_in_crop = face_center_in_crop / self.TARGET_WIDTH
                        
                        logger.info(
                            "Face auto-crop applied (ratio %.3f, offset %dpx of %dpx margin)",
                            auto_center_ratio,
                            offset_x,
                            margin,
                        )
                        logger.info(
                            "Crop window: [%dpx - %dpx] of scaled %dpx×%dpx. "
                            "Face center in final crop: %.1fpx (%.1f%% from left edge, 0%%=left 50%%=center 100%%=right)",
                            crop_start_x,
                            crop_end_x,
                            scaled_width,
                            self.TARGET_HEIGHT,
                            face_center_in_crop,
                            face_center_ratio_in_crop * 100,
                        )
                    else:
                        offset_x = self._resolve_crop_offset(margin, focus)

                    pipeline = (
                        ffmpeg
                        .input(video_path)
                        .filter('scale', scaled_width, self.TARGET_HEIGHT)
                        .filter('crop', self.TARGET_WIDTH, self.TARGET_HEIGHT, offset_x, 0)
                    )
                    
                    logger.info(
                        "Center crop focus=%s (offset %dpx of %dpx margin)",
                        focus,
                        offset_x,
                        margin,
                    )
                else:
                    # Видео уже/ближе к вертикальному → вписываем по ширине и обрезаем верх/низ
                    pipeline = (
                        ffmpeg
                        .input(video_path)
                        .filter('scale', self.TARGET_WIDTH, -1)
                        .filter('crop', self.TARGET_WIDTH, self.TARGET_HEIGHT)
                    )

                (
                    pipeline
                    .output(output_path, **{'c:v': 'libx264', 'preset': 'medium'})
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
            else:
                raise ValueError(f"Unsupported vertical conversion method: {method}")
            
            logger.info(f"Converted video to vertical format: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error converting to vertical: {e.stderr.decode(errors='ignore') if e.stderr else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error converting to vertical: {e}", exc_info=True)
            raise
    
    def _convert_with_focus_timeline(
        self,
        video_path: str,
        output_path: str,
        focus_timeline: list[dict],
        focus_timeline_y: list[dict] | None = None,
    ) -> str:
        """
        Multi-segment crop: for each time span apply its own horizontal offset,
        and optional vertical offset, then concat all segments back.
        No smoothing; hard cuts when focus jumps.
        """
        if not focus_timeline:
            raise ValueError("focus_timeline is empty")

        # Probe source
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        input_width = int(video_info['width'])
        input_height = int(video_info['height'])

        target_ratio = self.TARGET_WIDTH / self.TARGET_HEIGHT
        input_ratio = input_width / input_height if input_height else target_ratio
        # Allow vertical margin: scale a bit larger than target to slide window
        scale_factor = 1.08
        scaled_height = int(round(self.TARGET_HEIGHT * scale_factor))
        scaled_width = int(round(scaled_height * input_ratio))
        if scaled_width < self.TARGET_WIDTH:
            scaled_width = self.TARGET_WIDTH
            scaled_height = int(round(scaled_width / input_ratio))
        # Ensure even
        if scaled_width % 2 != 0:
            scaled_width += 1
        if scaled_height % 2 != 0:
            scaled_height += 1
        margin_x = max(scaled_width - self.TARGET_WIDTH, 0)
        margin_y = max(scaled_height - self.TARGET_HEIGHT, 0)

        # Build filter_complex with trim+scale+crop per segment
        # Skip ~3-4 frames (0.12s) at transitions to hide flicker
        TRANSITION_SKIP = 0.12
        
        parts = []
        labels = []
        for idx, seg in enumerate(focus_timeline):
            start = max(0.0, float(seg.get("start", 0.0)))
            end = float(seg.get("end", start))
            
            # Skip frames at transition (except first segment)
            if idx > 0:
                start = start + TRANSITION_SKIP
            
            focus = float(seg.get("focus", 0.5))
            # face_detector уже возвращает правильный focus, offset_x ограничен ниже
            
            offset_x = int(round(focus * scaled_width - (self.TARGET_WIDTH / 2)))
            offset_x = max(0, min(margin_x, offset_x))  # Гарантирует что кроп в пределах кадра

            # Vertical focus
            focus_y = 0.5
            if focus_timeline_y:
                seg_y = next((s for s in focus_timeline_y if s["start"] <= start < s["end"]), None)
                if seg_y:
                    focus_y = float(seg_y.get("focus_y", 0.5))
            offset_y = int(round(focus_y * scaled_height - (self.TARGET_HEIGHT / 2)))
            offset_y = max(0, min(margin_y, offset_y))

            logger.info(
                "Segment %d: t=[%.2f, %.2f] focus_x=%.3f offset_x=%d focus_y=%.3f offset_y=%d",
                idx,
                start,
                end,
                focus,
                offset_x,
                focus_y,
                offset_y,
            )

            label = f"v{idx}"
            labels.append(label)
            parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
                f"scale={scaled_width}:{scaled_height},"
                f"crop={self.TARGET_WIDTH}:{self.TARGET_HEIGHT}:{offset_x}:{offset_y}[{label}]"
            )

        concat = "".join([f"[{l}]" for l in labels]) + f"concat=n={len(labels)}:v=1:a=0[outv]"
        filter_complex = ";".join(parts + [concat])

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            output_path,
        ]

        logger.info("Applying multi-crop timeline with %d segments (%.0fms skip at transitions)", 
                    len(focus_timeline), TRANSITION_SKIP * 1000)
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
            stdout = exc.stdout.decode(errors="ignore") if exc.stdout else ""
            logger.error("FFmpeg multi-crop failed.\nSTDOUT:\n%s\nSTDERR:\n%s", stdout.strip(), stderr.strip())
            raise

        return output_path
    
    @staticmethod
    def _probe_duration(media_path: str) -> float | None:
        """Return media duration in seconds using ffmpeg probe."""
        try:
            probe = ffmpeg.probe(media_path)
            if probe and "format" in probe:
                duration = probe["format"].get("duration")
                if duration is not None:
                    return float(duration)
        except Exception as exc:
            logger.debug("Failed to probe duration for %s: %s", media_path, exc)
        return None

    def add_audio_and_subtitles(
        self,
        video_path: str,
        audio_path: str,
        subtitles: List[Dict],
        output_path: str,
        style: str = "capcut",
        animation: str = "fade",
        font_name: str = "Montserrat Light",
        font_size: int = 86,
        subtitle_position: str = "mid_low",
        subtitle_background: bool = False,
        subtitle_glow: bool = True,
        subtitle_gradient: bool = False,
        convert_to_vertical: bool = True,
        vertical_method: str = "letterbox",
        crop_focus: str = "center",
        auto_center_ratio: float | None = None,
        focus_timeline: list[dict] | None = None,
        focus_timeline_y: list[dict] | None = None,
        target_duration: float | None = None,
        background_audio_path: str | None = None,
        background_volume_db: float = -20.0,
    ) -> str:
        """
        Add TTS audio and stylized subtitles to video.
        
        Args:
            video_path: Path to video file
            audio_path: Path to TTS audio file
            subtitles: List of subtitle entries with word-level timing
            output_path: Path for output video
            style: Subtitle style ("capcut", "tiktok", "instagram", "youtube")
            animation: Animation preset for subtitles ("bounce", "slide", "spark")
            convert_to_vertical: Whether to convert to vertical format (9:16)
            vertical_method: Method for vertical conversion
            
        Returns:
            Path to processed video
        """
        try:
            # Check video/audio sync
            video_duration = self._probe_duration(video_path)
            audio_duration = self._probe_duration(audio_path)
            diff = abs((video_duration or 0) - (audio_duration or 0))
            if diff > 1.0:  # Only log if significant difference
                logger.warning(
                    "VIDEO/AUDIO SYNC: video=%.2fs, audio=%.2fs, diff=%.2fs",
                    video_duration or 0, audio_duration or 0, diff
                )
            
            # Step 1: Convert to vertical if needed
            working_video = video_path
            if convert_to_vertical:
                temp_vertical = Path(output_path).parent / f"{Path(output_path).stem}_vertical_temp.mp4"
                working_video = self.convert_to_vertical(
                    video_path, 
                    str(temp_vertical),
                    method=vertical_method,
                    crop_focus=crop_focus,
                    auto_center_ratio=auto_center_ratio,
                    focus_timeline=focus_timeline,
                    focus_timeline_y=focus_timeline_y,
                )
                
            # Diagnostic: check face positions in final cropped video
            # (always run for face_auto, even when using multi-segment timeline)
            if crop_focus == "face_auto":
                try:
                    detector = self._get_face_detector()
                    detector.diagnose_final_crop(working_video, max_samples=16)
                except Exception as diag_exc:
                    logger.warning("Post-crop diagnostic failed: %s", diag_exc)
            
            # Step 2: Create ASS subtitle file with styling
            subtitle_path = Path(output_path).with_suffix('.ass')
            self._create_stylized_subtitles(
                subtitles,
                subtitle_path,
                style,
                animation,
                font_name,
                font_size,
                subtitle_position,
                subtitle_background,
                subtitle_glow,
            )
            
            # Step 3: Process video with ffmpeg
            # Replace audio with TTS and burn in stylized subtitles
            video_input = ffmpeg.input(working_video)
            tts_audio_input = ffmpeg.input(audio_path)
            video_stream = video_input.video
            audio_stream = tts_audio_input.audio

            if background_audio_path:
                try:
                    bg_audio_input = ffmpeg.input(background_audio_path)
                    bg_audio_stream = bg_audio_input.audio.filter(
                        "volume",
                        f"{background_volume_db}dB",
                    )
                    audio_stream = ffmpeg.filter(
                        [audio_stream, bg_audio_stream],
                        "amix",
                        inputs=2,
                        duration="longest",
                        dropout_transition=0,
                    )
                except ffmpeg.Error as mix_err:
                    stderr = (mix_err.stderr or b"").decode(errors="ignore")
                    logger.warning(
                        "Failed to mix background audio %s: %s",
                        background_audio_path,
                        stderr.strip(),
                    )
                except Exception as generic_mix_err:
                    logger.warning(
                        "Background audio mix error for %s: %s",
                        background_audio_path,
                        generic_mix_err,
                    )

            video_duration = self._probe_duration(working_video)
            if target_duration and video_duration:
                extra = target_duration - video_duration
                if extra > 0.05:
                    video_stream = video_stream.filter(
                        "tpad",
                        stop_mode="clone",
                        stop_duration=max(0.0, extra),
                    )
                    logger.info(
                        "Extended video by %.2fs to match audio duration %.2fs",
                        extra,
                        target_duration,
                    )
            
            # Apply gradient overlay before subtitles (so subtitles appear on top)
            if subtitle_gradient:
                video_stream = self._apply_gradient_filter(video_stream)
            
            video_stream = self._apply_ass_filter(video_stream, subtitle_path)

            output = (
                ffmpeg
                .output(
                    video_stream,
                    audio_stream,
                    output_path,
                    shortest=None,  # Use shortest stream
                    **{'c:v': 'libx264', 'c:a': 'aac', 'b:a': '192k', 'preset': 'medium', 'crf': 23}
                )
                .overwrite_output()
            )
            
            output.run(quiet=True, capture_stdout=True, capture_stderr=True)
            
            logger.info(f"Processed video saved to: {output_path}")
            
            # Clean up temporary files
            subtitle_path.unlink(missing_ok=True)
            if convert_to_vertical and working_video != video_path:
                Path(working_video).unlink(missing_ok=True)
            
            return output_path
            
        except ffmpeg.Error as ff_err:
            stderr = (ff_err.stderr or b"").decode(errors="ignore")
            stdout = (ff_err.stdout or b"").decode(errors="ignore")
            logger.error("FFmpeg processing failed.\nSTDOUT:\n%s\nSTDERR:\n%s", stdout.strip(), stderr.strip())
            raise
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise

    def create_vertical_video(
        self,
        video_path: str,
        audio_path: str,
        text: str,
        start_time: float,
        end_time: float,
        target_duration: float | None = None,
        method: Literal["letterbox", "center_crop"] = "letterbox",
        subtitle_style: str = "capcut",
        subtitle_animation: str = "bounce",
        subtitle_position: str = "mid_low",
        subtitle_font: str = "Montserrat Light",
        subtitle_font_size: int = 86,
        subtitle_background: bool = False,
        subtitle_glow: bool = True,
        subtitle_gradient: bool = False,
        dialogue: list[dict] | None = None,
        preserve_background_audio: bool = False,
        crop_focus: str = "center",
        speaker_color_mode: str = "colored",
    ) -> str:
        """
        End-to-end helper that cuts the source video, converts it to vertical format,
        overlays subtitles and replaces audio with synthesized TTS.
        Returns path to the processed temporary file.
        """
        background_audio_path: Path | None = None
        try:
            original_duration = max(0.1, end_time - start_time)
            duration = max(original_duration, target_duration or 0.0)
            temp_segment = self.output_dir / f"segment_cut_{uuid.uuid4().hex}.mp4"
            temp_output = self.output_dir / f"segment_processed_{uuid.uuid4().hex}.mp4"
            effective_crop_focus = (crop_focus or "center").lower()
            auto_center_ratio: float | None = None
            focus_timeline: list[dict] | None = None
            dialogue_turns = dialogue or []

            # Step 1: cut source segment
            cut_path = self.cut_segment(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                output_path=str(temp_segment)
            )

            # Optional external diarization (separate venv) if no dialogue provided
            if not dialogue_turns and os.getenv("EXTERNAL_DIARIZATION_ENABLED", "0") == "1":
                diar_runner = get_diarization_runner()
                diar_segments = diar_runner.run(input_path=str(cut_path))
                dialogue_turns = [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": seg.get("speaker") or f"SPEAKER_{idx:02d}",
                        "text": seg.get("text") or "",
                        "text_ru": seg.get("text_ru") or "",
                    }
                    for idx, seg in enumerate(diar_segments)
                ]
                if dialogue_turns:
                    logger.info("External diarization attached %d turns", len(dialogue_turns))
                else:
                    logger.info("External diarization returned no segments; proceeding without dialogue.")

            # Optional: extract quiet background audio
            if preserve_background_audio:
                background_audio_path = self.output_dir / f"{Path(temp_segment).stem}_bg.wav"
                try:
                    (
                        ffmpeg
                        .input(str(cut_path))
                        .output(
                            str(background_audio_path),
                            acodec="pcm_s16le",
                            ac=1,
                            ar="44100",
                            t=duration,
                        )
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as bg_err:
                    stderr = (bg_err.stderr or b"").decode(errors="ignore")
                    logger.warning(
                        "Failed to extract background audio for %s: %s",
                        cut_path,
                        stderr.strip(),
                    )
                    background_audio_path = None
                except Exception as unexpected_bg_err:
                    logger.warning(
                        "Unexpected error extracting background audio: %s",
                        unexpected_bg_err,
                    )
                    background_audio_path = None

            focus_timeline_y = None

            if method == "center_crop" and effective_crop_focus == "face_auto":
                # === DIAGNOSTIC: what dialogue_turns are we passing? ===
                def _fmt_turn(t):
                    spk = t.get("speaker", "?")
                    start = t.get("start") or t.get("tts_start_offset") or 0
                    end = t.get("end") or t.get("tts_end_offset") or 0
                    return (spk, f"{start:.1f}-{end:.1f}")
                logger.info(
                    "DIAGNOSTIC: dialogue_turns passed to focus_timeline: len=%d, turns=%s",
                    len(dialogue_turns) if dialogue_turns else 0,
                    [_fmt_turn(t) for t in (dialogue_turns or [])[:5]],
                )
                # === END DIAGNOSTIC ===
                
                # Build timeline (dynamic crops). If timeline has 0 or 1 segment, fall back to single focus.
                focus_timeline = self._build_focus_timeline(
                    str(cut_path),
                    dialogue=dialogue_turns,
                    segment_start=start_time,
                    segment_end=end_time,
                    sample_period=0.10,  # частая детекция (0.10s) + scene detection для резких смен планов
                )
                focus_timeline_y = self._build_vertical_focus_timeline(
                    str(cut_path),
                    segment_start=start_time,
                    segment_end=end_time,
                    sample_period=0.10,
                )
                if focus_timeline:
                    if len(focus_timeline) == 1:
                        # Single stable focus - reuse classic flow
                        auto_center_ratio = focus_timeline[0]["focus"]
                        focus_timeline = None
                    else:
                        # Multiple segments: ignore auto_center_ratio, use timeline
                        auto_center_ratio = None
                        logger.info("Using multi-segment face timeline (%d segments)", len(focus_timeline))
                else:
                    # Fallback: single focus estimation
                    auto_center_ratio, two_speaker_positions = self._estimate_face_focus(
                        str(cut_path),
                        dialogue=dialogue_turns,
                        segment_start=start_time,
                        segment_end=end_time,
                    )
                    
                    # Log two-speaker detection but DON'T override auto_center_ratio
                    # The focus calculation in estimate_horizontal_focus already handles:
                    # - Primary speaker detection
                    # - Dominant face prioritization (3x+ size ratio)
                    # - Smart blending for wide speaker spans
                    if two_speaker_positions is not None:
                        left_pos, right_pos = two_speaker_positions
                        logger.info(
                            "Two-speaker setup detected: left=%.3f, right=%.3f, using smart focus=%.3f",
                            left_pos,
                            right_pos,
                            auto_center_ratio if auto_center_ratio else 0.5,
                        )
                    
                    if auto_center_ratio is None:
                        effective_crop_focus = "center"
                        logger.info(
                            "Face auto-crop fallback to center for %s (no face detected).",
                            cut_path,
                        )
            else:
                auto_center_ratio = None

            # Step 2: prepare basic subtitles
            # Use dynamic character limit based on font metrics for proper margins
            speaker_palette = self._assign_speaker_colors(dialogue_turns, speaker_color_mode)
            subtitles = self._generate_basic_subtitles(
                text=text,
                duration=duration,
                dialogue=dialogue_turns,
                segment_start=start_time,
                speaker_palette=speaker_palette,
                font_name=subtitle_font,
                font_size=subtitle_font_size,
            )

            # Step 3: add audio + subtitles + vertical conversion
            processed_path = self.add_audio_and_subtitles(
                video_path=cut_path,
                audio_path=audio_path,
                subtitles=subtitles,
                output_path=str(temp_output),
                style=subtitle_style,
                animation=subtitle_animation,
                font_name=subtitle_font,
                font_size=subtitle_font_size,
                subtitle_position=subtitle_position,
                subtitle_background=subtitle_background,
                subtitle_glow=subtitle_glow,
                subtitle_gradient=subtitle_gradient,
                convert_to_vertical=True,
                vertical_method=method,
                crop_focus=effective_crop_focus,
                auto_center_ratio=auto_center_ratio,
                focus_timeline=focus_timeline,
                focus_timeline_y=focus_timeline_y,
                target_duration=duration,
                background_audio_path=str(background_audio_path) if background_audio_path else None,
                background_volume_db=-20.0,
            )

            return processed_path

        finally:
            # Clean up intermediate cut segment
            if 'temp_segment' in locals():
                Path(temp_segment).unlink(missing_ok=True)
            if background_audio_path:
                background_audio_path.unlink(missing_ok=True)

    def save_video(self, processed_path: str, output_path: str) -> str:
        """
        Move processed temporary video into its final location.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Path(processed_path).replace(output_path)
        logger.info(f"Final video saved to: {output_path}")
        return str(output_path)

    def _apply_ass_filter(self, video_stream, subtitle_path: Path):
        """
        Apply ASS subtitles via ffmpeg filter, optionally wiring in custom fonts dir.
        """
        subtitle_arg = str(subtitle_path)
        if self.fonts_dir and self.fonts_dir.exists():
            fonts_dir = str(self.fonts_dir)
            # ass filter signature: ass=filename[:original_size[:fontsdir]]
            original_size = f"{self.TARGET_WIDTH}x{self.TARGET_HEIGHT}"
            return video_stream.filter("ass", subtitle_arg, original_size, fonts_dir)
        return video_stream.filter("ass", subtitle_arg)

    def _apply_gradient_filter(self, video_stream):
        """
        Apply a smooth dark gradient overlay at the bottom of the video for subtitle readability.
        
        Uses a pre-generated PNG with perfect gradient (no banding).
        Gradient covers bottom 40% of the frame, fading from transparent to 65% black.
        """
        # Path to pre-generated gradient PNG
        gradient_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets", "overlays", "gradient_bottom.png"
        )
        
        if not os.path.exists(gradient_path):
            logger.warning("Gradient overlay not found at %s, skipping gradient", gradient_path)
            return video_stream
        
        # Load gradient PNG and overlay on video
        gradient_input = ffmpeg.input(gradient_path, loop=1, framerate=30)
        
        # Overlay gradient on video (gradient has alpha channel for transparency)
        result = ffmpeg.overlay(video_stream, gradient_input, x=0, y=0, shortest=1)
        
        logger.info("Applied PNG gradient overlay from %s", gradient_path)
        return result

    @staticmethod
    def _assign_speaker_colors(
        dialogue: list[dict] | None,
        color_mode: str = "colored",
    ) -> dict[str, str]:
        """
        Return a deterministic color palette for every speaker in the segment.
        Colors are stored in ASS format (AABBGGRR without the leading &H).
        
        Args:
            dialogue: List of dialogue turns
            color_mode: "colored" for different colors per speaker, "white" for all white
        """
        if color_mode == "white":
            # All speakers get white (None = use default white)
            palette = [None, None]
        else:
            # Colored mode - different colors for speakers
            palette = [
                None,         # Primary speaker → белый
                "004DC0FF",   # Второй спикер → насыщенный жёлто-оранжевый (#FFC04D)
            ]
        
        assignment: dict[str, str] = {}
        if not dialogue:
            return assignment

        for turn in dialogue:
            speaker = turn.get("speaker")
            if not speaker or speaker in assignment:
                continue
            color_idx = len(assignment) % len(palette)
            assignment[speaker] = palette[color_idx]
        return assignment

    def _generate_basic_subtitles(
        self,
        text: str,
        duration: float,
        dialogue: list[dict] | None = None,
        segment_start: float = 0.0,
        speaker_palette: dict[str, str] | None = None,
        font_name: str = "Montserrat-Medium",
        font_size: int = 60,
    ) -> List[Dict]:
        """
        Create a multi-line subtitle track with approximate word-level timings.
        Uses dynamic character-based line splitting based on font metrics.
        """
        # Calculate max characters per line based on font metrics
        max_chars = get_max_chars_for_line(font_name, font_size, self.fonts_dir)
        logger.info(
            "Subtitle line limit: max %d chars (font='%s', size=%d)",
            max_chars, font_name, font_size
        )
        
        if dialogue:
            return self._generate_dialogue_subtitles(
                dialogue=dialogue,
                duration=duration,
                max_chars_per_line=max_chars,
                segment_start=segment_start,
                speaker_palette=speaker_palette or {},
            )

        cleaned_text = " ".join(text.strip().split())
        if not cleaned_text:
            cleaned_text = "..."

        words = cleaned_text.split(" ")
        total_words = len(words)
        if total_words == 0:
            words = ["..."]
            total_words = 1

        # Group words into chunks using character-based limit
        word_chunks: List[List[str]] = []
        current_chunk: List[str] = []
        current_chars = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_chars + word_len > max_chars and current_chunk:
                word_chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0
            current_chunk.append(word)
            current_chars += word_len

        if current_chunk:
            word_chunks.append(current_chunk)

        subtitles: List[Dict] = []
        elapsed = 0.0

        for chunk in word_chunks:
            chunk_word_count = len(chunk)
            # Allocate duration proportionally to number of words in chunk
            chunk_duration = duration * (chunk_word_count / total_words)
            # Ensure last chunk ends exactly at total duration
            if chunk is word_chunks[-1]:
                chunk_duration = max(0.05, duration - elapsed)

            start_time = elapsed
            end_time = min(duration, start_time + chunk_duration)
            elapsed = end_time

            # Word-level timings within the chunk
            word_entries = []
            if chunk_word_count == 0:
                chunk_word_count = 1

            per_word = chunk_duration / chunk_word_count if chunk_duration > 0 else 0
            for index, word in enumerate(chunk):
                word_start = start_time + index * per_word
                word_end = min(end_time, word_start + per_word)
                word_entries.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end
                })

            subtitles.append({
                'start': start_time,
                'end': end_time,
                'text': " ".join(chunk),
                'words': word_entries
            })

        # Guard: if due to rounding we didn't cover the entire duration,
        # extend the last subtitle slightly
        if subtitles and subtitles[-1]['end'] < duration:
            subtitles[-1]['end'] = duration
            if subtitles[-1]['words']:
                subtitles[-1]['words'][-1]['end'] = duration

        return subtitles

    def _generate_dialogue_subtitles(
        self,
        dialogue: list[dict],
        duration: float,
        max_chars_per_line: int,
        segment_start: float,
        speaker_palette: dict[str, str],
    ) -> List[Dict]:
        """
        Build subtitles directly from dialogue turns, preserving per-speaker timing
        and enabling color coding.
        Uses character-based line splitting for proper margin handling.
        """
        # Summary: check timing quality
        turns_with_timing = sum(1 for t in dialogue if t.get("tts_duration", 0) >= 0.1)
        turns_with_words = sum(1 for t in dialogue if len(t.get("tts_words", [])) > 0)
        total_turns = len(dialogue)
        
        # Check for overlapping turns (potential issue)
        overlaps = []
        for i in range(1, len(dialogue)):
            prev_end = dialogue[i-1].get("tts_end_offset", 0)
            curr_start = dialogue[i].get("tts_start_offset", 0)
            if curr_start < prev_end - 0.05:  # 50ms tolerance
                overlaps.append((i-1, i, prev_end - curr_start))
        
        # Only log if there are issues
        if len(overlaps) > 0:
            logger.warning(
                "SUBTITLE SYNC: %d/%d turns have timing, %d overlaps detected",
                turns_with_timing, total_turns, len(overlaps)
            )
            for prev_idx, curr_idx, overlap_sec in overlaps[:3]:
                logger.warning(
                    "  Overlap: turn %d--%d (%.2fs)",
                    prev_idx, curr_idx, overlap_sec
                )
        
        # Per-turn timing log disabled (enable for debugging)
        # for idx, turn in enumerate(dialogue):
        #     tts_words = turn.get("tts_words", [])
        #     logger.info(
        #         "Subtitle input turn %d: tts_start=%.2f, tts_end=%.2f, tts_dur=%.2f, tts_words=%d",
        #         idx, turn.get("tts_start_offset", -1), turn.get("tts_end_offset", -1),
        #         turn.get("tts_duration", -1), len(tts_words),
        #     )
        
        # Log subtitle coverage summary
        if dialogue:
            first_turn = dialogue[0]
            last_turn = dialogue[-1]
            subtitle_start = first_turn.get("tts_start_offset", 0)
            subtitle_end = last_turn.get("tts_end_offset", 0)
            subtitle_coverage = subtitle_end - subtitle_start
            
            # Warn if subtitle coverage seems too short
            if subtitle_coverage < 10 and len(dialogue) > 5:
                logger.warning(
                    "SUBTITLE SYNC: Only %.2fs coverage for %d turns!",
                    subtitle_coverage, len(dialogue)
                )
        
        subtitles: List[Dict] = []
        # Track when the last subtitle ends to prevent overlaps
        last_subtitle_end = 0.0
        
        # Minimum gap between subtitles (prevents visual overlap)
        MIN_SUBTITLE_GAP = 0.05  # 50ms

        for turn in dialogue:
            # Skip turns that were merged into previous turn (too short for separate subtitle)
            if turn.get("_subtitle_merged"):
                logger.debug(
                    "Skipping merged turn (merged into turn %d)",
                    turn.get("_merged_into", -1)
                )
                continue
            
            raw_text = (turn.get("text_ru") or turn.get("text") or "").strip()
            if not raw_text:
                continue

            speaker_id = turn.get("speaker")
            words = raw_text.split()
            if not words:
                continue

            relative_start = turn.get("tts_start_offset")
            if relative_start is None:
                start_abs = float(turn.get("start", segment_start))
                relative_start = max(0.0, start_abs - segment_start)

            duration_override = turn.get("tts_duration")
            if duration_override is None:
                start_abs = float(turn.get("start", segment_start))
                end_abs = float(turn.get("end", start_abs))
                duration_override = max(0.1, end_abs - start_abs)

            relative_end = turn.get("tts_end_offset")
            if relative_end is None:
                relative_end = relative_start + duration_override

            relative_end = min(duration, max(relative_end, relative_start + 0.1))

            # Check if we have precise word timestamps from ElevenLabs
            tts_words = turn.get("tts_words")
            
            if tts_words and len(tts_words) > 0:
                # Use precise word timestamps from ElevenLabs alignment
                logger.debug(
                    "Using %d ElevenLabs word timestamps for turn (speaker=%s)",
                    len(tts_words), speaker_id
                )
                
                # Build word entries with precise timestamps
                # Chunk words while preserving their timestamps
                word_idx = 0
                chunks_added = 0
                
                # Filter out emotion tags like [curiously], [excitedly], etc.
                filtered_tts_words = [
                    tw for tw in tts_words 
                    if not EMOTION_TAG_PATTERN.match(tw.get("word", ""))
                ]
                
                while word_idx < len(filtered_tts_words):
                    # Take words until we hit max_chars_per_line
                    chunk_tts_words = []
                    current_chars = 0
                    while word_idx < len(filtered_tts_words):
                        word = filtered_tts_words[word_idx]["word"]
                        word_len = len(word) + 1  # +1 for space
                        if current_chars + word_len > max_chars_per_line and chunk_tts_words:
                            # This word would exceed limit, stop here
                            break
                        chunk_tts_words.append(filtered_tts_words[word_idx])
                        current_chars += word_len
                        word_idx += 1
                    
                    if not chunk_tts_words:
                        break
                    
                    # Get timing from first and last word in chunk
                    chunk_start = chunk_tts_words[0]["start"]
                    chunk_end_time = chunk_tts_words[-1]["end"]
                    
                    # Build word entries
                    word_entries = []
                    for tw in chunk_tts_words:
                        word_entries.append({
                            "word": tw["word"],
                            "start": tw["start"],
                            "end": tw["end"],
                        })
                    
                    lane_idx = 0  # All subtitles at same position
                    
                    # Prevent overlap: if this subtitle starts before previous ends, adjust
                    if chunk_start < last_subtitle_end + MIN_SUBTITLE_GAP:
                        # Option 1: Trim previous subtitle's end
                        if subtitles:
                            old_end = subtitles[-1]["end"]
                            subtitles[-1]["end"] = max(subtitles[-1]["start"] + 0.3, chunk_start - MIN_SUBTITLE_GAP)
                            if old_end != subtitles[-1]["end"]:
                                logger.debug(
                                    "Trimmed previous subtitle end: %.2f -> %.2f to avoid overlap",
                                    old_end, subtitles[-1]["end"]
                                )
                    
                    subtitles.append({
                        "start": chunk_start,
                        "end": chunk_end_time,
                        "text": " ".join(tw["word"] for tw in chunk_tts_words),
                        "words": word_entries,
                        "speaker": speaker_id,
                        "color": speaker_palette.get(speaker_id),
                        "lane": lane_idx,
                    })
                    last_subtitle_end = chunk_end_time
                    chunks_added += 1
            else:
                # Fallback: distribute words proportionally using character-based chunking
                word_chunks: List[List[str]] = []
                chunk: List[str] = []
                current_chars = 0
                for word in words:
                    word_len = len(word) + 1  # +1 for space
                    if current_chars + word_len > max_chars_per_line and chunk:
                        word_chunks.append(chunk)
                        chunk = []
                        current_chars = 0
                    chunk.append(word)
                    current_chars += word_len
                if chunk:
                    word_chunks.append(chunk)

                total_words = len(words)
                total_window = max(relative_end - relative_start, 0.1)
                elapsed = relative_start

                chunks_added = 0

                for idx, chunk_words in enumerate(word_chunks):
                    chunk_word_count = len(chunk_words)
                    if total_words <= 0:
                        chunk_duration = total_window / max(len(word_chunks), 1)
                    else:
                        chunk_duration = total_window * (chunk_word_count / total_words)

                    if idx == len(word_chunks) - 1:
                        chunk_duration = max(0.05, relative_end - elapsed)

                    start_time = elapsed
                    end_time = min(relative_end, start_time + chunk_duration)
                    elapsed = end_time

                    per_word = chunk_duration / chunk_word_count if chunk_word_count else 0.0
                    word_entries = []
                    for word_idx, word in enumerate(chunk_words):
                        word_start = start_time + word_idx * per_word
                        word_entries.append(
                            {
                                "word": word,
                                "start": word_start,
                                "end": min(end_time, word_start + per_word),
                            }
                        )

                    lane_idx = 0  # All subtitles at same position

                    # Prevent overlap: if this subtitle starts before previous ends, adjust
                    if start_time < last_subtitle_end + MIN_SUBTITLE_GAP:
                        if subtitles:
                            old_end = subtitles[-1]["end"]
                            subtitles[-1]["end"] = max(subtitles[-1]["start"] + 0.3, start_time - MIN_SUBTITLE_GAP)
                            if old_end != subtitles[-1]["end"]:
                                logger.debug(
                                    "Trimmed previous subtitle end: %.2f -> %.2f to avoid overlap",
                                    old_end, subtitles[-1]["end"]
                                )

                    subtitles.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "text": " ".join(chunk_words),
                            "words": word_entries,
                            "speaker": speaker_id,
                            "color": speaker_palette.get(speaker_id),
                            "lane": lane_idx,
                        }
                    )
                    last_subtitle_end = end_time
                    chunks_added += 1
                chunks_added += 1

            if subtitles and subtitles[-1]["end"] < relative_end:
                # Extend last subtitle to cover remaining time
                subtitles[-1]["end"] = relative_end
                if subtitles[-1]["words"]:
                    subtitles[-1]["words"][-1]["end"] = relative_end

            if chunks_added == 0:
                logger.warning(
                    "No subtitle chunks generated for speaker %s (%.2f–%.2fs). Adding fallback line.",
                    speaker_id,
                    relative_start,
                    relative_end,
                )
                per_word = total_window / max(total_words, 1)
                word_entries = []
                for word_idx, word in enumerate(words):
                    word_start = relative_start + word_idx * per_word
                    word_entries.append(
                        {
                            "word": word,
                            "start": word_start,
                            "end": min(relative_end, word_start + per_word),
                        }
                    )
                lane_idx = 0  # All subtitles at same position
                
                # Prevent overlap for fallback subtitle too
                actual_start = relative_start
                if actual_start < last_subtitle_end + MIN_SUBTITLE_GAP:
                    if subtitles:
                        subtitles[-1]["end"] = max(subtitles[-1]["start"] + 0.3, actual_start - MIN_SUBTITLE_GAP)
                
                subtitles.append(
                    {
                        "start": actual_start,
                        "end": relative_end,
                        "text": " ".join(words),
                        "words": word_entries,
                        "speaker": speaker_id,
                        "color": speaker_palette.get(speaker_id),
                        "lane": lane_idx,
                    }
                )
                last_subtitle_end = relative_end

        return subtitles
    
    def _create_stylized_subtitles(
        self,
        subtitles: List[Dict],
        output_path: Path,
        style: str = "capcut",
        animation: str = "fade",
        font_name: str = "Montserrat Light",
        font_size: int = 86,
        subtitle_position: str = "mid_low",
        subtitle_background: bool = False,
        subtitle_glow: bool = True,
    ):
        """
        Create ASS subtitle file with TikTok/Instagram style.
        
        Subtitles format:
        [
            {
                'start': 0.5,
                'end': 1.2,
                'text': 'Привет',
                'words': [
                    {'word': 'Привет', 'start': 0.5, 'end': 1.2}
                ]
            },
            ...
        ]
        """
        # ASS subtitle styles
        styles = {
            'capcut': {
                'fontname': font_name,
                'fontsize': font_size,
                'primarycolor': '&H00FFFFFF',
                'outlinecolor': '&H00000000',
                'borderstyle': 1,
                'outline': 0,
                'shadow': 0,
                'alignment': 8,
                'marginv': 450,
                'animation': 'fade',
            },
            'tiktok': {
                'fontname': font_name,
                'fontsize': font_size,
                'primarycolor': '&H00FFFFFF',
                'outlinecolor': '&H00000000',
                'borderstyle': 1,
                'outline': 0,
                'shadow': 0,
                'alignment': 2,
                'marginv': 90,
                'animation': 'slide',
            },
            'instagram': {
                'fontname': font_name,
                'fontsize': font_size,
                'primarycolor': '&H00FFFFFF',
                'outlinecolor': '&H00000000',
                'borderstyle': 1,
                'outline': 0,
                'shadow': 0,
                'alignment': 2,
                'marginv': 80,
                'animation': 'spark',
            },
            'youtube': {
                'fontname': font_name,
                'fontsize': font_size,
                'primarycolor': '&H00FFFFFF',
                'outlinecolor': '&H00000000',
                'borderstyle': 1,
                'outline': 0,
                'shadow': 0,
                'alignment': 2,
                'marginv': 70,
                'animation': 'fade',
            }
        }
        
        selected_style = {**styles.get(style, styles['capcut'])}
        animation_style = animation or selected_style.get('animation', 'fade')
        position_config = self.SUBTITLE_POSITIONS.get(
            subtitle_position,
            self.SUBTITLE_POSITIONS['mid_low'],
        )
        selected_style['alignment'] = position_config.get('an', selected_style['alignment'])
        selected_style['marginv'] = position_config.get('marginv', selected_style['marginv'])
        
        # Style options priority:
        # 1. Background box (BorderStyle=3) - takes precedence
        # 2. Glow effect (outline + shadow) - soft halo around text
        # 3. Neither - plain white text
        
        if subtitle_background:
            # Opaque box behind text (BorderStyle=3)
            selected_style['borderstyle'] = 3
            selected_style['outline'] = 14  # Padding around text
            # OutlineColour becomes the box color in BorderStyle=3
            # Format: &HAABBGGRR (AA=alpha, BB=blue, GG=green, RR=red)
            box_color = "&HC0000000"  # ~75% transparent black
            back_color = "&H00000000"
        elif subtitle_glow:
            # Soft glow effect - black outline + shadow for readability
            selected_style['borderstyle'] = 1  # Outline + shadow mode
            selected_style['outline'] = 3  # Black stroke around text
            selected_style['shadow'] = 2  # Soft shadow offset
            box_color = "&H80000000"  # Semi-transparent black outline (50% alpha)
            back_color = "&H80000000"  # Semi-transparent black shadow
        else:
            # No effects - plain text
            box_color = selected_style['outlinecolor']
            back_color = "&H00000000"

        ass_content = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{selected_style['fontname']},{selected_style['fontsize']},{selected_style['primarycolor']},&H000000FF,{box_color},{back_color},0,0,0,0,100,100,0,0,{selected_style['borderstyle']},{selected_style['outline']},{selected_style['shadow']},{selected_style['alignment']},105,105,{selected_style['marginv']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        for subtitle in subtitles:
            start_time = self._format_timestamp(subtitle['start'])
            end_time = self._format_timestamp(subtitle['end'])
            color_value = subtitle.get('color')
            color_tag = rf"\1c&H{color_value}&" if color_value else ""

            lane = subtitle.get('lane', 0)
            # When background is enabled, use chunk-level animation (not word-by-word)
            # because BorderStyle=3 creates separate boxes around each animated word
            if style == 'capcut' and not subtitle_background:
                text = self._build_capcut_line(
                    subtitle,
                    animation_style,
                    position_config,
                    subtitle_background,
                    color_tag,
                    lane,
                )
                ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,105,105,0,,{text}\n"
            else:
                # For chunk mode (especially with background), we may need to split
                # into multiple dialogue lines with different Y positions
                lines = self._build_chunk_lines(
                    subtitle,
                    animation_style,
                    position_config,
                    subtitle_background,
                    color_tag,
                    lane,
                )
                for line_text in lines:
                    ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,105,105,0,,{line_text}\n"
        
        # Log subtitle line lengths for margin debugging (using dynamic font metrics)
        actual_font_name = selected_style.get('fontname', 'Montserrat-Medium')
        actual_font_size = selected_style.get('fontsize', 60)
        max_safe_chars = get_max_chars_for_line(actual_font_name, actual_font_size, self.fonts_dir)
        
        over_limit_count = 0
        for idx, subtitle in enumerate(subtitles):
            words = [w.get('word', '') for w in subtitle.get('words', [])]
            line_text = ' '.join(words)
            char_count = len(line_text)
            if char_count > max_safe_chars:
                over_limit_count += 1
                logger.warning(
                    "SUBTITLE MARGIN WARNING: line %d has %d chars (max=%d for font '%s' size %d): '%s'",
                    idx, char_count, max_safe_chars, actual_font_name, actual_font_size, line_text[:50]
                )
        
        # Only log if there were issues
        if over_limit_count > 0:
            logger.warning("SUBTITLE: %d/%d lines exceed margin limit", over_limit_count, len(subtitles))
        
        # Write to file
        output_path.write_text(ass_content, encoding='utf-8')
        
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to ASS timestamp format (H:MM:SS.CS)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    
    def _build_chunk_lines(
        self,
        subtitle: Dict,
        animation: str,
        position_conf: Dict,
        subtitle_background: bool,
        color_tag: str = "",
        lane: int = 0,
    ) -> List[str]:
        """
        Render text chunk(s) with a chosen animation preset.
        Returns a list of formatted lines. When background is enabled and text
        is long, returns two separate lines with different Y positions to prevent
        overlapping background boxes.
        """
        words = subtitle.get('text', '').split()
        if not words and subtitle.get('words'):
            words = [w.get('word', '') for w in subtitle['words']]

        if not words:
            return [self._get_base_animation_tag(animation, position_conf, subtitle_background, color_tag, lane)]

        # When background is enabled and we have many words, split into two
        # separate dialogue events with different Y positions
        if len(words) >= 6 and subtitle_background:
            split_index = len(words) // 2
            line1_words = words[:split_index]
            line2_words = words[split_index:]
            
            # Line spacing for background boxes (larger gap to prevent overlap)
            line_spacing = 90  # pixels between lines
            
            # Create position config for each line
            x = position_conf.get('x', 540)
            base_y = position_conf.get('y', 1250)
            lane_gap = 140
            y = base_y - lane * lane_gap
            
            # Upper line (line 1) - slightly above center
            y1 = y - line_spacing // 2
            # Lower line (line 2) - slightly below center  
            y2 = y + line_spacing // 2
            
            pos_conf_1 = {**position_conf, 'y': y1}
            pos_conf_2 = {**position_conf, 'y': y2}
            
            tag1 = self._get_base_animation_tag(animation, pos_conf_1, subtitle_background, color_tag, lane=0)
            tag2 = self._get_base_animation_tag(animation, pos_conf_2, subtitle_background, color_tag, lane=0)
            
            return [
                f"{tag1}{' '.join(line1_words)}",
                f"{tag2}{' '.join(line2_words)}",
            ]
        
        # Normal case: single line (or two lines with \N when no background)
        if len(words) >= 6 and not subtitle_background:
            split_index = len(words) // 2
            text = " ".join(words[:split_index]) + r"\N" + " ".join(words[split_index:])
        else:
            text = " ".join(words)

        return [f"{self._get_base_animation_tag(animation, position_conf, subtitle_background, color_tag, lane)}{text}"]

    def _build_capcut_line(
        self,
        subtitle: Dict,
        animation: str,
        position_conf: Dict,
        subtitle_background: bool,
        color_tag: str = "",
        lane: int = 0,
    ) -> str:
        """Animate each word sequentially according to animation preset."""
        words = subtitle.get('words', [])
        if not words:
            return self._build_chunk_line(subtitle, animation, position_conf, subtitle_background, color_tag, lane)

        chunk_start = subtitle.get('start', 0.0)
        chunk_end = subtitle.get('end', chunk_start)
        chunk_duration = max(0.01, chunk_end - chunk_start)

        rendered = [self._get_base_animation_tag(animation, position_conf, subtitle_background, color_tag, lane)]
        tokens: List[str] = []
        
        # Determine which words will be on the second line (after split)
        # Words on second line need extended visibility since they appear later
        split_index = len(words) // 2 if len(words) >= 6 else len(words)
        
        # Animations that benefit from extended visibility (like fade)
        EXTENDED_VISIBILITY_ANIMATIONS = ('fade', 'fade_short', 'highlight', 'boxed', 'bounce_word', 'readable')
        
        # Word timing logs disabled to reduce noise (enable for debugging)
        
        for idx, word in enumerate(words):
            rel_start = max(0.0, (word.get('start', chunk_start) - chunk_start) * 1000)
            rel_end = max(rel_start + 200.0, (word.get('end', chunk_start) - chunk_start) * 1000)
            highlight_start = int(rel_start)
            highlight_mid = int(min(rel_start + 160.0, chunk_duration * 1000))
            
            # Extend visibility for words that need more reading time:
            # 1. Last word of chunk: +700ms buffer
            # 2. Words on second line (idx >= split_index): +500ms buffer
            # This ensures second line words stay visible longer
            is_last_word = (idx == len(words) - 1)
            is_second_line = (idx >= split_index) and (len(words) >= 6)
            
            # Minimum visibility time for readability
            MIN_WORD_VISIBILITY = 1100  # ms - minimum time a word should be visible
            
            if animation in EXTENDED_VISIBILITY_ANIMATIONS:
                # All words stay until chunk ends + buffer
                if is_last_word:
                    highlight_end = int(chunk_duration * 1000 + 700)  # +700ms for last word
                else:
                    highlight_end = int(chunk_duration * 1000 + 300)  # +300ms for all other words
                
                # Enforce minimum visibility: if word appears late, extend its end time
                calculated_visibility = highlight_end - highlight_start
                if calculated_visibility < MIN_WORD_VISIBILITY:
                    highlight_end = highlight_start + MIN_WORD_VISIBILITY
            else:
                # Other animations: original behavior
                if is_last_word:
                    highlight_end = int(chunk_duration * 1000 + 700)
                elif is_second_line:
                    highlight_end = int(chunk_duration * 1000 + 500)
                else:
                    highlight_end = int(min(rel_end, chunk_duration * 1000))

            word_text = word.get('word', '')
            visibility_ms = highlight_end - highlight_start
            
            # Word timing log disabled (enable for debugging)
            # if animation in EXTENDED_VISIBILITY_ANIMATIONS:
            #     flag = "⚠️SHORT" if visibility_ms < 400 else ""
            #     logger.info(f"  [{idx}] '{word_text}' appear={highlight_start}ms visible={visibility_ms}ms {flag}")

            tag = self._get_word_animation_tag(
                animation,
                highlight_start,
                highlight_mid,
                highlight_end,
                subtitle_background,
                preserve_color=bool(color_tag),
            )

            if idx != len(words) - 1:
                word_text += " "

            tokens.append(f"{tag}{word_text}")

        # fade_short: never split into 2 lines (optimized for short chunks)
        # fade: split into 2 lines if 6+ words
        if animation != 'fade_short' and len(tokens) >= 6:
            split_index = len(tokens) // 2
            tokens.insert(split_index, r"\N")

        rendered.append(" ".join(tokens).replace(" \\N ", r"\N"))
        return "".join(rendered)

    def _get_base_animation_tag(
        self,
        animation: str,
        position_conf: Dict,
        subtitle_background: bool,
        color_tag: str = "",
        lane: int = 0,
    ) -> str:
        lane_gap = 140
        x = position_conf.get('x', 540)
        base_y = position_conf.get('y', 1250)
        y = base_y - lane * lane_gap
        an = position_conf.get('an', 8)
        pos_tag = rf"\pos({x},{y})"
        color_cmd = color_tag or ""
        # For slide/mask animations we need both start/end Y positions
        slide_start_y = y + 220
        move_down_y = y + 100
        
        # \3a sets outline/box alpha: 00=opaque, FF=transparent
        # Use ~70% transparent box when background is enabled
        bg_alpha = r"\3a&HB0&" if subtitle_background else ""
        
        presets = {
            'slide': rf"{{\an{an}\move({x},{slide_start_y},{x},{y},0,260)\fad(60,60){bg_alpha}{color_cmd}}}",
            'spark': rf"{{\an{an}{pos_tag}\fad(50,70)\blur2{bg_alpha}{color_cmd}}}",
            'fade': rf"{{\an{an}{pos_tag}\fad(100,100){bg_alpha}{color_cmd}}}",
            'fade_short': rf"{{\an{an}{pos_tag}\fad(100,100){bg_alpha}{color_cmd}}}",
            'readable': rf"{{\an{an}{pos_tag}\fad(200,200){bg_alpha}{color_cmd}}}",
            'highlight': rf"{{\an{an}{pos_tag}\fad(150,150){bg_alpha}{color_cmd}}}",
            'boxed': rf"{{\an{an}{pos_tag}\fad(100,100){bg_alpha}{color_cmd}}}",
            'bounce_word': rf"{{\an{an}{pos_tag}\fad(80,80){bg_alpha}{color_cmd}}}",
            'scale': rf"{{\an{an}{pos_tag}\fad(80,40){bg_alpha}{color_cmd}}}",
            'karaoke': rf"{{\an{an}{pos_tag}\fad(80,40){bg_alpha}{color_cmd}}}",
            'typewriter': rf"{{\an{an}{pos_tag}{bg_alpha}{color_cmd}}}",
            'mask': rf"{{\an{an}\move({x},{move_down_y},{x},{y},0,300){bg_alpha}{color_cmd}}}",
            'simple_fade': rf"{{\an{an}{pos_tag}\fad(150,150){bg_alpha}{color_cmd}}}",
            'word_pop': rf"{{\an{an}{pos_tag}\fad(80,40){bg_alpha}{color_cmd}}}",
        }
        return presets.get(animation, presets['fade'])

    def _get_word_animation_tag(
        self,
        animation: str,
        start_ms: int,
        mid_ms: int,
        end_ms: int,
        subtitle_background: bool,
        preserve_color: bool = False,
        is_current_word: bool = False,
    ) -> str:
        # Background box is handled by BorderStyle=3 in the style definition
        # Word animations should not override \bord or \shad
        if animation == 'slide':
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H40)"
                rf"\t({mid_ms},{end_ms},\alpha&H00)}}"
            )
        if animation == 'spark':
            if preserve_color:
                return (
                    r"{\alpha&HFF\blur4"
                    rf"\t({start_ms},{mid_ms},\alpha&H00\blur0)"
                    rf"\t({mid_ms},{end_ms},\alpha&H00)}}"
                )
            return (
                r"{\alpha&HFF\1c&H00F7FF\blur4"
                rf"\t({start_ms},{mid_ms},\alpha&H00\1c&HFFFFFF\blur0)"
                rf"\t({mid_ms},{end_ms},\alpha&H00)}}"
            )
        if animation == 'fade' or animation == 'fade_short':
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H00)"
                rf"\t({mid_ms},{end_ms},)}}"
            )
        if animation == 'readable':
            # All words appear together at chunk start - no per-word animation
            # Just return empty tag, the base animation handles fade in/out
            return ""
        if animation == 'highlight':
            # All words visible from start, current word highlighted in gold/yellow
            # Words start gray, become white when "active", current word is gold
            # \1c is primary fill color in BGR format
            gold = r"\1c&H45BAF4&"  # #F4BA45 in BGR
            white = r"\1c&HFFFFFF&"   # White
            gray = r"\1c&HBBBBBB&"    # Light gray
            return (
                rf"{{{gray}"
                rf"\t({start_ms},{start_ms + 80},{gold})"
                rf"\t({mid_ms},{end_ms},{white})}}"
            )
        if animation == 'boxed':
            # Word appears with a box/border effect that highlights it
            # Uses \bord for border thickness animation
            return (
                r"{\alpha&HFF\bord0"
                rf"\t({start_ms},{mid_ms},\alpha&H00\bord4)"
                rf"\t({mid_ms},{end_ms},\bord2)}}"
            )
        if animation == 'bounce_word':
            # Word bounces in with scale effect - appears and grows slightly then settles
            return (
                r"{\alpha&HFF\fscx80\fscy80"
                rf"\t({start_ms},{mid_ms},\alpha&H00\fscx115\fscy115)"
                rf"\t({mid_ms},{end_ms},\fscx100\fscy100)}}"
            )
        if animation == 'scale':
            return (
                r"{\alpha&HFF\fscx90\fscy90"
                rf"\t({start_ms},{mid_ms},\alpha&H00\fscx105\fscy105)"
                rf"\t({mid_ms},{end_ms},\fscx100\fscy100)}}"
            )
        if animation == 'word_pop':
            return (
                r"{\alpha&HFF\fscx0\fscy0"
                rf"\t({start_ms},{mid_ms},\alpha&H00\fscx115\fscy115)"
                rf"\t({mid_ms},{end_ms},\fscx100\fscy100)}}"
            )
        if animation == 'karaoke':
            # Word appears in Cyan/Gold, then fades to White
            if preserve_color:
                return (
                    r"{\alpha&HFF"
                    rf"\t({start_ms},{mid_ms},\alpha&H00)"
                    rf"\t({mid_ms},{end_ms},)}}"
                )
            return (
                r"{\alpha&HFF\1c&H00E1FF"
                rf"\t({start_ms},{mid_ms},\alpha&H00)"
                rf"\t({mid_ms},{end_ms},\1c&HFFFFFF)}}"
            )
        if animation == 'typewriter':
            # Sharp appearance
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{start_ms + 1},\alpha&H00)}}"
            )
        if animation == 'mask':
            # Clip reveal simulation: move slightly up and appear
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H00)}}"
            )
        if animation == 'simple_fade':
            # Smooth appearance opacity 0->1
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H00)}}"
            )
        # Default fallback
        return (
            r"{\alpha&HFF"
            rf"\t({start_ms},{mid_ms},\alpha&H00\fscx118\fscy118)"
            rf"\t({mid_ms},{end_ms},\fscx100\fscy100)}}"
        )
    
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video."""
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata."""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate']),
                'has_audio': audio_info is not None
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise

