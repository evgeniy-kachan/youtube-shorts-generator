"""Video processing service for cutting, adding audio and subtitles."""
import ffmpeg
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Literal
import subprocess
import uuid

from backend.services.face_detector import FaceDetector
from backend.services.diarization_runner import run_external_diarization

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process video segments: cut, add TTS audio, add stylized subtitles."""
    
    # Target dimensions for Reels/Shorts (9:16 aspect ratio)
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    SUBTITLE_POSITIONS = {
        "mid_low": {"x": 540, "y": 1050, "an": 8, "marginv": 480},
        "lower_center": {"x": 540, "y": 1250, "an": 8, "marginv": 420},
        "lower_left": {"x": 360, "y": 1350, "an": 7, "marginv": 380},
        "lower_right": {"x": 720, "y": 1350, "an": 9, "marginv": 380},
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
    ) -> float | None:
        try:
            detector = self._get_face_detector()
        except Exception as exc:
            logger.warning("Unable to initialize face detector: %s", exc)
            return None

        try:
            focus = detector.estimate_horizontal_focus(
                video_path,
                max_samples=max_samples,
                dialogue=dialogue,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            return focus
        except Exception as exc:
            logger.warning("Face focus estimation failed for %s: %s", video_path, exc)
            return None

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
            (
                ffmpeg
                .input(video_path, ss=start_time, t=duration)
                .output(
                    output_path,
                    codec='copy',  # Copy without re-encoding for speed
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
    ) -> str:
        """
        Multi-segment crop: for each time span apply its own horizontal offset,
        then concat all segments back. No smoothing; hard cuts when focus jumps.
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

        # Compute scaling params once
        if input_ratio >= target_ratio:
            scaled_width = max(self.TARGET_WIDTH, int(round(self.TARGET_HEIGHT * input_ratio)))
            if scaled_width % 2 != 0:
                scaled_width += 1
            margin = max(scaled_width - self.TARGET_WIDTH, 0)
        else:
            scaled_width = self.TARGET_WIDTH
            margin = 0

        # Build filter_complex with trim+scale+crop per segment
        parts = []
        labels = []
        for idx, seg in enumerate(focus_timeline):
            start = max(0.0, float(seg.get("start", 0.0)))
            end = float(seg.get("end", start))
            focus = float(seg.get("focus", 0.5))
            focus = max(0.0, min(1.0, focus))

            offset = int(round(focus * scaled_width - (self.TARGET_WIDTH / 2)))
            offset = max(0, min(margin, offset))

            label = f"v{idx}"
            labels.append(label)
            parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
                f"scale={scaled_width}:{self.TARGET_HEIGHT},"
                f"crop={self.TARGET_WIDTH}:{self.TARGET_HEIGHT}:{offset}:0[{label}]"
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

        logger.info("Applying multi-crop timeline with %d segments", len(focus_timeline))
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
        animation: str = "bounce",
        font_name: str = "Montserrat Light",
        font_size: int = 86,
        subtitle_position: str = "mid_low",
        subtitle_background: bool = False,
        convert_to_vertical: bool = True,
        vertical_method: str = "letterbox",
        crop_focus: str = "center",
        auto_center_ratio: float | None = None,
        focus_timeline: list[dict] | None = None,
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
            logger.info(f"Adding audio and subtitles to video")
            
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
        dialogue: list[dict] | None = None,
        preserve_background_audio: bool = False,
        crop_focus: str = "center",
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
                diar_segments = run_external_diarization(
                    input_path=str(cut_path),
                    diar_python=os.getenv("EXTERNAL_DIAR_PY"),
                    diar_script=os.getenv("EXTERNAL_DIAR_SCRIPT"),
                    hf_token=os.getenv("HUGGINGFACE_TOKEN"),
                )
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

            if method == "center_crop" and effective_crop_focus == "face_auto":
                # Build timeline (dynamic crops). If timeline has 0 or 1 segment, fall back to single focus.
                focus_timeline = self._build_focus_timeline(
                    str(cut_path),
                    dialogue=dialogue_turns,
                    segment_start=start_time,
                    segment_end=end_time,
                    sample_period=0.15,  # максимум частоты замеров для теста проблемных кадров
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
                    auto_center_ratio = self._estimate_face_focus(
                        str(cut_path),
                        dialogue=dialogue,
                        segment_start=start_time,
                        segment_end=end_time,
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
            speaker_palette = self._assign_speaker_colors(dialogue_turns)
            subtitles = self._generate_basic_subtitles(
                text=text,
                duration=duration,
                dialogue=dialogue_turns,
                segment_start=start_time,
                speaker_palette=speaker_palette,
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
                convert_to_vertical=True,
                vertical_method=method,
                crop_focus=effective_crop_focus,
                auto_center_ratio=auto_center_ratio,
                focus_timeline=focus_timeline,
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

    @staticmethod
    def _assign_speaker_colors(dialogue: list[dict] | None) -> dict[str, str]:
        """
        Return a deterministic color palette for every speaker in the segment.
        Colors are stored in ASS format (AABBGGRR without the leading &H).
        """
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
        max_words_per_line: int = 6,
        dialogue: list[dict] | None = None,
        segment_start: float = 0.0,
        speaker_palette: dict[str, str] | None = None,
    ) -> List[Dict]:
        """
        Create a multi-line subtitle track with approximate word-level timings.
        Each subtitle line contains up to `max_words_per_line` words so the viewer
        sees a few words at a time, synchronized with the TTS audio.
        """
        if dialogue:
            return self._generate_dialogue_subtitles(
                dialogue=dialogue,
                duration=duration,
                max_words_per_line=max_words_per_line,
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

        # Group words into chunks (a few words per subtitle line)
        word_chunks: List[List[str]] = []
        current_chunk: List[str] = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_words_per_line:
                word_chunks.append(current_chunk)
                current_chunk = []

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
        max_words_per_line: int,
        segment_start: float,
        speaker_palette: dict[str, str],
    ) -> List[Dict]:
        """
        Build subtitles directly from dialogue turns, preserving per-speaker timing
        and enabling color coding.
        """
        subtitles: List[Dict] = []
        lane_available_until = [0.0, 0.0]  # support two stacked rows

        def allocate_lane(start_time: float, end_time: float) -> int:
            tolerance = 0.03
            for idx in range(len(lane_available_until)):
                if start_time >= lane_available_until[idx] - tolerance:
                    lane_available_until[idx] = end_time
                    return idx
            lane_idx = lane_available_until.index(min(lane_available_until))
            lane_available_until[lane_idx] = end_time
            return lane_idx

        for turn in dialogue:
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

            word_chunks: List[List[str]] = []
            chunk: List[str] = []
            for word in words:
                chunk.append(word)
                if len(chunk) >= max_words_per_line:
                    word_chunks.append(chunk)
                    chunk = []
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

                lane_idx = allocate_lane(start_time, end_time)

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
                chunks_added += 1

            if subtitles and subtitles[-1]["end"] < relative_end:
                lane_idx = subtitles[-1].get("lane", 0)
                lane_available_until[lane_idx] = relative_end
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
                lane_idx = allocate_lane(relative_start, relative_end)
                subtitles.append(
                    {
                        "start": relative_start,
                        "end": relative_end,
                        "text": " ".join(words),
                        "words": word_entries,
                        "speaker": speaker_id,
                        "color": speaker_palette.get(speaker_id),
                        "lane": lane_idx,
                    }
                )

        return subtitles
    
    def _create_stylized_subtitles(
        self,
        subtitles: List[Dict],
        output_path: Path,
        style: str = "capcut",
        animation: str = "bounce",
        font_name: str = "Montserrat Light",
        font_size: int = 86,
        subtitle_position: str = "mid_low",
        subtitle_background: bool = False,
    ):
        logger.info(f"Generating subtitles with animation='{animation}', font='{font_name}', bg={subtitle_background}")
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
                'animation': 'bounce',
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
                'animation': 'bounce',
            }
        }
        
        selected_style = {**styles.get(style, styles['capcut'])}
        animation_style = animation or selected_style.get('animation', 'bounce')
        position_config = self.SUBTITLE_POSITIONS.get(
            subtitle_position,
            self.SUBTITLE_POSITIONS['mid_low'],
        )
        selected_style['alignment'] = position_config.get('an', selected_style['alignment'])
        selected_style['marginv'] = position_config.get('marginv', selected_style['marginv'])
        
        # Create ASS file
        back_color = "&H55000000" if subtitle_background else "&H00000000"

        ass_content = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{selected_style['fontname']},{selected_style['fontsize']},{selected_style['primarycolor']},&H000000FF,{selected_style['outlinecolor']},{back_color},0,0,0,0,100,100,0,0,{selected_style['borderstyle']},{selected_style['outline']},{selected_style['shadow']},{selected_style['alignment']},10,10,{selected_style['marginv']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        for subtitle in subtitles:
            start_time = self._format_timestamp(subtitle['start'])
            end_time = self._format_timestamp(subtitle['end'])
            color_value = subtitle.get('color')
            color_tag = rf"\1c&H{color_value}&" if color_value else ""

            lane = subtitle.get('lane', 0)
            if style == 'capcut':
                text = self._build_capcut_line(
                    subtitle,
                    animation_style,
                    position_config,
                    subtitle_background,
                    color_tag,
                    lane,
                )
            else:
                text = self._build_chunk_line(
                    subtitle,
                    animation_style,
                    position_config,
                    subtitle_background,
                    color_tag,
                    lane,
                )

            ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
        
        # Write to file
        output_path.write_text(ass_content, encoding='utf-8')
        
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to ASS timestamp format (H:MM:SS.CS)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    
    def _build_chunk_line(
        self,
        subtitle: Dict,
        animation: str,
        position_conf: Dict,
        subtitle_background: bool,
        color_tag: str = "",
        lane: int = 0,
    ) -> str:
        """Render text chunk with a chosen animation preset."""
        words = subtitle.get('text', '').split()
        if not words and subtitle.get('words'):
            words = [w.get('word', '') for w in subtitle['words']]

        if not words:
            return self._get_base_animation_tag(animation, position_conf)

        if len(words) >= 6:
            split_index = len(words) // 2
            text = " ".join(words[:split_index]) + r"\N" + " ".join(words[split_index:])
        else:
            text = " ".join(words)

        return f"{self._get_base_animation_tag(animation, position_conf, subtitle_background, color_tag, lane)}{text}"

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

        for idx, word in enumerate(words):
            rel_start = max(0.0, (word.get('start', chunk_start) - chunk_start) * 1000)
            rel_end = max(rel_start + 200.0, (word.get('end', chunk_start) - chunk_start) * 1000)
            highlight_start = int(rel_start)
            highlight_mid = int(min(rel_start + 160.0, chunk_duration * 1000))
            highlight_end = int(min(rel_end, chunk_duration * 1000))

            tag = self._get_word_animation_tag(
                animation,
                highlight_start,
                highlight_mid,
                highlight_end,
                subtitle_background,
                preserve_color=bool(color_tag),
            )

            word_text = word.get('word', '')
            if idx != len(words) - 1:
                word_text += " "

            tokens.append(f"{tag}{word_text}")

        if len(tokens) >= 6:
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
        background_tags = r"{\blur14\bord2\shad0\1c&H000000&}" if subtitle_background else ""
        color_cmd = color_tag or ""
        # For slide/mask animations we need both start/end Y positions
        slide_start_y = y + 220
        move_down_y = y + 100
        presets = {
            'bounce': rf"{{\an{an}{pos_tag}\fad(80,40){background_tags}{color_cmd}}}",
            'slide': rf"{{\an{an}\move({x},{slide_start_y},{x},{y},0,260)\fad(60,60){background_tags}{color_cmd}}}",
            'spark': rf"{{\an{an}{pos_tag}\fad(50,70)\blur2{background_tags}{color_cmd}}}",
            'fade': rf"{{\an{an}{pos_tag}\fad(100,100){background_tags}{color_cmd}}}",
            'scale': rf"{{\an{an}{pos_tag}\fad(80,40){background_tags}{color_cmd}}}",
            'karaoke': rf"{{\an{an}{pos_tag}\fad(80,40){background_tags}{color_cmd}}}",
            'typewriter': rf"{{\an{an}{pos_tag}{background_tags}{color_cmd}}}",
            'mask': rf"{{\an{an}\move({x},{move_down_y},{x},{y},0,300){background_tags}{color_cmd}}}",
            'simple_fade': rf"{{\an{an}{pos_tag}\fad(150,150){background_tags}{color_cmd}}}",
            'word_pop': rf"{{\an{an}{pos_tag}\fad(80,40){background_tags}{color_cmd}}}",
        }
        return presets.get(animation, presets['bounce'])

    def _get_word_animation_tag(
        self,
        animation: str,
        start_ms: int,
        mid_ms: int,
        end_ms: int,
        subtitle_background: bool,
        preserve_color: bool = False,
    ) -> str:
        background_tags = (
            r"\bord0\shad0"
            if subtitle_background
            else ""
        )
        if animation == 'bounce':
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H00\fscy120)"
                rf"\t({mid_ms},{end_ms},\fscy100{background_tags})}}"
            )
        if animation == 'slide':
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H40)"
                rf"\t({mid_ms},{end_ms},\alpha&H00{background_tags})}}"
            )
        if animation == 'spark':
            if preserve_color:
                return (
                    r"{\alpha&HFF\bord4\blur4"
                    rf"\t({start_ms},{mid_ms},\alpha&H00\bord0\blur0)"
                    rf"\t({mid_ms},{end_ms},\alpha&H00{background_tags})}}"
                )
            return (
                r"{\alpha&HFF\1c&H00F7FF\bord4\blur4"
                rf"\t({start_ms},{mid_ms},\alpha&H00\1c&HFFFFFF\bord0\blur0)"
                rf"\t({mid_ms},{end_ms},\alpha&H00{background_tags})}}"
            )
        if animation == 'fade':
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H00)"
                rf"\t({mid_ms},{end_ms},{background_tags})}}"
            )
        if animation == 'scale':
            return (
                r"{\alpha&HFF\fscx90\fscy90"
                rf"\t({start_ms},{mid_ms},\alpha&H00\fscx105\fscy105)"
                rf"\t({mid_ms},{end_ms},\fscx100\fscy100{background_tags})}}"
            )
        if animation == 'word_pop':
            return (
                r"{\alpha&HFF\fscx0\fscy0"
                rf"\t({start_ms},{mid_ms},\alpha&H00\fscx115\fscy115)"
                rf"\t({mid_ms},{end_ms},\fscx100\fscy100{background_tags})}}"
            )
        if animation == 'karaoke':
            # Word appears in Cyan/Gold, then fades to White
            if preserve_color:
                return (
                    r"{\alpha&HFF"
                    rf"\t({start_ms},{mid_ms},\alpha&H00)"
                    rf"\t({mid_ms},{end_ms},{background_tags})}}"
                )
            return (
                r"{\alpha&HFF\1c&H00E1FF"
                rf"\t({start_ms},{mid_ms},\alpha&H00)"
                rf"\t({mid_ms},{end_ms},\1c&HFFFFFF{background_tags})}}"
            )
        if animation == 'typewriter':
            # Sharp appearance
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{start_ms + 1},\alpha&H00{background_tags})}}"
            )
        if animation == 'mask':
            # Clip reveal simulation: move slightly up and appear
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H00{background_tags})}}"
            )
        if animation == 'simple_fade':
            # Smooth appearance opacity 0->1
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H00{background_tags})}}"
            )
        return (
            r"{\alpha&HFF"
            rf"\t({start_ms},{mid_ms},\alpha&H00\fscx118\fscy118)"
            rf"\t({mid_ms},{end_ms},\fscx100\fscy100{background_tags})}}"
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

