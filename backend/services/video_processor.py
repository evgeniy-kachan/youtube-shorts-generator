"""Video processing service for cutting, adding audio and subtitles."""
import ffmpeg
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Literal
import subprocess
import uuid

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
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        method: Literal["letterbox", "center_crop"] = "letterbox"
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

                if input_ratio >= target_ratio:
                    # Видео шире 9:16 → вписываем по высоте и обрезаем по бокам
                    pipeline = (
                        ffmpeg
                        .input(video_path)
                        .filter('scale', -1, self.TARGET_HEIGHT)
                        .filter('crop', self.TARGET_WIDTH, self.TARGET_HEIGHT)
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
        vertical_method: str = "letterbox"
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
                    method=vertical_method
                )
            
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
            audio_input = ffmpeg.input(audio_path)
            
            output = (
                ffmpeg
                .output(
                    video_input.video,
                    audio_input.audio,
                    output_path,
                    vf=f"ass={subtitle_path}",
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
        method: Literal["letterbox", "center_crop"] = "letterbox",
        subtitle_style: str = "capcut",
        subtitle_animation: str = "bounce",
        subtitle_position: str = "mid_low",
        subtitle_font: str = "Montserrat Light",
        subtitle_font_size: int = 86,
        subtitle_background: bool = False,
    ) -> str:
        """
        End-to-end helper that cuts the source video, converts it to vertical format,
        overlays subtitles and replaces audio with synthesized TTS.
        Returns path to the processed temporary file.
        """
        try:
            duration = max(0.1, end_time - start_time)
            temp_segment = self.output_dir / f"segment_cut_{uuid.uuid4().hex}.mp4"
            temp_output = self.output_dir / f"segment_processed_{uuid.uuid4().hex}.mp4"

            # Step 1: cut source segment
            cut_path = self.cut_segment(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                output_path=str(temp_segment)
            )

            # Step 2: prepare basic subtitles
            subtitles = self._generate_basic_subtitles(text=text, duration=duration)

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
                vertical_method=method
            )

            return processed_path

        finally:
            # Clean up intermediate cut segment
            if 'temp_segment' in locals():
                Path(temp_segment).unlink(missing_ok=True)

    def save_video(self, processed_path: str, output_path: str) -> str:
        """
        Move processed temporary video into its final location.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Path(processed_path).replace(output_path)
        logger.info(f"Final video saved to: {output_path}")
        return str(output_path)

    def _generate_basic_subtitles(
        self,
        text: str,
        duration: float,
        max_words_per_line: int = 6
    ) -> List[Dict]:
        """
        Create a multi-line subtitle track with approximate word-level timings.
        Each subtitle line contains up to `max_words_per_line` words so the viewer
        sees a few words at a time, synchronized with the TTS audio.
        """
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
    
    def _create_stylized_subtitles(
        self,
        subtitles: List[Dict],
        output_path: Path,
        style: str = "capcut",
        animation: str = "bounce",
        font_name: str = "Montserrat Light",
        font_size: int = 86,
        subtitle_position: str = "mid_low",
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
                'shadow': 8,
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
                'shadow': 6,
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
                'shadow': 5,
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
                'shadow': 5,
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

            if style == 'capcut':
                text = self._build_capcut_line(subtitle, animation_style, position_config)
            else:
                text = self._build_chunk_line(subtitle, animation_style, position_config)

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
    
    def _build_chunk_line(self, subtitle: Dict, animation: str, position_conf: Dict) -> str:
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

        return f"{self._get_base_animation_tag(animation, position_conf)}{text}"

    def _build_capcut_line(self, subtitle: Dict, animation: str, position_conf: Dict) -> str:
        """Animate each word sequentially according to animation preset."""
        words = subtitle.get('words', [])
        if not words:
            return self._build_chunk_line(subtitle, animation, position_conf)

        chunk_start = subtitle.get('start', 0.0)
        chunk_end = subtitle.get('end', chunk_start)
        chunk_duration = max(0.01, chunk_end - chunk_start)

        rendered = [self._get_base_animation_tag(animation, position_conf)]
        tokens: List[str] = []

        for idx, word in enumerate(words):
            rel_start = max(0.0, (word.get('start', chunk_start) - chunk_start) * 1000)
            rel_end = max(rel_start + 200.0, (word.get('end', chunk_start) - chunk_start) * 1000)
            highlight_start = int(rel_start)
            highlight_mid = int(min(rel_start + 160.0, chunk_duration * 1000))
            highlight_end = int(min(rel_end, chunk_duration * 1000))

            tag = self._get_word_animation_tag(animation, highlight_start, highlight_mid, highlight_end)

            word_text = word.get('word', '')
            if idx != len(words) - 1:
                word_text += " "

            tokens.append(f"{tag}{word_text}")

        if len(tokens) >= 6:
            split_index = len(tokens) // 2
            tokens.insert(split_index, r"\N")

        rendered.append(" ".join(tokens).replace(" \\N ", r"\N"))
        return "".join(rendered)

    def _get_base_animation_tag(self, animation: str, position_conf: Dict) -> str:
        x = position_conf.get('x', 540)
        y = position_conf.get('y', 1250)
        an = position_conf.get('an', 8)
        pos_tag = rf"\pos({x},{y})"
        presets = {
            'bounce': rf"{{\an{an}{pos_tag}\fad(80,40)}}",
            'slide': rf"{{\an{an}\move({x},{y + 220},{x},{y},0,260)\fad(60,60)}}",
            'spark': rf"{{\an{an}{pos_tag}\fad(50,70)\blur2}}",
        }
        return presets.get(animation, presets['bounce'])

    def _get_word_animation_tag(self, animation: str, start_ms: int, mid_ms: int, end_ms: int) -> str:
        if animation == 'slide':
            return (
                r"{\alpha&HFF"
                rf"\t({start_ms},{mid_ms},\alpha&H40)"
                rf"\t({mid_ms},{end_ms},\alpha&H00)}}"
            )
        if animation == 'spark':
            return (
                r"{\alpha&HFF\1c&H00F7FF\bord4\blur4"
                rf"\t({start_ms},{mid_ms},\alpha&H00\1c&HFFFFFF\bord0\blur0)"
                rf"\t({mid_ms},{end_ms},\alpha&H00)}}"
            )
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

