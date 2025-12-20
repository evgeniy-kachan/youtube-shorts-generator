"""Text-to-Speech services."""
import io
import logging
import os
import time
from pathlib import Path
import re
from datetime import datetime
from urllib.error import HTTPError

import ffmpeg
import httpx
import torch
import torchaudio
import soundfile as sf
from pydub import AudioSegment
from sentence_splitter import SentenceSplitter

from backend.config import TEMP_DIR

logger = logging.getLogger(__name__)


class BaseTTSService:
    """Common helpers shared between different TTS providers."""

    SPEECH_WORDS_PER_SEC = 140.0 / 60.0  # ~2.33 words per second
    MIN_CHUNK_DURATION = 0.12  # seconds
    MAX_PAUSE_FRACTION = 0.25  # pauses should not exceed 25% of the turn window

    def __init__(self, language: str = "ru", max_chunk_chars: int = 400):
        self.language = language
        self.max_chunk_chars = max_chunk_chars
        self.splitter = SentenceSplitter(language=language or "ru")
        self.failed_chunk_log = Path(TEMP_DIR) / "tts_failed_chunks.log"

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Normalize common punctuation and strip unsupported symbols."""
        replacements = {
            "“": '"',
            "”": '"',
            "„": '"',
            "’": "'",
            "‘": "'",
            "—": "-",
            "–": "-",
            "…": "...",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"[^0-9A-Za-zА-Яа-яЁё.,!?;:'\"()\- ]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into smaller chunks so the provider can process them.
        """
        if len(text) <= self.max_chunk_chars:
            return [text]

        sentences = self.splitter.split(text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) + 1 <= self.max_chunk_chars:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                if len(sentence) > self.max_chunk_chars:
                    # Split very long sentence into pieces
                    for i in range(0, len(sentence), self.max_chunk_chars):
                        chunks.append(sentence[i:i + self.max_chunk_chars])
                    current = ""
                else:
                    current = sentence

        if current:
            chunks.append(current)

        if not chunks:
            chunks = [text[: self.max_chunk_chars]]

        logger.info(
            "TTS text split into %s chunk(s), max length %s chars.",
            len(chunks),
            self.max_chunk_chars,
        )
        return chunks

    @staticmethod
    def _split_for_fallback(text: str) -> list[str]:
        """Split text into smaller sentences for fallback synthesis."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
        return cleaned if cleaned else [text]

    @staticmethod
    def _force_split(text: str, max_len: int = 150) -> list[str]:
        """Bruteforce split text into chunks without breaking words."""
        words = text.split()
        if not words:
            return [text]

        parts: list[str] = []
        current = ""

        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_len:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = word

        if current:
            parts.append(current)

        return parts if parts else [text]

    def _log_failed_chunk(self, text: str, context: str) -> None:
        """Persist failed chunk to a log file for manual inspection."""
        try:
            self.failed_chunk_log.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().isoformat()
            with self.failed_chunk_log.open("a", encoding="utf-8") as log_file:
                log_file.write(f"[{timestamp}] context={context}\n{text}\n---\n")
        except Exception as log_exc:
            logger.warning("Unable to write TTS failed chunk log: %s", log_exc)

    def _estimate_pause_ms(self, chunk_text: str) -> int:
        """Heuristic pause duration after a sentence chunk."""
        word_count = len(chunk_text.split())
        if word_count >= 18:
            return 180
        if word_count >= 12:
            return 140
        if word_count >= 7:
            return 100
        if word_count >= 4:
            return 70
        return 50

    def _estimate_duration_from_text(self, text: str) -> float:
        """Fallback duration estimation based on average speech rate."""
        words = max(1, len(text.split()))
        return max(0.5, words / self.SPEECH_WORDS_PER_SEC)

    def _split_turn_into_chunks(self, text: str) -> list[str]:
        """Split turn text into sentence-like chunks with basic normalization."""
        normalized = (text or "").strip()
        if not normalized:
            return []

        sentences = self.splitter.split(normalized) if self.splitter else [normalized]
        chunks: list[str] = []
        current = ""
        min_len = 45
        max_len = 220

        for sentence in sentences:
            cleaned = self._sanitize_text(sentence)
            if not cleaned:
                continue

            if not current:
                current = cleaned
                continue

            candidate = f"{current} {cleaned}".strip()
            if len(current) < min_len and len(candidate) <= max_len:
                current = candidate
                continue

            chunks.append(current.strip())
            current = cleaned

        if current:
            chunks.append(current.strip())

        if not chunks:
            chunks = [self._sanitize_text(normalized)]

        expanded: list[str] = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_chars:
                expanded.extend(self._chunk_text(chunk))
            else:
                expanded.append(chunk)

        return [piece.strip() for piece in expanded if piece.strip()]

    def _plan_turn_chunks(self, turn: dict) -> list[dict]:
        """
        Prepare a synthesis plan for a dialogue turn: small chunks with pauses and
        per-chunk target durations so the final audio matches diarized timings.
        """
        text = (turn.get("text_ru") or turn.get("text") or "").strip()
        if not text:
            return []

        raw_chunks = self._split_turn_into_chunks(text)
        plan: list[dict] = []
        for chunk_text in raw_chunks:
            cleaned = chunk_text.strip()
            if not cleaned:
                continue
            plan.append(
                {
                    "text": cleaned,
                    "word_count": max(1, len(cleaned.split())),
                    "pause_ms": self._estimate_pause_ms(cleaned),
                }
            )

        if not plan:
            return []

        plan[-1]["pause_ms"] = 0  # no trailing pause

        try:
            start_time = float(turn.get("start", 0.0))
        except (TypeError, ValueError):
            start_time = 0.0

        diarized_end = turn.get("end")
        end_time: float | None
        try:
            end_time = float(diarized_end) if diarized_end is not None else None
        except (TypeError, ValueError):
            end_time = None

        if end_time is not None and end_time > start_time:
            turn_window = max(0.2, end_time - start_time)
        else:
            turn_window = self._estimate_duration_from_text(text)

        raw_pause_ms = sum(chunk["pause_ms"] for chunk in plan[:-1])
        pause_budget_ms = min(
            raw_pause_ms,
            int(turn_window * 1000 * self.MAX_PAUSE_FRACTION),
        )

        if raw_pause_ms > 0 and pause_budget_ms >= 0:
            scale = pause_budget_ms / raw_pause_ms if raw_pause_ms else 0.0
            for chunk in plan[:-1]:
                chunk["pause_ms"] = int(chunk["pause_ms"] * scale)
        else:
            for chunk in plan[:-1]:
                chunk["pause_ms"] = 0

        pause_total_sec = sum(chunk["pause_ms"] for chunk in plan[:-1]) / 1000.0
        speech_budget = turn_window - pause_total_sec
        if speech_budget <= 0:
            speech_budget = self.MIN_CHUNK_DURATION * len(plan)

        total_weight = sum(chunk["word_count"] for chunk in plan) or len(plan)
        remaining = speech_budget

        for idx, chunk in enumerate(plan):
            weight = chunk["word_count"] / total_weight if total_weight else 1 / len(plan)
            if idx == len(plan) - 1:
                chunk_duration = max(self.MIN_CHUNK_DURATION, remaining)
            else:
                chunk_duration = max(self.MIN_CHUNK_DURATION, speech_budget * weight)
                remaining = max(0.0, remaining - chunk_duration)
            chunk["target_duration"] = chunk_duration

        return plan

    @staticmethod
    def _collect_filters_for_tempo(tempo: float) -> list[float]:
        """Split tempo factor into ffmpeg-compatible atempo filters."""
        filters: list[float] = []
        remaining = tempo

        while remaining > 2.0:
            filters.append(2.0)
            remaining /= 2.0
        while remaining < 0.5 and remaining > 0:
            filters.append(0.5)
            remaining /= 0.5

        if remaining > 0 and abs(remaining - 1.0) > 1e-3:
            filters.append(remaining)

        return [factor for factor in filters if 0.5 <= factor <= 2.0 and abs(factor - 1.0) > 1e-3]

    def _retime_audio_file(self, file_path: Path, tempo: float) -> bool:
        """
        Adjust audio duration via ffmpeg atempo filters.
        tempo > 1.0 speeds up (shorter), tempo < 1.0 slows down (longer).
        """
        if tempo <= 0 or abs(tempo - 1.0) < 0.02:
            return False

        filters = self._collect_filters_for_tempo(tempo)
        if not filters:
            return False

        temp_path = file_path.with_suffix(".retime.wav")
        try:
            stream = ffmpeg.input(str(file_path)).audio
            for factor in filters:
                stream = stream.filter("atempo", factor)

            (
                ffmpeg.output(
                    stream,
                    str(temp_path),
                    acodec="pcm_s16le",
                    ac=1,
                    ar="44100",
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            temp_path.replace(file_path)
            return True
        except ffmpeg.Error as exc:
            stderr = (exc.stderr or b"").decode(errors="ignore")
            logger.warning(
                "Failed to retime %s (tempo=%.2f): %s",
                file_path,
                tempo,
                stderr.strip(),
            )
            temp_path.unlink(missing_ok=True)
            return False

    def _render_turn_chunks(
        self,
        plan: list[dict],
        voice: str | None,
        destination: Path,
        turn_idx: int,
    ) -> AudioSegment | None:
        """Synthesize per-chunk audio and stitch it with pauses."""
        combined_segments: list[AudioSegment] = []
        speech_target = sum(chunk.get("target_duration", 0.0) for chunk in plan)
        pause_target = sum(chunk.get("pause_ms", 0) for chunk in plan[:-1]) / 1000.0
        turn_target_duration = speech_target + pause_target

        for chunk_idx, chunk in enumerate(plan):
            chunk_path = (
                destination.parent
                / f"dialogue_turn_{turn_idx}_chunk_{chunk_idx}_{destination.name}"
            )
            try:
                self.synthesize(chunk["text"], str(chunk_path), speaker=voice)
            except Exception as exc:
                logger.error(
                    "Failed to synthesize chunk %s/%s for turn %s: %s",
                    chunk_idx + 1,
                    len(plan),
                    turn_idx + 1,
                    exc,
                )
                chunk_path.unlink(missing_ok=True)
                continue

            try:
                if not chunk_path.exists():
                    logger.warning("Chunk path %s missing after synthesis.", chunk_path)
                    continue

                seg_audio = AudioSegment.from_file(str(chunk_path))
                current_duration = len(seg_audio) / 1000.0
                target_duration = chunk.get("target_duration")
                word_count = chunk.get("word_count") or len(chunk.get("text", "").split())

                should_retime_chunk = (
                    target_duration
                    and target_duration > 0
                    and current_duration > 0
                    and abs(current_duration - target_duration) > 0.08
                    and current_duration >= 0.6
                    and word_count >= 4
                )

                if should_retime_chunk:
                    tempo = current_duration / target_duration
                    if self._retime_audio_file(chunk_path, tempo):
                        seg_audio = AudioSegment.from_file(str(chunk_path))
                        current_duration = len(seg_audio) / 1000.0

                combined_segments.append(seg_audio)

                pause_ms = int(chunk.get("pause_ms") or 0)
                if pause_ms > 0 and chunk_idx != len(plan) - 1:
                    combined_segments.append(AudioSegment.silent(duration=pause_ms))
            finally:
                chunk_path.unlink(missing_ok=True)

        if not combined_segments:
            return None

        final_audio = combined_segments[0]
        for seg in combined_segments[1:]:
            final_audio += seg

        actual_duration = len(final_audio) / 1000.0
        if (
            turn_target_duration > 0
            and actual_duration > 0
            and abs(actual_duration - turn_target_duration) > 0.12
        ):
            temp_path = destination.parent / f"dialogue_turn_{turn_idx}_retime_{destination.name}"
            final_audio.export(str(temp_path), format="wav")
            tempo = actual_duration / turn_target_duration
            if self._retime_audio_file(temp_path, tempo):
                final_audio = AudioSegment.from_file(str(temp_path))
            temp_path.unlink(missing_ok=True)
        return final_audio

    def synthesize_dialogue(
        self,
        dialogue_turns: list[dict],
        output_path: str,
        voice_map: dict[str, str],
        pause_ms: int = 300,
        base_start: float | None = None,
    ) -> str:
        """
        Synthesize a dialogue with multiple speakers.

        If precise timestamps are available for each turn (start/end), we attempt to keep
        the natural overlaps by placing each synthesized phrase on a shared timeline.
        Otherwise, we fall back to sequential concatenation with fixed pauses.
        """
        if self._dialogue_has_offsets(dialogue_turns):
            try:
                return self._synthesize_dialogue_with_overlap(
                    dialogue_turns=dialogue_turns,
                    output_path=output_path,
                    voice_map=voice_map,
                    base_start=base_start,
                )
            except Exception as exc:
                logger.warning(
                    "Dialogue overlap mixing failed (%s). Falling back to linear merge.",
                    exc,
                )

        return self._synthesize_dialogue_linear(
            dialogue_turns=dialogue_turns,
            output_path=output_path,
            voice_map=voice_map,
            pause_ms=pause_ms,
        )

    @staticmethod
    def _dialogue_has_offsets(dialogue_turns: list[dict]) -> bool:
        if not dialogue_turns:
            return False
        for turn in dialogue_turns:
            start = turn.get("start")
            if start is None:
                return False
            try:
                float(start)
            except (TypeError, ValueError):
                return False
        return True

    def _synthesize_dialogue_linear(
        self,
        dialogue_turns: list[dict],
        output_path: str,
        voice_map: dict[str, str],
        pause_ms: int,
    ) -> str:
        """Previous behavior: concatenate turns sequentially with short pauses."""
        destination = Path(output_path)
        audio_segments: list[AudioSegment] = []
        voice_map = voice_map or {}
        timeline_ms = 0

        for idx, turn in enumerate(dialogue_turns):
            text = turn.get("text_ru") or turn.get("text")
            if not text:
                continue

            speaker_id = turn.get("speaker")
            default_voice = getattr(self, "voice_id", None) or getattr(self, "speaker", None)
            voice = voice_map.get(speaker_id) or voice_map.get("__default__") or default_voice

            plan = self._plan_turn_chunks(turn)
            if not plan:
                continue

            logger.info(
                "Synthesizing dialogue turn %s/%s: speaker=%s -> voice=%s, len=%s (linear, %s chunk(s))",
                idx + 1,
                len(dialogue_turns),
                speaker_id,
                voice,
                len(text),
                len(plan),
            )

            seg_audio = self._render_turn_chunks(plan, voice, destination, idx)
            if seg_audio is None:
                continue

            logger.info(
                "Dialogue turn %s rendered at %.2fs duration (linear)",
                idx + 1,
                len(seg_audio) / 1000.0,
            )

            if audio_segments:
                audio_segments.append(AudioSegment.silent(duration=pause_ms))
                timeline_ms += pause_ms

            start_seconds = timeline_ms / 1000.0
            audio_segments.append(seg_audio)
            duration_ms = len(seg_audio)
            timeline_ms += duration_ms

            turn["tts_start_offset"] = start_seconds
            turn["tts_duration"] = duration_ms / 1000.0
            turn["tts_end_offset"] = start_seconds + turn["tts_duration"]

        if not audio_segments:
            raise RuntimeError("No audio generated for dialogue")

        final_audio = audio_segments[0]
        for seg_audio in audio_segments[1:]:
            final_audio += seg_audio

        final_audio.export(str(destination), format="wav")
        logger.info("Dialogue audio saved to %s (linear mode)", output_path)
        return output_path

    def _synthesize_dialogue_with_overlap(
        self,
        dialogue_turns: list[dict],
        output_path: str,
        voice_map: dict[str, str],
        base_start: float | None,
    ) -> str:
        """Place every synthesized phrase on a common timeline to preserve overlaps."""
        destination = Path(output_path)
        voice_map = voice_map or {}
        offsets = [float(turn["start"]) for turn in dialogue_turns if turn.get("start") is not None]
        if not offsets:
            raise ValueError("Dialogue turns missing start timestamps.")

        reference = base_start if base_start is not None else min(offsets)
        reference = float(reference)

        layers: list[tuple[AudioSegment, int]] = []
        max_duration_ms = 0

        for idx, turn in enumerate(dialogue_turns):
            text = turn.get("text_ru") or turn.get("text")
            if not text:
                continue

            start_time = float(turn.get("start", reference))
            offset_ms = max(0, int(round((start_time - reference) * 1000)))

            speaker_id = turn.get("speaker")
            default_voice = getattr(self, "voice_id", None) or getattr(self, "speaker", None)
            voice = voice_map.get(speaker_id) or voice_map.get("__default__") or default_voice

            plan = self._plan_turn_chunks(turn)
            if not plan:
                continue

            logger.info(
                "Synthesizing dialogue turn %s/%s: speaker=%s -> voice=%s, len=%s, offset=%.2fs (%s chunk(s))",
                idx + 1,
                len(dialogue_turns),
                speaker_id,
                voice,
                len(text),
                offset_ms / 1000,
                len(plan),
            )

            seg_audio = self._render_turn_chunks(plan, voice, destination, idx)
            if seg_audio is None:
                continue

            duration_seconds = len(seg_audio) / 1000.0
            relative_start = offset_ms / 1000.0

            diarized_end = turn.get("end")
            if diarized_end is not None:
                try:
                    diarized_end = float(diarized_end)
                except (TypeError, ValueError):
                    diarized_end = None
            if diarized_end is not None:
                relative_end_target = max(0.0, diarized_end - reference)
                if relative_end_target > relative_start:
                    max_allowed = max(self.MIN_CHUNK_DURATION, relative_end_target - relative_start)
                    if duration_seconds > max_allowed + 0.05:
                        clip_ms = int(max_allowed * 1000)
                        seg_audio = seg_audio[:clip_ms]
                        duration_seconds = len(seg_audio) / 1000.0

            turn["tts_start_offset"] = relative_start
            turn["tts_duration"] = duration_seconds
            turn["tts_end_offset"] = relative_start + duration_seconds

            end_ms = offset_ms + len(seg_audio)
            max_duration_ms = max(max_duration_ms, end_ms)
            layers.append((seg_audio, offset_ms))

        if not layers:
            raise RuntimeError("No audio generated for dialogue")

        final_duration = max(max_duration_ms, 1)
        final_audio = AudioSegment.silent(duration=final_duration)

        for seg_audio, offset_ms in layers:
            final_audio = final_audio.overlay(seg_audio, position=offset_ms)

        final_audio.export(str(destination), format="wav")
        logger.info(
            "Dialogue audio saved to %s (overlap mode, duration %.2fs)",
            output_path,
            final_duration / 1000,
        )
        return output_path


class TTSService(BaseTTSService):
    """Text-to-Speech using Silero models."""
    
    def __init__(
        self,
        language: str = "ru",
        speaker: str = "xenia",
        model_version: str = "v3_1_ru",
        device: str = "cuda",
        max_chunk_chars: int = 400,
    ):
        """
        Initialize Silero TTS model.
        
        Args:
            language: Language code (ru, en, etc.)
            speaker: Speaker name for synthesis (aidar, baya, kseniya, xenia, eugene, random)
            model_version: Model version for loading (v3_1_ru, v3_1_en, etc.)
            device: Device to use (cuda, cpu)
            max_chunk_chars: Maximum characters per synthesized chunk
        """
        super().__init__(language=language, max_chunk_chars=max_chunk_chars)
        logger.info(f"Loading Silero TTS model: language={language}, speaker={speaker}, model_version={model_version}")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.language = language
        self.speaker = speaker  # Speaker name for synthesis
        self.sample_rate = 48000
        self.max_chunk_chars = max_chunk_chars
        
        # Load Silero TTS model with retry to survive transient network errors (GitHub 503, etc.)
        max_retries = 3
        retry_delay = 5  # seconds
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Calling torch.hub.load (attempt {attempt}/{max_retries}) "
                    f"with language={language}, model_version={model_version}"
                )
                model_result = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=language,
                    speaker=model_version,  # Use model_version for loading
                    trust_repo=True
                )
                
                logger.info(f"torch.hub.load returned type: {type(model_result)}")
                
                # Handle different return formats from torch.hub.load
                if isinstance(model_result, tuple):
                    logger.info(f"Result is tuple with {len(model_result)} elements")
                    self.model, example_text = model_result
                    logger.info(f"Extracted model type: {type(self.model)}")
                else:
                    self.model = model_result
                    logger.info(f"Result is not tuple, using directly. Type: {type(self.model)}")
                
                # Validate that model was loaded
                if self.model is None:
                    raise ValueError("torch.hub.load returned None - model failed to load")
                
                logger.info(f"Model before .to(device): {type(self.model)}, has apply_tts: {hasattr(self.model, 'apply_tts')}")
                
                if not hasattr(self.model, 'apply_tts'):
                    raise ValueError(f"Loaded object does not have 'apply_tts' method. Type: {type(self.model)}, dir: {dir(self.model)[:10]}")
                
                logger.info("Silero TTS model loaded successfully")
                break

            except HTTPError as http_err:
                last_error = http_err
                logger.warning(
                    f"HTTP error while loading Silero TTS model (attempt {attempt}/{max_retries}): {http_err}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                logger.error("Exhausted retries downloading Silero model from GitHub.")
                raise RuntimeError(
                    "Could not initialize TTS service: remote repository unavailable. "
                    "Please try again in a minute."
                ) from http_err
            except Exception as e:
                last_error = e
                logger.error(f"Failed to load Silero TTS model (attempt {attempt}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                raise RuntimeError(f"Could not initialize TTS service: {e}") from e
        
    def _synthesize_chunk(self, chunk: str, speaker_name: str) -> torch.Tensor:
        """Generate audio for a single chunk with fallback splitting."""
        try:
            audio = self.model.apply_tts(
                text=chunk,
                speaker=speaker_name,
                sample_rate=self.sample_rate
            )
            return audio if isinstance(audio, torch.Tensor) else torch.tensor(audio)
        except ValueError:
            self._log_failed_chunk(chunk, "chunk_value_error")
            fallback_sentences = self._split_for_fallback(chunk)
            if len(fallback_sentences) == 1:
                fallback_sentences = self._force_split(chunk)
                if len(fallback_sentences) == 1:
                    raise

            logger.warning("TTS chunk failed, retrying sentence-by-sentence fallback")
            audio_segments: list[torch.Tensor] = []

            for sentence in fallback_sentences:
                try:
                    sentence_audio = self.model.apply_tts(
                        text=sentence,
                        speaker=speaker_name,
                        sample_rate=self.sample_rate
                    )
                    if not isinstance(sentence_audio, torch.Tensor):
                        sentence_audio = torch.tensor(sentence_audio)
                    audio_segments.append(sentence_audio)
                except ValueError:
                    self._log_failed_chunk(sentence, "fallback_sentence_value_error")
                    safe_preview = sentence[:120].replace("\n", " ")
                    logger.error("Fallback sentence synthesis failed. Preview=%r", safe_preview)
                    continue

            if not audio_segments:
                raise

            return torch.cat(audio_segments, dim=-1)

    def synthesize(self, text: str, output_path: str, speaker: str | None = None) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            speaker: Speaker name (xenia, eugene, etc.)
            
        Returns:
            Path to generated audio file
        """
        try:
            # Validate model is still available
            if self.model is None:
                raise RuntimeError("TTS model is None - model was not loaded correctly")
            
            if not hasattr(self.model, 'apply_tts'):
                raise RuntimeError(f"Model does not have 'apply_tts' method. Model type: {type(self.model)}")
            
            speaker_name = speaker or self.speaker
            logger.info(f"Generating TTS audio for text length: {len(text)} chars, speaker: {speaker_name}")

            clean_text = self._sanitize_text(text)
            if not clean_text:
                raise ValueError("TTS input is empty after sanitization")

            chunks = self._chunk_text(clean_text)
            audio_segments = []

            for idx, chunk in enumerate(chunks, start=1):
                logger.info(f"Synthesizing TTS chunk {idx}/{len(chunks)} (len={len(chunk)} chars)")

                try:
                    chunk_audio = self._synthesize_chunk(chunk, speaker_name)
                except Exception as chunk_exc:
                    safe_preview = chunk[:200].replace("\n", " ")
                    logger.error(
                        "TTS chunk %s failed. Length=%s. Preview=%r",
                        idx,
                        len(chunk),
                        safe_preview,
                    )
                    raise

                audio_segments.append(chunk_audio)

            # Concatenate all chunks
            audio = torch.cat(audio_segments, dim=-1)
            
            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save via soundfile to avoid torchcodec/ffmpeg deps in torchaudio.save
            sf.write(
                file=str(output_path),
                data=audio.squeeze(0).cpu().numpy().T,
                samplerate=self.sample_rate,
                subtype="PCM_16",
            )
            
            logger.info(f"Audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise
    
    def synthesize_and_save(self, text: str, output_path: str, speaker: str | None = None) -> str:
        """Compatibility wrapper used elsewhere in the codebase."""
        return self.synthesize(text, output_path, speaker=speaker)
    
    def get_available_speakers(self) -> list:
        """Get list of available speakers for current language."""
        speakers = {
            'ru': ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'],
            'en': ['lj', 'random'],
        }
        return speakers.get(self.language, ['random'])


class ElevenLabsTTSService(BaseTTSService):
    """Cloud-based TTS powered by ElevenLabs."""

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        language: str = "ru",
        sample_rate: int = 44100,
        max_chunk_chars: int = 500,
        base_url: str = "https://api.elevenlabs.io/v1",
        request_timeout: float = 45.0,
        stability: float = 0.35,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        speaker_boost: bool = True,
        proxy_url: str | None = None,
    ):
        super().__init__(language=language, max_chunk_chars=max_chunk_chars)
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not configured.")
        if not voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID is not configured.")

        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.proxy_url = proxy_url
        self.voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": speaker_boost,
        }

    def _request_audio_chunk(self, text: str, voice_override: str | None = None) -> bytes:
        """Call ElevenLabs API and return raw MP3 bytes."""
        voice_id = voice_override or self.voice_id
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": self.voice_settings,
        }

        try:
            client_kwargs = {"timeout": self.request_timeout}
            transport = None
            if self.proxy_url:
                try:
                    transport = httpx.HTTPTransport(proxy=self.proxy_url)
                except TypeError:
                    transport = None
            if transport:
                client_kwargs["transport"] = transport
            with httpx.Client(**client_kwargs) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as http_exc:
            logger.error(
                "ElevenLabs API error (status=%s): %s",
                http_exc.response.status_code,
                http_exc.response.text,
            )
            raise
        except httpx.HTTPError as exc:
            logger.error("ElevenLabs API request failed: %s", exc)
            raise

    def _decode_audio(self, audio_bytes: bytes) -> AudioSegment:
        """Convert returned MP3 bytes into an AudioSegment."""
        try:
            return AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        except Exception as exc:
            logger.error("Failed to decode ElevenLabs audio chunk: %s", exc)
            raise

    def synthesize(self, text: str, output_path: str, speaker: str | None = None) -> str:
        clean_text = self._sanitize_text(text)
        if not clean_text:
            raise ValueError("TTS input is empty after sanitization")

        chunks = self._chunk_text(clean_text)
        audio_segments: list[AudioSegment] = []
        voice_override = speaker or self.voice_id

        for idx, chunk in enumerate(chunks, start=1):
            logger.info("ElevenLabs: synthesizing chunk %s/%s (len=%s)", idx, len(chunks), len(chunk))
            audio_bytes = self._request_audio_chunk(chunk, voice_override=voice_override)
            segment_audio = self._decode_audio(audio_bytes)
            audio_segments.append(segment_audio)

        if not audio_segments:
            raise RuntimeError("ElevenLabs returned no audio segments")

        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined += segment

        combined = combined.set_frame_rate(self.sample_rate).set_channels(1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path), format="wav")
        logger.info("ElevenLabs audio saved to %s", output_path)
        return str(output_path)

    def synthesize_and_save(self, text: str, output_path: str, speaker: str | None = None) -> str:
        return self.synthesize(text, output_path, speaker=speaker)

