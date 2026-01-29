"""Text-to-Speech services."""
import base64
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

    def _group_words_into_phrases(
        self, words: list[dict], pause_threshold: float = 0.3
    ) -> list[dict]:
        """
        Group words into phrases based on pauses between them.
        
        Args:
            words: List of word dicts with 'word', 'start', 'end'
            pause_threshold: Seconds of pause to split phrases (default 0.3s)
        
        Returns:
            List of phrase dicts with 'text', 'start', 'end', 'duration', 'word_count'
        """
        if not words:
            return []
        
        phrases: list[dict] = []
        current_words: list[dict] = [words[0]]
        phrase_start = words[0].get("start", 0.0)
        phrase_end = words[0].get("end", 0.0)
        
        for i in range(1, len(words)):
            prev_end = words[i - 1].get("end", 0.0)
            curr_start = words[i].get("start", 0.0)
            gap = curr_start - prev_end
            
            if gap > pause_threshold:
                # Finalize current phrase
                phrase_text = " ".join(w.get("word", "") for w in current_words)
                phrases.append({
                    "text": phrase_text,
                    "start": phrase_start,
                    "end": phrase_end,
                    "duration": max(0.1, phrase_end - phrase_start),
                    "word_count": len(current_words),
                    "pause_after": gap,  # Store pause duration for later use
                })
                
                # Start new phrase
                current_words = [words[i]]
                phrase_start = curr_start
                phrase_end = words[i].get("end", 0.0)
            else:
                current_words.append(words[i])
                phrase_end = words[i].get("end", 0.0)
        
        # Don't forget the last phrase
        if current_words:
            phrase_text = " ".join(w.get("word", "") for w in current_words)
            phrases.append({
                "text": phrase_text,
                "start": phrase_start,
                "end": phrase_end,
                "duration": max(0.1, phrase_end - phrase_start),
                "word_count": len(current_words),
                "pause_after": 0.0,
            })
        
        return phrases

    def _calculate_speech_rate_from_words(self, words: list[dict]) -> float:
        """
        Calculate actual words-per-second from word timestamps.
        Returns default rate if words are empty or invalid.
        """
        if not words or len(words) < 2:
            return self.SPEECH_WORDS_PER_SEC
        
        try:
            first_start = words[0].get("start", 0.0)
            last_end = words[-1].get("end", 0.0)
            total_duration = last_end - first_start
            
            if total_duration > 0.1:
                rate = len(words) / total_duration
                # Sanity check: speech rate should be between 1-5 words/sec
                return max(1.0, min(5.0, rate))
        except (TypeError, ValueError):
            pass
        
        return self.SPEECH_WORDS_PER_SEC

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
        
        Now uses word-level timestamps (if available) for better timing estimation.
        """
        text = (turn.get("text_ru") or turn.get("text") or "").strip()
        if not text:
            return []

        # Check for word-level timestamps from WhisperX
        original_words = turn.get("words") or []
        has_word_timestamps = bool(original_words)
        
        if has_word_timestamps:
            # Calculate actual speech rate from original audio
            original_speech_rate = self._calculate_speech_rate_from_words(original_words)
            # Group original words into phrases for pause analysis
            original_phrases = self._group_words_into_phrases(original_words, pause_threshold=0.3)
            logger.debug(
                "Turn has %d word timestamps, speech_rate=%.2f w/s, %d phrases detected",
                len(original_words), original_speech_rate, len(original_phrases)
            )
        else:
            original_speech_rate = self.SPEECH_WORDS_PER_SEC
            original_phrases = []

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
            # Use speech rate from original audio if available
            translated_word_count = max(1, len(text.split()))
            turn_window = max(0.5, translated_word_count / original_speech_rate)

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

        # Track if any chunk duration was clamped by MIN_CHUNK_DURATION
        # This would cause inconsistent speed between chunks
        any_clamped = False

        for idx, chunk in enumerate(plan):
            weight = chunk["word_count"] / total_weight if total_weight else 1 / len(plan)
            ideal_duration = speech_budget * weight
            
            if idx == len(plan) - 1:
                chunk_duration = max(self.MIN_CHUNK_DURATION, remaining)
            else:
                chunk_duration = max(self.MIN_CHUNK_DURATION, ideal_duration)
                if ideal_duration < self.MIN_CHUNK_DURATION:
                    any_clamped = True
                remaining = max(0.0, remaining - chunk_duration)
            chunk["target_duration"] = chunk_duration

        # Warn if speed might be inconsistent between chunks
        if any_clamped and len(plan) > 1:
            logger.debug(
                "Turn has %d chunks with MIN_CHUNK_DURATION override - speed may vary slightly",
                len(plan)
            )

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
    ) -> tuple[AudioSegment | None, list[dict]]:
        """Synthesize per-chunk audio and stitch it with pauses.
        
        IMPORTANT: Speed is UNIFORM across all chunks within a turn.
        
        How it works:
        - Each chunk's target_duration is proportional to its word count
        - ElevenLabs calculates speed = estimated_duration / target_duration
        - Since ratios are the same, speed is identical for all chunks
        
        Example:
        - Turn: 10 words, target = 4s
        - Chunk A (3 words): target = 1.2s → speed = (3/2.5)/1.2 = 1.0
        - Chunk B (7 words): target = 2.8s → speed = (7/2.5)/2.8 = 1.0
        """
        combined_segments: list[AudioSegment] = []
        all_turn_words: list[dict] = []
        speech_target = sum(chunk.get("target_duration", 0.0) for chunk in plan)
        pause_target = sum(chunk.get("pause_ms", 0) for chunk in plan[:-1]) / 1000.0
        turn_target_duration = speech_target + pause_target
        cumulative_duration = 0.0

        for chunk_idx, chunk in enumerate(plan):
            chunk_path = (
                destination.parent
                / f"dialogue_turn_{turn_idx}_chunk_{chunk_idx}_{destination.name}"
            )
            try:
                # Pass chunk's proportional target_duration
                # Speed will be uniform because word_count/target_duration ratio is the same
                chunk_target = chunk.get("target_duration", 0.0)
                _, chunk_words = self.synthesize(chunk["text"], str(chunk_path), speaker=voice, target_duration=chunk_target)
                
                # Adjust word timestamps for cumulative duration (previous chunks + pauses)
                for word in chunk_words:
                    all_turn_words.append({
                        "word": word["word"],
                        "start": word["start"] + cumulative_duration,
                        "end": word["end"] + cumulative_duration,
                    })
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

                # Check if this TTS service wants to skip ffmpeg post-retime
                # (e.g. ElevenLabs has native speed control, ffmpeg sounds unnatural)
                skip_retime = getattr(self, 'SKIP_POST_RETIME', False)
                
                should_retime_chunk = (
                    not skip_retime
                    and target_duration
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
                cumulative_duration += current_duration

                pause_ms = int(chunk.get("pause_ms") or 0)
                if pause_ms > 0 and chunk_idx != len(plan) - 1:
                    combined_segments.append(AudioSegment.silent(duration=pause_ms))
                    cumulative_duration += pause_ms / 1000.0
            finally:
                chunk_path.unlink(missing_ok=True)

        if not combined_segments:
            return None, []

        final_audio = combined_segments[0]
        for seg in combined_segments[1:]:
            final_audio += seg

        actual_duration = len(final_audio) / 1000.0
        
        # Skip final retime for services with native speed control
        skip_retime = getattr(self, 'SKIP_POST_RETIME', False)
        
        if (
            not skip_retime
            and turn_target_duration > 0
            and actual_duration > 0
            and abs(actual_duration - turn_target_duration) > 0.12
        ):
            temp_path = destination.parent / f"dialogue_turn_{turn_idx}_retime_{destination.name}"
            final_audio.export(str(temp_path), format="wav")
            tempo = actual_duration / turn_target_duration
            if self._retime_audio_file(temp_path, tempo):
                final_audio = AudioSegment.from_file(str(temp_path))
                # Adjust word timestamps for retiming
                retime_ratio = actual_duration / turn_target_duration
                for word in all_turn_words:
                    word["start"] = word["start"] / retime_ratio
                    word["end"] = word["end"] / retime_ratio
            temp_path.unlink(missing_ok=True)
        return final_audio, all_turn_words

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

            seg_audio, turn_words = self._render_turn_chunks(plan, voice, destination, idx)
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
            
            # Add word-level timestamps if available (adjusted for turn start time)
            if turn_words:
                tts_words = []
                for w in turn_words:
                    tts_words.append({
                        "word": w["word"],
                        "start": w["start"] + start_seconds,
                        "end": w["end"] + start_seconds,
                    })
                turn["tts_words"] = tts_words
                logger.debug(
                    "TTS turn %d: %d words with timestamps (first: %.2f-%.2fs, last: %.2f-%.2fs)",
                    idx,
                    len(tts_words),
                    tts_words[0]["start"] if tts_words else 0,
                    tts_words[0]["end"] if tts_words else 0,
                    tts_words[-1]["start"] if tts_words else 0,
                    tts_words[-1]["end"] if tts_words else 0,
                )

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

            seg_audio, turn_words = self._render_turn_chunks(plan, voice, destination, idx)
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
                        # Adjust word timestamps proportionally if audio is clipped
                        clip_ratio = max_allowed / duration_seconds
                        seg_audio = seg_audio[:clip_ms]
                        duration_seconds = len(seg_audio) / 1000.0
                        # Clip word timestamps to match clipped audio
                        for word in turn_words:
                            if word["end"] > max_allowed:
                                word["end"] = max_allowed
                            if word["start"] > max_allowed:
                                word["start"] = max_allowed

            turn["tts_start_offset"] = relative_start
            turn["tts_duration"] = duration_seconds
            turn["tts_end_offset"] = relative_start + duration_seconds
            
            # Add word-level timestamps if available (adjusted for turn start time)
            if turn_words:
                tts_words = []
                for w in turn_words:
                    tts_words.append({
                        "word": w["word"],
                        "start": w["start"] + relative_start,
                        "end": w["end"] + relative_start,
                    })
                turn["tts_words"] = tts_words
                logger.debug(
                    "TTS turn %d (overlap): %d words with timestamps (first: %.2f-%.2fs, last: %.2f-%.2fs)",
                    idx,
                    len(tts_words),
                    tts_words[0]["start"] if tts_words else 0,
                    tts_words[0]["end"] if tts_words else 0,
                    tts_words[-1]["start"] if tts_words else 0,
                    tts_words[-1]["end"] if tts_words else 0,
                )

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

    def synthesize(
        self, text: str, output_path: str, speaker: str | None = None, target_duration: float | None = None
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            speaker: Speaker name (xenia, eugene, etc.)
            target_duration: Ignored for local TTS (handled by ffmpeg post-processing)
            
        Returns:
            Path to generated audio file
        """
        # Note: target_duration is ignored here - speed adjustment done via ffmpeg in BaseTTSService
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
    
    # Skip ffmpeg post-processing - ElevenLabs has native speed control
    # Aggressive tempo changes (> 1.2x) sound unnatural, better have slight desync
    SKIP_POST_RETIME = True

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
        stability: float = 0.5,  # TTD requires 0.0, 0.5, or 1.0
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

    def _request_audio_chunk(
        self, text: str, voice_override: str | None = None, speed: float = 1.0
    ) -> bytes:
        """Call ElevenLabs API and return raw MP3 bytes.
        
        Args:
            text: Text to synthesize
            voice_override: Optional voice ID to use instead of default
            speed: Speech speed multiplier (0.7-1.2, default 1.0)
        """
        voice_id = voice_override or self.voice_id
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        # Only add speed if different from default (1.0) - we use FFmpeg for tempo adjustment
        voice_settings_final = {**self.voice_settings}
        if abs(speed - 1.0) > 0.02:
            voice_settings_final["speed"] = max(0.7, min(1.2, speed))
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": voice_settings_final,
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

    def _request_audio_with_timestamps(
        self, text: str, voice_override: str | None = None, speed: float = 1.0
    ) -> tuple[bytes, list[dict]]:
        """Call ElevenLabs TTS API with timestamps and return audio + word timings.
        
        Uses /text-to-speech/{voice_id}/with-timestamps endpoint.
        
        Args:
            text: Text to synthesize
            voice_override: Optional voice ID to use instead of default
            speed: Speech speed multiplier (0.7-1.2, default 1.0)
            
        Returns:
            Tuple of (audio_bytes, word_timestamps)
            word_timestamps: [{"word": "Hello", "start": 0.1, "end": 0.4}, ...]
        """
        voice_id = voice_override or self.voice_id
        url = f"{self.base_url}/text-to-speech/{voice_id}/with-timestamps"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        # Only add speed if different from default (1.0) - we use FFmpeg for tempo adjustment
        voice_settings_final = {**self.voice_settings}
        if abs(speed - 1.0) > 0.02:
            voice_settings_final["speed"] = max(0.7, min(1.2, speed))
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": voice_settings_final,
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
            
            # Parse JSON response with audio (base64) and alignment
            data = response.json()
            audio_base64 = data.get("audio_base64", "")
            alignment = data.get("alignment", {})
            
            # Decode audio from base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Parse alignment into word timestamps
            words = self._parse_single_speaker_alignment(alignment)
            
            logger.info(
                "ElevenLabs TTS with timestamps: %d chars, %d words parsed",
                len(text), len(words)
            )
            
            return audio_bytes, words
            
        except httpx.HTTPStatusError as http_exc:
            logger.error(
                "ElevenLabs TTS with-timestamps API error (status=%s): %s",
                http_exc.response.status_code,
                http_exc.response.text,
            )
            raise
        except httpx.HTTPError as exc:
            logger.error("ElevenLabs TTS with-timestamps API request failed: %s", exc)
            raise

    @staticmethod
    def _parse_single_speaker_alignment(alignment: dict) -> list[dict]:
        """Parse ElevenLabs character alignment into word-level timestamps.
        
        Args:
            alignment: Dict with characters, character_start_times_seconds, character_end_times_seconds
            
        Returns:
            List of word dicts: [{"word": "Hello", "start": 0.1, "end": 0.4}, ...]
        """
        characters = alignment.get("characters", [])
        char_starts = alignment.get("character_start_times_seconds", [])
        char_ends = alignment.get("character_end_times_seconds", [])
        
        if not characters or len(characters) != len(char_starts) or len(characters) != len(char_ends):
            logger.warning(
                "TTS alignment data incomplete: chars=%d, starts=%d, ends=%d",
                len(characters), len(char_starts), len(char_ends)
            )
            return []
        
        words = []
        current_word = []
        word_start_idx = 0
        
        for i, char in enumerate(characters):
            if char == " " or char == "\n":
                # Word boundary - save the word if we have one
                if current_word:
                    word_text = "".join(current_word)
                    word_start = char_starts[word_start_idx]
                    word_end = char_ends[i - 1] if i > 0 else char_starts[word_start_idx]
                    
                    words.append({
                        "word": word_text,
                        "start": word_start,
                        "end": word_end,
                    })
                    
                    current_word = []
                word_start_idx = i + 1
            else:
                current_word.append(char)
        
        # Don't forget the last word
        if current_word:
            word_text = "".join(current_word)
            word_start = char_starts[word_start_idx] if word_start_idx < len(char_starts) else 0
            word_end = char_ends[-1] if char_ends else word_start
            
            words.append({
                "word": word_text,
                "start": word_start,
                "end": word_end,
            })
        
        if words:
            logger.info(
                "TTS parsed %d words, range: %.2f-%.2fs",
                len(words), words[0]["start"], words[-1]["end"]
            )
        
        return words

    def _decode_audio(self, audio_bytes: bytes) -> AudioSegment:
        """Convert returned MP3 bytes into an AudioSegment."""
        try:
            return AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        except Exception as exc:
            logger.error("Failed to decode ElevenLabs audio chunk: %s", exc)
            raise

    def synthesize(
        self, text: str, output_path: str, speaker: str | None = None, target_duration: float | None = None
    ) -> tuple[str, list[dict]]:
        """Synthesize text with word-level timestamps.
        
        Returns:
            Tuple of (output_path, word_timestamps)
            word_timestamps: [{"word": "Hello", "start": 0.1, "end": 0.4}, ...]
        """
        clean_text = self._sanitize_text(text)
        if not clean_text:
            raise ValueError("TTS input is empty after sanitization")

        chunks = self._chunk_text(clean_text)
        audio_segments: list[AudioSegment] = []
        all_words: list[dict] = []
        voice_override = speaker or self.voice_id
        
        # Speed adjustment is handled by FFmpeg tempo in video.py (0.7-1.25x range)
        # ElevenLabs speed parameter is unreliable, so we generate at natural speed
        speed = 1.0

        # Track cumulative duration for multi-chunk word timestamp adjustment
        cumulative_duration = 0.0
        
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("ElevenLabs: synthesizing chunk %s/%s (len=%s) with timestamps", idx, len(chunks), len(chunk))
            
            try:
                audio_bytes, chunk_words = self._request_audio_with_timestamps(
                    chunk, voice_override=voice_override, speed=speed
                )
            except Exception as e:
                # Fallback to non-timestamp endpoint if with-timestamps fails
                logger.warning("TTS with-timestamps failed, falling back: %s", e)
                audio_bytes = self._request_audio_chunk(chunk, voice_override=voice_override, speed=speed)
                chunk_words = []
            
            segment_audio = self._decode_audio(audio_bytes)
            audio_segments.append(segment_audio)
            
            # Adjust word timestamps for multi-chunk scenario
            for word in chunk_words:
                all_words.append({
                    "word": word["word"],
                    "start": word["start"] + cumulative_duration,
                    "end": word["end"] + cumulative_duration,
                })
            
            cumulative_duration += len(segment_audio) / 1000.0  # AudioSegment length is in ms

        if not audio_segments:
            raise RuntimeError("ElevenLabs returned no audio segments")

        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined += segment

        combined = combined.set_frame_rate(self.sample_rate).set_channels(1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path), format="wav")
        
        logger.info(
            "ElevenLabs audio saved to %s (duration=%.2fs, %d words with timestamps)",
            output_path, len(combined) / 1000.0, len(all_words)
        )
        
        return str(output_path), all_words

    def synthesize_and_save(
        self, text: str, output_path: str, speaker: str | None = None, target_duration: float | None = None
    ) -> tuple[str, list[dict]]:
        """Synthesize and save with word timestamps.
        
        Args:
            text: Text to synthesize
            output_path: Where to save the audio file
            speaker: Optional voice ID override
            target_duration: Optional target duration for speed control
        
        Returns:
            Tuple of (output_path, word_timestamps)
        """
        return self.synthesize(text, output_path, speaker=speaker, target_duration=target_duration)


class ElevenLabsTTDService(ElevenLabsTTSService):
    """
    ElevenLabs Text-to-Dialogue service using Eleven v3 model.
    
    Advantages over regular TTS:
    - Synthesizes entire dialogue in ONE API call
    - Supports audio tags for emotions: [cheerfully], [sad], [laughing]
    - More natural transitions between speakers
    - No need for manual audio stitching
    
    Reference: https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert
    """
    
    @staticmethod
    def _parse_alignment_to_words(
        alignment: dict,
        voice_segments: list[dict],
    ) -> dict[int, list[dict]]:
        """
        Parse ElevenLabs character alignment into word-level timestamps.
        
        Can use either 'alignment' (original text) or 'normalized_alignment' (normalized text).
        normalized_alignment is preferred as it may be more accurate after text normalization.
        
        Reference: https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert-with-timestamps
        
        Returns a dict mapping dialogue_input_index to list of word dicts:
        {0: [{"word": "Привет", "start": 0.1, "end": 0.5}, ...], ...}
        """
        if not alignment:
            logger.warning("TTD alignment is empty or None")
            return {}
        
        characters = alignment.get("characters", [])
        char_starts = alignment.get("character_start_times_seconds", [])
        char_ends = alignment.get("character_end_times_seconds", [])
        
        if not characters or len(characters) != len(char_starts) or len(characters) != len(char_ends):
            logger.warning(
                "TTD alignment data incomplete: chars=%d, starts=%d, ends=%d",
                len(characters), len(char_starts), len(char_ends)
            )
            return {}
        
        logger.debug(
            "TTD parsing alignment: %d characters, %d voice_segments",
            len(characters), len(voice_segments)
        )
        
        # Build character-to-segment mapping from voice_segments
        # voice_segments have character_start_index and character_end_index
        # These indices point to characters array in the alignment (or normalized_alignment)
        segment_ranges = []
        max_char_idx = len(characters) - 1
        
        for vs in voice_segments:
            char_start = vs.get("character_start_index", 0)
            char_end = vs.get("character_end_index", 0)
            
            # Validate indices are within bounds
            if char_start < 0 or char_end > len(characters):
                logger.warning(
                    "TTD voice_segment: invalid char_range [%d, %d) for input_idx=%d (max=%d), clamping",
                    char_start, char_end, vs.get("dialogue_input_index", -1), max_char_idx
                )
                char_start = max(0, min(char_start, max_char_idx))
                char_end = max(char_start, min(char_end, len(characters)))
            
            segment_ranges.append({
                "input_idx": vs.get("dialogue_input_index", -1),
                "char_start": char_start,
                "char_end": char_end,
            })
        
        # Log segment ranges for debugging
        for sr in segment_ranges:
            logger.debug(
                "TTD voice_segment: input_idx=%d, char_range=[%d, %d), chars_in_range=%d",
                sr["input_idx"], sr["char_start"], sr["char_end"],
                sr["char_end"] - sr["char_start"]
            )
        
        def get_input_idx_for_char(char_idx: int) -> int:
            """Find which dialogue input a character belongs to."""
            for sr in segment_ranges:
                if sr["char_start"] <= char_idx < sr["char_end"]:
                    return sr["input_idx"]
            return -1
        
        # Parse characters into words with timestamps
        # Also split words at segment boundaries (ElevenLabs concatenates inputs without spaces)
        words_by_input: dict[int, list[dict]] = {}
        current_word = []
        word_start_idx = 0
        prev_input_idx = -1
        
        for i, char in enumerate(characters):
            current_input_idx = get_input_idx_for_char(i)
            
            # Check if we crossed a segment boundary
            crossed_boundary = (prev_input_idx >= 0 and 
                               current_input_idx >= 0 and 
                               current_input_idx != prev_input_idx)
            
            if char == " " or char == "\n" or crossed_boundary:
                # Word boundary - save the word if we have one
                if current_word:
                    word_text = "".join(current_word)
                    word_start = char_starts[word_start_idx]
                    word_end = char_ends[i - 1] if i > 0 else char_starts[word_start_idx]
                    
                    input_idx = get_input_idx_for_char(word_start_idx)
                    if input_idx >= 0:
                        if input_idx not in words_by_input:
                            words_by_input[input_idx] = []
                        words_by_input[input_idx].append({
                            "word": word_text,
                            "start": word_start,
                            "end": word_end,
                        })
                    
                    current_word = []
                
                # If crossed boundary (not space), start new word with current char
                if crossed_boundary and char != " " and char != "\n":
                    word_start_idx = i
                    current_word.append(char)
                else:
                    word_start_idx = i + 1
            else:
                current_word.append(char)
            
            if current_input_idx >= 0:
                prev_input_idx = current_input_idx
        
        # Don't forget the last word
        if current_word:
            word_text = "".join(current_word)
            word_start = char_starts[word_start_idx] if word_start_idx < len(char_starts) else 0
            word_end = char_ends[-1] if char_ends else word_start
            
            input_idx = get_input_idx_for_char(word_start_idx)
            if input_idx >= 0:
                if input_idx not in words_by_input:
                    words_by_input[input_idx] = []
                words_by_input[input_idx].append({
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                })
        
        # Log parsed words count (debug level for detailed logs)
        total_words = sum(len(w) for w in words_by_input.values())
        logger.debug(
            "TTD parsed %d words from alignment across %d inputs",
            total_words,
            len(words_by_input),
        )
        
        # Log first 3 words of each input for debugging
        for input_idx, words in sorted(words_by_input.items()):
            first_words = [w["word"] for w in words[:3]]
            last_words = [w["word"] for w in words[-2:]] if len(words) > 3 else []
            logger.debug(
                "TTD input %d: %d words, first=[%s], last=[%s]",
                input_idx,
                len(words),
                ", ".join(first_words),
                ", ".join(last_words),
            )
        
        return words_by_input

    @staticmethod
    def _validate_and_fix_timings_with_voice_segments(
        words_by_input: dict[int, list[dict]],
        voice_segments: list[dict],
    ) -> dict[int, list[dict]]:
        """
        Validate and fix word timings using voice_segments as a reference.
        
        voice_segments contain precise start_time_seconds and end_time_seconds for each segment,
        which can be used to validate and correct character-level timings from alignment.
        
        Reference: https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert-with-timestamps
        
        Args:
            words_by_input: Dict mapping dialogue_input_index to list of word dicts
            voice_segments: List of voice segment dicts with start_time_seconds, end_time_seconds,
                          dialogue_input_index, character_start_index, character_end_index
        
        Returns:
            Validated and corrected words_by_input dict
        """
        if not voice_segments or not words_by_input:
            return words_by_input
        
        # Build mapping from dialogue_input_index to segment timing
        segment_timings = {}
        for vs in voice_segments:
            input_idx = vs.get("dialogue_input_index", -1)
            if input_idx >= 0:
                # If multiple segments per input, use the earliest start and latest end
                if input_idx not in segment_timings:
                    segment_timings[input_idx] = {
                        "start": vs.get("start_time_seconds", 0),
                        "end": vs.get("end_time_seconds", 0),
                    }
                else:
                    segment_timings[input_idx]["start"] = min(
                        segment_timings[input_idx]["start"],
                        vs.get("start_time_seconds", 0)
                    )
                    segment_timings[input_idx]["end"] = max(
                        segment_timings[input_idx]["end"],
                        vs.get("end_time_seconds", 0)
                    )
        
        # Validate and fix word timings for each input
        corrected_words = {}
        for input_idx, words in words_by_input.items():
            if not words:
                continue
            
            segment_timing = segment_timings.get(input_idx)
            if not segment_timing:
                # No segment timing available, keep original words
                corrected_words[input_idx] = words
                continue
            
            segment_start = segment_timing["start"]
            segment_end = segment_timing["end"]
            segment_duration = segment_end - segment_start
            
            # Check if word timings are within segment boundaries
            first_word_start = words[0].get("start", segment_start)
            last_word_end = words[-1].get("end", segment_end)
            
            # If word timings are outside segment boundaries, scale them proportionally
            if first_word_start < segment_start or last_word_end > segment_end:
                # Calculate scale factor to fit words within segment
                word_span = last_word_end - first_word_start
                if word_span > 0.01:  # Avoid division by zero
                    scale = segment_duration / word_span
                    offset = segment_start - (first_word_start * scale)
                    
                    logger.debug(
                        "TTD: Scaling word timings for input %d: "
                        "words=[%.2f-%.2f]s, segment=[%.2f-%.2f]s, scale=%.3f",
                        input_idx, first_word_start, last_word_end,
                        segment_start, segment_end, scale
                    )
                    
                    # Scale all word timings
                    corrected_words[input_idx] = []
                    for word in words:
                        corrected_words[input_idx].append({
                            "word": word.get("word", ""),
                            "start": max(segment_start, min(segment_end, word.get("start", 0) * scale + offset)),
                            "end": max(segment_start, min(segment_end, word.get("end", 0) * scale + offset)),
                        })
                else:
                    # Word span too small, distribute evenly
                    corrected_words[input_idx] = []
                    word_duration = segment_duration / len(words) if words else 0.1
                    for i, word in enumerate(words):
                        corrected_words[input_idx].append({
                            "word": word.get("word", ""),
                            "start": segment_start + i * word_duration,
                            "end": segment_start + (i + 1) * word_duration,
                        })
            else:
                # Timings are within bounds, but clamp to be safe
                corrected_words[input_idx] = []
                for word in words:
                    corrected_words[input_idx].append({
                        "word": word.get("word", ""),
                        "start": max(segment_start, min(segment_end, word.get("start", segment_start))),
                        "end": max(segment_start, min(segment_end, word.get("end", segment_end))),
                    })
        
        return corrected_words

    # Audio tags for emotional delivery
    EMOTION_TAGS = {
        # Positive emotions
        "happy": "[cheerfully]",
        "excited": "[excitedly]",
        "enthusiastic": "[enthusiastically]",
        "laughing": "[laughing]",
        "amused": "[amused]",
        
        # Negative emotions
        "sad": "[sadly]",
        "angry": "[angrily]",
        "frustrated": "[frustrated]",
        "disappointed": "[disappointed]",
        
        # Neutral/other
        "thoughtful": "[thoughtfully]",
        "curious": "[curiously]",
        "surprised": "[surprised]",
        "hesitant": "[hesitantly]",
        "confident": "[confidently]",
        "whispering": "[whispering]",
        "shouting": "[shouting]",
        "sarcastic": "[sarcastically]",
    }
    
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        language: str = "ru",
        sample_rate: int = 44100,
        base_url: str = "https://api.elevenlabs.io/v1",
        request_timeout: float = 120.0,  # TTD may take longer
        proxy_url: str | None = None,
    ):
        # Initialize parent with eleven_v3 model (required for TTD)
        super().__init__(
            api_key=api_key,
            voice_id=voice_id,
            model_id="eleven_v3",  # TTD requires v3 model
            language=language,
            sample_rate=sample_rate,
            base_url=base_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
        )
        logger.info("ElevenLabsTTDService initialized with eleven_v3 model")

    def _add_emotion_tag(self, text: str, emotion: str | None = None) -> str:
        """
        Add audio tag for emotion based on punctuation or explicit emotion.
        
        Examples:
            "Hello!" → "[excitedly] Hello!"
            "Oh no..." → "[sadly] Oh no..."
            "What?!" → "[surprised] What?!"
        """
        text = text.strip()
        if not text:
            return text
        
        # If explicit emotion provided, use it
        if emotion and emotion.lower() in self.EMOTION_TAGS:
            return f"{self.EMOTION_TAGS[emotion.lower()]} {text}"
        
        # Infer emotion from punctuation
        if text.endswith("?!") or text.endswith("!?"):
            return f"[surprised] {text}"
        elif text.endswith("!"):
            return f"[enthusiastically] {text}"
        elif text.endswith("..."):
            return f"[thoughtfully] {text}"
        elif text.endswith("?"):
            return f"[curiously] {text}"
        
        # Check for laughing indicators
        if any(laugh in text.lower() for laugh in ["хаха", "ахах", "lol", "haha", "хех"]):
            return f"[laughing] {text}"
        
        return text

    def _log_dialogue_timeline(self, dialogue_turns: list[dict]) -> None:
        """
        Log timeline comparison between original and translated dialogue for debugging.
        Speed adjustment is handled by FFmpeg tempo in video.py.
        """
        # Log full timeline for debugging
        logger.debug("=" * 70)
        logger.debug("TTD TIMELINE COMPARISON")
        logger.debug("=" * 70)
        
        total_original_duration = 0.0
        total_translated_words = 0
        
        for idx, turn in enumerate(dialogue_turns):
            start = turn.get("start", 0.0)
            end = turn.get("end", 0.0)
            duration = end - start if end > start else 0.0
            
            text_en = turn.get("text") or ""
            text_ru = turn.get("text_ru") or text_en
            speaker = turn.get("speaker", "?")
            
            words_en = len(text_en.split())
            words_ru = len(text_ru.split())
            
            # Estimate natural duration for Russian (2.5 words/sec)
            est_ru_duration = words_ru / 2.5 if words_ru > 0 else 0.0
            
            logger.debug(
                "Turn %d [%s] %.1f-%.1fs (%.1fs):",
                idx, speaker, start, end, duration
            )
            logger.debug("  EN (%d words): %s", words_en, text_en[:80] + ("..." if len(text_en) > 80 else ""))
            logger.debug("  RU (%d words): %s", words_ru, text_ru[:80] + ("..." if len(text_ru) > 80 else ""))
            logger.debug("  Original: %.1fs | RU natural: %.1fs | Diff: %+.1fs", duration, est_ru_duration, est_ru_duration - duration)
            
            if duration > 0:
                total_original_duration += duration
            total_translated_words += words_ru
        
        logger.debug("-" * 70)
        
        if total_original_duration <= 0 or total_translated_words <= 0:
            logger.debug("TTD: No valid durations for timeline comparison")
            return
        
        # Estimate natural duration for Russian at ~1.8 words/sec (ElevenLabs actual rate)
        estimated_natural_duration = total_translated_words / 1.8
        
        # Calculate what FFmpeg tempo will need to apply
        required_tempo = estimated_natural_duration / total_original_duration
        
        logger.debug(
            "TTD TOTAL: %d RU words | Original: %.1fs | Est. generated: %.1fs | Required tempo: %.2fx",
            total_translated_words, total_original_duration, estimated_natural_duration, required_tempo
        )
        if required_tempo > 1.25:
            logger.warning("TTD: Audio will be clamped to 1.25x tempo (%.1fs extra)", 
                          estimated_natural_duration - total_original_duration * 1.25)
        logger.debug("=" * 70)

    def _calculate_gaps(self, dialogue_turns: list[dict]) -> list[float]:
        """
        Calculate gaps (pauses) between dialogue turns based on RELATIVE timestamps.
        
        Returns list of gap durations in seconds (one per turn, gap BEFORE the turn).
        First turn's gap is always 0 (we start from the first turn).
        
        Note: Timestamps in turns are ABSOLUTE (from video start), so we convert
        them to relative by using the first turn's start as base.
        """
        if not dialogue_turns:
            return []
        
        gaps = []
        
        # Use first turn's start as the base (segment start)
        segment_start = dialogue_turns[0].get("start", 0.0)
        prev_end = segment_start  # First turn starts at relative 0
        
        for i, turn in enumerate(dialogue_turns):
            start = turn.get("start", prev_end)
            
            if i == 0:
                # First turn: no leading gap (starts at 0)
                gaps.append(0.0)
            else:
                # Gap between previous turn end and this turn start
                gap = max(0.0, start - prev_end)
                gaps.append(gap)
            
            prev_end = turn.get("end", start)
        
        return gaps

    def _prepare_ttd_inputs(
        self,
        dialogue_turns: list[dict],
        voice_map: dict[str, str],
        add_emotions: bool = True,
        speed: float = 1.0,
    ) -> tuple[list[dict], list[float]]:
        """
        Convert dialogue turns to TTD API format.
        
        Input format (our internal):
            [{"speaker": "SPEAKER_00", "text_ru": "Привет!", "emotion": "happy"}, ...]
        
        Output format (TTD API):
            [{"text": "[cheerfully] Привет!", "voice_id": "abc123", "voice_settings": {...}}, ...]
            
        Returns:
            Tuple of (inputs list, gaps list for post-processing)
        """
        inputs = []
        default_voice = self.voice_id
        
        # Calculate gaps between turns for post-processing
        gaps = self._calculate_gaps(dialogue_turns)
        
        # Log gaps for debugging
        total_gap = sum(gaps)
        if total_gap > 0.5:
            logger.info("TTD: Total gaps between turns: %.1fs", total_gap)
            for i, gap in enumerate(gaps):
                if gap > 0.1:
                    logger.info("  Gap before turn %d: %.2fs", i, gap)
        
        # Voice settings with speed (same for all turns to maintain consistent rhythm)
        voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
        if abs(speed - 1.0) > 0.02:
            voice_settings["speed"] = speed
        
        for idx, turn in enumerate(dialogue_turns):
            text_ru = turn.get("text_ru")
            text_en = turn.get("text", "")
            
            # CRITICAL: Use translated text, warn if missing
            if text_ru and text_ru.strip():
                text = text_ru
            else:
                text = text_en
                if text_en.strip():
                    logger.warning(
                        "TTD turn %d: MISSING text_ru! Using original EN: '%s...'",
                        idx, text_en[:40]
                    )
            
            if not text.strip():
                continue
            
            speaker_id = turn.get("speaker")
            voice_id = voice_map.get(speaker_id) or voice_map.get("__default__") or default_voice
            
            # Add emotion tags if enabled
            if add_emotions:
                emotion = turn.get("emotion")
                text = self._add_emotion_tag(text, emotion)
            
            inp = {
                "text": text,
                "voice_id": voice_id,
            }
            
            # Add voice_settings only if we're adjusting speed
            if abs(speed - 1.0) > 0.02:
                inp["voice_settings"] = voice_settings
            
            inputs.append(inp)
        
        return inputs, gaps

    def _call_ttd_api(self, inputs: list[dict]) -> tuple[bytes, list[dict], dict]:
        """
        Make a single TTD API call.
        
        Returns:
            Tuple of (audio_bytes, voice_segments, alignment)
        """
        import base64
        
        url = f"{self.base_url}/text-to-dialogue/with-timestamps"
        headers = {
            "xi-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": inputs,
            "output_format": "mp3_44100_128",
            "apply_text_normalization": "off",  # Disable to preserve abbreviations like "ИИ" (AI)
        }
        
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
            response = client.post(url, headers=headers, json=payload)
        
        response.raise_for_status()
        
        response_data = response.json()
        audio_base64 = response_data.get("audio_base64", "")
        voice_segments = response_data.get("voice_segments", [])
        
        # Since apply_text_normalization is "off", we should use alignment (not normalized_alignment)
        # normalized_alignment contains timestamps for normalized text, which won't match our original text
        # Reference: https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert-with-timestamps
        normalized_alignment = response_data.get("normalized_alignment")
        alignment = response_data.get("alignment", {})
        
        # Always use alignment when normalization is disabled
        # normalized_alignment would have timestamps for normalized text that doesn't match our input
        final_alignment = alignment
        
        if normalized_alignment and not alignment:
            # Fallback: if alignment is missing but normalized_alignment exists, use it
            logger.warning("TTD: alignment missing, falling back to normalized_alignment (may cause sync issues)")
            final_alignment = normalized_alignment
        else:
            logger.info("TTD: Using alignment (text normalization disabled)")
        
        audio_bytes = base64.b64decode(audio_base64)
        
        return audio_bytes, voice_segments, final_alignment

    def _get_speaker_rate(
        self,
        voice_id: str | None,
        segment_timing: dict[int, dict],
        inputs: list[dict],
        dialogue_turns: list[dict],
    ) -> float:
        """
        Calculate personal speech rate for a speaker based on their valid turns.
        
        Instead of using fixed 2.5 words/sec, we analyze turns where this speaker
        has valid timing and calculate their actual speech rate.
        
        Args:
            voice_id: The voice ID to get rate for
            segment_timing: Dict mapping input_idx to {start, end} timing
            inputs: List of TTD inputs with voice_id and text
            dialogue_turns: Original dialogue turns
            
        Returns:
            Speech rate in words/second for this speaker (or default 2.5 if no data)
        """
        DEFAULT_RATE = 2.5  # Fallback rate
        
        if not voice_id or not segment_timing or not inputs:
            return DEFAULT_RATE
        
        # Collect word counts and durations for this speaker's valid turns
        total_words = 0
        total_duration = 0.0
        
        for idx, inp in enumerate(inputs):
            if inp.get("voice_id") != voice_id:
                continue
            
            timing = segment_timing.get(idx, {})
            duration = timing.get("end", 0) - timing.get("start", 0)
            
            # Only use turns with valid timing (> 0.1s)
            if duration < 0.1:
                continue
            
            text = inp.get("text", "")
            # Remove emotion tags like [cheerfully] before counting words
            clean_text = re.sub(r'\[[\w]+\]\s*', '', text)
            word_count = len(clean_text.split())
            
            if word_count > 0:
                total_words += word_count
                total_duration += duration
        
        if total_words > 0 and total_duration > 0.1:
            rate = total_words / total_duration
            # Sanity check: speech rate should be between 1.5-4.0 words/sec
            rate = max(1.5, min(4.0, rate))
            logger.debug(
                "Speaker %s rate: %.2f words/sec (from %d words in %.2fs)",
                voice_id[:8] if voice_id else "?", rate, total_words, total_duration
            )
            return rate
        
        logger.debug(
            "Speaker %s: no valid timing data, using default rate %.2f w/s",
            voice_id[:8] if voice_id else "?", DEFAULT_RATE
        )
        return DEFAULT_RATE

    def _fix_bunched_word_timestamps(
        self, words: list[dict], turn_start: float, turn_end: float
    ) -> list[dict]:
        """
        Fix word timestamps where multiple words are bunched together.
        ElevenLabs sometimes returns compressed timestamps that cause subtitle overlaps.
        
        Algorithm:
        PASS 0: If entire turn has clearly wrong timestamps (>10 words, <100ms each),
                redistribute ALL words evenly across turn_duration
        PASS 1: Find groups of words with very close start times (< 200ms apart)
        PASS 2: Redistribute within turn window to maintain audio sync
        """
        if not words or len(words) < 2:
            return words
        
        fixed = [dict(w) for w in words]  # Deep copy
        
        MIN_WORD_DURATION = 0.15  # 150ms target per word for readability
        BUNCH_THRESHOLD = 0.20    # Words within 200ms are considered bunched
        MAX_BORROW_RATIO = 0.5    # Can borrow up to 50% of gap after bunched group
        CLEARLY_WRONG_THRESHOLD = 0.10  # 100ms - if per_word < this, timestamps are wrong
        MIN_WORDS_FOR_FULL_REDISTRIBUTION = 8  # Only redistribute if turn has >= 8 words
        
        # PASS 0: Check if entire turn has clearly wrong timestamps
        # This happens when ElevenLabs returns very compressed character-level timestamps
        turn_duration = turn_end - turn_start
        word_count = len(fixed)
        
        if word_count >= MIN_WORDS_FOR_FULL_REDISTRIBUTION and turn_duration > 0:
            # Calculate total time span that words occupy according to timestamps
            first_word_start = fixed[0]["start"]
            last_word_end = fixed[-1]["end"]
            timestamps_span = last_word_end - first_word_start
            
            per_word_from_timestamps = timestamps_span / word_count if word_count > 0 else 0
            per_word_from_turn = turn_duration / word_count
            
            # If timestamps give <100ms per word but turn_duration would give >=150ms,
            # the timestamps are clearly wrong - redistribute evenly
            if per_word_from_timestamps < CLEARLY_WRONG_THRESHOLD and per_word_from_turn >= MIN_WORD_DURATION:
                logger.warning(
                    "SUBTITLE FIX: %d words bunched (%.0fms/word) → redistributed (%.0fms/word)",
                    word_count, per_word_from_timestamps * 1000, per_word_from_turn * 1000
                )
                
                # Redistribute ALL words evenly across turn_duration
                for i in range(word_count):
                    fixed[i]["start"] = turn_start + i * per_word_from_turn
                    fixed[i]["end"] = turn_start + (i + 1) * per_word_from_turn
                
                return fixed
        
        # PASS 1: Find bunched groups and redistribute within available window
        i = 0
        while i < len(fixed):
            # Find group of words bunched together (close start times)
            group_start = i
            last_start = fixed[i]["start"]
            
            while i + 1 < len(fixed):
                next_start = fixed[i + 1]["start"]
                if next_start - last_start < BUNCH_THRESHOLD:
                    last_start = next_start
                    i += 1
                else:
                    break
            
            group_end = i
            group_size = group_end - group_start + 1
            
            if group_size > 1:
                # Found a group of bunched words
                window_start = fixed[group_start]["start"]
                bunched_end = fixed[group_end]["end"]  # Original end of last bunched word
                
                # Determine available window end
                if group_end + 1 < len(fixed):
                    next_word_start = fixed[group_end + 1]["start"]
                    # Calculate gap between bunched group and next word
                    gap_after = next_word_start - bunched_end
                    
                    # Borrow up to MAX_BORROW_RATIO of the gap (to maintain some pause)
                    borrow_amount = min(gap_after * MAX_BORROW_RATIO, gap_after - 0.05)
                    borrow_amount = max(0, borrow_amount)  # Don't borrow negative
                    
                    window_end = bunched_end + borrow_amount
                    # But don't exceed next word's start
                    window_end = min(window_end, next_word_start - 0.02)
                else:
                    # Last group in turn - can extend to turn_end
                    window_end = turn_end
                
                window_duration = window_end - window_start
                per_word = window_duration / group_size if window_duration > 0 else 0.1
                
                # CRITICAL: If per_word is still < 100ms after borrow, the timestamps are severely wrong
                # Fall back to redistributing ALL words in the turn evenly
                if per_word < CLEARLY_WRONG_THRESHOLD and group_size >= 5:
                    per_word_from_turn = turn_duration / len(fixed) if len(fixed) > 0 else 0.2
                    if per_word_from_turn >= MIN_WORD_DURATION:
                        logger.warning(
                            "SUBTITLE FIX: %d bunched words → redistributed all %d words (%.0fms/word)",
                            group_size, len(fixed), per_word_from_turn * 1000
                        )
                        for k in range(len(fixed)):
                            fixed[k]["start"] = turn_start + k * per_word_from_turn
                            fixed[k]["end"] = turn_start + (k + 1) * per_word_from_turn
                        return fixed
                
                # Apply redistribution for this group only
                for j in range(group_size):
                    w_idx = group_start + j
                    fixed[w_idx]["start"] = window_start + j * per_word
                    fixed[w_idx]["end"] = window_start + (j + 1) * per_word
                
                # Only log if subtitles might be too fast
                if per_word < MIN_WORD_DURATION:
                    logger.warning(
                        "SUBTITLE FIX: %d bunched words → %.0fms/word (may be fast)",
                        group_size, per_word * 1000
                    )
            
            i += 1
        
        # PASS 2: Final cleanup - ensure no overlaps
        for j in range(len(fixed) - 1):
            if fixed[j]["end"] > fixed[j + 1]["start"]:
                # Overlap detected, adjust
                mid = (fixed[j]["end"] + fixed[j + 1]["start"]) / 2
                fixed[j]["end"] = mid - 0.01
                fixed[j + 1]["start"] = mid + 0.01
        
        # Clamp last word to turn_end
        if fixed and fixed[-1]["end"] > turn_end:
            fixed[-1]["end"] = turn_end
        
        return fixed

    def _validate_ttd_timing(self, voice_segments: list[dict], num_inputs: int) -> tuple[bool, list[int]]:
        """
        Check if TTD API returned valid timing for most inputs.
        
        Returns:
            Tuple of (is_valid, missing_indices)
            - is_valid: True if at least 70% of inputs have timing
            - missing_indices: List of input indices without valid timing
        """
        if not voice_segments:
            return False, list(range(num_inputs))
        
        # Build timing map
        segment_timing = {}
        for vs in voice_segments:
            input_idx = vs.get("dialogue_input_index", -1)
            if input_idx >= 0:
                if input_idx not in segment_timing:
                    segment_timing[input_idx] = {
                        "start": vs.get("start_time_seconds", 0),
                        "end": vs.get("end_time_seconds", 0),
                    }
                else:
                    segment_timing[input_idx]["end"] = max(
                        segment_timing[input_idx]["end"],
                        vs.get("end_time_seconds", 0)
                    )
        
        # Check each input has valid timing
        missing_timing = []
        for idx in range(num_inputs):
            timing = segment_timing.get(idx, {})
            duration = timing.get("end", 0) - timing.get("start", 0)
            if duration < 0.1:
                missing_timing.append(idx)
        
        # Accept if at least 70% have timing (allow some short phrases to be missing)
        coverage = (num_inputs - len(missing_timing)) / num_inputs if num_inputs > 0 else 0
        
        if missing_timing:
            logger.warning(
                "TTD API timing coverage: %.0f%% (%d/%d inputs missing, indices: %s)",
                coverage * 100, len(missing_timing), num_inputs, missing_timing[:10]
            )
        
        # Consider valid if we have at least 70% coverage
        is_valid = coverage >= 0.7
        return is_valid, missing_timing

    def _check_timing_quality(self, words_by_input: dict[int, list[dict]], dialogue_turns: list[dict], inputs: list[dict]) -> bool:
        """
        Check if TTD alignment timestamps are of acceptable quality.
        
        Returns True if timestamps are good enough, False if Whisper fallback is needed.
        
        Criteria for poor quality:
        - Many bunched words (< 100ms per word after fixes)
        - Many turns with very fast words (< 150ms per word)
        - Large gaps or overlaps between words
        """
        if not words_by_input:
            return False
        
        total_words = 0
        fast_words = 0
        bunched_groups = 0
        turns_with_issues = 0
        
        for turn_idx, words in words_by_input.items():
            if not words or turn_idx >= len(dialogue_turns):
                continue
            
            # Calculate turn duration from word timestamps
            if not words:
                continue
            
            turn_start = words[0]["start"]
            turn_end = words[-1]["end"]
            turn_duration = turn_end - turn_start
            
            if turn_duration <= 0:
                continue
            
            total_words += len(words)
            turn_has_issues = False
            
            # Check for bunched words (very close start times)
            for i in range(len(words) - 1):
                word_duration = words[i]["end"] - words[i]["start"]
                next_start = words[i + 1]["start"]
                gap = next_start - words[i]["end"]
                
                # Word is too fast
                if word_duration < 0.15:  # 150ms
                    fast_words += 1
                    turn_has_issues = True
                
                # Words are bunched (gap < 50ms and word is fast)
                if gap < 0.05 and word_duration < 0.15:
                    if i == 0 or (i > 0 and words[i]["start"] - words[i-1]["end"] >= 0.05):
                        bunched_groups += 1
                        turn_has_issues = True
            
            if turn_has_issues:
                turns_with_issues += 1
        
        if total_words == 0:
            return False
        
        # Calculate quality metrics
        fast_word_ratio = fast_words / total_words if total_words > 0 else 0
        bunched_ratio = bunched_groups / len(words_by_input) if words_by_input else 0
        turns_with_issues_ratio = turns_with_issues / len(words_by_input) if words_by_input else 0
        
        # Quality is poor if:
        # - More than 30% of words are too fast
        # - More than 50% of turns have bunched groups
        # - More than 60% of turns have any issues
        is_poor_quality = fast_word_ratio > 0.30 or bunched_ratio > 0.50 or turns_with_issues_ratio > 0.60
        
        if is_poor_quality:
            logger.warning(
                "TTD QUALITY CHECK: Poor timestamps detected (%.1f%% fast words, %.1f%% bunched turns, %.1f%% turns with issues) - using Whisper fallback",
                fast_word_ratio * 100, bunched_ratio * 100, turns_with_issues_ratio * 100
            )
            return False
        
        logger.debug(
            "TTD QUALITY CHECK: Timestamps acceptable (%.1f%% fast words, %.1f%% bunched turns, %.1f%% turns with issues)",
            fast_word_ratio * 100, bunched_ratio * 100, turns_with_issues_ratio * 100
        )
        return True

    def _get_timestamps_via_whisper(
        self,
        audio_path: str,
        dialogue_turns: list[dict],
        inputs: list[dict],
    ) -> dict[int, list[dict]]:
        """
        Get word-level timestamps by running Whisper on the generated TTD audio.
        
        This is more accurate than ElevenLabs TTD alignment which often has
        "bunched" timestamps (multiple words with same start time).
        
        Args:
            audio_path: Path to the generated TTD audio file
            dialogue_turns: Original dialogue turns with text_ru
            inputs: TTD API inputs (with text after emotion tags)
            
        Returns:
            Dict mapping turn index to list of word dicts:
            {0: [{"word": "Привет", "start": 0.1, "end": 0.5}, ...], ...}
        """
        import json
        import subprocess
        import tempfile
        
        logger.info("TTD WHISPER: Getting timestamps via WhisperX for %s", audio_path)
        
        # Get paths from environment (same as transcription_runner.py)
        python_path = os.getenv(
            "EXTERNAL_ASR_PY",
            "/opt/youtube-shorts-generator/venv-asr/bin/python"
        )
        script_path = os.getenv(
            "EXTERNAL_ASR_SCRIPT",
            "/opt/youtube-shorts-generator/backend/tools/transcribe.py"
        )
        
        # Check if external transcription is available
        if not os.path.exists(python_path) or not os.path.exists(script_path):
            logger.warning(
                "TTD WHISPER: External ASR not available (python=%s, script=%s), falling back to TTD alignment",
                python_path, script_path
            )
            return {}
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp_out:
            output_json_path = tmp_out.name
        
        try:
            # Run WhisperX on the generated audio (Russian language)
            cmd = [
                python_path,
                script_path,
                "--audio", audio_path,
                "--model", "large-v3",
                "--language", "ru",
                "--device", "cuda",
                "--compute_type", "float16",
                "--output", output_json_path,
                # No diarization needed - we know the turn structure
            ]
            
            logger.info("TTD WHISPER: Running WhisperX...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min max
                check=True,
            )
            
            if result.stderr:
                logger.debug("TTD WHISPER stderr: %s", result.stderr[:500])
            
            # Read Whisper output
            with open(output_json_path, "r", encoding="utf-8") as f:
                whisper_data = json.load(f)
            
            segments = whisper_data.get("segments", [])
            logger.info("TTD WHISPER: Got %d segments from WhisperX", len(segments))
            
            # Extract all words with timestamps from Whisper
            whisper_words = []
            for seg in segments:
                for word_info in seg.get("words", []):
                    word = word_info.get("word", "").strip()
                    start = word_info.get("start", 0)
                    end = word_info.get("end", 0)
                    if word and end > start:
                        whisper_words.append({
                            "word": word,
                            "start": start,
                            "end": end,
                        })
            
            logger.info("TTD WHISPER: Extracted %d words with timestamps", len(whisper_words))
            
            if not whisper_words:
                logger.warning("TTD WHISPER: No words extracted, falling back to TTD alignment")
                return {}
            
            # Now match Whisper words to original turns
            # Strategy: sequential matching by position
            words_by_input = self._match_whisper_words_to_turns(
                whisper_words, dialogue_turns, inputs
            )
            
            return words_by_input
            
        except subprocess.TimeoutExpired:
            logger.error("TTD WHISPER: WhisperX timed out")
            return {}
        except subprocess.CalledProcessError as e:
            logger.error("TTD WHISPER: WhisperX failed: %s", e.stderr[:500] if e.stderr else str(e))
            return {}
        except Exception as e:
            logger.error("TTD WHISPER: Unexpected error: %s", e)
            return {}
        finally:
            # Cleanup temp file
            try:
                os.unlink(output_json_path)
            except Exception:
                pass

    def _match_whisper_words_to_turns(
        self,
        whisper_words: list[dict],
        dialogue_turns: list[dict],
        inputs: list[dict],
    ) -> dict[int, list[dict]]:
        """
        Match Whisper-recognized words to original dialogue turns.
        
        Strategy:
        1. Get expected words for each turn (from inputs, cleaned of emotion tags)
        2. Go through Whisper words sequentially
        3. Match by position, using original text but Whisper timestamps
        
        Args:
            whisper_words: Words from Whisper with timestamps
            dialogue_turns: Original turns
            inputs: TTD inputs with text
            
        Returns:
            Dict mapping turn index to word list with timestamps
        """
        words_by_input: dict[int, list[dict]] = {}
        
        # Build expected words per turn
        expected_words_per_turn = []
        for idx, inp in enumerate(inputs):
            text = inp.get("text", "")
            # Remove emotion tags like [cheerfully], [sadly], etc.
            clean_text = re.sub(r'\[[\w]+\]\s*', '', text)
            words = clean_text.split()
            expected_words_per_turn.append(words)
        
        total_expected = sum(len(w) for w in expected_words_per_turn)
        logger.info(
            "TTD WHISPER MATCH: %d expected words across %d turns, %d Whisper words",
            total_expected, len(expected_words_per_turn), len(whisper_words)
        )
        
        # Sequential matching
        whisper_idx = 0
        
        for turn_idx, expected_words in enumerate(expected_words_per_turn):
            turn_words = []
            
            for word_idx, expected_word in enumerate(expected_words):
                if whisper_idx >= len(whisper_words):
                    # Ran out of Whisper words - estimate remaining
                    logger.warning(
                        "TTD WHISPER MATCH: Ran out of Whisper words at turn %d, word %d",
                        turn_idx, word_idx
                    )
                    break
                
                whisper_word = whisper_words[whisper_idx]
                
                # Use original word text but Whisper timestamps
                turn_words.append({
                    "word": expected_word,  # Original text (correct spelling)
                    "start": whisper_word["start"],
                    "end": whisper_word["end"],
                    "_whisper_word": whisper_word["word"],  # For debugging
                })
                
                whisper_idx += 1
            
            if turn_words:
                words_by_input[turn_idx] = turn_words
                logger.debug(
                    "TTD WHISPER MATCH: Turn %d: %d words (%.2f-%.2fs)",
                    turn_idx, len(turn_words),
                    turn_words[0]["start"], turn_words[-1]["end"]
                )
        
        # Log matching quality
        matched_count = sum(len(w) for w in words_by_input.values())
        logger.info(
            "TTD WHISPER MATCH: Matched %d/%d expected words (%.0f%%), used %d/%d Whisper words",
            matched_count, total_expected,
            (matched_count / total_expected * 100) if total_expected > 0 else 0,
            whisper_idx, len(whisper_words)
        )
        
        return words_by_input

    def synthesize_dialogue(
        self,
        dialogue_turns: list[dict],
        output_path: str,
        voice_map: dict[str, str],
        pause_ms: int = 300,  # ignored in TTD - natural pauses
        base_start: float | None = None,  # ignored in TTD
        add_emotions: bool = True,
    ) -> str:
        """
        Synthesize dialogue using ElevenLabs Text-to-Dialogue API.
        
        This is a single API call that generates the entire dialogue
        with natural transitions between speakers.
        
        Includes retry logic: if API doesn't return timing for all turns,
        retries up to 3 times with 30s wait before the 3rd attempt.
        
        Args:
            dialogue_turns: List of turns with speaker, text_ru/text, optional emotion
            output_path: Where to save the audio file
            voice_map: Mapping of speaker IDs to ElevenLabs voice IDs
            pause_ms: Ignored (TTD handles pauses naturally)
            base_start: Ignored (TTD handles timing internally)
            add_emotions: Whether to add audio tags based on punctuation
            
        Returns:
            Path to the saved audio file
            
        Raises:
            RuntimeError: If API fails to return valid timing after 3 attempts
        """
        import time
        
        # Log timeline info for debugging (speed adjustment is handled by FFmpeg in video.py)
        self._log_dialogue_timeline(dialogue_turns)
        
        # TTD API does NOT support speed parameter (unlike regular TTS)
        # See: https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert
        # Speed adjustment is handled entirely by FFmpeg tempo in video.py
        speed = 1.0
        inputs, gaps = self._prepare_ttd_inputs(dialogue_turns, voice_map, add_emotions, speed=speed)
        
        if not inputs:
            raise ValueError("No valid dialogue turns to synthesize")
        
        # Calculate leading silence (gap before first turn)
        leading_silence_ms = int(gaps[0] * 1000) if gaps else 0
        
        logger.info(
            "TTD: Synthesizing dialogue with %d turns, voices=%s",
            len(inputs),
            list(set(inp["voice_id"] for inp in inputs)),
        )
        
        # Log the inputs for debugging (debug level)
        for idx, inp in enumerate(inputs):
            vs = inp.get("voice_settings", {})
            speed_val = vs.get("speed", 1.0) if vs else 1.0
            logger.debug(
                "TTD input %d: voice=%s, speed=%.2f, text=%s",
                idx, inp["voice_id"], speed_val, inp["text"][:60]
            )
        
        # Retry logic: attempt up to 3 times
        max_attempts = 3
        audio_bytes = None
        voice_segments = []
        alignment = {}
        
        for attempt in range(1, max_attempts + 1):
            try:
                audio_bytes, voice_segments, alignment = self._call_ttd_api(inputs)
                
                # Validate timing (now returns tuple with missing indices)
                is_valid, missing_indices = self._validate_ttd_timing(voice_segments, len(inputs))
                
                if is_valid:
                    if missing_indices:
                        logger.info(
                            "TTD: attempt %d OK, %d voice_segments, %d missing (will interpolate)",
                            attempt, len(voice_segments), len(missing_indices)
                        )
                    break
                else:
                    if attempt < max_attempts:
                        if attempt == 2:
                            logger.warning("TTD: insufficient timing, waiting 30s...")
                            time.sleep(30)
                    else:
                        if audio_bytes and len(audio_bytes) > 1000:
                            logger.warning("TTD: timing incomplete, using fallback")
                            break
                        else:
                            raise RuntimeError(
                                "ElevenLabs TTD API не вернул корректные тайминги после 3 попыток."
                            )
                        
            except httpx.HTTPStatusError as http_exc:
                logger.error("TTD API error: %s", http_exc.response.status_code)
                if attempt == max_attempts:
                    raise
                if attempt == 2:
                    time.sleep(30)
            except httpx.HTTPError as exc:
                logger.error("TTD API failed: %s", exc)
                if attempt == max_attempts:
                    raise
                if attempt == 2:
                    time.sleep(30)
        
        if audio_bytes is None:
            raise RuntimeError("TTD API failed to return audio")
        
        # Decode and save audio
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        except Exception as exc:
            logger.error("Failed to decode TTD audio: %s", exc)
            raise
        
        # Convert to target sample rate
        audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
        
        # Add leading silence if first turn doesn't start at 0
        if leading_silence_ms > 100:  # Only add if gap > 100ms
            silence = AudioSegment.silent(duration=leading_silence_ms, frame_rate=self.sample_rate)
            audio = silence + audio
            logger.info("TTD: Added %.2fs leading silence", leading_silence_ms / 1000.0)
        
        leading_sec = leading_silence_ms / 1000.0
        
        # ============================================================
        # SYNC WITH ORIGINAL TIMINGS: Insert pauses between turns
        # This ensures TTS audio matches original video timing
        # ============================================================
        cumulative_offset = 0.0  # Track how much time we've added
        turn_offsets = [0.0] * len(dialogue_turns)  # Offset for each turn
        
        if voice_segments and len(gaps) > 1:
            # Build segment timing from API response
            segment_timing_raw = {}
            for vs in voice_segments:
                input_idx = vs.get("dialogue_input_index", -1)
                if input_idx >= 0:
                    if input_idx not in segment_timing_raw:
                        segment_timing_raw[input_idx] = {
                            "start": vs.get("start_time_seconds", 0),
                            "end": vs.get("end_time_seconds", 0),
                        }
                    else:
                        segment_timing_raw[input_idx]["end"] = max(
                            segment_timing_raw[input_idx]["end"],
                            vs.get("end_time_seconds", 0)
                        )
            
            # Calculate how much silence to insert between each turn
            # Gap[i] = original pause before turn i (from original video timing)
            # We need to insert (gap[i] - actual_gap[i]) silence
            #
            # DISABLED: Silence insertion causes audio artifacts (stuttering/cutting)
            # because ElevenLabs TTD generates continuous audio without gaps.
            # When we cut and insert silence, we may cut in the middle of a word.
            # Instead, we rely on FFmpeg tempo adjustment to match timing.
            ENABLE_SILENCE_INSERTION = False
            
            inserted_silences = []
            for i in range(1, len(dialogue_turns)):
                if i >= len(gaps):
                    break
                
                original_gap = gaps[i]  # Gap from original video
                
                # Calculate actual gap in TTS audio
                prev_timing = segment_timing_raw.get(i - 1, {})
                curr_timing = segment_timing_raw.get(i, {})
                prev_end = prev_timing.get("end", 0)
                curr_start = curr_timing.get("start", prev_end)
                actual_gap = max(0, curr_start - prev_end)
                
                # How much extra silence do we need?
                extra_silence = max(0, original_gap - actual_gap)
                
                if extra_silence > 0.1 and ENABLE_SILENCE_INSERTION:  # Only insert if > 100ms
                    inserted_silences.append((i, extra_silence, original_gap, actual_gap))
                    cumulative_offset += extra_silence
                    
                turn_offsets[i] = cumulative_offset
            
            # Actually insert silences into audio
            if inserted_silences and ENABLE_SILENCE_INSERTION:
                logger.info(
                    "TTD SYNC: Inserting %d silences (total %.2fs) to match original timing",
                    len(inserted_silences),
                    sum(s[1] for s in inserted_silences),
                )
                
                # Rebuild audio with inserted silences
                # We need to cut and splice the audio
                new_audio = AudioSegment.empty()
                last_cut_ms = 0
                
                for turn_idx, extra_sec, orig_gap, actual_gap in inserted_silences:
                    # Get timing for PREVIOUS turn and CURRENT turn
                    prev_timing = segment_timing_raw.get(turn_idx - 1, {})
                    curr_timing = segment_timing_raw.get(turn_idx, {})
                    
                    prev_end_sec = prev_timing.get("end", 0) + leading_sec
                    curr_start_sec = curr_timing.get("start", 0) + leading_sec
                    
                    # The gap in original TTS audio between prev turn end and curr turn start
                    tts_gap_sec = max(0, curr_start_sec - prev_end_sec)
                    
                    # We want to insert silence BETWEEN turns, not cut audio
                    # Strategy: include all audio up to curr_start, then insert silence
                    # This preserves any audio "tail" from the previous turn
                    
                    cut_point_ms = int(curr_start_sec * 1000)
                    
                    # Add audio from last cut to current turn start
                    if cut_point_ms > last_cut_ms:
                        new_audio += audio[last_cut_ms:cut_point_ms]
                    
                    # Insert the required silence
                    silence_ms = int(extra_sec * 1000)
                    new_audio += AudioSegment.silent(duration=silence_ms, frame_rate=self.sample_rate)
                    
                    logger.info(
                        "  Turn %d: cut at %.2fs, inserted %.2fs silence (prev_end=%.2fs, curr_start=%.2fs, tts_gap=%.2fs, orig_gap=%.2fs)",
                        turn_idx, curr_start_sec, extra_sec, prev_end_sec, curr_start_sec, tts_gap_sec, orig_gap
                    )
                    
                    # Continue from curr_start (we've already included audio up to this point)
                    last_cut_ms = cut_point_ms
                
                # Add remaining audio
                if last_cut_ms < len(audio):
                    new_audio += audio[last_cut_ms:]
                
                audio = new_audio
                logger.info("TTD SYNC: Audio extended from %.2fs to %.2fs", 
                           len(audio) / 1000.0 - cumulative_offset, len(audio) / 1000.0)
        
        # Add trailing silence buffer to prevent last word from being cut off
        # ElevenLabs sometimes returns audio shorter than the last voice segment end time
        TRAILING_SILENCE_MS = 200  # 200ms buffer at the end
        trailing_silence = AudioSegment.silent(duration=TRAILING_SILENCE_MS, frame_rate=audio.frame_rate)
        audio = audio + trailing_silence
        logger.info("TTD: Added %dms trailing silence buffer", TRAILING_SILENCE_MS)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(output_path), format="wav")
        
        duration_sec = len(audio) / 1000.0
        
        logger.info(
            "TTD: audio=%.2fs, %d turns, %d voice_segments",
            duration_sec, len(inputs), len(voice_segments) if voice_segments else 0
        )
        
        # Store timing info in turns for subtitle sync
        if voice_segments:
            
            # Build mapping from dialogue_input_index to segment timing
            segment_timing = {}
            for vs in voice_segments:
                input_idx = vs.get("dialogue_input_index", -1)
                if input_idx >= 0:
                    # Accumulate if multiple segments per input (shouldn't happen normally)
                    if input_idx not in segment_timing:
                        segment_timing[input_idx] = {
                            "start": vs.get("start_time_seconds", 0),
                            "end": vs.get("end_time_seconds", 0),
                        }
                    else:
                        # Extend end time if there are multiple segments
                        segment_timing[input_idx]["end"] = max(
                            segment_timing[input_idx]["end"],
                            vs.get("end_time_seconds", 0)
                        )
            
            # Parse character alignment into word-level timestamps
            # Strategy:
            # 1. First try ElevenLabs TTD alignment (fast, but can have "bunched" timestamps)
            # 2. Apply fixes for bunched words
            # 3. Check quality - if poor, fallback to Whisper for accurate timestamps
            
            logger.info("TTD: Using ElevenLabs TTD alignment for word timestamps")
            words_by_input = self._parse_alignment_to_words(alignment, voice_segments)
            
            # Validate and fix timings using voice_segments as reference
            if words_by_input and voice_segments:
                words_by_input = self._validate_and_fix_timings_with_voice_segments(
                    words_by_input, voice_segments
                )
            
            # Apply fixes for bunched words and check quality
            if words_by_input:
                # Calculate turn timings for quality check (before applying offsets)
                temp_turn_timings = {}
                for turn_idx in words_by_input.keys():
                    if turn_idx < len(inputs):
                        timing = segment_timing.get(turn_idx, {})
                        # Use timing without offsets for quality check
                        temp_turn_timings[turn_idx] = {
                            "start": timing.get("start", 0) + leading_sec,
                            "end": timing.get("end", 0) + leading_sec,
                        }
                
                # Apply fixes for bunched words
                temp_words_by_input = {}
                for turn_idx, words in words_by_input.items():
                    if turn_idx in temp_turn_timings:
                        turn_timing = temp_turn_timings[turn_idx]
                        fixed_words = self._fix_bunched_word_timestamps(
                            words, turn_timing["start"], turn_timing["end"]
                        )
                        temp_words_by_input[turn_idx] = fixed_words
                    else:
                        temp_words_by_input[turn_idx] = words
                
                # Check quality of fixed timestamps
                if not self._check_timing_quality(temp_words_by_input, dialogue_turns, inputs):
                    # Quality is poor - use Whisper fallback
                    logger.warning("TTD: TTD alignment quality is poor, using Whisper fallback")
                    whisper_words = self._get_timestamps_via_whisper(
                        str(output_path), dialogue_turns, inputs
                    )
                    
                    if whisper_words:
                        logger.info("TTD: Whisper fallback successful, using Whisper timestamps")
                        words_by_input = whisper_words
                    else:
                        logger.warning("TTD: Whisper fallback failed, using fixed TTD alignment")
                        words_by_input = temp_words_by_input
                else:
                    # Quality is acceptable - use fixed TTD alignment
                    logger.info("TTD: TTD alignment quality is acceptable after fixes")
                    words_by_input = temp_words_by_input
            
            # Apply timing to dialogue turns (with inserted silence offsets)
            for idx, turn in enumerate(dialogue_turns):
                if idx >= len(inputs):
                    break
                
                timing = segment_timing.get(idx, {})
                # Add offset from inserted silences before this turn
                offset = turn_offsets[idx] if idx < len(turn_offsets) else 0.0
                
                start_time = timing.get("start", 0) + leading_sec + offset
                end_time = timing.get("end", 0) + leading_sec + offset
                turn_duration = end_time - start_time
                
                # Fallback for missing timing: interpolate from neighbors or estimate
                if turn_duration < 0.1:
                    text = inputs[idx]["text"] if idx < len(inputs) else ""
                    word_count = len(text.split())
                    
                    # Get speaker's personal speech rate from valid turns
                    voice_id = inputs[idx].get("voice_id") if idx < len(inputs) else None
                    speaker_rate = self._get_speaker_rate(
                        voice_id, segment_timing, inputs, dialogue_turns
                    )
                    
                    estimated_duration = max(0.5, word_count / speaker_rate)
                    
                    # Try to use previous turn's end as start
                    if idx > 0 and dialogue_turns[idx - 1].get("tts_end_offset"):
                        start_time = dialogue_turns[idx - 1]["tts_end_offset"] + 0.1
                    else:
                        start_time = offset + leading_sec
                    
                    end_time = start_time + estimated_duration
                    
                    # Check if we overlap with next turn and clamp if needed
                    next_turn_start = None
                    for next_idx in range(idx + 1, len(dialogue_turns)):
                        next_timing = segment_timing.get(next_idx, {})
                        next_offset = turn_offsets[next_idx] if next_idx < len(turn_offsets) else 0.0
                        next_start = next_timing.get("start", 0) + leading_sec + next_offset
                        if next_start > 0.1:  # Valid timing
                            next_turn_start = next_start
                            break
                    
                    if next_turn_start and end_time > next_turn_start - 0.1:
                        # Clamp end_time to not overlap with next turn
                        old_end = end_time
                        end_time = next_turn_start - 0.1
                        
                        # If end_time is now before start_time, adjust start_time too
                        if end_time <= start_time:
                            # Ensure minimum 0.3s duration, shift start_time back
                            min_duration = 0.3
                            new_start = end_time - min_duration
                            
                            # But don't overlap with previous turn!
                            prev_end = 0.0
                            if idx > 0 and dialogue_turns[idx - 1].get("tts_end_offset"):
                                prev_end = dialogue_turns[idx - 1]["tts_end_offset"]
                            
                            # Ensure at least 0.05s gap from previous turn
                            min_start = prev_end + 0.05
                            start_time = max(min_start, new_start)
                            
                            logger.warning(
                                "TTD turn %d: shifting start to %.2f (end=%.2f, prev_end=%.2f)",
                                idx, start_time, end_time, prev_end
                            )
                        
                        logger.warning(
                            "TTD turn %d: clamped end %.2f->%.2f to avoid overlap with next turn at %.2f",
                            idx, old_end, end_time, next_turn_start
                        )
                    
                    # CRITICAL: Final validation - ensure start < end
                    # If there's no room (squeezed between prev and next turns), 
                    # place this turn right after prev with minimal duration
                    if start_time >= end_time:
                        prev_end = 0.0
                        if idx > 0 and dialogue_turns[idx - 1].get("tts_end_offset"):
                            prev_end = dialogue_turns[idx - 1]["tts_end_offset"]
                        
                        # Place immediately after previous turn
                        start_time = prev_end + 0.05
                        end_time = start_time + 0.3
                        
                        # But also check we don't overlap with NEXT turn!
                        if next_turn_start and end_time > next_turn_start - 0.05:
                            end_time = next_turn_start - 0.05
                            # Ensure minimum 0.1s duration
                            if end_time - start_time < 0.1:
                                # Truly no room - make it minimal
                                end_time = start_time + 0.1
                        
                        logger.warning(
                            "TTD turn %d: NO ROOM! Forcing %.2f-%.2f (%.2fs) after prev_end=%.2f, next_start=%.2f",
                            idx, start_time, end_time, end_time - start_time, prev_end, 
                            next_turn_start if next_turn_start else -1
                        )
                    
                    estimated_duration = end_time - start_time
                    
                    turn_duration = estimated_duration
                    logger.warning(
                        "TTD turn %d: interpolated timing %.2f-%.2fs (%.2fs) using speaker rate %.2f w/s for '%s...'",
                        idx, start_time, end_time, turn_duration, speaker_rate, text[:30]
                    )
                
                turn["tts_start_offset"] = start_time
                turn["tts_duration"] = turn_duration
                turn["tts_end_offset"] = end_time
                # Mark timing source based on whether we had valid API timing
                api_timing = segment_timing.get(idx, {})
                api_duration = api_timing.get("end", 0) - api_timing.get("start", 0)
                turn["_timing_source"] = "api" if api_duration >= 0.1 else "interpolated"
                
                # Add word-level timestamps if available
                if idx in words_by_input:
                    # Adjust word timestamps for leading silence + inserted silences
                    # Note: words are already fixed for bunched timestamps during quality check
                    tts_words = []
                    for w in words_by_input[idx]:
                        tts_words.append({
                            "word": w["word"],
                            "start": w["start"] + leading_sec + offset,
                            "end": w["end"] + leading_sec + offset,
                        })
                    
                    turn["tts_words"] = tts_words
                    logger.debug(
                        "TTD turn %d: %d words with timestamps (first: %.2f-%.2fs, last: %.2f-%.2fs)",
                        idx,
                        len(tts_words),
                        tts_words[0]["start"] if tts_words else 0,
                        tts_words[0]["end"] if tts_words else 0,
                        tts_words[-1]["start"] if tts_words else 0,
                        tts_words[-1]["end"] if tts_words else 0,
                    )
                else:
                    # Generate estimated word timestamps for turns without API timing
                    text = inputs[idx]["text"] if idx < len(inputs) else ""
                    # Remove emotion tags before splitting
                    clean_text = re.sub(r'\[[\w]+\]\s*', '', text)
                    words = clean_text.split()
                    if words and turn_duration > 0:
                        per_word = turn_duration / len(words)
                        tts_words = []
                        for w_idx, word in enumerate(words):
                            w_start = start_time + w_idx * per_word
                            w_end = w_start + per_word
                            tts_words.append({
                                "word": word,
                                "start": w_start,
                                "end": w_end,
                            })
                        turn["tts_words"] = tts_words
                        logger.debug(
                            "TTD turn %d: generated %d estimated word timestamps (%.2f-%.2fs)",
                            idx, len(tts_words), start_time, end_time
                        )
                
                logger.debug(
                    "TTD subtitle turn %d: %.2f-%.2fs (%.2fs, %d words) [from API]",
                    idx,
                    turn["tts_start_offset"],
                    turn["tts_end_offset"],
                    turn["tts_duration"],
                    len(turn.get("tts_words", [])),
                )
            
            # POST-PROCESSING: Fix bunched/overlapping turns
            # ElevenLabs TTD sometimes returns multiple turns ending at the same time
            # This causes subtitle overlaps. We need to detect and redistribute them.
            
            MIN_TURN_DURATION = 0.4  # Minimum 400ms per turn for subtitles
            TURN_GAP = 0.05  # 50ms gap between turns
            BUNCH_THRESHOLD = 0.15  # Turns ending within 150ms are "bunched"
            
            # Step 1: Find groups of bunched turns (same end time)
            num_turns = min(len(dialogue_turns), len(inputs))
            bunched_groups = []
            i = 0
            while i < num_turns:
                group_start = i
                group_end = i
                base_end_time = dialogue_turns[i].get("tts_end_offset", 0)
                
                # Find all consecutive turns that end at approximately the same time
                while group_end + 1 < num_turns:
                    next_end_time = dialogue_turns[group_end + 1].get("tts_end_offset", 0)
                    if abs(next_end_time - base_end_time) < BUNCH_THRESHOLD:
                        group_end += 1
                    else:
                        break
                
                if group_end > group_start:  # Found a bunched group
                    bunched_groups.append((group_start, group_end))
                    i = group_end + 1
                else:
                    i += 1
            
            # Step 2: Redistribute each bunched group
            total_fixes = 0
            for group_start_idx, group_end_idx in bunched_groups:
                group_size = group_end_idx - group_start_idx + 1
                
                # Window: from first turn's original start to a reasonable end
                first_turn = dialogue_turns[group_start_idx]
                last_turn = dialogue_turns[group_end_idx]
                
                window_start = first_turn.get("tts_start_offset", 0)
                window_end = last_turn.get("tts_end_offset", 0)
                
                # If window is too small, extend it
                # Check if there's a next turn to not overlap with
                if group_end_idx + 1 < num_turns:
                    next_turn_start = dialogue_turns[group_end_idx + 1].get("tts_start_offset", window_end + 2)
                    max_window_end = next_turn_start - TURN_GAP
                else:
                    max_window_end = duration_sec  # Use total audio duration
                
                # Calculate required duration for all turns
                required_duration = group_size * MIN_TURN_DURATION + (group_size - 1) * TURN_GAP
                current_duration = window_end - window_start
                
                if current_duration < required_duration:
                    # Need to extend the window
                    new_window_end = min(window_start + required_duration, max_window_end)
                    logger.warning(
                        "TTD BUNCHED: turns %d-%d need %.2fs but have %.2fs, extending window to %.2fs",
                        group_start_idx, group_end_idx, required_duration, current_duration, new_window_end - window_start
                    )
                    window_end = new_window_end
                
                # Calculate per-turn duration
                total_gaps = (group_size - 1) * TURN_GAP
                available_for_turns = (window_end - window_start) - total_gaps
                per_turn_duration = max(MIN_TURN_DURATION, available_for_turns / group_size)
                
                logger.warning(
                    "TTD BUNCHED: turns %d-%d redistributed (%.2f-%.2fs) → %.0fms/turn",
                    group_start_idx, group_end_idx, window_start, window_end, per_turn_duration * 1000
                )
                
                # Redistribute turns
                current_time = window_start
                for idx in range(group_start_idx, group_end_idx + 1):
                    turn = dialogue_turns[idx]
                    
                    new_start = current_time
                    new_end = current_time + per_turn_duration
                    new_duration = per_turn_duration
                    
                    turn["tts_start_offset"] = new_start
                    turn["tts_end_offset"] = new_end
                    turn["tts_duration"] = new_duration
                    
                    # Redistribute word timestamps within the new turn window
                    words = turn.get("tts_words", [])
                    if words:
                        word_count = len(words)
                        per_word_duration = new_duration / word_count
                        for w_idx, w in enumerate(words):
                            w["start"] = new_start + w_idx * per_word_duration
                            w["end"] = new_start + (w_idx + 1) * per_word_duration
                    
                    current_time = new_end + TURN_GAP
                    total_fixes += 1
            
            # Step 3: Simple overlap fix for any remaining non-bunched overlaps
            for idx in range(1, num_turns):
                prev_turn = dialogue_turns[idx - 1]
                curr_turn = dialogue_turns[idx]
                
                prev_end = prev_turn.get("tts_end_offset", 0)
                curr_start = curr_turn.get("tts_start_offset", 0)
                
                if prev_end > curr_start + 0.01:  # Still overlapping
                    shift = prev_end - curr_start + TURN_GAP
                    curr_turn["tts_start_offset"] = curr_start + shift
                    curr_turn["tts_end_offset"] = curr_turn.get("tts_end_offset", 0) + shift
                    
                    # Shift word timestamps
                    for w in curr_turn.get("tts_words", []):
                        w["start"] += shift
                        w["end"] += shift
                    
                    logger.warning(
                        "TTD OVERLAP FIX turn %d: shifted by %.2fs to avoid overlap with turn %d",
                        idx, shift, idx - 1
                    )
                    total_fixes += 1
            
            if total_fixes > 0:
                logger.info("TTD: Fixed %d turn timing issues (bunched/overlapping)", total_fixes)
            
            # POST-PROCESSING: Merge very short turns into previous turn for subtitle display
            # If a turn has < 0.5s duration, its text will be merged with the previous turn
            MIN_SUBTITLE_DURATION = 0.5  # Minimum 500ms for a readable subtitle
            merged_count = 0
            for idx in range(len(dialogue_turns) - 1, 0, -1):  # Go backwards to avoid index issues
                if idx >= len(inputs):
                    continue
                curr_turn = dialogue_turns[idx]
                prev_turn = dialogue_turns[idx - 1]
                
                curr_duration = curr_turn.get("tts_duration", 0)
                if curr_duration < MIN_SUBTITLE_DURATION and curr_duration > 0:
                    # This turn is too short - merge its subtitle into previous turn
                    # Keep the timing but mark for merging
                    prev_text = inputs[idx - 1]["text"] if idx - 1 < len(inputs) else ""
                    curr_text = inputs[idx]["text"] if idx < len(inputs) else ""
                    
                    # Extend previous turn's end time to include this turn
                    prev_turn["tts_end_offset"] = curr_turn.get("tts_end_offset", prev_turn.get("tts_end_offset", 0))
                    prev_turn["tts_duration"] = prev_turn["tts_end_offset"] - prev_turn.get("tts_start_offset", 0)
                    
                    # Merge word timestamps
                    prev_words = prev_turn.get("tts_words", [])
                    curr_words = curr_turn.get("tts_words", [])
                    if curr_words:
                        prev_turn["tts_words"] = prev_words + curr_words
                    
                    # Mark current turn as merged (will be skipped in subtitle generation)
                    curr_turn["_subtitle_merged"] = True
                    curr_turn["_merged_into"] = idx - 1
                    
                    merged_count += 1
                    logger.warning(
                        "TTD MERGE: turn %d (%.2fs, '%s...') merged INTO turn %d for subtitles",
                        idx, curr_duration, curr_text[:25], idx - 1
                    )
            
            if merged_count > 0:
                logger.info("TTD: Merged %d short turns into previous turns for subtitle display", merged_count)
            
            # Summary log: timing source for each turn
            api_count = sum(1 for t in dialogue_turns if t.get("_timing_source") == "api")
            interp_count = sum(1 for t in dialogue_turns if t.get("_timing_source") == "interpolated")
            logger.info(
                "TTD TIMING SUMMARY: %d turns from API, %d interpolated",
                api_count, interp_count
            )
        else:
            # Fallback: distribute proportionally by character count
            logger.warning("TTD: No voice_segments in response, using character-based estimation")
            
            ttd_speech_duration = duration_sec - leading_sec
            char_counts = []
            for idx, turn in enumerate(dialogue_turns):
                if idx >= len(inputs):
                    break
                text = inputs[idx]["text"]
                char_counts.append(len(text))
            
            total_chars = sum(char_counts)
            elapsed = leading_sec
            
            for idx, turn in enumerate(dialogue_turns):
                if idx >= len(char_counts):
                    break
                
                char_ratio = char_counts[idx] / total_chars if total_chars > 0 else 1.0 / len(char_counts)
                turn_duration = ttd_speech_duration * char_ratio
                
                turn["tts_start_offset"] = elapsed
                turn["tts_duration"] = turn_duration
                turn["tts_end_offset"] = elapsed + turn_duration
                elapsed += turn_duration
        
        return str(output_path)

