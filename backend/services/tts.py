"""Text-to-Speech services."""
import base64
import io
import json
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
                "TTS INPUT [turn %s/%s]: speaker=%s -> voice=%s, len=%s, text='%s' (linear, %s chunk(s))",
                idx + 1,
                len(dialogue_turns),
                speaker_id,
                voice,
                len(text),
                text,
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

        # Single-speaker mode: TTS may run longer than the English slot without issue
        # because the next turn starts after a pause — no real overlap to prevent.
        # Only clip when multiple distinct voices are actually interleaved.
        unique_speakers = {t.get("speaker") for t in dialogue_turns if t.get("speaker")}
        is_single_speaker = len(unique_speakers) <= 1
        if is_single_speaker:
            logger.info(
                "OVERLAP MODE: single-speaker (%s) — clip disabled, TTS may extend freely",
                next(iter(unique_speakers), "unknown"),
            )

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
                "TTS INPUT [turn %s/%s]: speaker=%s -> voice=%s, len=%s, offset=%.2fs, text='%s' (%s chunk(s))",
                idx + 1,
                len(dialogue_turns),
                speaker_id,
                voice,
                len(text),
                offset_ms / 1000,
                text,
                len(plan),
            )
            print(
                f"[TTS] turn {idx+1}/{len(dialogue_turns)} speaker={speaker_id} offset={offset_ms/1000:.2f}s text={text[:60]!r}",
                flush=True,
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
            if diarized_end is not None and not is_single_speaker:
                relative_end_target = max(0.0, diarized_end - reference)
                if relative_end_target > relative_start:
                    # Add 120 ms buffer so TTS slightly longer than the slot is not clipped mid-word
                    CLIP_BUFFER_MS = 120
                    CLIP_FADE_MS   = 40   # smooth fade-out instead of hard cut
                    max_allowed = max(self.MIN_CHUNK_DURATION, relative_end_target - relative_start)
                    max_allowed_with_buf = max_allowed + CLIP_BUFFER_MS / 1000.0
                    if duration_seconds > max_allowed_with_buf:
                        clip_ms = int(max_allowed_with_buf * 1000)
                        fade_ms = min(CLIP_FADE_MS, clip_ms // 4)
                        seg_audio = seg_audio[:clip_ms].fade_out(fade_ms)
                        duration_seconds = len(seg_audio) / 1000.0
                        logger.info(
                            "OVERLAP CLIP turn %d: allowed=%.3fs buf=+%dms clip=%.3fs (was %.3fs) fade=%dms",
                            idx, max_allowed, CLIP_BUFFER_MS, duration_seconds,
                            clip_ms / 1000.0 + 0.001, fade_ms,
                        )
                        # Clip word timestamps to match clipped audio
                        for word in turn_words:
                            if word["end"] > duration_seconds:
                                word["end"] = duration_seconds
                            if word["start"] > duration_seconds:
                                word["start"] = duration_seconds

            slot_sec = (float(turn.get("end", 0)) - float(turn.get("start", 0))) if turn.get("end") else 0.0
            print(
                f"[TTS] turn {idx+1} done: tts={duration_seconds:.3f}s slot={slot_sec:.3f}s"
                + (" FREE" if is_single_speaker else f" CLIP_CHECK"),
                flush=True,
            )

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
            speed: Speech speed multiplier (0.9-1.2, default 1.0)
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
            voice_settings_final["speed"] = max(0.9, min(1.2, speed))
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
            speed: Speech speed multiplier (0.9-1.2, default 1.0)
            
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
            voice_settings_final["speed"] = max(0.9, min(1.2, speed))
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
        
        # Speed adjustment is handled by FFmpeg tempo in video.py (0.9-1.25x range)
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
            turn_start = turn.get("start", 0)
            turn_end = turn.get("end", 0)
            turn_duration = turn_end - turn_start if turn_end > turn_start else 0
            
            # Log both EN and RU with timings for debugging
            logger.info(
                "TTD TURN %d [%.1f-%.1fs, %.1fs]: EN='%s' → RU='%s'",
                idx,
                turn_start, turn_end, turn_duration,
                text_en[:100] + ("..." if len(text_en) > 100 else ""),
                (text_ru[:100] + ("..." if len(text_ru) > 100 else "")) if text_ru else "<MISSING>"
            )
            
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
        segment_timing: dict[int, dict] | None = None,
        leading_sec: float = 0.0,
        force_transcribe: bool = False,
    ) -> dict[int, list[dict]]:
        """
        Get word-level timestamps by running WhisperX on the generated TTD audio.

        Modes (chosen automatically):

        1. **Full transcription** (``force_transcribe=True``):
           Runs Whisper large-v3 ASR → natural word timestamps from the main
           model (excellent Russian quality).  Results are matched to turns via
           time-based grouping + fuzzy alignment.  This is the preferred mode
           for subtitle generation because Whisper ASR produces far more
           accurate timestamps than wav2vec2 forced alignment for Russian.

        2. **Forced alignment** (``force_transcribe=False``, default):
           Passes known text + approximate boundaries to whisperx.align
           (wav2vec2).  Faster (~20-30 s) but mediocre Russian accuracy.
           Used for PHRASE_SYNC raw-audio pass where speed matters more.

        3. **Full transcription fallback**: when segment_timing is absent.

        Args:
            audio_path:        Path to the generated TTD audio file.
            dialogue_turns:    Original dialogue turns with text_ru.
            inputs:            TTD API inputs (text after emotion-tag stripping).
            segment_timing:    Dict {turn_idx: {start, end}} from TTD voice_segments.
            leading_sec:       Seconds of leading silence prepended to the audio.
            force_transcribe:  Always use full transcription (skip wav2vec2).

        Returns:
            Dict mapping turn index → list of word dicts
            {0: [{"word": "Привет", "start": 0.1, "end": 0.5}, ...], ...}
        """
        import json
        import subprocess
        import tempfile

        mode_label = "FULL-TRANSCRIBE" if force_transcribe else "FORCED-ALIGN"
        logger.info("TTD WHISPER [%s]: Getting timestamps for %s", mode_label, audio_path)

        python_path = os.getenv(
            "EXTERNAL_ASR_PY",
            "/opt/youtube-shorts-generator/venv-asr/bin/python"
        )
        script_path = os.getenv(
            "EXTERNAL_ASR_SCRIPT",
            "/opt/youtube-shorts-generator/backend/tools/transcribe.py"
        )

        if not os.path.exists(python_path) or not os.path.exists(script_path):
            logger.warning(
                "TTD WHISPER: External ASR not available (python=%s, script=%s), falling back to TTD alignment",
                python_path, script_path
            )
            return {}

        # ── Build forced-alignment segments (only when NOT force_transcribe) ──
        forced_segments_path: str | None = None
        use_forced = bool(segment_timing) and not force_transcribe

        if use_forced:
            _SPLIT_THRESHOLD = 10
            _MIN_SUB_WORDS = 4
            _SPLIT_PUNCT = set(".!?:;,\u2014\u2013")
            segments_for_align = []
            num_inputs = min(len(inputs), len(dialogue_turns))
            for idx in range(num_inputs):
                text = inputs[idx].get("text", "").strip() if idx < len(inputs) else ""
                if not text:
                    continue
                timing = segment_timing.get(idx, {})
                seg_start = timing.get("start", 0.0) + leading_sec
                seg_end   = timing.get("end",   0.0) + leading_sec
                if seg_end <= seg_start:
                    seg_end = seg_start + max(1.0, len(text) * 0.06)

                words = text.split()
                if len(words) > _SPLIT_THRESHOLD:
                    sub_texts: list[str] = []
                    current_words: list[str] = []
                    for w in words:
                        current_words.append(w)
                        last_char = w.rstrip()[-1:] if w.rstrip() else ""
                        is_punct = last_char in _SPLIT_PUNCT or w.strip() in ("\u2014", "\u2013")
                        if (
                            len(current_words) >= _MIN_SUB_WORDS
                            and is_punct
                        ):
                            sub_texts.append(" ".join(current_words))
                            current_words = []
                    if current_words:
                        if sub_texts and len(current_words) < _MIN_SUB_WORDS:
                            sub_texts[-1] += " " + " ".join(current_words)
                        else:
                            sub_texts.append(" ".join(current_words))

                    if len(sub_texts) > 1:
                        total_chars = max(1, sum(len(s) for s in sub_texts))
                        cursor = seg_start
                        total_dur = seg_end - seg_start
                        for si, sub in enumerate(sub_texts):
                            proportion = len(sub) / total_chars
                            sub_dur = total_dur * proportion
                            sub_end = cursor + sub_dur if si < len(sub_texts) - 1 else seg_end
                            segments_for_align.append({
                                "text":  sub,
                                "start": round(cursor, 3),
                                "end":   round(sub_end, 3),
                            })
                            cursor = sub_end
                        logger.info(
                            "TTD WHISPER: Turn %d split into %d sub-segments (%d words)",
                            idx, len(sub_texts), len(words),
                        )
                        continue

                segments_for_align.append({
                    "text":  text,
                    "start": round(seg_start, 3),
                    "end":   round(seg_end, 3),
                })

            if segments_for_align:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_forced.json", delete=False, encoding="utf-8"
                ) as tmp_forced:
                    json.dump({"segments": segments_for_align}, tmp_forced, ensure_ascii=False)
                    forced_segments_path = tmp_forced.name
                logger.info(
                    "TTD WHISPER: Forced-alignment mode — %d segments prepared",
                    len(segments_for_align),
                )
            else:
                use_forced = False
                logger.warning("TTD WHISPER: No segments for forced alignment, falling back to transcription")

        # Create temp file for output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp_out:
            output_json_path = tmp_out.name

        try:
            if use_forced and forced_segments_path:
                cmd = [
                    python_path,
                    script_path,
                    "--audio", audio_path,
                    "--language", "ru",
                    "--device", "cuda",
                    "--compute_type", "float16",
                    "--output", output_json_path,
                    "--forced-segments-json", forced_segments_path,
                ]
                logger.info("TTD WHISPER: Running forced alignment (wav2vec2)...")
            else:
                cmd = [
                    python_path,
                    script_path,
                    "--audio", audio_path,
                    "--model", "large-v3",
                    "--language", "ru",
                    "--device", "cuda",
                    "--compute_type", "float16",
                    "--output", output_json_path,
                ]
                logger.info("TTD WHISPER: Running full Whisper transcription (large-v3)...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )

            if result.stderr:
                logger.debug("TTD WHISPER stderr: %s", result.stderr[:500])

            with open(output_json_path, "r", encoding="utf-8") as f:
                whisper_data = json.load(f)

            segments = whisper_data.get("segments", [])
            logger.info("TTD WHISPER: Got %d segments from WhisperX", len(segments))

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

            # Choose matching strategy based on mode
            if force_transcribe and segment_timing:
                words_by_input = self._match_transcription_to_turns(
                    whisper_words, dialogue_turns, inputs, segment_timing,
                )
            else:
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
            try:
                os.unlink(output_json_path)
            except Exception:
                pass
            if forced_segments_path:
                try:
                    os.unlink(forced_segments_path)
                except Exception:
                    pass

    def _get_timestamps_via_forced_alignment(
        self,
        audio_path: str,
        dialogue_turns: list[dict],
        inputs: list[dict],
        segment_timing: dict[int, dict],
        leading_sec: float = 0.0,
        turn_offsets: list[float] | None = None,
    ) -> dict[int, list[dict]]:
        """
        Get word-level timestamps using ElevenLabs Forced Alignment API.

        Sends the post-PHRASE_SYNC audio together with concatenated Russian text
        to ``/v1/forced-alignment``.  The API returns word-level timestamps
        aligned to the *actual* audio — no extra offsets required.

        Returns:
            Dict mapping turn index -> list of word dicts
            {0: [{"word": "Привет", "start": 0.1, "end": 0.5}, ...], ...}
            or empty dict on failure.
        """
        import re

        num_inputs = min(len(inputs), len(dialogue_turns))
        if num_inputs == 0:
            return {}

        # Build per-turn text (strip emotion tags) and concatenate
        turn_texts: list[str] = []
        for idx in range(num_inputs):
            raw = inputs[idx].get("text", "").strip()
            clean = re.sub(r"\[[\w]+\]\s*", "", raw).strip()
            turn_texts.append(clean)

        full_text = " ".join(turn_texts)
        if not full_text.strip():
            logger.warning("FA: No text to align")
            return {}

        logger.info(
            "FA: Calling ElevenLabs Forced Alignment API (%d turns, %d chars)",
            num_inputs, len(full_text),
        )

        # Call the API
        url = f"{self.base_url}/forced-alignment"
        try:
            with open(audio_path, "rb") as audio_file:
                resp = httpx.post(
                    url,
                    headers={"xi-api-key": self.api_key},
                    files={"file": ("audio.wav", audio_file, "audio/wav")},
                    data={"text": full_text},
                    timeout=120.0,
                )
            resp.raise_for_status()
            fa_data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "FA: API returned %s: %s",
                exc.response.status_code,
                exc.response.text[:300] if exc.response.text else "",
            )
            return {}
        except Exception as exc:
            logger.error("FA: API call failed: %s", exc)
            return {}

        fa_words_raw = fa_data.get("words", [])
        overall_loss = fa_data.get("loss", -1)
        logger.info(
            "FA: Got %d words, overall loss=%.3f",
            len(fa_words_raw), overall_loss,
        )
        if not fa_words_raw:
            return {}

        # Convert to standard format
        fa_words = []
        for w in fa_words_raw:
            text = w.get("text", "").strip()
            start = w.get("start", 0.0)
            end = w.get("end", 0.0)
            if text and end > start:
                fa_words.append({"word": text, "start": start, "end": end})

        if not fa_words:
            logger.warning("FA: All words filtered out")
            return {}

        # Map words to turns using segment boundaries.
        # FA was run on post-PHRASE_SYNC audio, so turn boundaries need
        # leading_sec and turn_offsets applied.
        turn_boundaries: list[tuple[float, float]] = []
        for idx in range(num_inputs):
            timing = segment_timing.get(idx, {})
            t_start = timing.get("start", 0.0) + leading_sec
            t_end = timing.get("end", 0.0) + leading_sec
            if turn_offsets and idx < len(turn_offsets):
                t_start += turn_offsets[idx]
                t_end += turn_offsets[idx]
            turn_boundaries.append((t_start, t_end))

        # Assign words to turns: word belongs to the turn whose boundaries
        # contain its midpoint.  Fallback: nearest turn.
        words_by_input: dict[int, list[dict]] = {i: [] for i in range(num_inputs)}
        for w in fa_words:
            mid = (w["start"] + w["end"]) / 2
            best_turn = 0
            best_dist = float("inf")
            for idx, (tb_start, tb_end) in enumerate(turn_boundaries):
                if tb_start <= mid <= tb_end:
                    best_turn = idx
                    best_dist = 0
                    break
                dist = min(abs(mid - tb_start), abs(mid - tb_end))
                if dist < best_dist:
                    best_dist = dist
                    best_turn = idx
            words_by_input[best_turn].append(w)

        # Log summary
        for idx in range(num_inputs):
            wds = words_by_input[idx]
            if wds:
                logger.info(
                    "FA turn %d: %d words [%.2f-%.2fs] first='%s' last='%s'",
                    idx, len(wds), wds[0]["start"], wds[-1]["end"],
                    wds[0]["word"], wds[-1]["word"],
                )
            else:
                logger.warning("FA turn %d: no words assigned", idx)

        # Remove empty turns
        words_by_input = {k: v for k, v in words_by_input.items() if v}
        return words_by_input

    @staticmethod
    def _fix_whisper_overlaps(
        words_by_input: dict[int, list[dict]],
    ) -> tuple[dict[int, list[dict]], int]:
        """Fix overlapping and micro-duration words from Whisper alignment.

        Whisper sometimes produces:
        - Overlapping timestamps (word N starts before word N-1 ends)
        - Unreasonably short words (e.g. 89 ms for a 14-char word)

        For short words we expand into neighbouring gaps (forward first,
        then backward) so subtitles stay visible long enough to read.

        Returns (fixed_words, total_fixes).
        """
        _MIN_WORD_DUR = 0.15   # 150 ms absolute floor for any real word
        _CHAR_RATE = 0.035     # 35 ms per letter → target minimum
        _PUNCT_ONLY = re.compile(r'^[\W_]+$')
        total_fixes = 0

        for turn_idx, words in words_by_input.items():
            if not words:
                continue
            for i in range(len(words)):
                w = words[i]
                if i > 0:
                    prev_end = words[i - 1]["end"]
                    if w["start"] < prev_end:
                        words[i] = dict(w, start=prev_end)
                        w = words[i]
                        total_fixes += 1

                raw = w.get("word", "")
                if _PUNCT_ONLY.match(raw):
                    continue

                clean = re.sub(r'[^\w]', '', raw)
                target_dur = max(_MIN_WORD_DUR, len(clean) * _CHAR_RATE)
                dur = w["end"] - w["start"]

                if dur < target_dur:
                    deficit = target_dur - dur
                    new_start, new_end = w["start"], w["end"]

                    end_limit = (words[i + 1]["start"]
                                 if i + 1 < len(words) else new_end + deficit)
                    fwd = min(deficit, max(0.0, end_limit - new_end))
                    new_end += fwd
                    deficit -= fwd

                    if deficit > 0:
                        start_limit = (words[i - 1]["end"]
                                       if i > 0 else new_start - deficit)
                        bwd = min(deficit, max(0.0, new_start - start_limit))
                        new_start -= bwd

                    logger.info(
                        "_fix_whisper_overlaps: turn %d word %d '%s' %.0fms→%.0fms "
                        "(fwd=%.0fms bwd=%.0fms target=%.0fms)",
                        turn_idx, i, raw, dur * 1000,
                        (new_end - new_start) * 1000,
                        fwd * 1000, (w["start"] - new_start) * 1000,
                        target_dur * 1000,
                    )
                    words[i] = dict(w, start=new_start, end=new_end)
                    total_fixes += 1

        return words_by_input, total_fixes

    def _match_transcription_to_turns(
        self,
        whisper_words: list[dict],
        dialogue_turns: list[dict],
        inputs: list[dict],
        segment_timing: dict[int, dict],
    ) -> dict[int, list[dict]]:
        """
        Match words from full Whisper transcription to dialogue turns.

        Uses **global sequential alignment** (one ``SequenceMatcher`` across
        all turns) instead of per-turn matching.  This avoids the fragile
        time-boundary grouping step that broke when ``_post_segment_timing``
        boundaries drifted due to PHRASE_SYNC offset changes.

        Steps:
        1. Flatten all expected words (preserving turn membership).
        2. One ``SequenceMatcher`` call: flat expected ↔ all Whisper words.
        3. Rescue unmatched expected words via character-level similarity.
        4. Split matched results back into turns, interpolate gaps.

        Returns ``{turn_idx: [{word, start, end}, ...], ...}``
        """
        from difflib import SequenceMatcher

        num_inputs = min(len(inputs), len(dialogue_turns))

        # ── Build expected words per turn ─────────────────────────────────
        expected_words_per_turn: list[list[str]] = []
        for idx in range(num_inputs):
            text = inputs[idx].get("text", "").strip() if idx < len(inputs) else ""
            clean_text = re.sub(r'\[[\w]+\]\s*', '', text)
            expected_words_per_turn.append(clean_text.split())

        # Flatten with metadata: (turn_idx, index_within_turn, word)
        flat_items: list[tuple[int, int, str]] = []
        for t_idx, words in enumerate(expected_words_per_turn):
            for w_idx, word in enumerate(words):
                flat_items.append((t_idx, w_idx, word))

        total_expected = len(flat_items)
        logger.info(
            "TTD TRANSCRIBE MATCH: %d expected words across %d turns, %d Whisper words",
            total_expected, num_inputs, len(whisper_words),
        )

        # Turn boundaries (for interpolation only, NOT for grouping)
        turn_boundaries: list[tuple[float, float]] = []
        for idx in range(num_inputs):
            timing = segment_timing.get(idx, {})
            turn_boundaries.append((timing.get("start", 0.0), timing.get("end", 0.0)))

        # ── Global fuzzy match: all expected vs all Whisper ───────────────
        _PUNCT_STRIP = str.maketrans("", "", ".,!?;:—–-…\"'«»()\u2014\u2013")

        def _norm(word: str) -> str:
            return word.lower().strip().translate(_PUNCT_STRIP)

        flat_expected_norm = [_norm(item[2]) for item in flat_items]
        whisper_norm = [_norm(w["word"]) for w in whisper_words]

        matcher = SequenceMatcher(
            None, flat_expected_norm, whisper_norm, autojunk=False,
        )

        exp_to_wh: dict[int, int] = {}
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    exp_to_wh[i1 + k] = j1 + k
            elif op == "replace":
                n = min(i2 - i1, j2 - j1)
                for k in range(n):
                    exp_to_wh[i1 + k] = j1 + k
            elif op == "delete":
                _used_wh = set(exp_to_wh.values())
                for ei in range(i1, i2):
                    best_j, best_r = -1, 0.0
                    for ji in range(max(0, j1 - 3), min(len(whisper_words), j2 + 3)):
                        if ji in _used_wh:
                            continue
                        r = SequenceMatcher(
                            None, flat_expected_norm[ei], whisper_norm[ji],
                        ).ratio()
                        if r > best_r:
                            best_r = r
                            best_j = ji
                    if best_r >= 0.6 and best_j >= 0:
                        exp_to_wh[ei] = best_j
                        _used_wh.add(best_j)

        # ── Split results back into turns ─────────────────────────────────
        words_by_input: dict[int, list[dict]] = {}

        for turn_idx in range(num_inputs):
            expected = expected_words_per_turn[turn_idx]
            if not expected:
                continue

            # Find the flat indices belonging to this turn
            turn_offset = sum(len(expected_words_per_turn[t]) for t in range(turn_idx))

            turn_words: list[dict] = []
            for w_idx, exp_word in enumerate(expected):
                flat_idx = turn_offset + w_idx
                if flat_idx in exp_to_wh:
                    wh = whisper_words[exp_to_wh[flat_idx]]
                    turn_words.append({
                        "word": exp_word,
                        "start": wh["start"],
                        "end": wh["end"],
                        "_whisper_word": wh["word"],
                    })
                else:
                    turn_words.append({
                        "word": exp_word,
                        "_needs_interp": True,
                    })

            self._interpolate_gaps(turn_words, turn_boundaries[turn_idx])

            if turn_words:
                words_by_input[turn_idx] = turn_words
                matched = sum(1 for w in turn_words if "_whisper_word" in w)
                interp_words = [
                    w for w in turn_words if w.get("_interpolated")
                ]
                logger.info(
                    "TTD TRANSCRIBE MATCH: Turn %d: %d/%d matched (%.0f%%)",
                    turn_idx, matched, len(expected),
                    matched / len(expected) * 100 if expected else 0,
                )
                if interp_words:
                    interp_detail = ", ".join(
                        f"'{w['word']}'[{w['start']:.2f}-{w['end']:.2f}]"
                        for w in interp_words
                    )
                    logger.info(
                        "TTD TRANSCRIBE MATCH: Turn %d interpolated: %s",
                        turn_idx, interp_detail,
                    )
                if matched < len(expected):
                    logger.info(
                        "TTD TRANSCRIBE MATCH: Turn %d unmatched expected: %s",
                        turn_idx,
                        ", ".join(
                            f"'{w['word']}'" for w in turn_words
                            if w.get("_interpolated")
                        )[:200],
                    )

        total_matched = sum(
            sum(1 for w in wds if "_whisper_word" in w)
            for wds in words_by_input.values()
        )
        logger.info(
            "TTD TRANSCRIBE MATCH: Overall %d/%d words matched (%.0f%%)",
            total_matched, total_expected,
            total_matched / total_expected * 100 if total_expected else 0,
        )

        return words_by_input

    @staticmethod
    def _interpolate_gaps(
        turn_words: list[dict],
        turn_boundary: tuple[float, float],
    ) -> None:
        """Fill ``_needs_interp`` entries with timestamps from neighbours.

        Modifies *turn_words* in place.  Runs of consecutive unmatched words
        are spread evenly between the preceding and following anchor.
        """
        n = len(turn_words)
        i = 0
        while i < n:
            if not turn_words[i].get("_needs_interp"):
                i += 1
                continue
            # Found start of an unmatched run
            run_start = i
            while i < n and turn_words[i].get("_needs_interp"):
                i += 1
            run_end = i  # exclusive

            # Anchor before the run
            if run_start > 0 and "end" in turn_words[run_start - 1]:
                anchor_start = turn_words[run_start - 1]["end"]
            else:
                anchor_start = turn_boundary[0]

            # Anchor after the run
            if run_end < n and "start" in turn_words[run_end]:
                anchor_end = turn_words[run_end]["start"]
            else:
                anchor_end = turn_boundary[1]

            _MIN_INTERP_DUR = 0.25  # 250ms — readable subtitle minimum

            run_len = run_end - run_start
            dur = max(0.0, anchor_end - anchor_start)
            word_dur = dur / max(1, run_len) if dur > 0 else _MIN_INTERP_DUR
            word_dur = max(word_dur, _MIN_INTERP_DUR)

            for j in range(run_start, run_end):
                offset = j - run_start
                w_start = anchor_start + offset * word_dur
                w_end = w_start + word_dur
                turn_words[j] = {
                    "word": turn_words[j]["word"],
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "_interpolated": True,
                }

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

        # Per-turn quality check: drop turns where Whisper coverage is too low
        # or timestamps are wildly out of range (wav2vec2 alignment failure).
        _MIN_COVERAGE = 0.70
        bad_turns = []
        for turn_idx, expected_words in enumerate(expected_words_per_turn):
            if not expected_words or turn_idx not in words_by_input:
                continue
            matched = words_by_input[turn_idx]
            coverage = len(matched) / len(expected_words)
            if coverage < _MIN_COVERAGE:
                bad_turns.append(turn_idx)
                logger.warning(
                    "TTD WHISPER MATCH: Turn %d coverage %.0f%% < %.0f%% — dropping (will use fallback)",
                    turn_idx, coverage * 100, _MIN_COVERAGE * 100,
                )
                del words_by_input[turn_idx]
        if bad_turns:
            logger.info(
                "TTD WHISPER MATCH: Dropped %d low-coverage turns: %s",
                len(bad_turns), bad_turns,
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
        
        # Log the full text that will be sent to ElevenLabs (INFO level for diagnostics)
        for idx, inp in enumerate(inputs):
            vs = inp.get("voice_settings", {})
            speed_val = vs.get("speed", 1.0) if vs else 1.0
            text = inp["text"]
            logger.info(
                "TTD INPUT [turn %d]: voice=%s, speed=%.2f, text='%s'",
                idx, inp["voice_id"], speed_val, text
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

            # Run Whisper on the RAW ElevenLabs audio (before PHRASE_SYNC) to get
            # accurate per-turn word boundaries.  ElevenLabs alignment timestamps
            # are often "bunched" / compressed and can be off by several seconds,
            # causing PHRASE_SYNC cuts to land inside words.  Whisper on the raw
            # audio gives us reliable last-word-end times for each turn.
            #
            # NOTE: This Whisper call is ONLY for PHRASE_SYNC cut positions.
            # Subtitle timestamps come from ElevenLabs TTD alignment (Phase 3).
            import tempfile as _tempfile

            whisper_raw_words: dict[int, list[dict]] = {}
            _tmp_raw_path: str | None = None
            try:
                with _tempfile.NamedTemporaryFile(suffix="_raw.wav", delete=False) as _tmp_raw:
                    _tmp_raw_path = _tmp_raw.name
                audio.export(_tmp_raw_path, format="wav")
                _raw_result = self._get_timestamps_via_whisper(
                    _tmp_raw_path,
                    dialogue_turns,
                    inputs,
                    segment_timing=segment_timing_raw,
                    leading_sec=leading_sec,
                )
                if _raw_result:
                    whisper_raw_words = _raw_result
                    logger.info(
                        "PHRASE_SYNC: Whisper on raw audio succeeded — %d turns with accurate timestamps",
                        len(whisper_raw_words),
                    )
                else:
                    logger.warning("PHRASE_SYNC: Whisper on raw audio returned empty — falling back to alignment")
            except Exception as _wh_exc:
                logger.warning(
                    "PHRASE_SYNC: Whisper on raw audio failed (%s) — using alignment for cut positions",
                    _wh_exc,
                )
            finally:
                if _tmp_raw_path and os.path.exists(_tmp_raw_path):
                    try:
                        os.unlink(_tmp_raw_path)
                    except OSError:
                        pass

            # last_word_end_by_turn[i] = last word's end time for turn i (seconds, raw)
            # first_word_start_by_turn[i] = first word's start time for turn i (seconds, raw)
            # last_word_text_by_turn[i] = text of the last word in turn i
            # Prefer Whisper raw timestamps; fall back to ElevenLabs alignment.
            last_word_end_by_turn: dict[int, float] = {}
            first_word_start_by_turn: dict[int, float] = {}
            last_word_text_by_turn: dict[int, str] = {}
            if whisper_raw_words:
                for _ti, _wds in whisper_raw_words.items():
                    if _wds:
                        last_word_end_by_turn[_ti] = _wds[-1].get("end", 0.0)
                        first_word_start_by_turn[_ti] = _wds[0].get("start", 0.0)
                        last_word_text_by_turn[_ti] = _wds[-1].get("word", "")
            else:
                try:
                    _early_words = self._parse_alignment_to_words(alignment, voice_segments)
                except Exception:
                    _early_words = {}
                for _ti, _wds in _early_words.items():
                    if _wds:
                        last_word_end_by_turn[_ti] = _wds[-1].get("end", 0.0)
                        first_word_start_by_turn[_ti] = _wds[0].get("start", 0.0)
                        last_word_text_by_turn[_ti] = _wds[-1].get("word", "")
            
            # ── PHRASE-LEVEL SYNC ─────────────────────────────────────────────
            # Align each turn's START time with the original English start time.
            # When Russian TTS finishes a phrase faster than English, we insert
            # a pause so the NEXT phrase starts together with English.
            #
            # Rollback: set env var  TTD_PHRASE_SYNC=false  (no restart needed if
            # the service reads env at call-time; restart service to apply).
            #
            # Algorithm:
            #   en_start_i   = dialogue_turns[i]["start"] - segment_start   (English ref)
            #   tts_raw_i    = segment_timing_raw[i]["start"] + leading_sec  (TTD audio)
            #   output_pos_i = tts_raw_i + cumulative_offset                 (where it lands)
            #   silence      = en_start_i - output_pos_i   (positive → Russian was faster)
            #
            # We only insert silence when:
            #   • Russian was faster (silence > 0)
            #   • Gain is significant (> PHRASE_SYNC_MIN_MS)
            #   • Not the very last turn (no point inserting silence at the end)
            #
            PHRASE_SYNC_ENABLED = os.getenv("TTD_PHRASE_SYNC", "true").lower() in ("true", "1", "yes")
            PHRASE_SYNC_MIN_MS  = 250   # Ignore micro-gaps below 250 ms
            PHRASE_SYNC_MAX_MS  = 1500  # Cap single insertion at 1.5 s (3 s was too jarring)

            logger.info(
                "PHRASE_SYNC: %s (min=%dms, max=%dms) | "
                "Disable with: TTD_PHRASE_SYNC=false + service restart",
                "ENABLED" if PHRASE_SYNC_ENABLED else "DISABLED",
                PHRASE_SYNC_MIN_MS, PHRASE_SYNC_MAX_MS,
            )

            segment_start_abs = dialogue_turns[0].get("start", 0.0)
            insertions: list[tuple[int, int]] = []  # (cut_ms_in_original, silence_ms)

            # ── Per-turn table ────────────────────────────────────────────────
            # Log every turn so it's easy to see what's happening in the journal.
            # Format: turn | EN start→end (dur) | RU start→end (dur) | drift | action
            logger.info("PHRASE_SYNC turn breakdown (EN=English, RU=TTD output):")
            logger.info("  %-4s  %-20s  %-20s  %-9s  %s",
                        "turn", "EN start→end (dur)", "RU start→end (dur)", "drift", "action")

            for i in range(1, len(dialogue_turns)):
                # English reference: when this turn should start (relative to segment)
                en_start_i = dialogue_turns[i].get("start", 0.0) - segment_start_abs
                en_end_i   = dialogue_turns[i].get("end",   0.0) - segment_start_abs
                en_dur_i   = max(0.0, en_end_i - en_start_i)

                # TTD raw timing for this turn
                tts_raw_start = segment_timing_raw.get(i, {}).get("start", 0.0) + leading_sec
                tts_raw_end   = segment_timing_raw.get(i, {}).get("end",   0.0) + leading_sec
                tts_dur_i     = max(0.0, tts_raw_end - tts_raw_start)

                # Where this turn currently lands in output after prior insertions
                output_pos_i = tts_raw_start + cumulative_offset

                silence_needed_sec = en_start_i - output_pos_i

                if PHRASE_SYNC_ENABLED and silence_needed_sec * 1000 >= PHRASE_SYNC_MIN_MS:
                    _prev_word = last_word_text_by_turn.get(i - 1, "")
                    _is_sentence_end = bool(
                        _prev_word and _prev_word.rstrip()[-1:] in ".!?…"
                    )
                    if not _is_sentence_end:
                        logger.info(
                            "PHRASE_SYNC turn %d: SKIP — prev word %r not sentence-end "
                            "(need +%dms but unsafe to cut mid-speech)",
                            i, _prev_word, int(silence_needed_sec * 1000),
                        )
                        action = (
                            f"SKIP mid-speech (would +{int(silence_needed_sec*1000)}ms, "
                            f"last={_prev_word!r})"
                        )
                    else:
                        _MIN_SAFE_GAP_MS = 100  # Skip cuts if ElevenLabs gap is too tiny (risk of mid-word cut)

                        # ElevenLabs turn boundaries (authoritative source of truth)
                        el_prev_end = segment_timing_raw.get(i - 1, {}).get("end", 0.0) + leading_sec
                        el_next_start = segment_timing_raw.get(i, {}).get("start", 0.0) + leading_sec
                        el_gap_ms = int((el_next_start - el_prev_end) * 1000)

                        if el_gap_ms < _MIN_SAFE_GAP_MS:
                            logger.info(
                                "PHRASE_SYNC turn %d: SKIP — gap=%dms < %dms "
                                "(el_end=%.3fs, el_next=%.3fs, word=%r)",
                                i, el_gap_ms, _MIN_SAFE_GAP_MS,
                                el_prev_end, el_next_start, _prev_word,
                            )
                            action = (
                                f"SKIP tiny-gap ({el_gap_ms}ms < {_MIN_SAFE_GAP_MS}ms, "
                                f"word={_prev_word!r})"
                            )
                        else:
                            silence_ms = int(min(silence_needed_sec * 1000, PHRASE_SYNC_MAX_MS))
                            cut_ms = max(0, int(el_prev_end * 1000))
                            skip_to_ms = int(el_next_start * 1000)

                            discard_ms = skip_to_ms - cut_ms
                            insertions.append((cut_ms, silence_ms, skip_to_ms))
                            cumulative_offset += (silence_ms - discard_ms) / 1000.0

                            logger.info(
                                "PHRASE_SYNC turn %d: cut=%.3fs, skip_to=%.3fs "
                                "(gap=%dms, discard=%dms, +%dms) after %r ✓",
                                i, cut_ms / 1000.0, skip_to_ms / 1000.0,
                                el_gap_ms, discard_ms, silence_ms, _prev_word,
                            )
                            action = (
                                f"+{silence_ms}ms pause, cut@{cut_ms}ms, "
                                f"skip_to@{skip_to_ms}ms (discard {discard_ms}ms) "
                                f"after {_prev_word!r} ✓"
                            )
                elif silence_needed_sec * 1000 >= PHRASE_SYNC_MIN_MS and not PHRASE_SYNC_ENABLED:
                    action = f"SYNC OFF (would +{int(silence_needed_sec*1000)}ms)"
                elif silence_needed_sec < -0.1:
                    action = f"RU longer by {int(-silence_needed_sec*1000)}ms — skip"
                else:
                    action = "in sync"

                logger.info(
                    "  %-4d  %5.2f→%5.2f (%4.2fs)   %5.2f→%5.2f (%4.2fs)  %+6.0fms  %s",
                    i,
                    en_start_i, en_end_i, en_dur_i,
                    output_pos_i, output_pos_i + tts_dur_i, tts_dur_i,
                    silence_needed_sec * 1000,
                    action,
                )

                turn_offsets[i] = cumulative_offset

            # ── Apply insertions with crossfade ───────────────────────────────
            if insertions:
                logger.info(
                    "PHRASE_SYNC: Applying %d pause(s), total +%.2fs",
                    len(insertions), sum(s for _, s, _ in insertions) / 1000.0,
                )
                offset_ms = 0
                for idx_ins, (cut_ms_orig, silence_ms, skip_to_ms_orig) in enumerate(insertions):
                    cut_ms = cut_ms_orig + offset_ms
                    skip_to_ms = skip_to_ms_orig + offset_ms
                    discarded_ms = skip_to_ms_orig - cut_ms_orig

                    logger.info(
                        "PHRASE_SYNC cut #%d: before=%.3fs, after_from=%.3fs, "
                        "discard=%dms, silence=+%dms",
                        idx_ins + 1,
                        cut_ms / 1000.0,
                        skip_to_ms / 1000.0,
                        discarded_ms,
                        silence_ms,
                    )

                    before = audio[:cut_ms]
                    after  = audio[skip_to_ms:]

                    silence = AudioSegment.silent(duration=silence_ms, frame_rate=audio.frame_rate)
                    audio = before + silence + after
                    offset_ms += silence_ms - discarded_ms

                audio_before_sec = (len(audio) - offset_ms) / 1000.0
                logger.info(
                    "PHRASE_SYNC: Done. %.2fs → %.2fs (+%.2fs total)",
                    audio_before_sec, len(audio) / 1000.0, offset_ms / 1000.0,
                )
            else:
                logger.info("PHRASE_SYNC: No pauses inserted (all turns in sync or Russian longer)")
        
        # Trailing silence so the last word's acoustic tail isn't clipped.
        # video_processor trims to speech_end + 1.5s; 300ms here is enough.
        TRAILING_SILENCE_MS = 300
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
            logger.info(
                "TTD alignment check: words_by_input=%s turns, voice_segments=%s",
                len(words_by_input) if words_by_input else 0,
                len(voice_segments) if voice_segments else 0,
            )
            # ── Choose word-level timestamp source ──────────────────────
            # Priority:
            #   1. ElevenLabs Forced Alignment API (word-level, aligned to actual audio)
            #   2. [DISABLED] ElevenLabs TTD character-level alignment
            #   3. [DISABLED] Whisper post-PHRASE_SYNC
            #   4. [DISABLED] Whisper on raw audio

            _el_alignment_active = False
            _fa_alignment_active = False
            whisper_raw_words = None  # Whisper path is disabled; kept for future use

            # 1. ElevenLabs Forced Alignment API — sends post-PHRASE_SYNC audio
            #    + text to /v1/forced-alignment, returns precise word-level timestamps.
            fa_words = self._get_timestamps_via_forced_alignment(
                str(output_path), dialogue_turns, inputs,
                segment_timing=segment_timing,
                leading_sec=leading_sec, turn_offsets=turn_offsets,
            )
            if fa_words:
                logger.info(
                    "TTD: Using ElevenLabs Forced Alignment API for subtitles (%d turns)",
                    len(fa_words),
                )
                words_by_input = fa_words
                _fa_alignment_active = True
            else:
                logger.warning("TTD: Forced Alignment API returned no data — falling back to TTD character alignment")

            # 2. [DISABLED] ElevenLabs TTD character-level alignment (fallback)
            # if not _fa_alignment_active and words_by_input:
            #     quality_ok = self._check_timing_quality(
            #         words_by_input, dialogue_turns, inputs
            #     )
            #     if not quality_ok:
            #         logger.info(
            #             "TTD: ElevenLabs char alignment quality poor — applying bunched fix"
            #         )
            #         for turn_idx, words in list(words_by_input.items()):
            #             if turn_idx < len(inputs):
            #                 timing = segment_timing.get(turn_idx, {})
            #                 words_by_input[turn_idx] = self._fix_bunched_word_timestamps(
            #                     words, timing.get("start", 0), timing.get("end", 0)
            #                 )
            #     else:
            #         logger.info("TTD: ElevenLabs char alignment quality OK")
            #     logger.info(
            #         "TTD: Using ElevenLabs TTD char alignment for subtitles (%d turns)",
            #         len(words_by_input),
            #     )
            #     _el_alignment_active = True

            # 3. [DISABLED] Whisper on POST-PHRASE_SYNC audio
            #    Kept for future use — enable by uncommenting.
            # _post_segment_timing: dict[int, dict] = {}
            # for _ti in range(min(len(inputs), len(dialogue_turns))):
            #     _raw_t = segment_timing.get(_ti, {})
            #     _off = turn_offsets[_ti] if _ti < len(turn_offsets) else 0.0
            #     _post_segment_timing[_ti] = {
            #         "start": _raw_t.get("start", 0.0) + leading_sec + _off,
            #         "end": _raw_t.get("end", 0.0) + leading_sec + _off,
            #     }
            # whisper_post_words = self._get_timestamps_via_whisper(
            #     str(output_path),
            #     dialogue_turns,
            #     inputs,
            #     segment_timing=_post_segment_timing,
            #     leading_sec=0.0,
            #     force_transcribe=True,
            # )
            # if whisper_post_words:
            #     whisper_post_words, _overlap_fixes = self._fix_whisper_overlaps(whisper_post_words)
            #     if _overlap_fixes:
            #         logger.info("TTD: Fixed %d overlapping/micro-duration Whisper words", _overlap_fixes)
            #     logger.info(
            #         "TTD: Using Whisper on post-PHRASE_SYNC audio for subtitles (%d turns)",
            #         len(whisper_post_words),
            #     )
            #     words_by_input = whisper_post_words

            # 4. [DISABLED] Whisper on raw audio fallback
            # elif whisper_raw_words:
            #     whisper_raw_words, _raw_fixes = self._fix_whisper_overlaps(whisper_raw_words)
            #     if _raw_fixes:
            #         logger.info("TTD: Fixed %d overlapping/micro-duration words in raw Whisper", _raw_fixes)
            #     logger.info(
            #         "TTD: Post-PHRASE_SYNC Whisper failed — using raw Whisper timestamps (%d turns)",
            #         len(whisper_raw_words),
            #     )
            #     words_by_input = whisper_raw_words
            
            # Apply timing to dialogue turns (with inserted silence offsets)
            num_inputs = min(len(dialogue_turns), len(inputs))

            # Derive turn-level timing from word positions or voice_segments.
            api_timed = [False] * num_inputs

            if _fa_alignment_active:
                # FA API: timestamps are absolute (aligned to post-PHRASE_SYNC audio).
                for idx in range(num_inputs):
                    wds = words_by_input.get(idx)
                    if wds:
                        s = wds[0]["start"]
                        e = wds[-1]["end"]
                        api_timed[idx] = True
                        turn = dialogue_turns[idx]
                        turn["tts_start_offset"] = s
                        turn["tts_duration"] = e - s
                        turn["tts_end_offset"] = e
                        turn["_timing_source"] = "forced_alignment"
            else:
                # ElevenLabs char alignment or fallback: use voice_segment timing
                # with leading_sec + turn_offsets applied.
                for idx in range(num_inputs):
                    timing = segment_timing.get(idx, {})
                    offset = turn_offsets[idx] if idx < len(turn_offsets) else 0.0
                    s = timing.get("start", 0) + leading_sec + offset
                    e = timing.get("end", 0) + leading_sec + offset
                    if (e - s) >= 0.1:
                        api_timed[idx] = True
                        turn = dialogue_turns[idx]
                        turn["tts_start_offset"] = s
                        turn["tts_duration"] = e - s
                        turn["tts_end_offset"] = e
                        turn["_timing_source"] = "elevenlabs_char"

            api_count = sum(api_timed)
            interp_count = num_inputs - api_count
            
            # Phase 2: distribute groups of consecutive non-API turns
            INTERP_GAP = 0.05
            idx = 0
            while idx < num_inputs:
                if api_timed[idx]:
                    idx += 1
                    continue
                
                group_start = idx
                group_end = idx
                while group_end + 1 < num_inputs and not api_timed[group_end + 1]:
                    group_end += 1
                
                # Find available window
                window_start = leading_sec
                if group_start > 0:
                    prev = dialogue_turns[group_start - 1]
                    window_start = prev.get("tts_end_offset", 0) + INTERP_GAP
                
                window_end = duration_sec
                if group_end + 1 < num_inputs:
                    nxt = dialogue_turns[group_end + 1]
                    window_end = nxt.get("tts_start_offset", duration_sec) - INTERP_GAP
                
                group_size = group_end - group_start + 1
                available = max(0.1, window_end - window_start)
                gap_total = max(0, (group_size - 1)) * INTERP_GAP
                available_speech = available - gap_total
                
                word_counts = []
                total_words = 0
                for g in range(group_start, group_end + 1):
                    text = inputs[g]["text"] if g < len(inputs) else ""
                    wc = max(1, len(text.split()))
                    word_counts.append(wc)
                    total_words += wc
                
                if available_speech < group_size * 0.3:
                    logger.warning(
                        "TTD INTERPOLATE: turns %d-%d only %.2fs for %d turns (%d words)",
                        group_start, group_end, available, group_size, total_words
                    )
                
                current = window_start
                for i, g_idx in enumerate(range(group_start, group_end + 1)):
                    turn = dialogue_turns[g_idx]
                    proportion = word_counts[i] / total_words if total_words > 0 else 1.0 / group_size
                    dur = max(0.3, available_speech * proportion)
                    
                    turn["tts_start_offset"] = current
                    turn["tts_duration"] = dur
                    turn["tts_end_offset"] = current + dur
                    turn["_timing_source"] = "interpolated"
                    
                    text = inputs[g_idx]["text"] if g_idx < len(inputs) else ""
                    logger.info(
                        "TTD turn %d: interpolated %.2f-%.2fs (%.2fs, %d words) [group %d-%d, %.1f%%]",
                        g_idx, current, current + dur, dur, word_counts[i],
                        group_start, group_end, proportion * 100,
                    )
                    current = current + dur + INTERP_GAP
                
                idx = group_end + 1
            
            # Phase 3: assign word-level timestamps to ALL turns
            # FA timestamps are already absolute (post-PHRASE_SYNC audio) → no offsets.
            # Whisper raw timestamps need turn_offsets. EL alignment needs leading_sec.
            if _fa_alignment_active:
                _base = 0.0
            elif _el_alignment_active or not whisper_raw_words:
                _base = leading_sec
            else:
                _base = 0.0
            logger.info(
                "TTD Phase 3: _base=%.3fs (fa=%s, el_align=%s, whisper=%s, leading=%.3fs)",
                _base, _fa_alignment_active, _el_alignment_active,
                bool(whisper_raw_words), leading_sec,
            )
            for idx in range(num_inputs):
                turn = dialogue_turns[idx]
                # FA: offsets already baked into timestamps. Others: apply turn_offsets.
                offset = 0.0 if _fa_alignment_active else (
                    turn_offsets[idx] if idx < len(turn_offsets) else 0.0
                )
                start_time = turn.get("tts_start_offset", 0)
                end_time = turn.get("tts_end_offset", 0)
                turn_duration = end_time - start_time
                
                if idx in words_by_input:
                    tts_words = []
                    for w in words_by_input[idx]:
                        tts_words.append({
                            "word": w["word"],
                            "start": w["start"] + _base + offset,
                            "end": w["end"] + _base + offset,
                        })
                    
                    # Cap ALL words with anomalously long durations.
                    # Whisper forced alignment sometimes gives absurd results
                    # (e.g., "что" lasting 15 seconds). Cap based on char count.
                    _MAX_WORD_DUR = 3.0
                    for i, tw in enumerate(tts_words):
                        char_count = max(1, len(tw["word"]))
                        # ~100ms per char + 200ms base, capped at 3s
                        natural_max_dur = min(_MAX_WORD_DUR, char_count * 0.10 + 0.2)
                        actual_dur = tw["end"] - tw["start"]
                        if actual_dur > natural_max_dur:
                            new_end = tw["start"] + natural_max_dur
                            logger.info(
                                "TTD turn %d word %d: capping '%s' duration %.2f→%.2fs",
                                idx, i, tw["word"], actual_dur, natural_max_dur
                            )
                            tts_words[i] = dict(tw, end=new_end)

                    # Safety net: expand micro-duration words into neighbouring gaps.
                    # Mirrors _fix_whisper_overlaps but runs on final tts_words
                    # (after _base/offset/scaling), catching any words that slipped
                    # through earlier stages.
                    _PHASE3_MIN_DUR = 0.15
                    _PHASE3_CHAR_RATE = 0.035
                    _PHASE3_PUNCT = re.compile(r'^[\W_]+$')
                    _phase3_fixes = 0
                    for i, tw in enumerate(tts_words):
                        if _PHASE3_PUNCT.match(tw["word"]):
                            continue
                        clean = re.sub(r'[^\w]', '', tw["word"])
                        target = max(_PHASE3_MIN_DUR, len(clean) * _PHASE3_CHAR_RATE)
                        dur = tw["end"] - tw["start"]
                        if dur >= target:
                            continue
                        deficit = target - dur
                        ns, ne = tw["start"], tw["end"]
                        # Expand forward
                        end_lim = tts_words[i + 1]["start"] if i + 1 < len(tts_words) else ne + deficit
                        fwd = min(deficit, max(0.0, end_lim - ne))
                        ne += fwd
                        deficit -= fwd
                        # Expand backward
                        if deficit > 0:
                            start_lim = tts_words[i - 1]["end"] if i > 0 else ns - deficit
                            bwd = min(deficit, max(0.0, ns - start_lim))
                            ns -= bwd
                        if ne - ns > dur + 0.001:
                            logger.info(
                                "TTD Phase3 expand turn %d word %d '%s': %.0fms→%.0fms (target %.0fms)",
                                idx, i, tw["word"], dur * 1000, (ne - ns) * 1000, target * 1000,
                            )
                            tts_words[i] = dict(tw, start=ns, end=ne)
                            _phase3_fixes += 1
                    if _phase3_fixes:
                        logger.info("TTD Phase3: expanded %d micro-duration words in turn %d", _phase3_fixes, idx)

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
                    
                    turn_text_ru = turn.get("text_ru") or turn.get("text", "")
                    clean_text = re.sub(r'\[[\w]+\]\s*', '', turn_text_ru)
                    yandex_format = {
                        "text": clean_text,
                        "words": tts_words,
                        "duration": turn_duration,
                        "language": "ru"
                    }
                    logger.info(
                        "TTD TURN %d JSON (Yandex format):\n%s",
                        idx,
                        json.dumps(yandex_format, ensure_ascii=False, indent=2)
                    )
                else:
                    text = inputs[idx]["text"] if idx < len(inputs) else ""
                    clean_text = re.sub(r'\[[\w]+\]\s*', '', text)
                    words = clean_text.split()
                    if words and turn_duration > 0:
                        # Proportional timing by character count (longer words get more time)
                        char_counts = [max(1, len(w)) for w in words]
                        total_chars = sum(char_counts)
                        tts_words = []
                        cursor = start_time
                        for w_idx, word in enumerate(words):
                            word_dur = turn_duration * char_counts[w_idx] / total_chars
                            w_start = cursor
                            w_end = cursor + word_dur
                            tts_words.append({
                                "word": word,
                                "start": w_start,
                                "end": w_end,
                            })
                            cursor = w_end
                        turn["tts_words"] = tts_words
                        logger.debug(
                            "TTD turn %d: generated %d proportional word timestamps (%.2f-%.2fs)",
                            idx, len(tts_words), start_time, end_time
                        )
                        
                        yandex_format = {
                            "text": clean_text,
                            "words": tts_words,
                            "duration": turn_duration,
                            "language": "ru"
                        }
                        logger.info(
                            "TTD TURN %d JSON (Yandex format, estimated):\n%s",
                            idx,
                            json.dumps(yandex_format, ensure_ascii=False, indent=2)
                        )
                
                tts_words = turn.get("tts_words", [])
                if tts_words:
                    words_str = ", ".join([
                        f'"{w["word"]}" [{w["start"]:.3f}-{w["end"]:.3f}]'
                        for w in tts_words[:20]
                    ])
                    if len(tts_words) > 20:
                        words_str += f" ... (+{len(tts_words) - 20} more)"
                    logger.info(
                        "TTD TURN %d WORDS [%d total]: %s",
                        idx, len(tts_words), words_str
                    )
            
            # POST-PROCESSING: Fix bunched/overlapping turns.
            # Run when ElevenLabs char alignment is the fallback source (bunched timestamps).
            # Skip when FA API was used (its word-level timestamps are already accurate).
            if _el_alignment_active and not _fa_alignment_active:
                MIN_TURN_DURATION = 0.4
                TURN_GAP = 0.05
                BUNCH_THRESHOLD = 0.15

                num_turns = min(len(dialogue_turns), len(inputs))
                bunched_groups = []
                i = 0
                while i < num_turns:
                    group_start = i
                    group_end = i
                    base_end_time = dialogue_turns[i].get("tts_end_offset", 0)

                    while group_end + 1 < num_turns:
                        next_end_time = dialogue_turns[group_end + 1].get("tts_end_offset", 0)
                        if abs(next_end_time - base_end_time) < BUNCH_THRESHOLD:
                            group_end += 1
                        else:
                            break

                    if group_end > group_start:
                        bunched_groups.append((group_start, group_end))
                        i = group_end + 1
                    else:
                        i += 1

                total_fixes = 0
                for group_start_idx, group_end_idx in bunched_groups:
                    group_size = group_end_idx - group_start_idx + 1

                    first_turn = dialogue_turns[group_start_idx]
                    last_turn = dialogue_turns[group_end_idx]

                    window_start = first_turn.get("tts_start_offset", 0)
                    window_end = last_turn.get("tts_end_offset", 0)

                    if group_end_idx + 1 < num_turns:
                        next_turn_start = dialogue_turns[group_end_idx + 1].get("tts_start_offset", window_end + 2)
                        max_window_end = next_turn_start - TURN_GAP
                    else:
                        max_window_end = duration_sec

                    required_duration = group_size * MIN_TURN_DURATION + (group_size - 1) * TURN_GAP
                    current_duration = window_end - window_start

                    if current_duration < required_duration:
                        new_window_end = min(window_start + required_duration, max_window_end)
                        logger.warning(
                            "TTD BUNCHED: turns %d-%d need %.2fs but have %.2fs, extending window to %.2fs",
                            group_start_idx, group_end_idx, required_duration, current_duration, new_window_end - window_start
                        )
                        window_end = new_window_end

                    total_gaps = (group_size - 1) * TURN_GAP
                    available_for_turns = (window_end - window_start) - total_gaps
                    per_turn_duration = max(MIN_TURN_DURATION, available_for_turns / group_size)

                    logger.warning(
                        "TTD BUNCHED: turns %d-%d redistributed (%.2f-%.2fs) → %.0fms/turn",
                        group_start_idx, group_end_idx, window_start, window_end, per_turn_duration * 1000
                    )

                    current_time = window_start
                    for idx in range(group_start_idx, group_end_idx + 1):
                        turn = dialogue_turns[idx]

                        new_start = current_time
                        new_end = current_time + per_turn_duration
                        new_duration = per_turn_duration

                        turn["tts_start_offset"] = new_start
                        turn["tts_end_offset"] = new_end
                        turn["tts_duration"] = new_duration

                        words = turn.get("tts_words", [])
                        if words:
                            word_count = len(words)
                            per_word_duration = new_duration / word_count
                            for w_idx, w in enumerate(words):
                                w["start"] = new_start + w_idx * per_word_duration
                                w["end"] = new_start + (w_idx + 1) * per_word_duration

                        current_time = new_end + TURN_GAP
                        total_fixes += 1

                for idx in range(1, num_turns):
                    prev_turn = dialogue_turns[idx - 1]
                    curr_turn = dialogue_turns[idx]

                    prev_end = prev_turn.get("tts_end_offset", 0)
                    curr_start = curr_turn.get("tts_start_offset", 0)

                    if prev_end > curr_start + 0.01:
                        shift = prev_end - curr_start + TURN_GAP
                        curr_turn["tts_start_offset"] = curr_start + shift
                        curr_turn["tts_end_offset"] = curr_turn.get("tts_end_offset", 0) + shift

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
            else:
                logger.info("TTD: Whisper/FA-timed — skipping bunched-turn post-processing")
            
            # POST-PROCESSING: Merge very short turns into previous turn for subtitle display
            MIN_SUBTITLE_DURATION = 0.5
            merged_count = 0
            for idx in range(len(dialogue_turns) - 1, 0, -1):
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
            _src_counts: dict[str, int] = {}
            for t in dialogue_turns:
                src = t.get("_timing_source", "unknown")
                _src_counts[src] = _src_counts.get(src, 0) + 1
            _src_str = ", ".join(f"{v} {k}" for k, v in sorted(_src_counts.items()))
            logger.info("TTD TIMING SUMMARY: %s", _src_str or "no turns")
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

