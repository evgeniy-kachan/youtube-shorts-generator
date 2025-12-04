"""Text-to-Speech services."""
import io
import logging
import time
from pathlib import Path
import re
from datetime import datetime
from urllib.error import HTTPError

import httpx
import torch
import torchaudio
from pydub import AudioSegment
from sentence_splitter import SentenceSplitter

from backend.config import TEMP_DIR

logger = logging.getLogger(__name__)


class BaseTTSService:
    """Common helpers shared between different TTS providers."""

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

            logger.info(
                "Synthesizing dialogue turn %s/%s: speaker=%s -> voice=%s, len=%s (linear)",
                idx + 1,
                len(dialogue_turns),
                speaker_id,
                voice,
                len(text),
            )

            temp_turn_path = Path(output_path).parent / f"dialogue_turn_{idx}_{Path(output_path).name}"

            try:
                self.synthesize(text, str(temp_turn_path), speaker=voice)
                if temp_turn_path.exists():
                    seg_audio = AudioSegment.from_file(str(temp_turn_path))

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

                    try:
                        temp_turn_path.unlink()
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to synthesize dialogue turn %s: %s", idx, exc)
                continue

        if not audio_segments:
            raise RuntimeError("No audio generated for dialogue")

        final_audio = audio_segments[0]
        for seg_audio in audio_segments[1:]:
            final_audio += seg_audio

        final_audio.export(output_path, format="wav")
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

            logger.info(
                "Synthesizing dialogue turn %s/%s: speaker=%s -> voice=%s, len=%s, offset=%.2fs",
                idx + 1,
                len(dialogue_turns),
                speaker_id,
                voice,
                len(text),
                offset_ms / 1000,
            )

            temp_turn_path = Path(output_path).parent / f"dialogue_turn_{idx}_{Path(output_path).name}"
            try:
                self.synthesize(text, str(temp_turn_path), speaker=voice)
                if not temp_turn_path.exists():
                    continue

                seg_audio = AudioSegment.from_file(str(temp_turn_path))
                duration_seconds = len(seg_audio) / 1000.0
                relative_start = offset_ms / 1000.0
                turn["tts_start_offset"] = relative_start
                turn["tts_duration"] = duration_seconds
                turn["tts_end_offset"] = relative_start + duration_seconds

                end_ms = offset_ms + len(seg_audio)
                max_duration_ms = max(max_duration_ms, end_ms)
                layers.append((seg_audio, offset_ms))
            finally:
                try:
                    temp_turn_path.unlink(missing_ok=True)
                except Exception:
                    pass

        if not layers:
            raise RuntimeError("No audio generated for dialogue")

        final_duration = max(max_duration_ms, 1)
        final_audio = AudioSegment.silent(duration=final_duration)

        for seg_audio, offset_ms in layers:
            final_audio = final_audio.overlay(seg_audio, position=offset_ms)

        final_audio.export(output_path, format="wav")
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
            
            torchaudio.save(
                str(output_path),
                audio.unsqueeze(0),
                self.sample_rate
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

