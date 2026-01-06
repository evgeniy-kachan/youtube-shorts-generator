"""Translation service powered by DeepSeek."""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import httpx

from backend.config import (
    DEEPSEEK_TRANSLATION_CHUNK_SIZE,
    DEEPSEEK_TRANSLATION_CONCURRENCY,
    DEEPSEEK_TRANSLATION_MAX_GROUP_CHARS,
    DEEPSEEK_TRANSLATION_MODEL,
    DEEPSEEK_TRANSLATION_TEMPERATURE,
    DEEPSEEK_TRANSLATION_TIMEOUT,
)
from backend.services.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


class Translator:
    """Translate English segments to Russian using DeepSeek."""

    def __init__(
        self,
        model_name: str = None,
        chunk_size: int = None,
        temperature: float = None,
        max_group_chars: int = None,
        timeout: float = None,
        max_parallel: int = None,
    ):
        self.model_name = model_name or DEEPSEEK_TRANSLATION_MODEL
        self.chunk_size = chunk_size or DEEPSEEK_TRANSLATION_CHUNK_SIZE
        self.temperature = temperature or DEEPSEEK_TRANSLATION_TEMPERATURE
        self.max_group_chars = max_group_chars or DEEPSEEK_TRANSLATION_MAX_GROUP_CHARS
        self.timeout = timeout or DEEPSEEK_TRANSLATION_TIMEOUT
        self.max_parallel = max_parallel or DEEPSEEK_TRANSLATION_CONCURRENCY
        self.client = DeepSeekClient(model=self.model_name, timeout=self.timeout)
        logger.info(
            "Translator initialized with DeepSeek model %s (chunk_size=%s, max_group_chars=%s, timeout=%ss, max_parallel=%s)",
            self.model_name,
            self.chunk_size,
            self.max_group_chars,
            self.timeout,
            self.max_parallel,
        )

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts, preserving order."""
        if not texts:
            return []

        payloads: List[List[Dict[str, str]]] = []
        index = 0
        while index < len(texts):
            payload, consumed = self._build_payload_group(texts, index)
            payloads.append(payload)
            index += consumed

        results_map = self._process_payloads(payloads)

        translations: List[str] = []
        for idx, original_text in enumerate(texts):
            segment_id = f"segment_{idx}"
            data = results_map.get(segment_id)
            if data and isinstance(data.get("subtitle_text"), str):
                translations.append(data["subtitle_text"].strip())
            else:
                if data is None:
                    logger.warning(
                        "DeepSeek translation missing for %s, using original text",
                        segment_id,
                    )
                translations.append(original_text)

        return translations

    def _process_payloads(self, payloads: List[List[Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
        """Translate payload groups sequentially or in parallel depending on config."""
        results: Dict[str, Dict[str, str]] = {}
        if not payloads:
            return results

        if len(payloads) == 1 or self.max_parallel <= 1:
            for payload in payloads:
                results.update(self._translate_chunk(payload))
            return results

        logger.info(
            "Translating %s payload groups with up to %s concurrent requests",
            len(payloads),
            self.max_parallel,
        )

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_payload = {
                executor.submit(self._translate_chunk, payload): payload for payload in payloads
            }
            for future in as_completed(future_to_payload):
                payload = future_to_payload[future]
                try:
                    chunk_result = future.result()
                    results.update(chunk_result)
                except Exception as exc:
                    logger.error(
                        "DeepSeek translation chunk failed for ids %s: %s",
                        [item["id"] for item in payload],
                        exc,
                        exc_info=True,
                    )

        return results

    def _build_payload_group(self, texts: List[str], start_index: int):
        """
        Build a group of segments respecting chunk_size and max_group_chars limits.
        Returns (payload_list, consumed_count).
        """
        payload: List[Dict[str, str]] = []
        consumed = 0
        total_chars = 0

        for offset in range(self.chunk_size):
            idx = start_index + consumed
            if idx >= len(texts):
                break

            text = texts[idx]
            text_len = len(text)

            # Avoid empty strings blocking grouping
            if not text:
                payload.append({"id": f"segment_{idx}", "text": ""})
                consumed += 1
                continue

            # If adding this text would exceed char limit and we already have entries, stop grouping
            if payload and total_chars + text_len > self.max_group_chars:
                break

            payload.append({"id": f"segment_{idx}", "text": text})
            consumed += 1
            total_chars += text_len

        if not payload:
            # Fallback to at least one item to avoid infinite loop
            payload.append({"id": f"segment_{start_index}", "text": texts[start_index]})
            consumed = 1

        return payload, consumed

    def _translate_chunk(self, payload: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Send a chunk of segments to DeepSeek and parse the JSON response.
        If DeepSeek times out, the payload is split into smaller chunks and retried.
        Returns a mapping {id: {"subtitle_text": str}}.
        """
        try:
            return self._request_translations(payload)
        except httpx.ReadTimeout as timeout_exc:
            if len(payload) <= 1:
                logger.error(
                    "DeepSeek timed out on single segment %s. Giving up.",
                    payload[0]["id"],
                )
                raise timeout_exc

            mid = max(1, len(payload) // 2)
            logger.warning(
                "DeepSeek timed out on %d segments. Splitting into %d + %d chunks.",
                len(payload),
                mid,
                len(payload) - mid,
            )

            left_result = self._translate_chunk(payload[:mid])
            right_result = self._translate_chunk(payload[mid:])

            merged = {**left_result, **right_result}
            return merged

    def _request_translations(
        self, payload: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        prompt = (
            "You are a professional RU<>EN translator specializing in DUBBING and VOICEOVER. "
            "Translate each segment to natural Russian with these CRITICAL rules:\n\n"
            "1. KEEP IT SHORT: Russian voiceover must fit the SAME duration as English original. "
            "Russian is typically 15-25% longer than English, so COMPENSATE by using:\n"
            "   - Shorter synonyms (использовать → брать, осуществлять → делать)\n"
            "   - Remove filler words (ну, вот, как бы, в общем)\n"
            "   - Simplify constructions (для того чтобы → чтобы)\n"
            "   - TARGET: translation should be ~SAME character count as original or SHORTER\n\n"
            "2. NATURAL SPEECH: This is for voice dubbing, not written text. Use spoken Russian.\n\n"
            "3. Return clean text suitable for on-screen subtitles (no markup).\n\n"
            "Respond ONLY with valid JSON in the format:\n"
            '{\n  "results": [\n    {"id": "...", "subtitle_text": "..."},\n    ...\n  ]\n}\n\n'
            "Segments:\n"
        )

        segments_str = "\n".join(
            [f"- id: {item['id']}\n  text: {item['text']}" for item in payload]
        )

        response_json = self.client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You provide accurate Russian localizations with consistent style.",
                },
                {"role": "user", "content": f"{prompt}{segments_str}"},
            ],
            temperature=self.temperature,
        )
        response_text = DeepSeekClient.extract_text(response_json)

        try:
            parsed = DeepSeekClient.extract_json(response_text)
            results_list = parsed.get("results", [])
        except Exception as exc:
            logger.error("Failed to parse DeepSeek translation response: %s", exc, exc_info=True)
            return {}

        result_map: Dict[str, Dict[str, str]] = {}
        for item in results_list:
            seg_id = item.get("id")
            subtitle_text = item.get("subtitle_text")
            if seg_id and isinstance(subtitle_text, str):
                result_map[seg_id] = {"subtitle_text": subtitle_text}
            else:
                logger.warning("Malformed translation item: %s", item)

        return result_map

    def translate_with_timings(
        self, dialogue_turns: list[dict], segment_context: str = ""
    ) -> list[dict]:
        """
        Second pass translation: isochronic translation with precise timings.
        
        Takes dialogue turns with start/end timestamps and translates each turn
        to fit approximately the same duration as the original.
        
        Args:
            dialogue_turns: List of turns with 'speaker', 'text', 'start', 'end'
            segment_context: Optional context about the video/conversation
            
        Returns:
            Same list with 'text_ru' field added to each turn
        """
        if not dialogue_turns:
            return dialogue_turns
        
        # Build dialogue representation with timings
        dialogue_parts = []
        for i, turn in enumerate(dialogue_turns):
            start = turn.get("start", 0.0)
            end = turn.get("end", 0.0)
            duration = end - start
            text = turn.get("text", "")
            speaker = turn.get("speaker", f"Speaker_{i}")
            word_count = len(text.split())
            
            dialogue_parts.append(
                f"Turn {i} [{speaker}] ({duration:.1f}s, {word_count} words):\n"
                f"  EN: {text}"
            )
        
        dialogue_str = "\n\n".join(dialogue_parts)
        
        prompt = (
            "You are a professional DUBBING translator. Your task is ISOCHRONIC translation - "
            "the Russian voiceover must match the EXACT TIMING of the original English.\n\n"
            "CRITICAL RULES:\n"
            "1. Each turn has a DURATION in seconds. Your Russian translation must be speakable "
            "in approximately that same time.\n"
            "2. If a turn is LONG (>10s) - you can EXPAND the translation, add natural phrases.\n"
            "3. If a turn is SHORT (<3s) - keep it concise.\n"
            "4. Russian speech rate: ~2.5 words/second. Calculate accordingly.\n"
            "   Example: 6.0s turn → aim for ~15 Russian words\n"
            "   Example: 18.0s turn → aim for ~45 Russian words\n"
            "5. Maintain natural conversational Russian. This is for voice dubbing.\n"
            "6. Keep the meaning and emotional tone of the original.\n\n"
        )
        
        if segment_context:
            prompt += f"CONTEXT: {segment_context}\n\n"
        
        prompt += (
            f"DIALOGUE TO TRANSLATE:\n{dialogue_str}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{\n  "translations": [\n'
            '    {"turn": 0, "text_ru": "..."},\n'
            '    {"turn": 1, "text_ru": "..."},\n'
            '    ...\n'
            '  ]\n}\n'
        )
        
        logger.info("Isochronic translation: %d turns", len(dialogue_turns))
        
        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert dubbing translator who matches speech timing precisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent output
            )
            response_text = DeepSeekClient.extract_text(response_json)
            parsed = DeepSeekClient.extract_json(response_text)
            translations = parsed.get("translations", [])
            
            # Apply translations to turns
            for item in translations:
                turn_idx = item.get("turn")
                text_ru = item.get("text_ru", "")
                if turn_idx is not None and 0 <= turn_idx < len(dialogue_turns):
                    dialogue_turns[turn_idx]["text_ru"] = text_ru
                    
                    # Log comparison
                    original = dialogue_turns[turn_idx]
                    duration = original.get("end", 0) - original.get("start", 0)
                    en_words = len(original.get("text", "").split())
                    ru_words = len(text_ru.split())
                    logger.info(
                        "Turn %d: %.1fs | EN %d words → RU %d words (target ~%d)",
                        turn_idx, duration, en_words, ru_words, int(duration * 2.5)
                    )
            
            return dialogue_turns
            
        except Exception as exc:
            logger.error("Isochronic translation failed: %s", exc, exc_info=True)
            # Fallback: keep original text_ru if exists, or use English
            for turn in dialogue_turns:
                if "text_ru" not in turn:
                    turn["text_ru"] = turn.get("text", "")
            return dialogue_turns