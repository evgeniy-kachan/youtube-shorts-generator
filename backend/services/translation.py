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
        Two-stage translation: quality first, then expansion if needed.
        
        Stage 1: High-quality translation without timing constraints
        Stage 2: Expand turns that are too short, maintaining meaning and context
        
        Args:
            dialogue_turns: List of turns with 'speaker', 'text', 'start', 'end'
            segment_context: Optional context about the video/conversation
            
        Returns:
            Same list with 'text_ru' field added to each turn
        """
        if not dialogue_turns:
            return dialogue_turns
        
        logger.info("Two-stage translation: %d turns", len(dialogue_turns))
        
        # ===== STAGE 1: Quality translation =====
        dialogue_turns = self._translate_quality_first(dialogue_turns, segment_context)
        
        # ===== STAGE 2: Expand short turns =====
        dialogue_turns = self._expand_short_turns(dialogue_turns, segment_context)
        
        return dialogue_turns
    
    def _translate_quality_first(
        self, dialogue_turns: list[dict], segment_context: str = ""
    ) -> list[dict]:
        """
        Stage 1: Translate for quality without timing constraints.
        """
        # Build dialogue representation
        dialogue_parts = []
        for i, turn in enumerate(dialogue_turns):
            text = turn.get("text", "")
            speaker = turn.get("speaker", f"Speaker_{i}")
            
            dialogue_parts.append(
                f"Turn {i} [{speaker}]:\n  EN: {text}"
            )
        
        dialogue_str = "\n\n".join(dialogue_parts)
        
        prompt = (
            "You are a professional translator specializing in natural, conversational Russian.\n\n"
            "Translate the following dialogue to Russian:\n"
            "- Maintain natural conversational flow\n"
            "- Preserve the exact meaning and tone\n"
            "- Use clear, precise language (avoid filler words)\n"
            "- Keep context and coherence between turns\n\n"
        )
        
        if segment_context:
            prompt += f"CONTEXT: {segment_context}\n\n"
        
        prompt += (
            f"DIALOGUE:\n{dialogue_str}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{\n  "translations": [\n'
            '    {"turn": 0, "text_ru": "..."},\n'
            '    {"turn": 1, "text_ru": "..."},\n'
            '    ...\n'
            '  ]\n}\n'
        )
        
        logger.info("Stage 1: Quality translation for %d turns", len(dialogue_turns))
        
        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert translator who produces natural, precise Russian translations. "
                                   "Focus on quality and meaning, not length.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            response_text = DeepSeekClient.extract_text(response_json)
            parsed = DeepSeekClient.extract_json(response_text)
            translations = parsed.get("translations", [])
            
            # Apply translations
            for item in translations:
                turn_idx = item.get("turn")
                text_ru = item.get("text_ru", "")
                if turn_idx is not None and 0 <= turn_idx < len(dialogue_turns):
                    dialogue_turns[turn_idx]["text_ru"] = text_ru
                    
                    # Log result
                    duration = dialogue_turns[turn_idx].get("end", 0) - dialogue_turns[turn_idx].get("start", 0)
                    ru_words = len(text_ru.split())
                    estimated_duration = ru_words / 2.5  # Russian speech rate
                    
                    logger.info(
                        "Stage 1 Turn %d: %.1fs target | %d RU words (est. %.1fs) %s",
                        turn_idx, duration, ru_words, estimated_duration,
                        "✓" if estimated_duration >= duration * 0.8 else "⚠ SHORT"
                    )
            
            return dialogue_turns
            
        except Exception as exc:
            logger.error("Stage 1 translation failed: %s", exc, exc_info=True)
            for turn in dialogue_turns:
                if "text_ru" not in turn:
                    turn["text_ru"] = turn.get("text", "")
            return dialogue_turns
    
    def _expand_short_turns(
        self, dialogue_turns: list[dict], segment_context: str = ""
    ) -> list[dict]:
        """
        Stage 2: Expand turns that are too short for their target duration.
        """
        # Identify turns that need expansion
        turns_to_expand = []
        for i, turn in enumerate(dialogue_turns):
            text_ru = turn.get("text_ru", "")
            if not text_ru:
                continue
            
            duration = turn.get("end", 0) - turn.get("start", 0)
            ru_words = len(text_ru.split())
            estimated_duration = ru_words / 2.5
            
            # Need expansion if translation is <80% of target duration
            if estimated_duration < duration * 0.8 and duration > 2.0:
                target_words = int(duration * 2.5)
                missing_words = target_words - ru_words
                turns_to_expand.append({
                    "index": i,
                    "current_ru": text_ru,
                    "duration": duration,
                    "current_words": ru_words,
                    "target_words": target_words,
                    "missing_words": missing_words,
                })
        
        if not turns_to_expand:
            logger.info("Stage 2: No turns need expansion")
            return dialogue_turns
        
        logger.info("Stage 2: Expanding %d short turns", len(turns_to_expand))
        
        # Build expansion request with context
        expansion_parts = []
        for item in turns_to_expand:
            idx = item["index"]
            turn = dialogue_turns[idx]
            
            # Get context from neighbors
            prev_text = dialogue_turns[idx - 1].get("text_ru", "") if idx > 0 else ""
            next_text = dialogue_turns[idx + 1].get("text_ru", "") if idx < len(dialogue_turns) - 1 else ""
            
            context_lines = []
            if prev_text:
                context_lines.append(f"  Previous: \"{prev_text}\"")
            context_lines.append(f"  Current: \"{item['current_ru']}\" ({item['current_words']} words)")
            if next_text:
                context_lines.append(f"  Next: \"{next_text}\"")
            
            expansion_parts.append(
                f"Turn {idx} (need {item['target_words']} words, have {item['current_words']}):\n" +
                "\n".join(context_lines)
            )
        
        expansion_str = "\n\n".join(expansion_parts)
        
        prompt = (
            "You are expanding Russian translations to match speech timing.\n\n"
            "CRITICAL RULES:\n"
            "1. PRESERVE THE EXACT MEANING - never change the core message\n"
            "2. Add ONLY:\n"
            "   - Clarifying details that enhance understanding\n"
            "   - More descriptive words (e.g. 'видео' → 'видео в реальном времени')\n"
            "   - Natural transitions between ideas\n"
            "3. Maintain coherence with previous/next turns\n"
            "4. Keep the conversational tone natural\n\n"
        )
        
        if segment_context:
            prompt += f"CONTEXT: {segment_context}\n\n"
        
        prompt += (
            f"TURNS TO EXPAND:\n{expansion_str}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{\n  "expansions": [\n'
            '    {"turn": 0, "text_ru": "expanded version"},\n'
            '    ...\n'
            '  ]\n}\n'
        )
        
        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at expanding translations while preserving meaning. "
                                   "Add only meaningful details.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            response_text = DeepSeekClient.extract_text(response_json)
            parsed = DeepSeekClient.extract_json(response_text)
            expansions = parsed.get("expansions", [])
            
            # Apply expansions
            for item in expansions:
                turn_idx = item.get("turn")
                expanded_text = item.get("text_ru", "")
                if turn_idx is not None and 0 <= turn_idx < len(dialogue_turns):
                    old_text = dialogue_turns[turn_idx].get("text_ru", "")
                    dialogue_turns[turn_idx]["text_ru"] = expanded_text
                    
                    # Log expansion
                    old_words = len(old_text.split())
                    new_words = len(expanded_text.split())
                    duration = dialogue_turns[turn_idx].get("end", 0) - dialogue_turns[turn_idx].get("start", 0)
                    target_words = int(duration * 2.5)
                    
                    logger.info(
                        "Stage 2 Turn %d: %d → %d words (target %d) | %.1fs",
                        turn_idx, old_words, new_words, target_words, duration
                    )
                    logger.debug("  Old: %s", old_text)
                    logger.debug("  New: %s", expanded_text)
            
            return dialogue_turns
            
        except Exception as exc:
            logger.error("Stage 2 expansion failed: %s", exc, exc_info=True)
            return dialogue_turns

    def translate_single_with_timing(
        self, text: str, target_duration: float, context: str = ""
    ) -> str:
        """
        Isochronic translation for single-speaker segments.
        
        Translates text to fit approximately the target duration when spoken.
        
        Args:
            text: English text to translate
            target_duration: Target duration in seconds for the spoken result
            context: Optional context about the video/conversation
            
        Returns:
            Russian translation optimized for target duration
        """
        if not text or not text.strip():
            return text
        
        en_word_count = len(text.split())
        # Target Russian words: ~2.5 words/second
        target_ru_words = int(target_duration * 2.5)
        
        prompt = (
            "You are a professional DUBBING translator. Your task is ISOCHRONIC translation - "
            "the Russian voiceover must match the EXACT TIMING of the original English.\n\n"
            "CRITICAL RULES:\n"
            f"1. Target duration: {target_duration:.1f} seconds\n"
            f"2. Russian speech rate: ~2.5 words/second → aim for ~{target_ru_words} Russian words\n"
            "3. If the target is LONGER than natural translation - EXPAND naturally:\n"
            "   - Add clarifying phrases\n"
            "   - Use more descriptive language\n"
            "   - Add natural filler words (ну, вот, значит, конечно)\n"
            "4. If the target is SHORTER - keep it concise\n"
            "5. Maintain the meaning and emotional tone\n"
            "6. This is for voice dubbing - use natural conversational Russian\n\n"
        )
        
        if context:
            prompt += f"CONTEXT: {context}\n\n"
        
        prompt += (
            f"ENGLISH TEXT ({en_word_count} words):\n{text}\n\n"
            f"TARGET: ~{target_ru_words} Russian words to fill {target_duration:.1f} seconds\n\n"
            "Respond ONLY with the Russian translation, nothing else."
        )
        
        logger.info(
            "Isochronic single-speaker translation: %.1fs target, %d EN words → ~%d RU words",
            target_duration, en_word_count, target_ru_words
        )
        
        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert dubbing translator who matches speech timing precisely. "
                                   "Respond only with the translation, no explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            response_text = DeepSeekClient.extract_text(response_json)
            
            # Clean up response - remove quotes if present
            translation = response_text.strip().strip('"').strip("'")
            
            ru_word_count = len(translation.split())
            logger.info(
                "Isochronic translation result: %d RU words (target %d, diff %+d)",
                ru_word_count, target_ru_words, ru_word_count - target_ru_words
            )
            
            return translation
            
        except Exception as exc:
            logger.error("Isochronic single-speaker translation failed: %s", exc)
            # Fallback: return original text
            return text