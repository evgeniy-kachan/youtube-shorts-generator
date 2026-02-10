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

# ============================================================================
# ISOCHRONIC TRANSLATION PROMPTS v3.0 (with dynamic timing)
# ============================================================================

ISOCHRONIC_SYSTEM_MESSAGE = (
    "You are an expert isochronic translator for professional podcast dubbing. "
    "You produce natural Russian that fits precisely into given time constraints. "
    "You understand technical, business, and scientific terminology. "
    "You prioritize meaning over literal translation, but never distort facts."
)

STAGE1_PROMPT = """You are an expert ISOCHRONIC TRANSLATOR for professional podcast dubbing.

=== GOAL ===
Translate English dialogue to Russian that:
1. FITS into the specified TARGET WORD COUNT for each turn
2. PRESERVES exact meaning (especially technical/business terms)
3. Sounds natural when spoken aloud by TTS

=== TIMING RULE (CRITICAL) ===

Each turn has a TARGET WORD COUNT calculated from its duration.
Your translation MUST stay within ±15% of the target.

WHY: Russian TTS speaks ~30% slower than English.
To fit the same duration, Russian text must be SHORTER.

Formula used: target_ru_words = duration_seconds × 2.3

Example:
• Turn duration: 5.2 sec → target: ~12 RU words
• Acceptable range: 10-14 words
• Too long (>14 words): audio will be rushed or cut off

=== TEMPO CORRECTION BUFFER ===

After translation, the system can adjust audio speed by ±20%:
• Translation 10-20% SHORT → audio slows down slightly (OK)
• Translation 10-20% LONG → audio speeds up slightly (OK)  
• Translation 30%+ LONG → audio quality suffers (AVOID!)

PRIORITY: Slightly SHORT is better than TOO LONG!
When in doubt — use fewer words.

=== SHORTENING STRATEGIES ===

1. REMOVE speech artifacts (fillers):
   - 'well', 'um', 'uh', 'like', 'you know', 'kind of' → remove
   - 'I mean', 'so', 'actually' → keep only if critical for meaning

2. USE compact phrasing:
   - 'there is a possibility that' → 'возможно'
   - 'in order to' → 'чтобы'
   - 'it is important to note that' → 'важно'
   - 'the fact that' → remove or 'то, что'

3. MERGE short sentences:
   - 'He came. He saw. He conquered.' → 'Он пришёл, увидел и победил.'

4. USE dashes for pauses (instead of adding words):
   - 'which is, in fact, the main reason' → '— вот причина'

5. PREFER short synonyms:
   - 'в настоящее время' → 'сейчас'
   - 'осуществлять' → 'делать'
   - 'является' → 'это'

=== PUNCTUATION FOR RHYTHM ===

Use punctuation to control TTS pacing without adding words:
• Dash (—) — medium pause, use for emphasis
• Comma (,) — light pause
• Period (.) — sentence end, natural pause

=== TERMINOLOGY ===

• Abbreviations: 'AI' → 'ИИ', 'ML' → 'ML', 'API' → 'API'
• Tech terms: use established Russian equivalents
• Companies: 'Apple' → 'Эппл', 'Google' → 'Гугл', 'OpenAI' → 'OpenAI'
• People: transliterate — 'Elon Musk' → 'Илон Маск'
• Places: 'Silicon Valley' → 'Кремниевая долина'

=== TERM CONSISTENCY ===

Within a dialogue, use CONSISTENT translations:
• If 'efficiency' → 'эффективность' in turn 1, keep it in all turns
• Don't switch between synonyms for the same term

=== FORBIDDEN ===

Never ADD what's not in original:
• Quantifiers: 'многие', 'большинство'
• Modality: 'должен', 'нужно', 'следует'
• Certainty: 'точно', 'очевидно', 'безусловно'
• Scope: 'фундаментально', 'радикально'
• Evaluation: 'серьёзный', 'критический' (unless in original)

=== SELF-CHECK ===

Before output, verify:
1. Each turn is within ±15% of target word count
2. No meaning is lost or distorted
3. No claims are stronger than original
4. Terminology is consistent across turns

=== INPUT FORMAT ===

You will receive turns with timing info:

Turn 0 [duration: 5.2s, target: ~12 words]:
"English text here"

Turn 1 [duration: 8.1s, target: ~19 words]:
"More English text"

=== OUTPUT FORMAT ===

Respond ONLY with valid JSON:
{
  "translations": [
    {"turn": 0, "text_ru": "...", "word_count": 11},
    {"turn": 1, "text_ru": "...", "word_count": 18},
    ...
  ]
}
"""

STAGE2_EXPANSION_PROMPT = """=== TRIGGER CONDITION ===

USE THIS STAGE ONLY IF the Stage 1 translation falls BELOW the LOWER BOUND:
• Short phrases (EN < 10 words): RU < 0.75× EN words
• Medium phrases (EN 10–25 words): RU < 0.9× EN words  
• Long phrases (EN > 25 words): RU < 0.95× EN words

Your goal: bring the translation to the MIDPOINT of the acceptable range.

---

The Russian translation is TOO SHORT for the target spoken duration.

=== TASK ===
Expand the translation to fill the timing gap.

=== ALLOWED EXPANSION TYPES ===

1. SYNTACTIC EXPANSION (longer grammar):
   • "Это важно" → "Это является важным"
   • "Мы видим" → "Мы наблюдаем это"
   • "ИИ меняет" → "ИИ сейчас меняет"

2. RHYTHMIC EXPANSION (spoken cadence):
   • "сейчас" → "на данный момент"
   • "быстро" → "достаточно быстро"
   • "важно" → "важно отметить"

3. PUNCTUATION EXPANSION (add pauses without words):
   • "Это работает и даёт результат" → "Это работает — и даёт результат"
   • "Мы видим рост" → "Мы видим рост — в данный момент"

4. FILLERS — ONLY WITH SOURCE MATCH:
   • "Well," in original → "Ну," allowed
   • "I mean," in original → "то есть" allowed  
   • "So," (conclusion) → "Итак," allowed
   • NO SOURCE = NO FILLER — accept shorter translation
   
   WRONG: Adding "собственно" without source phrase
   WRONG: "И, так, мы..." (filler after conjunction)
   CORRECT: "Well, we..." → "Ну, мы..." (source match)

=== FORBIDDEN EXPANSION — STRICT ===

The expansion must NOT introduce:

1. QUANTIFICATION:
   • 'многие', 'большинство', 'значительная часть', 'немало'
   ✗ "researchers say" → "многие исследователи говорят"

2. MODALITY or OBLIGATION:
   • 'должен', 'нужно', 'следует', 'обязан', 'необходимо'
   ✗ "we can do" → "мы должны делать"

3. EVALUATION or JUDGMENT:
   • DO NOT add evaluative adjectives NOT present in original:
     'серьёзный', 'успешный', 'проблемный', 'опасный', 'критический'
   • If original says "important" → "важно/важный" IS allowed (literal)
   • If original does NOT evaluate → do NOT add evaluation
   ✗ "this issue" → "эта серьёзная проблема"

4. CERTAINTY ESCALATION:
   • 'точно', 'определённо', 'безусловно', 'несомненно', 'очевидно', 'ясно'
   ✗ "может сработать" → "точно сработает"
   ✗ "вероятно" → "очевидно"
   ✗ "мы считаем" → "мы уверены"

5. SCOPE CHANGE:
   • 'фундаментально', 'радикально', 'полностью', 'кардинально'
   ✗ "меняет" → "фундаментально меняет"

6. TEMPORAL ADDITION:
   • 'в будущем', 'со временем', 'в перспективе', 'уже' (as "already")
   ✗ "это работает" → "это будет работать в будущем"
   ✗ "ИИ меняет" → "ИИ уже меняет"
   Note: "сейчас", "в настоящее время" ARE allowed (neutral present)

7. CAUSALITY INVENTION:
   ✗ "X растёт" → "X растёт, потому что Y"
   ✗ "это важно" → "это важно для развития"

=== FINAL SELF-CHECK (internal, before output) ===

Ask yourself:

1. "Does the Russian text make any claim STRONGER, BROADER, 
    MORE CERTAIN, or MORE GENERAL than the original?"

2. "Does the Russian text introduce any NEW claim, assumption, 
    implication, or conclusion NOT explicitly stated in the original?"

If YES to EITHER — revise to more literal translation.

=== OUTPUT FORMAT ===

Respond ONLY with valid JSON:
{
  "expansions": [
    {"turn": 0, "text_ru": "expanded version"},
    ...
  ]
}
"""


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
            logger.warning("_process_payloads: No payloads provided")
            return results

        total_items = sum(len(p) for p in payloads)
        logger.info("_process_payloads: Processing %d payload groups with %d total items", len(payloads), total_items)

        if len(payloads) == 1 or self.max_parallel <= 1:
            for payload in payloads:
                chunk_result = self._translate_chunk(payload)
                logger.info("_process_payloads: Chunk returned %d results for %d items", len(chunk_result), len(payload))
                results.update(chunk_result)
            logger.info("_process_payloads: Total results after sequential processing: %d", len(results))
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
                    logger.info("_process_payloads: Chunk returned %d results for %d items", len(chunk_result), len(payload))
                    results.update(chunk_result)
                except Exception as exc:
                    logger.error(
                        "DeepSeek translation chunk failed for ids %s: %s",
                        [item["id"] for item in payload],
                        exc,
                        exc_info=True,
                    )
        
        logger.info("_process_payloads: Total results after parallel processing: %d", len(results))
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
        
        logger.debug("DeepSeek translation response (first 500 chars): %s", response_text[:500])

        try:
            parsed = DeepSeekClient.extract_json(response_text)
            results_list = parsed.get("results", [])
            logger.info("DeepSeek returned %d translation results for payload with %d items", len(results_list), len(payload))
        except Exception as exc:
            logger.error("Failed to parse DeepSeek translation response: %s", exc, exc_info=True)
            logger.error("Response text (first 1000 chars): %s", response_text[:1000])
            return {}

        result_map: Dict[str, Dict[str, str]] = {}
        for item in results_list:
            seg_id = item.get("id")
            subtitle_text = item.get("subtitle_text")
            if seg_id and isinstance(subtitle_text, str):
                result_map[seg_id] = {"subtitle_text": subtitle_text}
            else:
                logger.warning("Malformed translation item: %s", item)
        
        logger.info("Translation result map: %d items mapped from %d payload items", len(result_map), len(payload))
        if len(result_map) < len(payload):
            missing_ids = [item["id"] for item in payload if item["id"] not in result_map]
            logger.warning("Missing translations for IDs: %s", missing_ids)

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
        Stage 1: Isochronic translation with dynamic timing targets.
        Uses STAGE1_PROMPT (v3.0) with per-turn duration and target word count.
        """
        # Build dialogue representation with timing info for each turn
        dialogue_parts = []
        for i, turn in enumerate(dialogue_turns):
            text = turn.get("text", "")
            speaker = turn.get("speaker", f"Speaker_{i}")
            en_words = len(text.split())
            
            # Calculate turn duration from timestamps
            start_time = turn.get("start", 0)
            end_time = turn.get("end", 0)
            duration = end_time - start_time if end_time > start_time else len(text.split()) / 2.5
            
            # Calculate target RU word count based on duration
            # Russian TTS speaks ~2.3 words/second on average
            target_ru_words = max(3, int(duration * 2.3))
            
            # Store for later validation
            turn["_en_words"] = en_words
            turn["_duration"] = duration
            turn["_target_ru_words"] = target_ru_words
            
            dialogue_parts.append(
                f"Turn {i} [{speaker}, duration: {duration:.1f}s, target: ~{target_ru_words} words]:\n\"{text}\""
            )
        
        dialogue_str = "\n\n".join(dialogue_parts)
        
        # Use the new isochronic prompt
        prompt = STAGE1_PROMPT
        
        if segment_context:
            prompt += f"\n=== CONTEXT ===\nThis is a podcast about: {segment_context}\n"
        
        prompt += f"\n=== DIALOGUE TO TRANSLATE ===\n{dialogue_str}\n"
        
        logger.info("Stage 1: Isochronic translation for %d turns", len(dialogue_turns))
        
        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": ISOCHRONIC_SYSTEM_MESSAGE,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            response_text = DeepSeekClient.extract_text(response_json)
            parsed = DeepSeekClient.extract_json(response_text)
            translations = parsed.get("translations", [])
            
            # Apply translations and check against target word counts
            for item in translations:
                turn_idx = item.get("turn")
                text_ru = item.get("text_ru", "")
                if turn_idx is not None and 0 <= turn_idx < len(dialogue_turns):
                    dialogue_turns[turn_idx]["text_ru"] = text_ru
                    
                    # Get timing info
                    turn = dialogue_turns[turn_idx]
                    en_words = turn.get("_en_words", 1)
                    duration = turn.get("_duration", 5.0)
                    target_ru = turn.get("_target_ru_words", int(en_words * 0.7))
                    ru_words = len(text_ru.split())
                    
                    # Check if within ±15% of target
                    lower_bound = int(target_ru * 0.85)
                    upper_bound = int(target_ru * 1.15)
                    
                    if ru_words < lower_bound:
                        status = "⚠ SHORT"
                    elif ru_words > upper_bound:
                        status = "⚠ LONG"
                    else:
                        status = "✓"
                    
                    # Calculate how much tempo adjustment would be needed
                    # Assuming 2.3 words/sec for Russian TTS
                    estimated_duration = ru_words / 2.3
                    tempo_ratio = estimated_duration / duration if duration > 0 else 1.0
                    
                    logger.info(
                        "Stage 1 Turn %d [%.1fs]: %d EN → %d RU words (target: %d±15%%, tempo: %.2fx) %s",
                        turn_idx, duration, en_words, ru_words, target_ru, tempo_ratio, status
                    )
            
            # CRITICAL: Check for missing translations and retry individually
            missing_turns = []
            for idx, turn in enumerate(dialogue_turns):
                if "text_ru" not in turn or not turn["text_ru"].strip():
                    missing_turns.append(idx)
            
            # Retry translating missing turns individually
            if missing_turns:
                logger.warning(
                    "Stage 1: %d turns missing translation, retrying individually: %s",
                    len(missing_turns), missing_turns
                )
                for idx in missing_turns:
                    turn = dialogue_turns[idx]
                    original_text = turn.get("text", "")
                    if not original_text.strip():
                        continue
                    
                    try:
                        # Simple retry for single turn
                        retry_prompt = f"""Translate this single phrase to Russian (isochronic translation for dubbing):

English: "{original_text}"

Respond with ONLY the Russian translation, no JSON, no quotes, just the text."""
                        
                        retry_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are a professional translator. Translate to natural Russian."},
                                {"role": "user", "content": retry_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=200,
                        )
                        
                        translated = retry_response.choices[0].message.content.strip()
                        if translated:
                            turn["text_ru"] = translated
                            logger.info(
                                "Stage 1 Turn %d: RETRY SUCCESS! '%s' → '%s'",
                                idx, original_text[:30], translated[:30]
                            )
                        else:
                            turn["text_ru"] = original_text
                            logger.warning(
                                "Stage 1 Turn %d: RETRY EMPTY, using original: '%s...'",
                                idx, original_text[:50]
                            )
                    except Exception as e:
                        turn["text_ru"] = original_text
                        logger.error(
                            "Stage 1 Turn %d: RETRY FAILED (%s), using original: '%s...'",
                            idx, e, original_text[:50]
                        )
            
            # Final check for any remaining missing translations
            still_missing = [
                idx for idx, turn in enumerate(dialogue_turns)
                if not turn.get("text_ru", "").strip()
            ]
            if still_missing:
                logger.error(
                    "Stage 1: %d/%d turns STILL missing translation after retry (indices: %s)",
                    len(still_missing), len(dialogue_turns), still_missing
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
        Stage 2: Expand turns that fall BELOW the timing heuristic lower bound.
        
        Trigger conditions (from STAGE2_EXPANSION_PROMPT):
        - Short (EN < 10): RU < 0.75× EN words
        - Medium (EN 10-25): RU < 0.9× EN words  
        - Long (EN > 25): RU < 0.95× EN words
        """
        # Identify turns that need expansion based on timing heuristic
        turns_to_expand = []
        for i, turn in enumerate(dialogue_turns):
            text_ru = turn.get("text_ru", "")
            if not text_ru:
                continue
            
            en_words = turn.get("_en_words", len(turn.get("text", "").split()))
            ru_words = len(text_ru.split())
            ratio = ru_words / en_words if en_words > 0 else 1.0
            
            # Determine if below lower bound based on EN length
            # Russian TTS is ~30% slower, so target ratios are lower
            needs_expansion = False
            target_ratio = 0.75
            
            if en_words < 10:
                # Short: lower bound = 0.65×, target = 0.70×
                if ratio < 0.65:
                    needs_expansion = True
                    target_ratio = 0.70
            elif en_words <= 25:
                # Medium: lower bound = 0.70×, target = 0.75×
                if ratio < 0.70:
                    needs_expansion = True
                    target_ratio = 0.75
            else:
                # Long: lower bound = 0.75×, target = 0.80×
                if ratio < 0.75:
                    needs_expansion = True
                    target_ratio = 0.80
            
            if needs_expansion:
                target_words = int(en_words * target_ratio)
                turns_to_expand.append({
                    "index": i,
                    "current_ru": text_ru,
                    "en_words": en_words,
                    "current_words": ru_words,
                    "target_words": max(target_words, ru_words + 1),  # At least +1 word
                    "current_ratio": ratio,
                    "target_ratio": target_ratio,
                })
        
        if not turns_to_expand:
            logger.info("Stage 2: No turns need expansion (all within timing bounds)")
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
            original_en = turn.get("text", "")
            
            context_lines = []
            context_lines.append(f"  Original EN: \"{original_en}\" ({item['en_words']} words)")
            if prev_text:
                context_lines.append(f"  Previous RU: \"{prev_text}\"")
            context_lines.append(
                f"  Current RU: \"{item['current_ru']}\" "
                f"({item['current_words']} words, ratio={item['current_ratio']:.2f})"
            )
            if next_text:
                context_lines.append(f"  Next RU: \"{next_text}\"")
            
            expansion_parts.append(
                f"Turn {idx} (need ~{item['target_words']} RU words for ratio={item['target_ratio']}, "
                f"have {item['current_words']}):\n" +
                "\n".join(context_lines)
            )
        
        expansion_str = "\n\n".join(expansion_parts)
        
        # Use the new expansion prompt
        prompt = STAGE2_EXPANSION_PROMPT
        
        if segment_context:
            prompt += f"\n=== CONTEXT ===\nThis is a podcast about: {segment_context}\n"
        
        prompt += f"\n=== TURNS TO EXPAND ===\n{expansion_str}\n"
        
        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": ISOCHRONIC_SYSTEM_MESSAGE,
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
                    
                    # Log expansion with ratio
                    old_words = len(old_text.split())
                    new_words = len(expanded_text.split())
                    en_words = dialogue_turns[turn_idx].get("_en_words", 1)
                    old_ratio = old_words / en_words if en_words > 0 else 0
                    new_ratio = new_words / en_words if en_words > 0 else 0
                    
                    logger.info(
                        "Stage 2 Turn %d: %d → %d RU words (ratio: %.2f → %.2f)",
                        turn_idx, old_words, new_words, old_ratio, new_ratio
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