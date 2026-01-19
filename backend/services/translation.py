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
# ISOCHRONIC TRANSLATION PROMPTS v2.1
# ============================================================================

ISOCHRONIC_SYSTEM_MESSAGE = (
    "You are an expert isochronic translator for professional podcast dubbing. "
    "You produce natural Russian that matches the original spoken duration precisely. "
    "You understand technical, business, and scientific terminology. "
    "You clean speech artifacts but never distort meaning. "
    "You treat punctuation as a timing tool for TTS, not just grammar."
)

STAGE1_PROMPT = """You are an expert ISOCHRONIC TRANSLATOR for professional podcast dubbing.

=== GOAL ===
Translate English dialogue to Russian that:
1. MATCHES the original spoken duration (see TIMING TARGET below)
2. PRESERVES exact meaning (especially technical/business terms)
3. Sounds natural when spoken aloud by TTS

=== TIMING HEURISTIC (CRITICAL) ===

• Short phrases (EN < 10 words):
  RU range = 0.75–0.95× EN word count
  TARGET: aim for 0.85× (midpoint)
  Prefer abbreviations and compact phrasing.

• Medium phrases (EN 10–25 words):
  RU range = 0.9–1.1× EN word count
  TARGET: aim for 1.0× (midpoint)
  This is the ideal isochronic zone.

• Long phrases (EN > 25 words):
  RU range = 0.95–1.1× EN word count
  TARGET: aim for 1.0× (midpoint)
  Prefer punctuation and rhythm over added words.

Treat ranges as SOFT BOUNDS. Prioritize naturalness within them.
If exact midpoint sounds unnatural, stay within range.

Examples:
• EN: 8 words → TARGET: 7 RU words (0.85×), acceptable: 6–8
• EN: 15 words → TARGET: 15 RU words (1.0×), acceptable: 14–17
• EN: 30 words → TARGET: 30 RU words (1.0×), acceptable: 28–33

=== PUNCTUATION FOR TIMING (USE ACTIVELY) ===

Use punctuation to control rhythm and TTS breathing without adding words:

• Dash (—) for spoken pause:
  "Это важно — особенно в данном контексте"
  
• Comma for light pause:
  "Мы видим рост, и это стоит учитывать"
  
• Sentence split for longer pause:
  "Это важно. Особенно сейчас."

Note: Punctuation is a TIMING TOOL for TTS, not just grammar.
Long sentences without dashes cause TTS to rush unnaturally.

=== TERMINOLOGY ===

• Use ESTABLISHED Russian abbreviations: ИИ, ИТ, ВВП, API, ML
• Keep full form for terms WITHOUT common abbreviation
• Examples:
  - 'artificial intelligence' → 'ИИ'
  - 'machine learning' → 'машинное обучение' or 'ML'
  - 'quantum computing' → 'квантовые вычисления'
  - 'venture capital' → 'венчурный капитал'
  - 'blockchain' → 'блокчейн'
  - 'startup' → 'стартап'

=== NAMES & REALIA ===

• COMPANY NAMES: 
  - Use established official translation if widely known in RU media:
    'Apple' → 'Эппл', 'Microsoft' → 'Майкрософт', 'Google' → 'Гугл'
  - Otherwise, transliterate or keep original if commonly used as-is:
    'OpenAI' → 'OpenAI', 'Tesla' → 'Тесла'

• PRODUCT NAMES: Keep in English unless established RU name exists:
  'iPhone' → 'iPhone', 'ChatGPT' → 'ChatGPT'

• PERSON NAMES: Transliterate to Russian phonetics:
  'Elon Musk' → 'Илон Маск', 'Sam Altman' → 'Сэм Альтман'

• PLACES: Use established Russian names:
  'Silicon Valley' → 'Кремниевая долина', 'Wall Street' → 'Уолл-стрит'

=== TERM CONSISTENCY ===

Within a single dialogue, use CONSISTENT translations for key terms:
• If you translate 'efficiency' as 'эффективность' in turn 1, 
  use 'эффективность' (not 'производительность') in all subsequent turns.
• If a technical term appears multiple times, keep the same Russian equivalent.

This prevents "patchwork" style across turns.

=== SPEECH CLEANUP ===

Remove ONLY meaningless speech artifacts:

• False starts: 
  'Well, first, the part, I will say...' → 'Скажу так...'
  
• Fillers: 
  'um', 'uh', 'like', 'you know', 'I mean' → remove
  
• Self-corrections: 
  'What I wanted, I mean, what I want to say...' → 'Хочу сказать...'
  
• Repetitions: 
  'We need to, we need to focus...' → 'Нам нужно сосредоточиться...'

BUT KEEP all meaningful content, even if speaker hesitates.

=== TIMING FILLERS (USE SPARINGLY) ===

If cleanup removes significant time, you MAY compensate 
WITHOUT adding artificial filler words.

=== FILLER WORDS — ONLY WITH SOURCE MATCH ===

Fillers are ALLOWED only when translating a corresponding English phrase:

ALLOWED (source match required):
• "Well," → "Ну," (at start of reply)
• "I mean," / "you know," → "то есть" (actual clarification)
• "So," (beginning conclusion) → "Итак,"
• "Let me say," / "I'll say" → "Скажу так:"

FORBIDDEN (no source = no filler):
• Adding 'так', 'собственно', 'в общем-то' for timing
• Adding 'то есть' without actual clarification in original
• Any filler in the MIDDLE of a sentence after conjunctions (И, А, Но)

WRONG: "Мы в гонке" → "И, так, мы в гонке" (no source for 'так')
WRONG: "Мы участвуем" → "Мы, собственно, участвуем" (no source)
CORRECT: "Well, we're in a race" → "Ну, мы в гонке" (source match)
CORRECT: "I mean, it's important" → "То есть, это важно" (source match)

=== PRIORITY ORDER ===

1st: Technical terms, facts, numbers — NEVER change
2nd: Core meaning and speaker's arguments — PRESERVE exactly  
3rd: Natural spoken Russian flow
4th: Duration matching — via:
     • SYNONYMS from the SAME register (no meaning shift)
     • Punctuation (—) for spoken pauses
     • Syntactic expansion (add verb, pronoun if natural)
     • If still short — ACCEPT IT (tempo adjustment will handle)
     
CLARIFICATION on "word choice":
You may choose between synonyms ONLY if they:
• Have the SAME meaning
• Belong to the SAME register (formal/informal)
• Do NOT change connotation or emphasis

Example:
✓ "важный" ↔ "значимый" (same meaning, same register)
✗ "важный" → "критический" (adds emphasis — FORBIDDEN)

=== FORBIDDEN ===

Never add what's not in original:
• Quantifiers: 'многие', 'большинство'
• Modality: 'должен', 'нужно', 'следует'
• Certainty: 'точно', 'очевидно', 'безусловно'
• Scope: 'фундаментально', 'радикально'
• Temporal: 'в будущем', 'уже' (as "already")
• Evaluation: 'серьёзный', 'критический' (unless in original)

=== WORD COUNTING RULE ===

Count words as space-separated tokens:
• Hyphenated forms (какой-либо) = ONE word
• Abbreviations (ИИ, API) = ONE word
• Numbers (2024) = ONE word

=== SELF-CHECK (before output) ===

Ask yourself:

1. "Does the Russian text make any claim STRONGER, BROADER, 
    MORE CERTAIN, or MORE GENERAL than the original?"

2. "Does the Russian text introduce any NEW claim, assumption, 
    implication, or conclusion NOT explicitly stated in the original?"

3. "Did I use consistent terminology across all turns?"

If YES to #1 or #2 — revise to more literal translation.
If NO to #3 — harmonize terminology.

=== OUTPUT FORMAT ===

Respond ONLY with valid JSON:
{
  "translations": [
    {"turn": 0, "text_ru": "..."},
    {"turn": 1, "text_ru": "..."},
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
        Stage 1: Isochronic translation with timing heuristics.
        Uses STAGE1_PROMPT (v2.1) for professional podcast dubbing.
        """
        # Build dialogue representation with word counts for timing heuristic
        dialogue_parts = []
        for i, turn in enumerate(dialogue_turns):
            text = turn.get("text", "")
            speaker = turn.get("speaker", f"Speaker_{i}")
            en_words = len(text.split())
            
            # Store EN word count for later timing check
            turn["_en_words"] = en_words
            
            dialogue_parts.append(
                f"Turn {i} [{speaker}] ({en_words} EN words):\n  EN: {text}"
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
            
            # Apply translations and check timing ratios
            for item in translations:
                turn_idx = item.get("turn")
                text_ru = item.get("text_ru", "")
                if turn_idx is not None and 0 <= turn_idx < len(dialogue_turns):
                    dialogue_turns[turn_idx]["text_ru"] = text_ru
                    
                    # Calculate timing ratio
                    en_words = dialogue_turns[turn_idx].get("_en_words", 1)
                    ru_words = len(text_ru.split())
                    ratio = ru_words / en_words if en_words > 0 else 1.0
                    
                    # Determine expected range based on EN length
                    if en_words < 10:
                        target_ratio = 0.85
                        range_str = "0.75-0.95"
                        in_range = 0.75 <= ratio <= 0.95
                    elif en_words <= 25:
                        target_ratio = 1.0
                        range_str = "0.9-1.1"
                        in_range = 0.9 <= ratio <= 1.1
                    else:
                        target_ratio = 1.0
                        range_str = "0.95-1.1"
                        in_range = 0.95 <= ratio <= 1.1
                    
                    status = "✓" if in_range else ("⚠ SHORT" if ratio < target_ratio else "⚠ LONG")
                    
                    logger.info(
                        "Stage 1 Turn %d: %d EN → %d RU words (ratio=%.2f, target=%s) %s",
                        turn_idx, en_words, ru_words, ratio, range_str, status
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
            needs_expansion = False
            target_ratio = 1.0
            
            if en_words < 10:
                # Short: lower bound = 0.75×, target midpoint = 0.85×
                if ratio < 0.75:
                    needs_expansion = True
                    target_ratio = 0.85
            elif en_words <= 25:
                # Medium: lower bound = 0.9×, target midpoint = 1.0×
                if ratio < 0.9:
                    needs_expansion = True
                    target_ratio = 1.0
            else:
                # Long: lower bound = 0.95×, target midpoint = 1.0×
                if ratio < 0.95:
                    needs_expansion = True
                    target_ratio = 1.0
            
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