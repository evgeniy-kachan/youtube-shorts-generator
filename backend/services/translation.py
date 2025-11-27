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
            "You are a professional RU<>EN translator. Translate each segment to natural Russian, "
            "allowing light adaptation for clarity but preserving meaning. Return clean text suitable "
            "for on-screen subtitles (no markup). Respond ONLY with valid JSON in the format:\n"
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