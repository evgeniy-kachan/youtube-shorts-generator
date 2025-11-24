"""Translation service powered by DeepSeek."""
import logging
from typing import List, Dict

from backend.config import (
    DEEPSEEK_MODEL,
    DEEPSEEK_TRANSLATION_CHUNK_SIZE,
    DEEPSEEK_TRANSLATION_TEMPERATURE,
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
    ):
        self.model_name = model_name or DEEPSEEK_MODEL
        self.chunk_size = chunk_size or DEEPSEEK_TRANSLATION_CHUNK_SIZE
        self.temperature = temperature or DEEPSEEK_TRANSLATION_TEMPERATURE
        self.client = DeepSeekClient(model=self.model_name)
        logger.info(
            "Translator initialized with DeepSeek model %s (chunk_size=%s)",
            self.model_name,
            self.chunk_size,
        )

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts, preserving order."""
        if not texts:
            return []

        translations: List[str] = []
        for start in range(0, len(texts), self.chunk_size):
            chunk = texts[start : start + self.chunk_size]
            payload = [
                {"id": f"segment_{start + idx}", "text": text}
                for idx, text in enumerate(chunk)
            ]
            chunk_result = self._translate_chunk(payload)

            for item in payload:
                data = chunk_result.get(item["id"])
                if data and isinstance(data.get("subtitle_text"), str):
                    translations.append(data["subtitle_text"].strip())
                else:
                    logger.warning(
                        "DeepSeek translation missing for %s, using original text",
                        item["id"],
                    )
                    translations.append(item["text"])

        return translations

    def _translate_chunk(self, payload: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Send a chunk of segments to DeepSeek and parse the JSON response.
        Returns a mapping {id: {"subtitle_text": str}}.
        """
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