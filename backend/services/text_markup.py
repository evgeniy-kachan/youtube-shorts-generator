"""LLM-powered text markup service for TTS enhancements."""
import logging

import ollama

from backend.config import (
    OLLAMA_HOST,
    OLLAMA_PORT,
    TTS_MARKUP_MODEL,
    TTS_MARKUP_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class TextMarkupService:
    """
    Uses Ollama-hosted LLM to lightly mark up text for TTS.
    - Adds pauses ("...") между фразами
    - Оборачивает важные слова в _подчёркивания_
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or TTS_MARKUP_MODEL
        host = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
        logger.info(f"Initializing TextMarkupService with model={self.model_name} on {host}")
        self.client = ollama.Client(host=host)

    def mark_text(self, text: str) -> str:
        if not text or not text.strip():
            return text

        prompt = (
            "Сделай текст более выразительным для синтеза речи Silero.\n"
            "Применяй только эти приёмы:\n"
            "- ставь `...` там, где естественная пауза;\n"
            "- оборачивай ключевые слова в `_` (например, _важно_);\n"
            "- не добавляй новых слов и не меняй смысл.\n"
            "Верни только обработанный текст.\n\n"
            f"Текст: {text}\n"
            "Размеченный текст:"
        )

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.2,
                    "num_predict": TTS_MARKUP_MAX_TOKENS,
                },
            )
            processed = response["response"].strip()
            if not processed:
                return text
            return processed
        except Exception as exc:
            logger.warning(f"Text markup failed, returning original text: {exc}")
            return text

