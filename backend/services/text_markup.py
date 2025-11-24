"""LLM-powered text polishing service for TTS enhancements."""
import logging
import re

from backend.config import (
    DEEPSEEK_MARKUP_TEMPERATURE,
    TTS_MARKUP_MODEL,
    TTS_MARKUP_MAX_TOKENS,
)
from backend.services.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


class TextMarkupService:
    """
    Uses DeepSeek LLM to slightly smooth Russian text for TTS.
    The output must stay plain (no Markdown, ellipses, or special markers).
    """

    def __init__(self, model_name: str = None, temperature: float = None):
        self.model_name = model_name or TTS_MARKUP_MODEL
        self.temperature = temperature or DEEPSEEK_MARKUP_TEMPERATURE
        self.client = DeepSeekClient(model=self.model_name)
        logger.info("Initialized TextMarkupService with DeepSeek model: %s", self.model_name)

    def mark_text(self, text: str) -> str:
        if not text or not text.strip():
            return text

        prompt = (
            "Сделай текст более естественным для русской речи (легко перефразируй, убери повторы), "
            "но не добавляй форматирование.\n"
            "- НИКАКОГО Markdown, кавычек для акцентов, подчёркиваний, `...` и других специальных символов.\n"
            "- Возвращай только отредактированный текст, без пояснений и без пустых строк.\n\n"
            f"Текст: {text}\n"
            "Обновлённый текст:"
        )

        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Ты помогаешь создавать выразительную русскую речь для TTS.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=TTS_MARKUP_MAX_TOKENS,
            )
            processed = DeepSeekClient.extract_text(response_json)
            processed = self._sanitize(processed) if processed else text
            return processed if processed else text
        except Exception as exc:
            logger.warning("Text markup failed, returning original text: %s", exc)
            return text

    @staticmethod
    def _sanitize(text: str) -> str:
        """Remove any markdown-like symbols and ellipses."""
        cleaned = text.strip()
        cleaned = re.sub(r'[*_`~]+', '', cleaned)
        cleaned = re.sub(r'\.{3,}', '.', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

