"""LLM-powered text polishing service for TTS enhancements."""
import logging
import re

import os
from pathlib import Path
import json

from backend.config import (
    DEEPSEEK_MARKUP_TEMPERATURE,
    TTS_MARKUP_MODEL,
    TTS_MARKUP_MAX_TOKENS,
    TEMP_DIR,
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
            "Ты помогаешь сделать русский текст более естественным для озвучки.\n"
            "Правила:\n"
            "- Используй только исходный текст, не добавляй новые мысли.\n"
            "- Легко перефразируй, убери повторы, но сохраняй смысл.\n"
            "- НИКАКОГО форматирования: никаких Markdown, кавычек для акцентов, подчёркиваний, `...`.\n"
            "- Ответ строго в формате JSON:\n"
            '{"text": "<итоговый текст без кавычек и пояснений>"}\n\n'
            f"Текст: {text}\n"
            "Верни JSON с итоговым текстом:"
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
            cleaned_text = self._extract_text_from_json(processed)
            cleaned_text = self._sanitize(cleaned_text) if cleaned_text else text
            final_text = cleaned_text if cleaned_text else text
            debug_flag = os.getenv("DEBUG_SAVE_MARKUP", "0") == "1"
            logger.debug("TextMarkup debug flag=%s", debug_flag)
            if debug_flag:
                debug_dir = Path(TEMP_DIR) / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / "text_markup.log"
                with open(debug_path, "a", encoding="utf-8") as debug_file:
                    debug_file.write("=== INPUT ===\n")
                    debug_file.write(text.strip() + "\n")
                    debug_file.write("--- OUTPUT ---\n")
                    debug_file.write(final_text.strip() + "\n\n")
            return final_text
        except Exception as exc:
            logger.warning("Text markup failed, returning original text: %s", exc)
            return text

    @staticmethod
    def _extract_text_from_json(raw_text: str) -> str:
        try:
            parsed = json.loads(raw_text)
            text = parsed.get("text")
            if isinstance(text, str):
                return text
        except Exception:
            pass
        return None

    @staticmethod
    def _sanitize(text: str) -> str:
        """Remove any markdown-like symbols and ellipses."""
        cleaned = text.strip()
        cleaned = re.sub(r'[*_`~]+', '', cleaned)
        cleaned = re.sub(r'\.{3,}', '.', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

