"""Reusable DeepSeek API client."""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from backend import config

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Lightweight wrapper around the DeepSeek chat/completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set. Please configure it in .env.")

        self.base_url = (base_url or config.DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or config.DEEPSEEK_MODEL
        self.timeout = timeout or config.DEEPSEEK_TIMEOUT

        self._client = httpx.Client(
            base_url=f"{self.base_url}/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call DeepSeek chat completion API."""
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        response = self._client.post("/chat/completions", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("DeepSeek API error: %s", exc.response.text)
            raise
        return response.json()

    @staticmethod
    def extract_text(response_json: Dict[str, Any]) -> str:
        """Get assistant text from DeepSeek response."""
        try:
            return (
                response_json["choices"][0]["message"]["content"]
                .strip()
            )
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected DeepSeek response format: {response_json}") from exc

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON payload from a string that may contain extra characters/markdown.
        """
        text = text.strip()
        if not text:
            raise ValueError("Empty response body")

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt to locate JSON object/array inside the text
        start_obj = text.find("{")
        start_arr = text.find("[")
        candidates = [idx for idx in (start_obj, start_arr) if idx != -1]
        if not candidates:
            raise ValueError(f"JSON not found in response: {text[:200]}...")

        start = min(candidates)
        # Walk backwards from end to find closing brace/bracket
        end = max(text.rfind("}"), text.rfind("]"))
        if end == -1 or end < start:
            raise ValueError(f"Incomplete JSON in response: {text[:200]}...")

        snippet = text[start : end + 1]
        return json.loads(snippet)

    def close(self):
        self._client.close()

    def generate_shorts_description(
        self,
        text_en: str,
        text_ru: str,
        duration: float,
        highlight_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate a catchy title, description, and hashtags for a short-form video.
        
        Returns:
            {
                "title": "Catchy hook title",
                "description": "2-3 sentence description",
                "hashtags": ["#shorts", "#viral", ...]
            }
        """
        prompt = f"""Ты — SMM-специалист, создающий вирусные описания для коротких видео (Shorts/Reels/TikTok).

КОНТЕНТ ВИДЕО:
Английский текст: {text_en[:1500]}
Русский перевод: {text_ru[:1500]}
Длительность: {duration:.0f} сек

ЗАДАЧА:
Создай описание для этого видео на РУССКОМ языке.

ТРЕБОВАНИЯ:
1. ЗАГОЛОВОК (title):
   - Цепляющий хук, вызывающий любопытство
   - 5-10 слов максимум
   - Можно использовать числа, вопросы, провокации
   - НЕ спойлерить главную мысль

2. ОПИСАНИЕ (description):
   - 2-3 коротких предложения
   - Интрига + призыв досмотреть
   - Эмодзи уместны (1-2 штуки)

3. ХЭШТЕГИ (hashtags):
   - 5-7 релевантных хэштегов
   - Первый всегда #shorts
   - Микс популярных и нишевых
   - На русском языке

ФОРМАТ ОТВЕТА (строго JSON):
{{
  "title": "Заголовок видео",
  "description": "Описание видео с эмодзи",
  "hashtags": ["#shorts", "#тема", "#ниша", "#viral", "#рекомендации"]
}}"""

        messages = [
            {"role": "system", "content": "Ты создаёшь вирусные описания для коротких видео. Отвечай только JSON."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(
            messages,
            temperature=0.7,  # More creative
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        text = self.extract_text(response)
        result = self.extract_json(text)
        
        # Ensure hashtags is a list
        if isinstance(result.get("hashtags"), str):
            result["hashtags"] = [h.strip() for h in result["hashtags"].split() if h.startswith("#")]
        
        # Ensure #shorts is first
        hashtags = result.get("hashtags", [])
        if hashtags and hashtags[0] != "#shorts":
            hashtags = ["#shorts"] + [h for h in hashtags if h != "#shorts"]
            result["hashtags"] = hashtags[:7]  # Max 7 hashtags
        
        logger.info("Generated description: title='%s', %d hashtags", result.get("title", "")[:30], len(result.get("hashtags", [])))
        return result


