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
        Attempts to fix common JSON errors like missing quotes.
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
        
        # Try to fix common JSON errors before parsing
        fixed_snippet = DeepSeekClient._fix_json_errors(snippet)
        
        try:
            return json.loads(fixed_snippet)
        except json.JSONDecodeError:
            # If fixed version also fails, try original
            raise ValueError(f"JSON parse error in response: {snippet[:200]}...")
    
    @staticmethod
    def _fix_json_errors(text: str) -> str:
        """
        Fix common JSON errors that DeepSeek sometimes produces:
        - Missing quotes around hashtags in arrays: [#tag] -> ["#tag"]
        - Missing quotes around string values in arrays
        """
        import re
        
        # Fix missing quotes around hashtags in arrays
        # Pattern: finds #hashtag not in quotes, after comma or opening bracket, before ] or , or "
        # Example: ["#shorts", #экономика] -> ["#shorts", "#экономика"]
        # Example: [#tag] -> ["#tag"]
        # Example: ["#shorts", #ии"] -> ["#shorts", "#ии"]  (missing quote before, but has quote after)
        def fix_hashtag_quotes(match):
            prefix = match.group(1)  # ", " or "["
            hashtag = match.group(2)  # "#tag"
            suffix = match.group(3)  # "]" or "," or """
            # If suffix is a quote, we already have the closing quote, just add opening
            if suffix == '"':
                return f'{prefix}"{hashtag}"'
            return f'{prefix}"{hashtag}"{suffix}'
        
        # Find patterns like: , #tag] or , #tag, or [ #tag] or , #tag ] or , #tag"
        # This handles the case where DeepSeek forgets quotes around hashtags
        # Pattern matches: comma/bracket, optional whitespace, # followed by word chars (including unicode), 
        # optional whitespace, then ] or , or "
        text = re.sub(r'([,\[])\s*(#[\w\u0400-\u04FF]+)\s*([,\]"])', fix_hashtag_quotes, text)
        
        return text

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str:
        """
        Simple text generation from a prompt.
        Returns just the text response (not full JSON).
        """
        messages = [{"role": "user", "content": prompt}]
        response_json = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self.extract_text(response_json)

    def close(self):
        self._client.close()

    def generate_shorts_description(
        self,
        text_en: str,
        text_ru: str,
        duration: float,
        highlight_score: float = 0.0,
        guest_name: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a catchy title, description, and hashtags for a short-form video.
        
        Returns:
            {
                "title": "Catchy hook title",
                "description": "2-3 sentence description",
                "guest_bio": "1-3 sentences about guest (if guest_name provided)",
                "hashtags": ["#kachancuts_category", ...]
            }
        """
        # Log input data for debugging
        logger.info(
            "generate_shorts_description: text_en=%d chars, text_ru=%d chars, duration=%.1f, guest=%s",
            len(text_en or ""), len(text_ru or ""), duration, guest_name or "none"
        )
        
        guest_block = ""
        guest_json_field = ""
        guest_section = ""
        hashtag_num = "4"
        if guest_name:
            guest_block = "\nГОСТЬ ПОДКАСТА: " + guest_name + "\n"
            guest_json_field = '\n  "guest_bio": "1-3 предложения о госте на русском языке",'
            guest_section = (
                "\n4. ИНФОРМАЦИЯ О ГОСТЕ (guest_bio):\n"
                "   - 1-3 предложения о госте \"" + guest_name + "\" на русском языке\n"
                "   - Кто он, чем известен, что делает\n"
                "   - Факты из открытых источников, без выдумок\n"
                "   - Если не знаешь этого человека, напиши пустую строку \"\"\n"
            )
            hashtag_num = "5"

        prompt = f"""Ты — SMM-специалист, создающий вирусные описания для коротких видео (Shorts/Reels/TikTok).

КОНТЕНТ ВИДЕО:
Английский текст: {text_en[:1500]}
Русский перевод: {text_ru[:1500]}
Длительность: {duration:.0f} сек{guest_block}

ЗАДАЧА:
Создай описание для этого видео на РУССКОМ языке.

ТРЕБОВАНИЯ:
1. КАТЕГОРИЯ (category):
   - Определи одну категорию из списка:
     саморазвитие, психология, мышление, продуктивность, мотивация,
     бизнес, финансы, карьера, лидерство,
     отношения, здоровье,
     технологии, наука, философия, общество
   - Выбери наиболее подходящую категорию по основной теме видео
   - Категория должна быть одним словом из списка выше, строчными буквами
   - Примеры: habits/mindset → мышление, startup → бизнес, mental health → психология

2. ЗАГОЛОВОК (title):
   - Цепляющий хук, вызывающий любопытство
   - 5-10 слов максимум
   - Можно использовать числа, вопросы, провокации
   - НЕ спойлерить главную мысль

3. ОПИСАНИЕ (description):
   - 1-2 коротких предложения
   - Интрига, раскрывающая суть видео
   - Эмодзи уместны (1-2 штуки)
   - ЗАПРЕЩЕНО: призывы "смотри до конца", "досмотри", "не пропусти" и подобные
{guest_section}
{hashtag_num}. ХЭШТЕГИ (hashtags):
   - 5-7 релевантных хэштегов
   - Микс популярных и нишевых
   - На русском языке

ФОРМАТ ОТВЕТА (строго JSON):
{{
  "category": "мышление",
  "title": "Заголовок видео",
  "description": "Описание видео с эмодзи",{guest_json_field}
  "hashtags": ["#тема", "#ниша", "#viral", "#рекомендации"]
}}"""

        messages = [
            {"role": "system", "content": "Ты создаёшь вирусные описания для коротких видео. Отвечай только JSON."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(
            messages,
            model="deepseek-chat",  # Force non-reasoning model for simple tasks
            temperature=0.7,  # More creative
            max_tokens=2000,  # Increased to ensure full JSON response
            response_format={"type": "json_object"}
        )
        
        # Log full response for debugging
        finish_reason = response.get("choices", [{}])[0].get("finish_reason", "unknown")
        usage = response.get("usage", {})
        logger.info(
            "DeepSeek description response: finish_reason=%s, prompt_tokens=%s, completion_tokens=%s",
            finish_reason, usage.get("prompt_tokens"), usage.get("completion_tokens")
        )
        
        text = self.extract_text(response)
        
        # Handle empty response
        if not text or text.strip() == "":
            logger.warning(
                "DeepSeek returned empty response for description generation. "
                "finish_reason=%s, full_response=%s",
                finish_reason, json.dumps(response, ensure_ascii=False)[:500]
            )
            # Return fallback
            return {
                "category": "другое",
                "title": "Интересный момент из видео",
                "description": "Мудрость, которая меняет взгляд на вещи 🔥",
                "hashtags": ["#kachancuts_другое", "#подкаст", "#мудрость"]
            }
        
        try:
            result = self.extract_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse DeepSeek response as JSON: %s, text: %s", e, text[:300])
            # Try to extract partial data from truncated JSON
            result = self._extract_partial_description(text)
            if result:
                logger.info("Recovered partial description from truncated JSON: title='%s'", result.get("title", "")[:30])
            else:
                return {
                    "category": "другое",
                    "title": "Интересный момент из видео",
                    "description": "Мудрость, которая меняет взгляд на вещи 🔥",
                    "hashtags": ["#kachancuts_другое", "#подкаст", "#мудрость"]
                }
        
        # Ensure hashtags is a list
        if isinstance(result.get("hashtags"), str):
            result["hashtags"] = [h.strip() for h in result["hashtags"].split() if h.startswith("#")]
        
        # Ensure category is set (default to "другое" if missing)
        if "category" not in result or not result.get("category"):
            result["category"] = "другое"
        
        # Build final hashtags: #kachancuts_category + 2 relevant ones
        category = result.get("category", "другое").lower().replace(" ", "_")
        brand_hashtag = f"#kachancuts_{category}"
        
        # Pick 2 best hashtags from DeepSeek response (skip generic ones)
        generic_tags = {"#shorts", "#viral", "#рекомендации", "#интересное", "#факты", "#trending", "#fyp"}
        original_hashtags = result.get("hashtags", [])
        relevant_hashtags = [h for h in original_hashtags if h.lower() not in generic_tags][:2]
        
        # If not enough relevant ones, add defaults based on category
        if len(relevant_hashtags) < 2:
            category_defaults = {
                "саморазвитие":  ["#рост", "#успех"],
                "психология":    ["#психология", "#осознанность"],
                "мышление":      ["#мышление", "#саморазвитие"],
                "продуктивность":["#продуктивность", "#привычки"],
                "мотивация":     ["#мотивация", "#цели"],
                "бизнес":        ["#бизнес", "#предпринимательство"],
                "финансы":       ["#финансы", "#деньги"],
                "карьера":       ["#карьера", "#успех"],
                "лидерство":     ["#лидерство", "#управление"],
                "отношения":     ["#отношения", "#общение"],
                "здоровье":      ["#здоровье", "#энергия"],
                "технологии":    ["#технологии", "#будущее"],
                "наука":         ["#наука", "#знания"],
                "философия":     ["#философия", "#смысл"],
                "общество":      ["#общество", "#культура"],
            }
            defaults = category_defaults.get(result.get("category", "").lower(), ["#подкаст", "#мудрость"])
            relevant_hashtags.extend(defaults)
            relevant_hashtags = relevant_hashtags[:2]
        
        result["hashtags"] = [brand_hashtag] + relevant_hashtags
        
        logger.info(
            "Generated description: category='%s', title='%s', %d hashtags",
            result.get("category", "другое"),
            result.get("title", "")[:30],
            len(result.get("hashtags", []))
        )
        return result

    def _extract_partial_description(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract title and description from a truncated JSON response.
        Returns None if extraction fails.
        """
        import re
        result = {}
        
        # Extract title
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', text)
        if title_match:
            result["title"] = title_match.group(1)
        
        # Extract description (may be truncated)
        desc_match = re.search(r'"description"\s*:\s*"([^"]*)', text)
        if desc_match:
            desc = desc_match.group(1)
            # Clean up truncated description
            if not desc.endswith('"'):
                desc = desc.rstrip() + "..."
            result["description"] = desc
        
        # Extract hashtags if present
        hashtags_match = re.search(r'"hashtags"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if hashtags_match:
            hashtags_str = hashtags_match.group(1)
            # First try to find quoted hashtags
            hashtags = re.findall(r'"(#[^"]+)"', hashtags_str)
            # If not found, try to find unquoted hashtags (common DeepSeek error)
            # Pattern matches # followed by word chars (including unicode), before ] or ,
            if not hashtags:
                hashtags = re.findall(r'(#[\w\u0400-\u04FF]+)', hashtags_str)
            if hashtags:
                result["hashtags"] = hashtags
        
        # If we got at least title, return with defaults for missing fields
        if result.get("title"):
            if "category" not in result:
                result["category"] = "другое"
            if "description" not in result:
                result["description"] = "Смотрите до конца! 🔥"
            if "hashtags" not in result:
                category = result.get("category", "другое").lower().replace(" ", "_")
                result["hashtags"] = [f"#kachancuts_{category}", "#подкаст", "#мудрость"]
            return result
        
        return None


