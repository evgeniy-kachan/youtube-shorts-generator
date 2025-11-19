"""Translation service using NLLB (No Language Left Behind)."""
import torch
from transformers import pipeline
import logging
from typing import List
from sentence_splitter import SentenceSplitter

from backend.config import TRANSLATION_DEVICE, TRANSLATION_MODEL_NAME

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, model_name=TRANSLATION_MODEL_NAME, device=TRANSLATION_DEVICE):
        self.device_obj = torch.device(device)
        print(f"Loading translation model {model_name} on {self.device_obj}...")
        self.translator = pipeline(
            'translation',
            model=model_name,
            device=0 if device == 'cuda' else -1,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        self.splitter = SentenceSplitter(language='en')
        print(f"Translation model loaded on {self.device_obj}")

    def _translate_single_text(self, text: str, max_length: int) -> str:
        """Helper to translate a single text, splitting if too long."""
        # NLLB has a max sequence length, translating long texts at once can lead to poor results.
        # Let's set a practical character limit to split by sentences.
        CHAR_LIMIT = 500  

        if len(text) < CHAR_LIMIT:
            # If text is short enough, translate it directly
            return self.translator(text, src_lang="eng_Latn", tgt_lang="rus_Cyrl", max_length=max_length)[0]['translation_text']
        else:
            # If text is too long, split into sentences, translate, and rejoin
            sentences = self.splitter.split(text=text)
            translated_sentences = self.translator(
                sentences,
                src_lang="eng_Latn",
                tgt_lang="rus_Cyrl",
                max_length=max_length,
                truncation=True
            )
            return " ".join([item['translation_text'] for item in translated_sentences])

    def translate(self, text: str, max_length: int = 1024) -> str:
        """Translate text from English to Russian."""
        try:
            return self._translate_single_text(text, max_length)
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            raise

    def translate_batch(self, texts: List[str], max_length: int = 1024, batch_size: int = 10) -> List[str]:
        """Translate multiple texts, handling long texts by splitting them."""
        print(f"Translating {len(texts)} segments...")
        translated_texts = []
        
        for text in texts:
            try:
                if self.device_obj.type == 'cuda':
                    torch.cuda.empty_cache()
                
                translated_text = self._translate_single_text(text, max_length)
                translated_texts.append(translated_text)
                
            except Exception as e:
                print(f"Error translating text: '{text[:50]}...'. Error: {e}")
                translated_texts.append("[Translation Error]")

        print("Batch translation completed.")
        return translated_texts