"""Translation service using NLLB (No Language Left Behind)."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging
from typing import List
from backend.config import TRANSLATION_DEVICE, TRANSLATION_MODEL_NAME

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, model_name=TRANSLATION_MODEL_NAME, device=TRANSLATION_DEVICE):
        self.device_obj = torch.device(device)
        print(f"Loading translation model {model_name} on {self.device_obj}...")
        self.translator = pipeline(
            'translation',  # Use generic 'translation' task for NLLB
            model=model_name, 
            device=0 if device == 'cuda' else -1,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        print(f"Translation model loaded on {self.device_obj}")

    def translate(self, text: str, max_length: int = 1024) -> str:
        """
        Translate text from English to Russian.
        
        Args:
            text: Text to translate
            max_length: Maximum length of translation
            
        Returns:
            Translated text
        """
        try:
            # Specify source and target languages for NLLB model
            return self.translator(text, src_lang="eng_Latn", tgt_lang="rus_Cyrl", max_length=max_length)[0]['translation_text']
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise
    
    def translate_batch(self, texts: List[str], max_length: int = 1024, batch_size: int = 10, num_beams: int = 3) -> List[str]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            max_length: Maximum length of translations
            batch_size: Number of texts to translate at once (default: 10, reduced for GPU memory)
            
        Returns:
            List of translated texts
        """
        print(f"Translating {len(texts)} segments in batches of {batch_size}...")
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # Clear CUDA cache before processing a new batch
                if self.device_obj.type == 'cuda':
                    torch.cuda.empty_cache()

                translated_batch = self.translator(
                    batch, 
                    src_lang="eng_Latn",  # Specify source language for NLLB
                    tgt_lang="rus_Cyrl",  # Specify target language for NLLB
                    max_length=max_length,
                    num_beams=num_beams,
                    truncation=True
                )
                translations.extend([item['translation_text'] for item in translated_batch])
                
                # Clear CUDA cache after processing a batch
                if self.device_obj.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error translating batch {i//batch_size + 1}: {e}")
                # Add placeholders for failed translations
                translations.extend(["[Translation Error]" for _ in batch])

        print("Batch translation completed.")
        return translations