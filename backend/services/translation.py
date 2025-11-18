"""Translation service using NLLB (No Language Left Behind)."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging
from typing import List

logger = logging.getLogger(__name__)


class TranslationService:
    """Translate text using NLLB model."""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cuda"):
        """
        Initialize NLLB translation model.
        
        Args:
            model_name: NLLB model name
                - facebook/nllb-200-distilled-600M (smaller, faster)
                - facebook/nllb-200-1.3B
                - facebook/nllb-200-3.3B (best quality)
            device: Device to use (cuda, cpu)
        """
        logger.info(f"Loading NLLB model: {model_name}")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # NLLB language codes
        # English: eng_Latn
        # Russian: rus_Cyrl
        self.src_lang = "eng_Latn"
        self.tgt_lang = "rus_Cyrl"
        
        logger.info(f"NLLB model loaded on {self.device}")
        
    def translate(self, text: str, max_length: int = 512) -> str:
        """
        Translate text from English to Russian.
        
        Args:
            text: Text to translate
            max_length: Maximum length of translation
            
        Returns:
            Translated text
        """
        try:
            # Set source language
            self.tokenizer.src_lang = self.src_lang
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translation
            translated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode
            translation = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            
            return translation
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise
    
    def translate_batch(self, texts: List[str], max_length: int = 512, batch_size: int = 10) -> List[str]:
        """
        Translate multiple texts in batches to avoid OOM.
        
        Args:
            texts: List of texts to translate
            max_length: Maximum length of translations
            batch_size: Number of texts to translate at once (default: 10, reduced for GPU memory)
            
        Returns:
            List of translated texts
        """
        try:
            self.tokenizer.src_lang = self.src_lang
            all_translations = []
            
            # Process in chunks to avoid OOM
            total = len(texts)
            logger.info(f"Translating {total} texts in batches of {batch_size}...")
            
            for i in range(0, total, batch_size):
                chunk = texts[i:i + batch_size]
                chunk_num = (i // batch_size) + 1
                total_chunks = (total + batch_size - 1) // batch_size
                
                logger.info(f"Translating batch {chunk_num}/{total_chunks} ({len(chunk)} texts)...")
                
                # Clear CUDA cache before each batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Tokenize chunk
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Generate translations (num_beams=3 for less GPU memory usage)
                translated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                    max_length=max_length,
                    num_beams=3,
                    early_stopping=True
                )
                
                # Decode chunk
                chunk_translations = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
                all_translations.extend(chunk_translations)
                
                # Clear cache after each batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            logger.info(f"Translation completed: {len(all_translations)} texts translated")
            return all_translations
            
        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            # Clear cache on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise

