# llm.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Dict, Any, List, Optional

class LLM:
    def __init__(self, model_name: str, device: str = None, **kwargs):  # Add **kwargs
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1, **kwargs)  # Use pipeline and pass kwargs

    def generate(self, texts: List[str], **kwargs) -> List[str]:  # Add **kwargs
        """
        Generates text using the LLM.

        Args:
            texts (List[str]): List of input texts for the LLM.
            **kwargs: Additional keyword arguments to pass to the LLM's generate method.

        Returns:
            List[str]: List of generated texts.
        """

        outputs = self.pipe(texts, **kwargs)  # Pass kwargs to pipe
        return [output[0]['generated_text'] for output in outputs]

    def generate_from_prompt(self, prompt: str, **kwargs) -> str:
        """Generates text from a single prompt string."""

        output = self.pipe(prompt, **kwargs)  # Pass kwargs to pipe
        return output[0]['generated_text']