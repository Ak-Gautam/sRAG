import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from typing import List, Dict, Optional, Union, Iterator, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLM:
    """
    Wrapper for interacting with large language models (LLMs) from Hugging Face.
    """
    def __init__(self, model_name: str, use_gpu: bool = True, task: str = "text-generation", **kwargs):
        """
        Initializes the LLM with the specified model name and task.
        """
        self.model_name = model_name
        self.task = task.lower()  # Ensure task is always lowercase
        self.device = 0 if torch.cuda.is_available() and use_gpu else -1
        self.kwargs = kwargs  # Store additional keyword arguments

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                self.is_encoder_decoder = True
            except ValueError:  
                self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                self.is_encoder_decoder = False
        except Exception as e:  
            logger.error(f"Error loading model: {e}. Check if model name/path is correct.")
            raise

        self.model.eval()

    def _generate_text(self, prompts: Union[str, List[str]], **generation_kwargs) -> Union[str, List[str]]:
        """
        Generates text for a single prompt or list of prompts.
        """

        if isinstance(prompts, str):
            input_ = self.tokenizer(prompts, return_tensors="pt").to(self.device)
            output = self.model.generate(**input_, **generation_kwargs)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        elif isinstance(prompts, list):
            outputs = []
            for prompt in prompts:  # Process prompts individually
                input_ = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output = self.model.generate(**input_, **generation_kwargs)
                outputs.append(self.tokenizer.decode(output[0], skip_special_tokens=True))  
            return outputs
        else:
            raise TypeError("prompts must be a string or a list of strings.")

    def generate(
        self, 
        prompts: Union[str, List[str]], 
        stream_output: bool = False, 
        **generation_kwargs
    ) -> Union[str, List[str], Generator]:
        """
        Generates text for given prompts, with optional streaming output.
        """

        generation_kwargs = {**self.kwargs, **generation_kwargs}  # Combine default and specific kwargs

        # Set defaults based on encoder-decoder model 
        generation_kwargs["max_new_tokens"] = generation_kwargs.get("max_new_tokens", 256 if self.is_encoder_decoder else 512)
        generation_kwargs["num_beams"] = generation_kwargs.get("num_beams", 1 if self.is_encoder_decoder else None)

        # Streaming support
        if stream_output:
            if isinstance(prompts, list):
                return (self._generate_text(prompt, **generation_kwargs) for prompt in prompts) # Streaming
            elif isinstance(prompts, str):
                return self._generate_text(prompts, **generation_kwargs)
            else:
                raise TypeError("prompts must be a string or list of strings.")
        else:
            return self._generate_text(prompts, **generation_kwargs)