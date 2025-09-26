import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from typing import List, Dict, Optional, Union, Iterator, Generator

from .exceptions import LLMError, ModelLoadError

logger = logging.getLogger(__name__)


class LLM:
    """
    Wrapper for interacting with large language models (LLMs) from Hugging Face.
    
    Provides a unified interface for both encoder-decoder and decoder-only models,
    with support for text generation, streaming, and batch processing.
    
    Attributes:
        model_name: Name of the Hugging Face model
        task: Task type (currently supports "text-generation")
        device: Device for model execution (GPU/CPU)
        is_encoder_decoder: Whether the model is encoder-decoder architecture
    """
    
    def __init__(
        self, 
        model_name: str, 
        use_gpu: Optional[bool] = None, 
        task: str = "text-generation", 
        **kwargs
    ):
        """
        Initializes the LLM with the specified model name and task.
        
        Args:
            model_name: Hugging Face model name or path
            use_gpu: Whether to use GPU. If None, auto-detects CUDA availability
            task: Task type (currently only "text-generation" supported)
            **kwargs: Additional keyword arguments for model generation
            
        Raises:
            ModelLoadError: If model loading fails
            LLMError: If initialization fails
        """
        if not model_name or not isinstance(model_name, str):
            raise LLMError("Model name must be a non-empty string")
            
        if task.lower() != "text-generation":
            raise LLMError(f"Unsupported task: {task}. Only 'text-generation' is supported")
            
        self.model_name = model_name
        self.task = task.lower()
        
        # Auto-detect GPU availability if not specified
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
            
        self.device = 0 if torch.cuda.is_available() and use_gpu else -1
        self.kwargs = kwargs
        
        logger.info(f"Initializing LLM with model: {model_name}, device: {self.device}")
        
        try:
            self._load_model()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_name}", details=str(e))

    def _load_model(self) -> None:
        """Load tokenizer and model components."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try to load as seq2seq model first, then as causal LM
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            ).to(self.device)
            self.is_encoder_decoder = True
            logger.info("Loaded as encoder-decoder model")
        except ValueError:  
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            ).to(self.device)
            self.is_encoder_decoder = False
            logger.info("Loaded as causal language model")

        self.model.eval()

    def _generate_text(
        self, 
        prompts: Union[str, List[str]], 
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        Generates text for a single prompt or list of prompts.
        
        Args:
            prompts: Single prompt string or list of prompts
            **generation_kwargs: Additional arguments for text generation
            
        Returns:
            Generated text(s) corresponding to input prompt(s)
            
        Raises:
            LLMError: If text generation fails
            ValueError: If prompts have invalid type
        """
        if not prompts:
            raise ValueError("Prompts cannot be empty")
            
        try:
            if isinstance(prompts, str):
                return self._generate_single(prompts, **generation_kwargs)
            elif isinstance(prompts, list):
                if not all(isinstance(p, str) for p in prompts):
                    raise ValueError("All prompts must be strings")
                return self._generate_batch(prompts, **generation_kwargs)
            else:
                raise ValueError("prompts must be a string or a list of strings")
                
        except Exception as e:
            if not isinstance(e, (ValueError, LLMError)):
                raise LLMError("Text generation failed", details=str(e))
            raise

    def _generate_single(self, prompt: str, **generation_kwargs) -> str:
        """Generate text for a single prompt."""
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.tokenizer.model_max_length
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**input_ids, **generation_kwargs)
            
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # For causal LMs, remove the input prompt from output
        if not self.is_encoder_decoder and generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
        return generated_text

    def _generate_batch(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate text for a batch of prompts."""
        outputs = []
        for prompt in prompts:
            try:
                output = self._generate_single(prompt, **generation_kwargs)
                outputs.append(output)
            except Exception as e:
                logger.error(f"Failed to generate text for prompt: {prompt[:50]}...")
                outputs.append("")  # Return empty string for failed generations
                
        return outputs

    def generate(
        self, 
        prompts: Union[str, List[str]], 
        stream_output: bool = False, 
        **generation_kwargs
    ) -> Union[str, List[str], Generator]:
        """
        Generates text for given prompts, with optional streaming output.
        
        Args:
            prompts: Single prompt string or list of prompts
            stream_output: Whether to return a generator for streaming output
            **generation_kwargs: Additional arguments for text generation
            
        Returns:
            Generated text(s) or generator for streaming
            
        Raises:
            LLMError: If text generation fails
            ValueError: If inputs are invalid
        """
        if not prompts:
            raise ValueError("Prompts cannot be empty")

        # Merge default and specific generation arguments
        merged_kwargs = {**self.kwargs, **generation_kwargs}
        
        # Set model-specific defaults
        if self.is_encoder_decoder:
            merged_kwargs.setdefault("max_new_tokens", 256)
            merged_kwargs.setdefault("num_beams", 1)
        else:
            merged_kwargs.setdefault("max_new_tokens", 512)
            merged_kwargs.setdefault("do_sample", True)
            merged_kwargs.setdefault("temperature", 0.7)
            merged_kwargs.setdefault("top_p", 0.9)

        logger.info(f"Generating text for {len(prompts) if isinstance(prompts, list) else 1} prompt(s)")

        try:
            if stream_output:
                return self._generate_streaming(prompts, **merged_kwargs)
            else:
                return self._generate_text(prompts, **merged_kwargs)
        except Exception as e:
            raise LLMError("Text generation failed", details=str(e))

    def _generate_streaming(
        self, 
        prompts: Union[str, List[str]], 
        **generation_kwargs
    ) -> Generator:
        """Generate text with streaming output."""
        if isinstance(prompts, str):
            yield self._generate_text(prompts, **generation_kwargs)
        elif isinstance(prompts, list):
            for prompt in prompts:
                yield self._generate_text(prompt, **generation_kwargs)
        else:
            raise ValueError("prompts must be a string or list of strings")

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "task": self.task,
            "is_encoder_decoder": self.is_encoder_decoder,
            "device": self.device,
            "model_max_length": getattr(self.tokenizer, 'model_max_length', None),
            "vocab_size": getattr(self.tokenizer, 'vocab_size', None)
        }