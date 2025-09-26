import gc
import torch
import logging
import numpy as np
from contextlib import contextmanager
from typing import List, Optional, Iterator, Union
from transformers import AutoTokenizer, AutoModel

from .chunk_node import Node
from .exceptions import EmbeddingError, ModelLoadError

# Configure logging
logger = logging.getLogger(__name__)


class Embeddings:
    """
    Handles embedding generation using Hugging Face models.
    
    Provides efficient batch processing, memory management, and lazy loading
    capabilities for generating text embeddings using transformer models.
    
    Attributes:
        model_name: Name of the Hugging Face model to use
        device: Device to run the model on (cuda/cpu)
    """

    def __init__(
        self, 
        model_name: str = "nomic-ai/nomic-embed-text-v1.5", 
        device: Optional[str] = None
    ):
        """
        Initialize the Embeddings class.
        
        Args:
            model_name: Name of the Hugging Face model to use for embeddings
            device: Device to run the model on. If None, auto-detects CUDA availability
            
        Raises:
            ModelLoadError: If model initialization fails
        """
        if not model_name or not isinstance(model_name, str):
            raise ModelLoadError("Model name must be a non-empty string")
            
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self._model_loaded = False
        
        logger.info(f"Embeddings initialized with model: {self.model_name} on device: {self.device}")
        
        # Lazy load model components
        try:
            self._lazy_load_model()
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize model {model_name}", details=str(e))

    def _lazy_load_model(self) -> None:
        """Load model and tokenizer only when needed."""
        if not self._model_loaded:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                ).to(self.device)
                self.model.eval()
                self._model_loaded = True
                logger.info("Model loaded successfully")
            except Exception as e:
                raise ModelLoadError(f"Failed to load model {self.model_name}", details=str(e))

    def _get_device(self, device: str) -> str:
        """Get appropriate device with validation."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return device

    def embed(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding computation

        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If inputs are invalid
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
            
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
            
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in texts must be strings")

        self._lazy_load_model()
        
        try:
            all_embeddings = []
            
            logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
            
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    try:
                        encoded_input = self.tokenizer(
                            batch, 
                            padding=True, 
                            truncation=True, 
                            return_tensors="pt"
                        ).to(self.device)
                        
                        model_output = self.model(**encoded_input)
                        batch_embeddings = self.mean_pooling(
                            model_output, 
                            encoded_input['attention_mask']
                        ).cpu().numpy()
                        
                        all_embeddings.append(batch_embeddings)
                        
                        # Clear GPU cache periodically
                        if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        raise EmbeddingError(
                            f"Failed to process batch {i//batch_size + 1}", 
                            details=str(e)
                        )
                        
            result = np.concatenate(all_embeddings, axis=0)
            logger.info(f"Successfully generated embeddings with shape {result.shape}")
            return result
            
        except Exception as e:
            if not isinstance(e, EmbeddingError):
                raise EmbeddingError("Failed to generate embeddings", details=str(e))
            raise

    def embed_nodes(self, nodes: List[Node], batch_size: int = 4) -> List[Node]:
        """
        Generates embeddings for a list of nodes and updates the nodes in-place.

        Args:
            nodes: List of Node objects to embed
            batch_size: Batch size for embedding computation

        Returns:
            The same list of Node objects, with their 'embedding' attribute updated
            
        Raises:
            EmbeddingError: If node embedding fails
            ValueError: If inputs are invalid
        """
        if not nodes:
            logger.warning("Empty node list provided for embedding")
            return nodes
            
        if not all(isinstance(node, Node) for node in nodes):
            raise ValueError("All items must be Node objects")
            
        try:
            texts = [node.text for node in nodes]
            embeddings = self.embed(texts, batch_size=batch_size)
            
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
                
            logger.info(f"Successfully embedded {len(nodes)} nodes")
            return nodes
            
        except Exception as e:
            raise EmbeddingError("Failed to embed nodes", details=str(e))

    @contextmanager
    def temporary_model_load(self) -> Iterator['Embeddings']:
        """
        Context manager for temporary model usage with automatic cleanup.
        
        Yields:
            The Embeddings instance with loaded model
            
        Example:
            with embeddings.temporary_model_load() as emb:
                result = emb.embed(texts)
        """
        try:
            self._lazy_load_model()
            yield self
        finally:
            if self._model_loaded:
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                self._model_loaded = False
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Model unloaded and memory cleared")

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Applies mean pooling to the model outputs.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask tensor
            
        Returns:
            Mean-pooled embeddings tensor
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity between two embeddings or batches of embeddings.
        
        Args:
            embedding1: First embedding(s) with shape (..., embedding_dim)
            embedding2: Second embedding(s) with shape (..., embedding_dim)
            
        Returns:
            Cosine similarity scores
            
        Raises:
            ValueError: If embeddings have incompatible shapes
        """
        if embedding1.shape[-1] != embedding2.shape[-1]:
            raise ValueError(
                f"Embedding dimensions don't match: {embedding1.shape[-1]} vs {embedding2.shape[-1]}"
            )
            
        embedding1_norm = np.linalg.norm(embedding1, axis=-1, keepdims=True)
        embedding2_norm = np.linalg.norm(embedding2, axis=-1, keepdims=True)

        # Avoid division by zero
        embedding1_norm = np.where(embedding1_norm == 0, 1e-9, embedding1_norm)
        embedding2_norm = np.where(embedding2_norm == 0, 1e-9, embedding2_norm)

        return np.dot(embedding1, embedding2.T) / (embedding1_norm * embedding2_norm.T)