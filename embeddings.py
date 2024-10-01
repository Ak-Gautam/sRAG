# embeddings.py
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embeddings:
    """Handles embedding generation using Hugging Face models."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Embeddings with model: {self.model_name} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): List of text strings.
            batch_size (int, optional): Batch size for embedding computation. Defaults to 4.

        Returns:
            np.ndarray: A NumPy array of embeddings.
        """
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                model_output = self.model(**encoded_input)
                batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
                all_embeddings.append(batch_embeddings)
                expected_dim = self.model.config.hidden_size
                if batch_embeddings.shape[1] != expected_dim:
                    raise ValueError(f"Unexpected embedding dimension. Expected {expected_dim}, but got {batch_embeddings.shape[1]}.")
        return np.concatenate(all_embeddings, axis=0)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """Applies mean pooling to the model outputs."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> np.ndarray:
        embedding1_norm = np.linalg.norm(embedding1, axis=-1, keepdims=True)
        embedding2_norm = np.linalg.norm(embedding2, axis=-1, keepdims=True)

        embedding1_norm[embedding1_norm == 0] = 1e-9
        embedding2_norm[embedding2_norm == 0] = 1e-9

        return np.dot(embedding1, embedding2.T) / (embedding1_norm * embedding2_norm.T)