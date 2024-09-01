from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import chromadb
from ChunkNode import Node  # Import your Node class
from typing import List, Tuple

class EmbeddingGenerator:
    def __init__(self, model_name: str = "liddlefish/privacy_embedding_rag_10k_base_15_final"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, nodes: List[Node]) -> np.ndarray:
        embeddings = self.model.encode([node.text for node in nodes])
        return embeddings

class FaissIndex:
    def __init__(self, embeddings: np.ndarray):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[int]:
        distances, indices = self.index.search(query_embedding, k)
        return indices[0].tolist()