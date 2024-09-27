# This is an attempt at creating embeddings and saving it in a vector database.

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
        return indices[0].tolist()  # Return the indices of the top-k closest nodes

class ChromaVectorStore:
    def __init__(self, collection_name: str = "vec_docs"):
        # Persist directory will store the database, if you don't want persistence
        # remove this parameter
        client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = client.get_or_create_collection(name=collection_name)


    def add_documents(self, nodes: List[Node], embeddings: np.ndarray):
        ids = [id(node) for node in nodes]
        metadatas = [node.metadata for node in nodes]
        documents = [node.text for node in nodes]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches both FAISS and ChromaDB indices for the top_k nearest neighbors to the query.

        Args:
            query (str): The query text.
            top_k (int, optional): Number of top similar vectors to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of metadata for the top_k nearest neighbors.
        """
        query_embedding = self.embed_nodes([{'text': query, 'metadata': {}}])[0]

        results = []

        if self.use_faiss:
            faiss_indices = self.search_faiss(query_embedding, top_k)
            # FAISS does not store metadata; user needs to map indices to nodes externally
            results.extend([{'faiss_index': idx} for idx in faiss_indices])

        if self.use_chroma:
            chroma_results = self.search_chromadb(query_embedding, top_k)
            results.extend(chroma_results)

        return results
    