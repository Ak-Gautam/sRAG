# VectorIndex.py

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from typing import List
from ChunkNode import Node

class EmbeddingGenerator:
    def __init__(self, model_name: str = "liddlefish/privacy_embedding_rag_10k_base_15_final"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, nodes: List[Node]) -> np.ndarray:
        embeddings = self.model.encode([node.text for node in nodes])
        return embeddings

class ChromaVectorStore:
    def __init__(self, collection_name: str = "vec_docs"):
        client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = client.get_or_create_collection(name=collection_name)

    def add_documents(self, nodes: List[Node], embeddings: np.ndarray):
        ids = [node.metadata['document_id'] for node in nodes]
        metadatas = [node.metadata for node in nodes]
        documents = [node.text for node in nodes]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

class VectorIndex:
    def __init__(self, collection_name: str = "vec_docs"):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = ChromaVectorStore(collection_name)

    def build_index_from_nodes(self, nodes: List[Node]):
        embeddings = self.embedding_generator.generate_embeddings(nodes)
        self.vector_store.add_documents(nodes, embeddings)