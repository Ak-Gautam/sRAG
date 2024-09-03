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

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        # Return a list of tuples (document_id, score)
        return list(zip(results['ids'][0], results['distances'][0]))
    

# Test
from CreateEmbeddings import EmbeddingGenerator, FaissIndex, ChromaVectorStore

# 1. Generate embeddings for the nodes:
embedding_generator = EmbeddingGenerator()
embeddings = embedding_generator.generate_embeddings(nodes)

# 2. (Optional) Create a FAISS index for quick in-memory searches:
faiss_index = FaissIndex(embeddings)