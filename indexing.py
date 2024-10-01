# indexing.py
import faiss
import chromadb
import numpy as np
import logging
import os
from typing import List, Dict, Any
from ChunkNode import Node

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages indexing and retrieval from vector databases."""

    def __init__(self, vector_store_type: str = "faiss", index_path: str = "faiss_index.bin", chroma_persist_dir: str = "chromadb"):
        """Initializes the VectorStore with either FAISS or ChromaDB."""

        self.vector_store_type = vector_store_type
        self.index_path = index_path

        if self.vector_store_type == "faiss":
            self.index = faiss.IndexFlatL2(768)  # Initialize with default dimension, change later
            logger.info("Initialized FAISS index.")
        elif self.vector_store_type == "chroma":
            self.client = chromadb.PersistentClient(path=chroma_persist_dir)
            self.collection = self.client.get_or_create_collection("documents")
            logger.info("Initialized ChromaDB client and collection.")
        else:
            raise ValueError("Invalid vector_store_type. Choose 'faiss' or 'chroma'.")

    def index_chunks(self, chunks: List[Node]):
        """Indexes chunks of text."""
        if self.vector_store_type == "faiss":
            embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
            if self.index.d != embeddings.shape[1]:  # Set/reset dimension if needed
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Indexed {len(chunks)} chunks in FAISS and saved to {self.index_path}.")
        elif self.vector_store_type == "chroma":
            ids = [chunk.metadata.get('node_id', str(i)) for i, chunk in enumerate(chunks)]
            embeddings = [chunk.embedding.tolist() for chunk in chunks]  # ChromaDB expects lists
            metadatas = [chunk.metadata for chunk in chunks]  # Pass metadata to ChromaDB
            documents = [chunk.text for chunk in chunks]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Indexed {len(chunks)} chunks in ChromaDB.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the vector store for similar chunks."""
        if self.vector_store_type == "faiss":
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            _, indices = self.index.search(query_embedding, top_k)
            results = [{'faiss_index': int(idx)} for idx in indices[0]]
            logger.info(f"FAISS search returned {len(results)} results.")
        elif self.vector_store_type == "chroma":
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "distances", "documents"]
            )
            results = [
                {
                    "node_id": result_id,
                    "metadata": metadata,
                    "score": 1 - distance,
                    "document": doc
                }
                for result_id, metadata, distance, doc in zip(
                    results['ids'][0], results['metadatas'][0], results['distances'][0], results['documents'][0]
                )
            ]
            logger.info(f"ChromaDB search returned {len(results)} results.")
        return results

    def save_index(self):
        if self.vector_store_type == "faiss":
          faiss.write_index(self.index, self.index_path)
          logger.info(f"Saved FAISS index to {self.index_path}")


    def load_index(self):
        if self.vector_store_type == "faiss":
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                logger.warning(f"Index not found at {self.index_path}")