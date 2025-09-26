import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    import faiss
except ImportError:
    faiss = None

try:
    import chromadb
except ImportError:
    chromadb = None

from .chunk_node import Node
from .exceptions import VectorStoreError, VectorStoreNotInitializedError

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages indexing and retrieval from vector databases.
    
    Supports both FAISS and ChromaDB backends for vector storage and similarity search.
    Provides a unified interface for different vector store implementations.
    
    Attributes:
        vector_store_type: Type of vector store backend ('faiss' or 'chroma')
        index_path: Path to save/load FAISS index
        embedding_dim: Dimension of embeddings (auto-detected if not provided)
    """

    def __init__(
        self, 
        vector_store_type: str = "faiss", 
        index_path: str = "faiss_index.bin", 
        chroma_persist_dir: str = "chromadb", 
        embedding_dim: Optional[int] = None
    ):
        """
        Initializes the VectorStore with either FAISS or ChromaDB.
        
        Args:
            vector_store_type: Backend type ('faiss' or 'chroma')
            index_path: Path to save/load FAISS index file
            chroma_persist_dir: Directory for ChromaDB persistence
            embedding_dim: Dimension of embeddings (auto-detected if None)
            
        Raises:
            VectorStoreError: If initialization fails or dependencies are missing
        """
        if vector_store_type not in ["faiss", "chroma"]:
            raise VectorStoreError(f"Invalid vector_store_type: {vector_store_type}. Choose 'faiss' or 'chroma'")

        self.vector_store_type = vector_store_type
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self._is_initialized = False

        try:
            if self.vector_store_type == "faiss":
                if faiss is None:
                    raise VectorStoreError("FAISS not available. Install with: pip install faiss-cpu")
                self.index = None  # Initialized later 
                logger.info("FAISS vector store initialized")
                
            elif self.vector_store_type == "chroma":
                if chromadb is None:
                    raise VectorStoreError("ChromaDB not available. Install with: pip install chromadb")
                self.client = chromadb.PersistentClient(path=chroma_persist_dir)
                self.collection = self.client.get_or_create_collection("documents")
                logger.info("ChromaDB vector store initialized")
                
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize {vector_store_type} vector store", details=str(e))

    def add_documents(self, chunks: List[Node], batch_size: int = 100) -> None:
        """
        Indexes chunks of text in the vector store.
        
        Args:
            chunks: List of Node objects with embeddings to index
            batch_size: Batch size for processing (useful for large datasets)
            
        Raises:
            VectorStoreError: If indexing fails
            ValueError: If input validation fails
        """
        if not chunks:
            logger.warning("Empty chunks list provided for indexing")
            return
            
        if not all(isinstance(chunk, Node) for chunk in chunks):
            raise ValueError("All items must be Node objects")
            
        # Validate that all chunks have embeddings
        missing_embeddings = [i for i, chunk in enumerate(chunks) if chunk.embedding is None]
        if missing_embeddings:
            raise VectorStoreError(
                f"Chunks at indices {missing_embeddings[:10]} are missing embeddings"
            )

        try:
            if self.vector_store_type == "faiss":
                self._index_faiss(chunks)
            elif self.vector_store_type == "chroma":
                self._index_chroma(chunks, batch_size)
                
            self._is_initialized = True
            logger.info(f"Successfully indexed {len(chunks)} chunks in {self.vector_store_type}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to index chunks in {self.vector_store_type}", details=str(e))

    def _index_faiss(self, chunks: List[Node]) -> None:
        """Index chunks in FAISS."""
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        
        # Auto-detect embedding dimension
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
            
        if embeddings.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )
            
        # Initialize or reset index if needed
        if self.index is None or self.index.d != embeddings.shape[1]:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            logger.info(f"Created FAISS index with dimension {embeddings.shape[1]}")
            
        self.index.add(embeddings)
        self.save()  # Auto-save after indexing
        
    def _index_chroma(self, chunks: List[Node], batch_size: int) -> None:
        """Index chunks in ChromaDB with batching."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk.metadata.get('node_id', f'node_{i + j}') for j, chunk in enumerate(batch)]
            embeddings = [chunk.embedding.tolist() for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            documents = [chunk.text for chunk in batch]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.debug(f"Indexed batch {i//batch_size + 1} ({len(batch)} chunks)")

    # Deprecated method with backward compatibility
    def index(self, chunks: List[Node]) -> None:
        """
        Deprecated: Use add_documents() instead.
        
        Args:
            chunks: List of Node objects to index
        """
        import warnings
        warnings.warn(
            "index() is deprecated, use add_documents() instead", 
            DeprecationWarning, 
            stacklevel=2
        )
        self.add_documents(chunks)

    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Searches the vector store for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of dictionaries containing search results with scores and metadata
            
        Raises:
            VectorStoreNotInitializedError: If vector store is not initialized
            VectorStoreError: If search fails
            ValueError: If inputs are invalid
        """
        if not self._is_initialized and (
            (self.vector_store_type == "faiss" and self.index is None) or
            (self.vector_store_type == "chroma" and not hasattr(self, 'collection'))
        ):
            raise VectorStoreNotInitializedError("Vector store is not initialized. Add documents first.")
            
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
            
        if query_embedding is None:
            raise ValueError("Query embedding cannot be None")

        try:
            if self.vector_store_type == "faiss":
                return self._search_faiss(query_embedding, k)
            elif self.vector_store_type == "chroma":
                return self._search_chroma(query_embedding, k)
        except Exception as e:
            raise VectorStoreError(f"Search failed in {self.vector_store_type}", details=str(e))

    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search in FAISS index."""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 indicates no result found
                results.append({
                    'faiss_index': int(idx),
                    'score': 1.0 / (1.0 + distance),  # Convert distance to similarity score
                    'distance': float(distance)
                })
        
        logger.debug(f"FAISS search returned {len(results)} results")
        return results

    def _search_chroma(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search in ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["metadatas", "distances", "documents"]
        )
        
        formatted_results = []
        for result_id, metadata, distance, doc in zip(
            results['ids'][0], 
            results['metadatas'][0], 
            results['distances'][0], 
            results['documents'][0]
        ):
            formatted_results.append({
                "node_id": result_id,
                "metadata": metadata,
                "score": 1.0 - distance,  # Convert distance to similarity
                "distance": distance,
                "document": doc
            })
        
        logger.debug(f"ChromaDB search returned {len(formatted_results)} results")
        return formatted_results

    # Deprecated method with backward compatibility  
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Deprecated: Use similarity_search() instead.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        import warnings
        warnings.warn(
            "search() is deprecated, use similarity_search() instead", 
            DeprecationWarning, 
            stacklevel=2
        )
        return self.similarity_search(query_embedding, k=top_k)

    def save(self) -> None:
        """
        Saves the vector store state.
        
        For FAISS: Saves the index to disk
        For ChromaDB: Data is automatically persisted
        
        Raises:
            VectorStoreError: If saving fails
        """
        try:
            if self.vector_store_type == "faiss" and self.index is not None:
                # Ensure directory exists
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(self.index_path))
                logger.info(f"FAISS index saved to {self.index_path}")
            elif self.vector_store_type == "chroma":
                # ChromaDB auto-persists, no action needed
                logger.info("ChromaDB data is automatically persisted")
        except Exception as e:
            raise VectorStoreError(f"Failed to save {self.vector_store_type} vector store", details=str(e))

    def load(self) -> bool:
        """
        Loads the vector store state from disk.
        
        For FAISS: Loads the index from file
        For ChromaDB: Collection is automatically loaded
        
        Returns:
            True if successfully loaded, False if no saved state exists
            
        Raises:
            VectorStoreError: If loading fails
        """
        try:
            if self.vector_store_type == "faiss":
                if self.index_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                    self._is_initialized = True
                    logger.info(f"FAISS index loaded from {self.index_path}")
                    return True
                else:
                    logger.info(f"No FAISS index found at {self.index_path}")
                    return False
            elif self.vector_store_type == "chroma":
                # ChromaDB collections are automatically loaded
                try:
                    count = self.collection.count()
                    if count > 0:
                        self._is_initialized = True
                        logger.info(f"ChromaDB collection loaded with {count} documents")
                        return True
                    else:
                        logger.info("ChromaDB collection is empty")
                        return False
                except Exception:
                    logger.info("ChromaDB collection not found or empty")
                    return False
        except Exception as e:
            raise VectorStoreError(f"Failed to load {self.vector_store_type} vector store", details=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing vector store statistics
        """
        stats = {
            'type': self.vector_store_type,
            'initialized': self._is_initialized,
            'embedding_dim': self.embedding_dim
        }
        
        try:
            if self.vector_store_type == "faiss" and self.index is not None:
                stats.update({
                    'total_vectors': self.index.ntotal,
                    'index_path': str(self.index_path)
                })
            elif self.vector_store_type == "chroma" and hasattr(self, 'collection'):
                stats.update({
                    'total_vectors': self.collection.count(),
                    'collection_name': self.collection.name
                })
        except Exception as e:
            stats['error'] = str(e)
            
        return stats