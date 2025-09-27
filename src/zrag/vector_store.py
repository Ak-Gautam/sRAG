"""Vector store abstraction supporting multiple backends.

This module provides a unified API for persisting and querying embeddings
across different vector store technologies. The refactor aligns the
implementation with the shared data models defined in ``models.py`` and adds
quality-of-life improvements such as metadata persistence for FAISS, a
lightweight numpy-based fallback, and richer retrieval results.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import chromadb
except ImportError:
    chromadb = None

from .models import Node, RetrievalResult, VectorStoreConfig
from .exceptions import VectorStoreError, VectorStoreNotInitializedError

logger = logging.getLogger(__name__)


class TinyNumpyIndex:
    """A minimal in-memory index that mimics FAISS's API surface.

    This acts as a soft fallback when FAISS isn't installed. It uses basic
    L2-distance search implemented with numpy operations. While this won't
    scale to millions of vectors, it keeps the development workflow and
    smaller deployments unblocked.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self._embeddings = np.empty((0, embedding_dim), dtype="float32")

    @property
    def ntotal(self) -> int:
        return self._embeddings.shape[0]

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.embedding_dim}), got {embeddings.shape}"
            )
        if self.ntotal == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype="float32")
        if query.ndim != 2:
            raise ValueError("Query must be a 2D array with shape (1, dim)")

        if self.ntotal == 0:
            distances = np.full((1, k), np.inf, dtype="float32")
            indices = np.full((1, k), -1, dtype="int32")
            return distances, indices

        diffs = self._embeddings[None, :, :] - query[:, None, :]
        distances = np.linalg.norm(diffs, axis=-1)
        top_k = min(k, self.ntotal)
        idx_sorted = np.argpartition(distances, top_k - 1, axis=1)[:, :top_k]

        # Sort the top-k distances for stability
        top_distances = np.take_along_axis(distances, idx_sorted, axis=1)
        order = np.argsort(top_distances, axis=1)
        sorted_indices = np.take_along_axis(idx_sorted, order, axis=1)
        sorted_distances = np.take_along_axis(top_distances, order, axis=1)

        # Pad to ensure consistent shape
        if top_k < k:
            pad_size = k - top_k
            sorted_indices = np.hstack(
                [sorted_indices, np.full((1, pad_size), -1, dtype="int32")]
            )
            sorted_distances = np.hstack(
                [sorted_distances, np.full((1, pad_size), np.inf, dtype="float32")]
            )

        return sorted_distances, sorted_indices


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
        config: Optional[VectorStoreConfig] = None,
        *,
        vector_store_type: Optional[str] = None,
        index_path: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        metadata_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        """Initialise the vector store with the desired backend.

        Parameters mirror :class:`VectorStoreConfig` but keep the legacy
        keyword arguments for backwards compatibility. Explicit keyword
        overrides take precedence over the provided configuration instance.
        """

        base_config = config or VectorStoreConfig()
        self.config = replace(
            base_config,
            backend=(vector_store_type or base_config.backend).lower(),
            index_path=index_path or base_config.index_path,
            chroma_persist_dir=chroma_persist_dir or base_config.chroma_persist_dir,
            metadata_path=metadata_path or base_config.metadata_path,
        )

        if self.config.backend not in {"faiss", "chroma", "numpy"}:
            raise VectorStoreError(
                f"Invalid vector_store_type: {self.config.backend}. Choose 'faiss', 'chroma', or 'numpy'"
            )

        self.vector_store_type = self.config.backend
        self.index_path = Path(self.config.index_path)
        self.metadata_path = (
            Path(self.config.metadata_path)
            if self.config.metadata_path
            else self.index_path.with_suffix(".meta.json")
        )
        self.embedding_dim = embedding_dim
        self._is_initialized = False
        self._records: List[Dict[str, Any]] = []

        backend = self.vector_store_type
        try:
            if backend == "faiss":
                if faiss is None:
                    logger.warning(
                        "FAISS backend requested but not installed; falling back to numpy index"
                    )
                    self.vector_store_type = "numpy"
                    self.index = None
                else:
                    self.index = None
                    logger.info("FAISS vector store initialised")
            if backend == "numpy" or self.vector_store_type == "numpy":
                logger.info("Using TinyNumpyIndex vector store")
                self.index = None  # Lazy initialisation once we know embedding dim
                self.vector_store_type = "numpy"
            elif backend == "chroma":
                if chromadb is None:
                    raise VectorStoreError(
                        "ChromaDB not available. Install with: pip install chromadb"
                    )
                self.client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
                self.collection = self.client.get_or_create_collection("documents")
                logger.info("ChromaDB vector store initialised")
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise VectorStoreError(
                f"Failed to initialise {self.vector_store_type} vector store",
                details=str(exc),
            )

    def add_documents(self, chunks: Iterable[Node], batch_size: int = 100) -> None:
        """
        Indexes chunks of text in the vector store.
        
        Args:
            chunks: List of Node objects with embeddings to index
            batch_size: Batch size for processing (useful for large datasets)
            
        Raises:
            VectorStoreError: If indexing fails
            ValueError: If input validation fails
        """
        chunk_list = list(chunks)
        if not chunk_list:
            logger.warning("Empty chunks list provided for indexing")
            return
            
        if not all(isinstance(chunk, Node) for chunk in chunk_list):
            raise ValueError("All items must be Node objects")
            
        missing_embeddings = [i for i, chunk in enumerate(chunk_list) if chunk.embedding is None]
        if missing_embeddings:
            raise VectorStoreError(
                f"Chunks at indices {missing_embeddings[:10]} are missing embeddings"
            )

        try:
            if self.vector_store_type == "faiss":
                self._index_faiss(chunk_list)
            elif self.vector_store_type == "chroma":
                self._index_chroma(chunk_list, batch_size)
            elif self.vector_store_type == "numpy":
                self._index_numpy(chunk_list)
            else:
                raise VectorStoreError(f"Unsupported vector store type: {self.vector_store_type}")
                
            self._is_initialized = True
            logger.info(
                "Successfully indexed %s chunks in %s",
                len(chunk_list),
                self.vector_store_type,
            )
            
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to index chunks in {self.vector_store_type}", details=str(exc)
            )

    def _index_faiss(self, chunks: List[Node]) -> None:
        """Index chunks in FAISS."""
        if faiss is None:
            raise VectorStoreError("FAISS backend requested but dependency is missing")

        embeddings = self._stack_embeddings(chunks)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            logger.info("Created FAISS index with dimension %s", embeddings.shape[1])
        elif self.index.d != embeddings.shape[1]:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.index.d}, got {embeddings.shape[1]}"
            )

        self._records.extend(self._serialise_records(chunks))
        self.index.add(embeddings)
        self._persist_metadata()
        self.save()

    def _index_numpy(self, chunks: List[Node]) -> None:
        embeddings = self._stack_embeddings(chunks)
        if self.index is None:
            self.index = TinyNumpyIndex(embeddings.shape[1])
            logger.info("Initialised TinyNumpyIndex with dimension %s", embeddings.shape[1])
        elif getattr(self.index, "embedding_dim", embeddings.shape[1]) != embeddings.shape[1]:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.index.embedding_dim}, got {embeddings.shape[1]}"
            )

        self._records.extend(self._serialise_records(chunks))
        self.index.add(embeddings)
        self._persist_metadata()
        
    def _index_chroma(self, chunks: List[Node], batch_size: int) -> None:
        """Index chunks in ChromaDB with batching."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [self._ensure_node_id(chunk) for chunk in batch]
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
        **_: Any,
    ) -> List[RetrievalResult]:
        """
        Searches the vector store for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of :class:`RetrievalResult` instances sorted by similarity
            
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
            if self.vector_store_type == "chroma":
                return self._search_chroma(query_embedding, k)
            if self.vector_store_type == "numpy":
                return self._search_numpy(query_embedding, k)
            raise VectorStoreError(f"Unsupported vector store type: {self.vector_store_type}")
        except Exception as exc:
            raise VectorStoreError(
                f"Search failed in {self.vector_store_type}", details=str(exc)
            )

    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        results = self._results_from_indices(distances[0], indices[0])
        logger.debug("FAISS search returned %s results", len(results))
        return results

    def _search_numpy(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        if self.index is None:
            raise VectorStoreNotInitializedError("TinyNumpyIndex is not initialised")
        query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        results = self._results_from_indices(distances[0], indices[0])
        logger.debug("TinyNumpyIndex search returned %s results", len(results))
        return results

    def _search_chroma(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Search in ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["metadatas", "distances", "documents"]
        )
        
        retrieval_results: List[RetrievalResult] = []
        for result_id, metadata, distance, doc in zip(
            results['ids'][0], 
            results['metadatas'][0], 
            results['distances'][0], 
            results['documents'][0]
        ):
            retrieval_results.append(
                RetrievalResult(
                    node_id=result_id,
                    metadata=metadata or {},
                    text=doc,
                    distance=float(distance),
                    score=max(0.0, 1.0 - float(distance)),
                )
            )

        logger.debug("ChromaDB search returned %s results", len(retrieval_results))
        return retrieval_results

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
        return [result.to_dict() for result in self.similarity_search(query_embedding, k=top_k)]

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
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(self.index_path))
                logger.info("FAISS index saved to %s", self.index_path)
            elif self.vector_store_type == "numpy":
                # Metadata persistence already handled; embeddings stored in JSON for simplicity
                self._persist_metadata()
                logger.info("TinyNumpyIndex metadata persisted to %s", self.metadata_path)
            elif self.vector_store_type == "chroma":
                logger.info("ChromaDB data is automatically persisted")
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to save {self.vector_store_type} vector store", details=str(exc)
            )

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
                    self._records = self._load_metadata()
                    self._is_initialized = True
                    self._validate_metadata_alignment()
                    logger.info("FAISS index loaded from %s", self.index_path)
                    return True
                logger.info("No FAISS index found at %s", self.index_path)
                return False
            if self.vector_store_type == "numpy":
                if self.metadata_path.exists():
                    self._records = self._load_metadata()
                    if self._records:
                        first_embedding = self._records[0].get("embedding")
                        if first_embedding is None:
                            raise VectorStoreError(
                                "TinyNumpyIndex metadata missing embeddings; cannot restore"
                            )
                        embeddings = np.asarray([rec.pop("embedding") for rec in self._records], dtype="float32")
                        self.index = TinyNumpyIndex(embeddings.shape[1])
                        self.index.add(embeddings)
                        self._is_initialized = True
                        logger.info("TinyNumpyIndex restored from %s", self.metadata_path)
                        return True
                logger.info("No TinyNumpyIndex metadata found at %s", self.metadata_path)
                return False
            if self.vector_store_type == "chroma":
                try:
                    count = self.collection.count()
                    if count > 0:
                        self._is_initialized = True
                        logger.info("ChromaDB collection loaded with %s documents", count)
                        return True
                    logger.info("ChromaDB collection is empty")
                    return False
                except Exception:  # pragma: no cover - defensive
                    logger.info("ChromaDB collection not found or empty")
                    return False
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to load {self.vector_store_type} vector store", details=str(exc)
            )

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
            if self.vector_store_type in {"faiss", "numpy"} and self.index is not None:
                stats.update({
                    'total_vectors': getattr(self.index, 'ntotal', len(self._records)),
                    'index_path': str(self.index_path),
                    'metadata_path': str(self.metadata_path),
                })
            elif self.vector_store_type == "chroma" and hasattr(self, 'collection'):
                stats.update({
                    'total_vectors': self.collection.count(),
                    'collection_name': self.collection.name
                })
        except Exception as exc:
            stats['error'] = str(exc)
            
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stack_embeddings(self, chunks: List[Node]) -> np.ndarray:
        embeddings = np.stack([np.asarray(chunk.embedding, dtype="float32") for chunk in chunks])
        if embeddings.ndim != 2:
            raise VectorStoreError("Embeddings must be rank-2 arrays")

        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
        elif embeddings.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )
        return embeddings

    def _serialise_records(self, chunks: List[Node]) -> List[Dict[str, Any]]:
        serialised = []
        for chunk in chunks:
            node_id = self._ensure_node_id(chunk)
            record = {
                "node_id": node_id,
                "metadata": chunk.metadata.copy(),
                "text": chunk.text,
            }
            if self.vector_store_type == "numpy":
                record["embedding"] = np.asarray(chunk.embedding, dtype="float32").tolist()
            serialised.append(record)
        return serialised

    def _ensure_node_id(self, chunk: Node) -> str:
        node_id = chunk.metadata.get("node_id")
        if not node_id:
            node_id = str(uuid.uuid4())
            chunk.metadata["node_id"] = node_id
        return node_id

    def _results_from_indices(self, distances: np.ndarray, indices: np.ndarray) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        total_records = len(self._records)
        for dist, idx in zip(distances.tolist(), indices.tolist()):
            if idx < 0 or idx >= total_records:
                continue
            record = self._records[idx]
            results.append(
                RetrievalResult(
                    node_id=record["node_id"],
                    metadata=record.get("metadata", {}),
                    text=record.get("text", ""),
                    distance=float(dist),
                    score=self._distance_to_score(dist),
                )
            )
        return results

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        if distance in (np.inf, float("inf")):
            return 0.0
        return 1.0 / (1.0 + float(distance))

    def _persist_metadata(self) -> None:
        if self.vector_store_type not in {"faiss", "numpy"}:
            return

        payload = {
            "records": self._records,
            "embedding_dim": self.embedding_dim,
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except TypeError:
            with self.metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        if not self.metadata_path.exists():
            return []
        with self.metadata_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.embedding_dim = payload.get("embedding_dim", self.embedding_dim)
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise VectorStoreError("Invalid metadata format: 'records' must be a list")
        return records

    def _validate_metadata_alignment(self) -> None:
        if self.vector_store_type not in {"faiss", "numpy"} or self.index is None:
            return
        total_vectors = getattr(self.index, "ntotal", len(self._records))
        if total_vectors != len(self._records):
            raise VectorStoreError(
                f"Metadata count {len(self._records)} does not match index size {total_vectors}"
            )