# Coding Guidelines for zRAG

This document outlines eight essential coding guidelines for maintaining code quality, consistency, and reliability in the zRAG project.

## 1. Type Hints and Documentation

**Always use type hints and comprehensive docstrings for public APIs.**

```python
from typing import List, Optional, Dict, Any

class DocumentLoader:
    """Loads and processes documents from various sources.
    
    This class provides methods to load documents from different file formats
    and convert them into a standardized Document format for processing.
    """
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from specified file paths.
        
        Args:
            file_paths: List of absolute file paths to load documents from.
            
        Returns:
            List of Document objects containing the loaded content.
            
        Raises:
            FileNotFoundError: If any file path does not exist.
            UnsupportedFormatError: If file format is not supported.
        """
```

**Why**: Type hints improve IDE support, catch errors early, and make the API self-documenting. Documentation helps users understand functionality without reading implementation.

## 2. Error Handling and Validation

**Implement comprehensive error handling with custom exceptions and input validation.**

```python
class ZRAGError(Exception):
    """Base exception for zRAG library."""
    pass

class InvalidChunkSizeError(ZRAGError):
    """Raised when chunk size is invalid."""
    pass

class VectorStoreError(ZRAGError):
    """Raised when vector store operations fail."""
    pass

def create_chunks(self, text: str, chunk_size: int = 512) -> List[Node]:
    """Create text chunks with validation."""
    if chunk_size <= 0:
        raise InvalidChunkSizeError(f"Chunk size must be positive, got {chunk_size}")
    
    if not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    try:
        # Chunking logic here
        return chunks
    except Exception as e:
        raise ZRAGError(f"Failed to create chunks: {str(e)}") from e
```

**Why**: Clear error messages help users debug issues quickly. Custom exceptions allow for specific error handling in downstream applications.

## 3. Configuration and Dependency Injection

**Use dependency injection and configuration objects to make components modular and testable.**

```python
@dataclass
class RAGConfig:
    """Configuration for RAG pipeline components."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_type: str = "faiss"
    llm_temperature: float = 0.7
    max_retrieval_docs: int = 5

class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        document_loader: DocumentLoader,
        embeddings: Embeddings,
        vector_store: VectorStore,
        llm: LLM,
    ):
        """Initialize pipeline with injected dependencies."""
        self.config = config
        self.document_loader = document_loader
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
```

**Why**: Dependency injection makes code more modular, testable, and allows users to customize components without modifying core logic.

## 4. Comprehensive Testing Strategy

**Write unit tests, integration tests, and use test doubles for external dependencies.**

```python
import unittest
from unittest.mock import Mock, patch, MagicMock

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_loader = Mock(spec=DocumentLoader)
        self.mock_embeddings = Mock(spec=Embeddings)
        self.mock_vector_store = Mock(spec=VectorStore)
        self.mock_llm = Mock(spec=LLM)
        
        self.config = RAGConfig(chunk_size=256, max_retrieval_docs=3)
        
        self.rag = RAGPipeline(
            config=self.config,
            document_loader=self.mock_loader,
            embeddings=self.mock_embeddings,
            vector_store=self.mock_vector_store,
            llm=self.mock_llm,
        )
    
    def test_query_with_valid_input(self):
        """Test successful query processing."""
        # Arrange
        query = "What is machine learning?"
        mock_docs = [Mock(), Mock()]
        self.mock_vector_store.similarity_search.return_value = mock_docs
        self.mock_llm.generate.return_value = "ML is a subset of AI..."
        
        # Act
        result = self.rag.query(query)
        
        # Assert
        self.mock_vector_store.similarity_search.assert_called_once_with(
            query, k=self.config.max_retrieval_docs
        )
        assert "ML is a subset of AI" in result

    @patch('zrag.embeddings.SentenceTransformer')
    def test_embeddings_model_loading(self, mock_transformer):
        """Test embeddings initialization with mocked external dependency."""
        embeddings = Embeddings(model_name="test-model")
        mock_transformer.assert_called_once_with("test-model")
```

**Why**: Tests ensure reliability, prevent regressions, and document expected behavior. Mocking external dependencies makes tests fast and deterministic.

## 5. Logging and Observability

**Implement structured logging with appropriate levels and context.**

```python
import logging
from typing import Dict, Any

# Configure structured logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    def query(self, query_text: str, **kwargs) -> str:
        """Process query with comprehensive logging."""
        query_id = kwargs.get('query_id', 'unknown')
        
        logger.info(
            "Starting RAG query processing",
            extra={
                'query_id': query_id,
                'query_length': len(query_text),
                'component': 'RAGPipeline'
            }
        )
        
        try:
            # Retrieve relevant documents
            start_time = time.time()
            retrieved_docs = self.vector_store.similarity_search(
                query_text, 
                k=self.config.max_retrieval_docs
            )
            retrieval_time = time.time() - start_time
            
            logger.info(
                "Document retrieval completed",
                extra={
                    'query_id': query_id,
                    'docs_retrieved': len(retrieved_docs),
                    'retrieval_time_ms': retrieval_time * 1000,
                    'component': 'VectorStore'
                }
            )
            
            # Generate response
            response = self.llm.generate(query_text, retrieved_docs)
            
            logger.info(
                "RAG query completed successfully",
                extra={
                    'query_id': query_id,
                    'response_length': len(response),
                    'component': 'RAGPipeline'
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "RAG query failed",
                extra={
                    'query_id': query_id,
                    'error': str(e),
                    'component': 'RAGPipeline'
                },
                exc_info=True
            )
            raise
```

**Why**: Good logging helps debug issues in production, monitor performance, and understand system behavior.

## 6. Resource Management and Memory Efficiency

**Implement proper resource management, especially for ML models and large datasets.**

```python
import gc
from contextlib import contextmanager
from typing import Iterator, List
import torch

class Embeddings:
    def __init__(self, model_name: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self._model_loaded = False
    
    def _lazy_load_model(self):
        """Load model only when needed."""
        if not self._model_loaded:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            self._model_loaded = True
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches to manage memory usage."""
        self._lazy_load_model()
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.array(embeddings)
    
    @contextmanager
    def temporary_model_load(self) -> Iterator['Embeddings']:
        """Context manager for temporary model usage."""
        try:
            self._lazy_load_model()
            yield self
        finally:
            if self._model_loaded:
                del self.model
                self.model = None
                self._model_loaded = False
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Usage
def process_large_dataset(embeddings: Embeddings, documents: List[str]):
    """Process large datasets with proper memory management."""
    with embeddings.temporary_model_load() as emb:
        return emb.encode_batch(documents, batch_size=16)
```

**Why**: ML models can consume significant memory. Proper resource management prevents out-of-memory errors and improves performance.

## 7. API Design and Backward Compatibility

**Design clean, consistent APIs and maintain backward compatibility.**

```python
from typing import Union, Deprecated
import warnings

class VectorStore:
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Node]:
        """Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of Node objects containing similar documents
        """
        pass
    
    # Deprecated method with clear migration path
    @Deprecated("Use similarity_search() instead. Will be removed in v0.3.0")
    def search(self, query: str, limit: int = 5) -> List[Node]:
        """Legacy search method."""
        warnings.warn(
            "search() is deprecated, use similarity_search() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.similarity_search(query, k=limit)
    
    # Method with backward-compatible parameter evolution
    def add_documents(
        self,
        nodes: List[Node],
        ids: Optional[List[str]] = None,
        # New parameter with default to maintain compatibility
        batch_size: int = 100,
        # Deprecated parameter
        chunk_size: Optional[int] = None,
    ):
        """Add documents to the vector store."""
        if chunk_size is not None:
            warnings.warn(
                "chunk_size parameter is deprecated, use batch_size instead",
                DeprecationWarning
            )
            batch_size = chunk_size
        
        # Implementation here
```

**Why**: Consistent APIs improve developer experience. Backward compatibility prevents breaking changes that would frustrate users.

## 8. Performance Optimization and Monitoring

**Implement performance optimizations and provide monitoring capabilities.**

```python
import time
import functools
from typing import Callable, Any, Dict
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0

class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
    
    def track_performance(self, operation_name: str):
        """Decorator to track method performance."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self._update_metrics(operation_name, execution_time)
            return wrapper
        return decorator
    
    def _update_metrics(self, operation: str, execution_time: float):
        """Update performance metrics for an operation."""
        metrics = self.metrics[operation]
        metrics.call_count += 1
        metrics.total_time += execution_time
        metrics.avg_time = metrics.total_time / metrics.call_count
        metrics.min_time = min(metrics.min_time, execution_time)
        metrics.max_time = max(metrics.max_time, execution_time)
    
    def get_report(self) -> Dict[str, Dict[str, float]]:
        """Get performance report."""
        return {
            op: {
                'calls': m.call_count,
                'total_time': round(m.total_time, 4),
                'avg_time': round(m.avg_time, 4),
                'min_time': round(m.min_time, 4),
                'max_time': round(m.max_time, 4),
            }
            for op, m in self.metrics.items()
        }

# Usage in components
monitor = PerformanceMonitor()

class VectorStore:
    @monitor.track_performance("vector_search")
    def similarity_search(self, query: str, k: int = 5) -> List[Node]:
        """Optimized similarity search with caching."""
        # Check cache first
        cache_key = f"{hash(query)}_{k}"
        if hasattr(self, '_cache') and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Perform search
        results = self._perform_search(query, k)
        
        # Cache results (with size limit)
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if len(self._cache) > 100:  # Simple cache eviction
            self._cache.clear()
        
        self._cache[cache_key] = results
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return monitor.get_report()
```

**Why**: Performance optimization ensures the library scales well. Monitoring helps identify bottlenecks and track improvements over time.

## Summary

These guidelines focus on:
1. **Clear APIs** with type hints and documentation
2. **Robust error handling** with custom exceptions
3. **Modular design** through dependency injection
4. **Comprehensive testing** with proper mocking
5. **Observability** through structured logging
6. **Resource efficiency** for ML workloads
7. **API stability** and backward compatibility
8. **Performance monitoring** and optimization

Following these guidelines will help maintain high code quality, make the library more reliable, and improve the developer experience for users of zRAG.