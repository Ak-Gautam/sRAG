"""Custom exceptions for the zRAG library."""

from typing import Optional, Any


class ZRAGError(Exception):
    """Base exception for zRAG library."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details
        
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class DocumentLoadError(ZRAGError):
    """Raised when document loading fails."""
    pass


class UnsupportedFileFormatError(DocumentLoadError):
    """Raised when file format is not supported."""
    pass


class InvalidChunkSizeError(ZRAGError):
    """Raised when chunk size is invalid."""
    pass


class EmbeddingError(ZRAGError):
    """Raised when embedding generation fails."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when model loading fails."""
    pass


class VectorStoreError(ZRAGError):
    """Raised when vector store operations fail."""
    pass


class VectorStoreNotInitializedError(VectorStoreError):
    """Raised when vector store is not properly initialized."""
    pass


class LLMError(ZRAGError):
    """Raised when LLM operations fail."""
    pass


class PromptError(ZRAGError):
    """Raised when prompt operations fail."""
    pass


class TemplateNotFoundError(PromptError):
    """Raised when a template is not found."""
    pass


class RAGPipelineError(ZRAGError):
    """Raised when RAG pipeline operations fail."""
    pass