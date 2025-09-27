from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import inspect
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from .exceptions import InvalidChunkSizeError, ZRAGError
from .models import Document, Node, ChunkerConfig

# Optional heavy dependencies are loaded lazily and provide informative
# error messages if missing. This prevents hard import failures for users
# who only need non-token chunkers.
_spacy_spec = importlib.util.find_spec("spacy")
if _spacy_spec is None:  # pragma: no cover - environment dependent
    spacy = None  # type: ignore[assignment]
    _SPACY_IMPORT_ERROR = ModuleNotFoundError("No module named 'spacy'")
else:  # pragma: no cover - import has side effects not needing tests
    spacy = importlib.import_module("spacy")
    _SPACY_IMPORT_ERROR = None

_nltk_spec = importlib.util.find_spec("nltk")
if _nltk_spec is None:  # pragma: no cover - environment dependent
    nltk = None  # type: ignore[assignment]
    _NLTK_IMPORT_ERROR = ModuleNotFoundError("No module named 'nltk'")
else:  # pragma: no cover
    nltk = importlib.import_module("nltk")
    _NLTK_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class ChunkSplitter(ABC):
    """
    Abstract base class for different chunking strategies.
    
    Provides interface for splitting documents into smaller chunks (nodes)
    with different strategies like token-based, sentence-based, or paragraph-based.
    """

    @abstractmethod
    def split_document(self, document: Document) -> List[Node]:
        """
        Splits a single document into chunks (Nodes).
        
        Args:
            document: Document object to split
            
        Returns:
            List of Node objects representing chunks
            
        Raises:
            ZRAGError: If document splitting fails
        """
        pass

    def split(self, documents: List[Document]) -> List[Node]:
        """
        Splits a list of documents into chunks, returning a list of nodes.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of Node objects from all documents
            
        Raises:
            ZRAGError: If document splitting fails
        """
        if not documents:
            logger.warning("Empty document list provided for chunking")
            return []
            
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All items must be Document objects")
            
        nodes = []
        failed_documents = 0
        
        for i, document in enumerate(documents):
            try:
                document_nodes = self.split_document(document)
                nodes.extend(document_nodes)
            except Exception as e:
                failed_documents += 1
                logger.error(f"Failed to split document {i}: {e}")
                
        logger.info(
            f"Chunking completed. Processed: {len(documents) - failed_documents} documents, "
            f"Generated: {len(nodes)} chunks, Failed: {failed_documents}"
        )
        
        return nodes


def _create_node(
    document: Document,
    text: str,
    start: int,
    chunk_type: str,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Node:
    """Create a Node with consistent metadata enrichment."""

    metadata: Dict[str, str] = {
        "document_id": document.document_id,
        "page_label": document.metadata.get("page_label", "1"),
        "start_index": start,
        "end_index": start + len(text),
        "chunk_type": chunk_type,
        "chunk_strategy": chunk_type,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Node(text.strip(), metadata)


class TokenChunkSplitter(ChunkSplitter):
    """
    Splits documents into chunks of text based on tokens (words).
    
    Uses spaCy for tokenization to handle linguistic boundaries properly.
    
    Attributes:
        chunk_size: The desired size of each chunk in tokens
        nlp: spaCy language model for tokenization
    """

    def __init__(self, chunk_size: int = 256):
        """
        Initializes the TokenChunkSplitter object.

        Args:
            chunk_size: The desired size of each chunk (in tokens)
            
        Raises:
            InvalidChunkSizeError: If chunk_size is invalid
            ZRAGError: If spaCy model loading fails
        """
        if chunk_size <= 0:
            raise InvalidChunkSizeError(f"Chunk size must be positive, got {chunk_size}")
            
        self.chunk_size = chunk_size

        if spacy is None:
            raise ZRAGError(
                "spaCy is required for token-based chunking. Install it with 'pip install spacy en-core-web-sm'",
                details=str(_SPACY_IMPORT_ERROR),
            )
        
        try:
            self.nlp = spacy.load("en_core_web_sm")  # Load a small English model
            logger.info(f"TokenChunkSplitter initialized with chunk_size={chunk_size}")
        except OSError as e:
            raise ZRAGError(
                "Failed to load spaCy model. Please install with: python -m spacy download en_core_web_sm",
                details=str(e)
            )

    def split_document(self, document: Document) -> List[Node]:
        """
        Splits a document into chunks of text based on tokens.
        
        Args:
            document: Document object to split
            
        Returns:
            List of Node objects representing token-based chunks
            
        Raises:
            ZRAGError: If tokenization fails
        """
        if not document.text.strip():
            logger.warning(f"Empty text in document {document.document_id}")
            return []
            
        try:
            doc = self.nlp(document.text)

            if not doc:
                return []

            nodes: List[Node] = []
            current_tokens: List[spacy.tokens.token.Token] = []  # type: ignore[attr-defined]
            current_start: Optional[int] = None

            def flush() -> None:
                nonlocal current_tokens, current_start
                if not current_tokens:
                    return
                chunk_text = " ".join(token.text for token in current_tokens)
                start_index = current_start if current_start is not None else 0
                metadata = {
                    "chunk_size": len(current_tokens),
                }
                nodes.append(
                    _create_node(document, chunk_text, start_index, "token", metadata)
                )
                current_tokens = []
                current_start = None

            for token in doc:  # type: ignore[attr-defined]
                if current_start is None:
                    current_start = token.idx
                current_tokens.append(token)
                if len(current_tokens) >= self.chunk_size:
                    flush()

            flush()

            logger.debug(
                "Split document %s into %d token chunks", document.document_id, len(nodes)
            )
            return nodes
            
        except Exception as e:
            raise ZRAGError(f"Failed to split document {document.document_id} into token chunks", details=str(e))


class SentenceChunkSplitterWithOverlap(ChunkSplitter):
    """
    Splits documents into chunks based on sentences with overlap.
    
    Provides overlapping chunks to maintain context across chunk boundaries,
    which can improve retrieval performance.
    
    Attributes:
        chunk_size: The desired size of each chunk in characters
        overlap: The number of characters to overlap between chunks
        tokenizer: NLTK sentence tokenizer
    """

    def __init__(self, chunk_size: int = 1024, overlap: int = 128):
        """
        Initializes the SentenceChunkSplitterWithOverlap object.

        Args:
            chunk_size: The desired size of each chunk (in characters)
            overlap: The number of characters to overlap between chunks
            
        Raises:
            InvalidChunkSizeError: If chunk_size or overlap values are invalid
            ZRAGError: If NLTK initialization fails
        """
        if chunk_size <= 0:
            raise InvalidChunkSizeError(f"Chunk size must be positive, got {chunk_size}")
            
        if overlap < 0:
            raise InvalidChunkSizeError(f"Overlap must be non-negative, got {overlap}")
            
        if overlap >= chunk_size:
            raise InvalidChunkSizeError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
            
        self.chunk_size = chunk_size
        self.overlap = overlap

        if nltk is None:
            raise ZRAGError(
                "NLTK is required for sentence-based chunking. Install it with 'pip install nltk'",
                details=str(_NLTK_IMPORT_ERROR),
            )
        
        try:
            nltk.download('punkt', quiet=True)  # Download Punkt sentence tokenizer
            self.tokenizer = nltk.tokenize.PunktSentenceTokenizer()
            logger.info(f"SentenceChunkSplitterWithOverlap initialized with chunk_size={chunk_size}, overlap={overlap}")
        except Exception as e:
            raise ZRAGError("Failed to initialize NLTK tokenizer", details=str(e))

    def split_document(self, document: Document) -> List[Node]:
        """
        Splits a document into chunks based on sentences with overlap.
        
        Args:
            document: Document object to split
            
        Returns:
            List of Node objects representing sentence-based chunks with overlap
            
        Raises:
            ZRAGError: If sentence tokenization fails
        """
        if not document.text.strip():
            logger.warning(f"Empty text in document {document.document_id}")
            return []
            
        try:
            text = document.text
            sentence_spans = list(self.tokenizer.span_tokenize(text))

            if not sentence_spans:
                return []

            nodes: List[Node] = []
            current_sentences: List[Tuple[str, int, int]] = []

            def current_text_length() -> int:
                return len(" ".join(sentence for sentence, _, _ in current_sentences))

            def flush() -> None:
                if not current_sentences:
                    return
                chunk_text = " ".join(sentence for sentence, _, _ in current_sentences)
                chunk_start = current_sentences[0][1]
                metadata = {
                    "chunk_size": len(chunk_text),
                    "overlap_size": self.overlap,
                }
                nodes.append(
                    _create_node(document, chunk_text, chunk_start, "sentence_overlap", metadata)
                )

            for span_start, span_end in sentence_spans:
                sentence_text = text[span_start:span_end].strip()
                if not sentence_text:
                    continue

                proposed_length = current_text_length() + (1 if current_sentences else 0) + len(sentence_text)
                if current_sentences and proposed_length > self.chunk_size:
                    flush()
                    if self.overlap > 0 and current_sentences:
                        overlap_chars = self.overlap
                        retained: List[Tuple[str, int, int]] = []
                        for sent in reversed(current_sentences):
                            retained.insert(0, sent)
                            overlap_chars -= len(sent[0])
                            if overlap_chars <= 0:
                                break
                        current_sentences = retained
                    else:
                        current_sentences = []

                current_sentences.append((sentence_text, span_start, span_end))

            flush()

            logger.debug(
                "Split document %s into %d sentence chunks with overlap",
                document.document_id,
                len(nodes),
            )
            return nodes
            
        except Exception as e:
            raise ZRAGError(f"Failed to split document {document.document_id} into sentence chunks", details=str(e))


class ParagraphChunkSplitter(ChunkSplitter):
    """
    Splits documents into chunks of text based on paragraphs.
    
    Splits text on paragraph boundaries (double newlines) while respecting
    maximum chunk size constraints.
    
    Attributes:
        chunk_size: The desired size of each chunk in characters
    """

    def __init__(self, chunk_size: int = 2048):
        """
        Initializes the ParagraphChunkSplitter object.

        Args:
            chunk_size: The desired size of each chunk (in characters)
            
        Raises:
            InvalidChunkSizeError: If chunk_size is invalid
        """
        if chunk_size <= 0:
            raise InvalidChunkSizeError(f"Chunk size must be positive, got {chunk_size}")
            
        self.chunk_size = chunk_size
        logger.info(f"ParagraphChunkSplitter initialized with chunk_size={chunk_size}")

    def split_document(self, document: Document) -> List[Node]:
        """
        Splits a document into chunks of text based on paragraphs.
        
        Args:
            document: Document object to split
            
        Returns:
            List of Node objects representing paragraph-based chunks
            
        Raises:
            ZRAGError: If paragraph splitting fails
        """
        if not document.text.strip():
            logger.warning(f"Empty text in document {document.document_id}")
            return []
            
        try:
            text = document.text
            pattern = re.compile(r"(.+?)(?:\n\s*\n|\Z)", re.DOTALL)
            matches = [m for m in pattern.finditer(text) if m.group(1).strip()]

            if not matches:
                return []

            nodes: List[Node] = []
            current_chunk_parts: List[Tuple[str, int, int]] = []

            def chunk_length() -> int:
                return sum(len(part[0]) for part in current_chunk_parts) + max(0, len(current_chunk_parts) - 1) * 2

            def flush() -> None:
                if not current_chunk_parts:
                    return
                chunk_text = "\n\n".join(part[0] for part in current_chunk_parts)
                chunk_start = current_chunk_parts[0][1]
                metadata = {
                    "chunk_size": len(chunk_text),
                }
                nodes.append(
                    _create_node(document, chunk_text, chunk_start, "paragraph", metadata)
                )
                current_chunk_parts.clear()

            for match in matches:
                paragraph_text = match.group(1).strip()
                start, end = match.span(1)

                if current_chunk_parts and chunk_length() + len(paragraph_text) + 2 > self.chunk_size:
                    flush()

                current_chunk_parts.append((paragraph_text, start, end))

            flush()

            logger.debug(
                "Split document %s into %d paragraph chunks",
                document.document_id,
                len(nodes),
            )
            return nodes
            
        except Exception as e:
            raise ZRAGError(f"Failed to split document {document.document_id} into paragraph chunks", details=str(e))


_CHUNK_SPLITTERS: Dict[str, Type[ChunkSplitter]] = {
    "token": TokenChunkSplitter,
    "sentence_overlap": SentenceChunkSplitterWithOverlap,
    "overlap": SentenceChunkSplitterWithOverlap,
    "paragraph": ParagraphChunkSplitter,
}


def register_chunk_splitter(name: str, splitter_cls: Type[ChunkSplitter]) -> None:
    """Register a custom chunk splitter implementation."""

    if not issubclass(splitter_cls, ChunkSplitter):
        raise TypeError("splitter_cls must inherit from ChunkSplitter")
    _CHUNK_SPLITTERS[name] = splitter_cls


def get_chunk_splitter(
    strategy: Union[str, ChunkerConfig],
    **kwargs: Dict[str, Any],
) -> ChunkSplitter:
    """
    Factory function to select the desired chunking strategy.
    
    Args:
        strategy: Name of the chunking strategy ('token', 'overlap', 'paragraph')
        **kwargs: Additional arguments to pass to the splitter constructor
        
    Returns:
        ChunkSplitter instance of the requested type
        
    Raises:
        ValueError: If strategy is unknown
        
    Example:
        splitter = get_chunk_splitter('token', chunk_size=512)
        splitter = get_chunk_splitter('overlap', chunk_size=1024, overlap=128)
    """
    if isinstance(strategy, ChunkerConfig):
        splitter_kwargs = {**strategy.as_kwargs(), **kwargs}
        strategy_name = strategy.strategy
    else:
        splitter_kwargs = kwargs
        strategy_name = strategy

    splitter_class = _CHUNK_SPLITTERS.get(strategy_name)
    if splitter_class is None:
        available = ", ".join(sorted(_CHUNK_SPLITTERS))
        raise ValueError(
            f"Unknown chunking strategy: {strategy_name}. Available strategies: {available}"
        )

    try:
        signature = inspect.signature(splitter_class.__init__)
        valid_params = {
            name for name, param in signature.parameters.items() if name not in {"self"}
        }
        filtered_kwargs = {k: v for k, v in splitter_kwargs.items() if k in valid_params}
        return splitter_class(**filtered_kwargs)
    except Exception as exc:
        raise ZRAGError(f"Failed to create {strategy_name} chunk splitter", details=str(exc))