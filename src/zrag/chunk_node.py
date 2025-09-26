from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import spacy
import re
import nltk
import logging

from .doc_loader import Document
from .exceptions import InvalidChunkSizeError, ZRAGError

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """
    Represents a single node containing a chunk of text and metadata.
    
    Attributes:
        text: The text content of the node
        metadata: Dictionary containing metadata about the node (document_id, indices, etc.)
        embedding: Optional embedding vector for the text content
    """
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[Any] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if not isinstance(self.text, str):
            raise ValueError("Text must be a string")
        self.text = self.text.strip()  # Remove leading/trailing spaces
        
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")

    def __repr__(self) -> str:
        return f"Node(text='{self.text[:20]}...', metadata={self.metadata})"


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
            text = document.text
            doc = self.nlp(text)
            tokens = [token.text for token in doc]

            if not tokens:
                return []

            nodes = []
            current_chunk = []
            current_chunk_start_index = 0

            for i, token in enumerate(tokens):
                # If adding the token would exceed the chunk size, start a new chunk
                if len(current_chunk) >= self.chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunk_metadata = {
                        'document_id': document.document_id,
                        'page_label': document.metadata.get('page_label', '1'),
                        'start_index': current_chunk_start_index,
                        'end_index': current_chunk_start_index + len(chunk_text),
                        'chunk_type': 'token',
                        'chunk_size': len(current_chunk)
                    }
                    nodes.append(Node(chunk_text, chunk_metadata))
                    current_chunk = [token]  # Start a new chunk 
                    current_chunk_start_index = doc[i].idx if i < len(doc) else current_chunk_start_index + len(chunk_text)
                else:
                    current_chunk.append(token)

            # Append the last chunk if it's not empty
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = {
                    'document_id': document.document_id,
                    'page_label': document.metadata.get('page_label', '1'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(chunk_text),
                    'chunk_type': 'token',
                    'chunk_size': len(current_chunk)
                }
                nodes.append(Node(chunk_text, chunk_metadata))

            logger.debug(f"Split document {document.document_id} into {len(nodes)} token chunks")
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
            sentences = self.tokenizer.tokenize(text)

            if not sentences:
                return []

            nodes = []
            current_chunk = ""
            current_chunk_start_index = 0

            for sentence in sentences:
                sentence_len = len(sentence)

                # If adding the sentence would exceed the chunk size, start a new chunk
                if len(current_chunk) + sentence_len > self.chunk_size:
                    if current_chunk.strip():  # Only create node if chunk has content
                        chunk_metadata = {
                            'document_id': document.document_id,
                            'page_label': document.metadata.get('page_label', '1'),
                            'start_index': current_chunk_start_index,
                            'end_index': current_chunk_start_index + len(current_chunk),
                            'chunk_type': 'sentence_overlap',
                            'chunk_size': len(current_chunk),
                            'overlap_size': self.overlap
                        }
                        nodes.append(Node(current_chunk.strip(), chunk_metadata))
                    
                    # Overlap logic
                    overlap_start = max(0, len(current_chunk) - self.overlap)
                    current_chunk = current_chunk[overlap_start:] + " " + sentence
                    current_chunk_start_index += overlap_start

                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            # Append the last chunk if it's not empty
            if current_chunk.strip():
                chunk_metadata = {
                    'document_id': document.document_id,
                    'page_label': document.metadata.get('page_label', '1'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(current_chunk),
                    'chunk_type': 'sentence_overlap',
                    'chunk_size': len(current_chunk),
                    'overlap_size': self.overlap
                }
                nodes.append(Node(current_chunk.strip(), chunk_metadata))

            logger.debug(f"Split document {document.document_id} into {len(nodes)} sentence chunks with overlap")
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
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]  # Split by double newline and filter empty

            if not paragraphs:
                return []

            nodes = []
            current_chunk = ""
            current_chunk_start_index = 0

            for paragraph in paragraphs:
                paragraph_len = len(paragraph)

                # If adding the paragraph would exceed the chunk size, start a new chunk
                if current_chunk and len(current_chunk) + paragraph_len + 2 > self.chunk_size:  # +2 for \n\n
                    chunk_metadata = {
                        'document_id': document.document_id,
                        'page_label': document.metadata.get('page_label', '1'),
                        'start_index': current_chunk_start_index,
                        'end_index': current_chunk_start_index + len(current_chunk),
                        'chunk_type': 'paragraph',
                        'chunk_size': len(current_chunk)
                    }
                    nodes.append(Node(current_chunk.strip(), chunk_metadata))
                    current_chunk = paragraph  # Start a new chunk 
                    current_chunk_start_index += len(current_chunk) + 2  # Add 2 for the double newline
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph

            # Append the last chunk if it's not empty
            if current_chunk.strip():
                chunk_metadata = {
                    'document_id': document.document_id,
                    'page_label': document.metadata.get('page_label', '1'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(current_chunk),
                    'chunk_type': 'paragraph',
                    'chunk_size': len(current_chunk)
                }
                nodes.append(Node(current_chunk.strip(), chunk_metadata))

            logger.debug(f"Split document {document.document_id} into {len(nodes)} paragraph chunks")
            return nodes
            
        except Exception as e:
            raise ZRAGError(f"Failed to split document {document.document_id} into paragraph chunks", details=str(e))


def get_chunk_splitter(strategy: str, **kwargs) -> ChunkSplitter:
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
    strategies = {
        'token': TokenChunkSplitter,
        'overlap': SentenceChunkSplitterWithOverlap,
        'paragraph': ParagraphChunkSplitter,
    }
    
    if strategy not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown chunking strategy: {strategy}. Available strategies: {available}")
    
    try:
        splitter_class = strategies[strategy]
        return splitter_class(**kwargs)
    except Exception as e:
        raise ZRAGError(f"Failed to create {strategy} chunk splitter", details=str(e))