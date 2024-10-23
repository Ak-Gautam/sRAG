from pathlib import Path
from typing import List, Dict, Optional
import spacy
import fitz
import markdown
import re
import nltk
from zrag.doc_loader import Document


class Node:
    """Represents a single node containing a chunk of text and metadata."""

    def __init__(self, text: str, metadata: Dict):
        """
        Initializes a Node object.

        Args:
            text (str): The text content of the node.
            metadata (Dict): A dictionary containing metadata about the node.
        """
        self.text = text.strip()  # Remove leading/trailing spaces
        self.metadata = metadata

    def __repr__(self):
        return f"Node(text='{self.text[:20]}...', metadata={self.metadata})"


class ChunkSplitter:
    """A base class for different chunking strategies."""

    def split_document(self, document: Document) -> List[Node]:
        """Splits a single document into chunks (Nodes)."""
        raise NotImplementedError("Subclasses should implement this method.")

    def split(self, documents: List[Document]) -> List[Node]:
        """Splits a list of documents into chunks, returning a list of nodes."""
        nodes = []
        for document in documents:
            nodes.extend(self.split_document(document))
        return nodes


class TokenChunkSplitter(ChunkSplitter):
    """Splits documents into chunks of text based on tokens (words)."""

    def __init__(self, chunk_size: int = 256):
        """
        Initializes the TokenChunkSplitter object.

        Args:
            chunk_size (int): The desired size of each chunk (in tokens).
        """
        self.chunk_size = chunk_size
        self.nlp = spacy.load("en_core_web_sm")  # Load a small English model

    def split_document(self, document: Document) -> List[Node]:
        """Splits a document into chunks of text based on tokens."""
        text = document.text
        doc = self.nlp(text)
        tokens = [token.text for token in doc]

        nodes = []
        current_chunk = []
        current_chunk_start_index = 0

        for i, token in enumerate(tokens):
            # If adding the token would exceed the chunk size, start a new chunk
            if len(current_chunk) >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = {
                    'document_id': document.document_id,
                    'page_label': document.metadata.get('page_label'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(chunk_text)
                }
                nodes.append(Node(chunk_text, chunk_metadata))
                current_chunk = [token]  # Start a new chunk 
                current_chunk_start_index = doc[i].idx  # Get the character index
            else:
                current_chunk.append(token)

        # Append the last chunk if it's not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = {
                'document_id': document.document_id,
                'page_label': document.metadata.get('page_label'),
                'start_index': current_chunk_start_index,
                'end_index': current_chunk_start_index + len(chunk_text)
            }
            nodes.append(Node(chunk_text, chunk_metadata))

        return nodes


class SentenceChunkSplitterWithOverlap(ChunkSplitter):
    """Splits documents into chunks based on sentences with overlap."""

    def __init__(self, chunk_size: int = 1024, overlap: int = 128):
        """
        Initializes the SentenceChunkSplitterWithOverlap object.

        Args:
            chunk_size (int): The desired size of each chunk (in characters).
            overlap (int): The number of characters to overlap.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        nltk.download('punkt', quiet=True)  # Download Punkt sentence tokenizer
        self.tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    def split_document(self, document: Document) -> List[Node]:
        """Splits a document into chunks based on sentences with overlap."""
        text = document.text
        sentences = self.tokenizer.tokenize(text)

        nodes = []
        current_chunk = ""
        current_chunk_start_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding the sentence would exceed the chunk size, start a new chunk
            if len(current_chunk) + sentence_len > self.chunk_size:
                chunk_metadata = {
                    'document_id': document.document_id,
                    'page_label': document.metadata.get('page_label'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(current_chunk)
                }
                nodes.append(Node(current_chunk.strip(), chunk_metadata))
                
                # Overlap logic
                overlap_start = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[overlap_start:] + " " + sentence
                current_chunk_start_index += len(current_chunk) - sentence_len - 1 

            else:
                current_chunk += " " + sentence

        # Append the last chunk if it's not empty
        if current_chunk:
            chunk_metadata = {
                'document_id': document.document_id,
                'page_label': document.metadata.get('page_label'),
                'start_index': current_chunk_start_index,
                'end_index': current_chunk_start_index + len(current_chunk)
            }
            nodes.append(Node(current_chunk.strip(), chunk_metadata))

        return nodes


class ParagraphChunkSplitter(ChunkSplitter):
    """Splits documents into chunks of text based on paragraphs."""

    def __init__(self, chunk_size: int = 2048):
        """
        Initializes the ParagraphChunkSplitter object.

        Args:
            chunk_size (int): The desired size of each chunk (in characters).
        """
        self.chunk_size = chunk_size

    def split_document(self, document: Document) -> List[Node]:
        """Splits a document into chunks of text based on paragraphs."""
        text = document.text
        paragraphs = text.split('\n\n')  # Split by double newline

        nodes = []
        current_chunk = ""
        current_chunk_start_index = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)

            # If adding the paragraph would exceed the chunk size, start a new chunk
            if len(current_chunk) + paragraph_len > self.chunk_size:
                chunk_metadata = {
                    'document_id': document.document_id,
                    'page_label': document.metadata.get('page_label'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(current_chunk)
                }
                nodes.append(Node(current_chunk.strip(), chunk_metadata))
                current_chunk = paragraph  # Start a new chunk 
                current_chunk_start_index += len(current_chunk) + 2  # Add 2 for the double newline
            else:
                current_chunk += paragraph + "\n\n" 

        # Append the last chunk if it's not empty
        if current_chunk:
            chunk_metadata = {
                'document_id': document.document_id,
                'page_label': document.metadata.get('page_label'),
                'start_index': current_chunk_start_index,
                'end_index': current_chunk_start_index + len(current_chunk)
            }
            nodes.append(Node(current_chunk.strip(), chunk_metadata))

        return nodes


def get_chunk_splitter(strategy: str, **kwargs) -> ChunkSplitter:
    """Factory function to select the desired chunking strategy."""
    if strategy == 'token':
        return TokenChunkSplitter(**kwargs)
    elif strategy == 'overlap':
        return SentenceChunkSplitterWithOverlap(**kwargs)
    elif strategy == 'paragraph':
        return ParagraphChunkSplitter(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")