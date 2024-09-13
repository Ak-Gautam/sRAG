from pathlib import Path
from typing import List, Dict, Optional
import spacy
import fitz
import markdown
from DocLoader import Document
import re


class Node:
    """Represents a single node containing a chunk of text and metadata."""

    def __init__(self, text: str, metadata: Dict):
        """
        Initializes a Node object.

        Args:
            text (str): The text content of the node.
            metadata (Dict): A dictionary containing metadata about the node.
        """
        self.text = text.strip()  # Bug fix: remove leading/trailing spaces
        self.metadata = metadata

    def __repr__(self):
        return f"Node(text='{self.text[:20]}...', metadata={self.metadata})"

class ChunkSplitter:
    """A base class for different chunking strategies."""

    def split_document(self, document: Document) -> List[Node]:
        raise NotImplementedError("Subclasses should implement this method.")

    def get_nodes_from_documents(self, documents: List[Document]) -> List[Node]:
        """Splits a list of documents into chunks, returning a list of nodes.

        Args:
            documents (List[Document]): A list of Document objects.

        Returns:
            List[Node]: A list of Node objects representing the chunks from all documents.
        """
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
        """Splits a document into chunks of text based on tokens.

        Args:
            document (Document): The Document to split.

        Returns:
            List[Node]: A list of Node objects representing the chunks.
        """
        text = document.text
        doc = self.nlp(text)
        tokens = [token.text for token in doc]

        nodes = []
        current_chunk = []
        current_chunk_start_index = 0

        for token in tokens:
            token_len = len(token)

            # If adding the token would exceed the chunk size, start a new chunk
            if len(current_chunk) + 1 > self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = {
                    'document_id': document.id,
                    'page_label': document.metadata.get('page_label'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(chunk_text)
                }
                nodes.append(Node(chunk_text, chunk_metadata))
                current_chunk = [token]
                current_chunk_start_index += len(chunk_text) + 1  # Include space
            else:
                current_chunk.append(token)

        # Append the last chunk if it's not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = {
                'document_id': document.id,
                'page_label': document.metadata.get('page_label'),
                'start_index': current_chunk_start_index,
                'end_index': current_chunk_start_index + len(chunk_text)
            }
            nodes.append(Node(chunk_text, chunk_metadata))

        return nodes

class SentenceChunkSplitterWithOverlap(ChunkSplitter):
    """Splits documents into chunks of text based on sentences with overlap."""

    def __init__(self, chunk_size: int = 1024, overlap: int = 128):
        """
        Initializes the SentenceChunkSplitterWithOverlap object.

        Args:
            chunk_size (int): The desired size of each chunk (in characters).
            overlap (int): The number of characters to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_document(self, document: Document) -> List[Node]:
        """Splits a document into chunks of text based on sentences with overlap.

        Args:
            document (Document): The Document to split.

        Returns:
            List[Node]: A list of Node objects representing the chunks.
        """
        text = document.text
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

        nodes = []
        current_chunk = ""
        current_chunk_start_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding the sentence would exceed the chunk size, start a new chunk
            if len(current_chunk) + sentence_len > self.chunk_size:
                chunk_metadata = {
                    'document_id': document.id,
                    'page_label': document.metadata.get('page_label'),
                    'start_index': current_chunk_start_index,
                    'end_index': current_chunk_start_index + len(current_chunk)
                }
                nodes.append(Node(current_chunk, chunk_metadata))
                current_chunk = sentence
                current_chunk_start_index = current_chunk_start_index + len(current_chunk) - self.overlap
            else:
                current_chunk += " " + sentence

        # Append the last chunk if it's not empty
        if current_chunk:
            chunk_metadata = {
                'document_id': document.id,
                'page_label': document.metadata.get('page_label'),
                'start_index': current_chunk_start_index,
                'end_index': current_chunk_start_index + len(current_chunk)
            }
            nodes.append(Node(current_chunk, chunk_metadata))

        return nodes
    
class ParagraphChunkSplitter(ChunkSplitter):
    """Splits documents into chunks of text based on paragraphs."""

    def split_document(self, document: Document) -> List[Node]:
        """Splits a document into chunks of text based on paragraphs.

        Args:
            document (Document): The Document to split.

        Returns:
            List[Node]: A list of Node objects representing the chunks.
        """
        text = document.text
        paragraphs = text.split('\n\n')  # Split by double newline

        nodes = []
        current_chunk_start_index = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            chunk_metadata = {
                'document_id': document.id,
                'page_label': document.metadata.get('page_label'),
                'start_index': current_chunk_start_index,
                'end_index': current_chunk_start_index + paragraph_len
            }
            nodes.append(Node(paragraph, chunk_metadata))
            current_chunk_start_index += paragraph_len + 2  # Include double newline

        return nodes

# Factory function to select the desired chunking strategy
def get_chunk_splitter(strategy: str, **kwargs) -> ChunkSplitter:
    if strategy == 'token':
        return TokenChunkSplitter(**kwargs)
    elif strategy == 'overlap':
        return SentenceChunkSplitterWithOverlap(**kwargs)
    elif strategy == 'paragraph':
        return ParagraphChunkSplitter()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")