import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional
import uuid
import datetime

import fitz  # PyMuPDF for PDF reading
import markdown  # For Markdown parsing
from file_loader import FileLoader, Document  # Import FileLoader and Document

class Node:
    """Represents a single node containing a chunk of text and metadata."""

    def __init__(self, text: str, metadata: Dict):
        """
        Initializes a Node object.

        Args:
            text (str): The text content of the node.
            metadata (Dict): A dictionary containing metadata about the node.
        """
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Node(text='{self.text[:20]}...', metadata={self.metadata})"


class ChunkSplitter:
    """Splits documents into chunks of text."""

    def __init__(self, chunk_size: int = 1024):
        """
        Initializes the ChunkSplitter object.

        Args:
            chunk_size (int): The desired size of each chunk (in characters).
        """
        self.chunk_size = chunk_size

    def split_document(self, document: Document) -> List[Node]:
        """Splits a document into chunks of text.

        Args:
            document (Document): The Document to split.

        Returns:
            List[Node]: A list of Node objects representing the chunks.
        """
        text = document.text
        nodes = []
        start_index = 0
        while start_index < len(text):
            end_index = min(start_index + self.chunk_size, len(text))
            chunk_text = text[start_index:end_index]
            chunk_metadata = {
                'document_id': document.id,
                'page_label': document.metadata.get('page_label'),
                'start_index': start_index,
                'end_index': end_index
            }
            nodes.append(Node(chunk_text, chunk_metadata))
            start_index = end_index
        return nodes

    def get_nodes_from_documents(self, documents: List[Document]) -> List[Node]:
        """Splits a list of documents into chunks of text, returning a list of nodes.

        Args:
            documents (List[Document]): A list of Document objects.

        Returns:
            List[Node]: A list of Node objects representing the chunks from all documents.
        """
        nodes = []
        for document in documents:
            nodes.extend(self.split_document(document))
        return nodes