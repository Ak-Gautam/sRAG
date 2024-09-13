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