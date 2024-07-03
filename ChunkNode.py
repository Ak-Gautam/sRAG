import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional
import uuid
import datetime

import fitz  
import markdown  
from DocLoader import FileLoader, Document  # Import FileLoader and Document

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
        self.text = text.strip() # Bug fix remove leading/trailing spaces
        self.metadata = metadata

    def __repr__(self):
        return f"Node(text='{self.text[:20]}...', metadata={self.metadata})"


class SentenceChunkSplitter:
    """Splits documents into chunks of text based on sentences."""

    def __init__(self, chunk_size: int = 1024):
        """
        Initializes the SentenceChunkSplitter object.

        Args:
            chunk_size (int): The desired size of each chunk (in characters).
                          Chunks will be formed from complete sentences and 
                          may exceed this size to maintain context.
        """
        self.chunk_size = chunk_size

    def split_document(self, document: Document) -> List[Node]:
        """Splits a document into chunks of text based on sentences.

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
                current_chunk_start_index = current_chunk_start_index + len(current_chunk) 
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