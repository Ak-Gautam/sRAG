import unittest
from srag.chunk_node import (
    Node,
    get_chunk_splitter,
    TokenChunkSplitter,
    SentenceChunkSplitterWithOverlap,
    ParagraphChunkSplitter,
)
from srag.document_loader import Document


class TestChunkNode(unittest.TestCase):
    """Comprehensive tests for the Node class and different ChunkSplitter classes."""

    def setUp(self):
        """Sets up test data (text and a Document object) for the tests."""
        self.test_text = """This is a test document. 
It has multiple sentences. 
And a few paragraphs.

This is the second paragraph with some special characters like %, $, and &."""
        self.test_document = Document("test_id", {"page_label": "1"}, self.test_text)

    def test_node_creation(self):
        """Tests the creation of a Node object with text and metadata."""
        node = Node("Test text", {"key": "value"})
        self.assertEqual(node.text, "Test text")
        self.assertEqual(node.metadata["key"], "value")

    def test_token_chunk_splitter(self):
        """Tests the TokenChunkSplitter with different chunk sizes."""
        splitter = TokenChunkSplitter(chunk_size=5)
        nodes = splitter.split_document(self.test_document)
        self.assertEqual(len(nodes), 5)
        self.assertEqual(nodes[0].text, "This is a test document.")
        # Add more assertions to check other chunks and metadata

        splitter = TokenChunkSplitter(chunk_size=10)  # Test with a different chunk size
        nodes = splitter.split_document(self.test_document)
        # Add assertions for the new chunk size

    def test_sentence_chunk_splitter_with_overlap(self):
        """Tests the SentenceChunkSplitterWithOverlap with different chunk sizes and overlaps."""
        splitter = SentenceChunkSplitterWithOverlap(chunk_size=50, overlap=10)
        nodes = splitter.split_document(self.test_document)
        self.assertEqual(len(nodes), 4)
        self.assertTrue(nodes[0].text.startswith("This is a test document."))
        # Add assertions to check other chunks and overlap

        splitter = SentenceChunkSplitterWithOverlap(
            chunk_size=70, overlap=15
        )  # Test with different parameters
        nodes = splitter.split_document(self.test_document)
        # Add assertions

    def test_paragraph_chunk_splitter(self):
        """Tests the ParagraphChunkSplitter with different chunk sizes."""
        splitter = ParagraphChunkSplitter(chunk_size=50)
        nodes = splitter.split_document(self.test_document)
        self.assertEqual(len(nodes), 2)
        self.assertTrue(nodes[0].text.startswith("This is a test document."))
        # Add assertions to check other chunks

        splitter = ParagraphChunkSplitter(chunk_size=100)  # Test with a different size
        nodes = splitter.split_document(self.test_document)
        # Add assertions

    def test_get_chunk_splitter(self):
        """Tests the get_chunk_splitter factory function."""
        splitter = get_chunk_splitter("token", chunk_size=10)
        self.assertIsInstance(splitter, TokenChunkSplitter)
        splitter = get_chunk_splitter("overlap", chunk_size=100, overlap=20)
        self.assertIsInstance(splitter, SentenceChunkSplitterWithOverlap)
        splitter = get_chunk_splitter("paragraph", chunk_size=200)
        self.assertIsInstance(splitter, ParagraphChunkSplitter)

        with self.assertRaises(ValueError):
            get_chunk_splitter("invalid_strategy")  # Test with an invalid strategy

    def test_special_characters(self):
        """Tests chunking with text containing special characters."""
        splitter = TokenChunkSplitter(chunk_size=10)
        nodes = splitter.split_document(self.test_document)
        # Add assertions to verify that special characters are handled correctly
        

    def test_empty_document(self):
        """Tests chunking with an empty document."""
        empty_document = Document("empty_id", {"page_label": "1"}, "")
        splitter = TokenChunkSplitter(chunk_size=10)
        nodes = splitter.split_document(empty_document)
        self.assertEqual(len(nodes), 0)  # Expect an empty list of nodes