# test_chunk_node.py
import unittest
from zrag.chunk_node import get_chunk_splitter, Node
from zrag.doc_loader import Document


class TestChunkNode(unittest.TestCase):
    def test_get_chunk_splitter(self):
        splitter = get_chunk_splitter("token", chunk_size=100)
        self.assertIsNotNone(splitter)
        splitter = get_chunk_splitter("overlap", chunk_size=100, overlap=10)
        self.assertIsNotNone(splitter)
        splitter = get_chunk_splitter("paragraph", chunk_size=100)
        self.assertIsNotNone(splitter)

    def test_token_chunk_splitter(self):
        splitter = get_chunk_splitter("token", chunk_size=10)
        doc = Document("1", {"page_label": "1"}, "This is a test document.")
        chunks = splitter.split_document(doc)
        self.assertEqual(len(chunks), 2)

    def test_overlap_chunk_splitter(self):
        splitter = get_chunk_splitter("overlap", chunk_size=20, overlap=5)
        doc = Document("1", {"page_label": "1"}, "This is a test. Another test.")
        chunks = splitter.split_document(doc)
        self.assertEqual(len(chunks), 2)

    def test_paragraph_chunk_splitter(self):
        splitter = get_chunk_splitter("paragraph", chunk_size=20)
        doc = Document("1", {"page_label": "1"}, "This is a test.\n\nAnother test.")
        chunks = splitter.split_document(doc)
        self.assertEqual(len(chunks), 2)

    def test_node_creation(self):
        node = Node("Test text", {"metadata": "value"})
        self.assertEqual(node.text, "Test text")
        self.assertEqual(node.metadata, {"metadata": "value"})