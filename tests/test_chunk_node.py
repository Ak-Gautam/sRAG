import unittest
from unittest.mock import patch, MagicMock

from srag.chunk_node import (
    Node,
    ChunkSplitter,
    TokenChunkSplitter,
    SentenceChunkSplitterWithOverlap,
    ParagraphChunkSplitter,
    get_chunk_splitter,
)
from srag.doc_loader import Document


class TestNode(unittest.TestCase):
    """Tests for the Node class."""

    def test_node_creation(self):
        """Tests the creation of a Node object."""
        node = Node("Test text", {"metadata": "value"})
        self.assertEqual(node.text, "Test text")
        self.assertEqual(node.metadata, {"metadata": "value"})

    def test_node_repr(self):
        """Tests the string representation (__repr__) of a Node."""
        node = Node("This is a long text that should be truncated in repr", {})
        self.assertTrue(node.__repr__().startswith("Node(text='This is a long text"))
        self.assertTrue(node.__repr__().endswith("...', metadata={})"))


class TestTokenChunkSplitter(unittest.TestCase):
    """Tests for the TokenChunkSplitter class."""

    @patch("srag.chunk_node.spacy.load")
    def setUp(self, mock_spacy_load):
        """Sets up a mock spaCy NLP pipeline for testing."""
        self.mock_nlp = MagicMock()
        mock_spacy_load.return_value = self.mock_nlp
        self.splitter = TokenChunkSplitter(chunk_size=5)

    def test_split_document(self):
        """Tests splitting a document into token chunks."""
        doc = Document(
            "1", {"page_label": "1"}, "This is a test document with more than five tokens."
        )
        self.mock_nlp.return_value = [MagicMock(text=word) for word in doc.text.split()]

        nodes = self.splitter.split_document(doc)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].text, "This is a test document")
        self.assertEqual(nodes[0].metadata["start_index"], 0)
        self.assertEqual(nodes[0].metadata["end_index"], 24)
        self.assertEqual(nodes[1].text, "with more than five tokens.")
        self.assertEqual(nodes[1].metadata["start_index"], 25)
        self.assertEqual(nodes[1].metadata["end_index"], 54)

    def test_empty_document(self):
        """Tests splitting an empty document."""
        doc = Document("1", {"page_label": "1"}, "")
        nodes = self.splitter.split_document(doc)
        self.assertEqual(len(nodes), 0)  # Expect an empty list of nodes


class TestSentenceChunkSplitterWithOverlap(unittest.TestCase):
    """Tests for the SentenceChunkSplitterWithOverlap class."""

    @patch("srag.chunk_node.nltk.download")
    @patch("srag.chunk_node.nltk.tokenize.PunktSentenceTokenizer")
    def setUp(self, mock_punkt, mock_download):
        """Sets up a mock sentence tokenizer for testing."""
        self.mock_tokenizer = mock_punkt.return_value
        self.splitter = SentenceChunkSplitterWithOverlap(chunk_size=20, overlap=5)

    def test_split_document(self):
        """Tests splitting a document into sentence chunks with overlap."""
        doc = Document(
            "1",
            {"page_label": "1"},
            "Short sentence. Another short one. This one is longer.",
        )
        self.mock_tokenizer.tokenize.return_value = [
            "Short sentence.",
            "Another short one.",
            "This one is longer.",
        ]

        nodes = self.splitter.split_document(doc)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].text, "Short sentence. Another short one.")
        self.assertEqual(nodes[0].metadata["start_index"], 0)
        self.assertEqual(nodes[0].metadata["end_index"], 36)
        self.assertTrue(nodes[1].text.startswith("short one. This one"))
        self.assertEqual(nodes[1].metadata["start_index"], 21)
        self.assertEqual(nodes[1].metadata["end_index"], 57)

    def test_empty_document(self):
        """Tests splitting an empty document."""
        doc = Document("1", {"page_label": "1"}, "")
        nodes = self.splitter.split_document(doc)
        self.assertEqual(len(nodes), 0)  # Expect an empty list of nodes


class TestParagraphChunkSplitter(unittest.TestCase):
    """Tests for the ParagraphChunkSplitter class."""

    def setUp(self):
        """Sets up a ParagraphChunkSplitter instance for testing."""
        self.splitter = ParagraphChunkSplitter(chunk_size=50)

    def test_split_document(self):
        """Tests splitting a document into paragraph chunks."""
        doc = Document(
            "1",
            {"page_label": "1"},
            "First paragraph.\n\nSecond paragraph.\n\nThird very long paragraph that exceeds the chunk size.",
        )

        nodes = self.splitter.split_document(doc)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].text, "First paragraph.\n\nSecond paragraph.")
        self.assertEqual(nodes[0].metadata["start_index"], 0)
        self.assertEqual(nodes[0].metadata["end_index"], 33)
        self.assertEqual(nodes[1].text, "Third very long paragraph that exceeds the chunk size.")
        self.assertEqual(nodes[1].metadata["start_index"], 35)
        self.assertEqual(nodes[1].metadata["end_index"], 88)

    def test_empty_document(self):
        """Tests splitting an empty document."""
        doc = Document("1", {"page_label": "1"}, "")
        nodes = self.splitter.split_document(doc)
        self.assertEqual(len(nodes), 0)  # Expect an empty list of nodes


class TestGetChunkSplitter(unittest.TestCase):
    """Tests for the get_chunk_splitter factory function."""

    def test_get_token_splitter(self):
        """Tests getting a TokenChunkSplitter instance."""
        splitter = get_chunk_splitter("token", chunk_size=100)
        self.assertIsInstance(splitter, TokenChunkSplitter)

    def test_get_overlap_splitter(self):
        """Tests getting a SentenceChunkSplitterWithOverlap instance."""
        splitter = get_chunk_splitter("overlap", chunk_size=100, overlap=10)
        self.assertIsInstance(splitter, SentenceChunkSplitterWithOverlap)

    def test_get_paragraph_splitter(self):
        """Tests getting a ParagraphChunkSplitter instance."""
        splitter = get_chunk_splitter("paragraph", chunk_size=100)
        self.assertIsInstance(splitter, ParagraphChunkSplitter)

    def test_invalid_strategy(self):
        """Tests that an invalid strategy raises a ValueError."""
        with self.assertRaises(ValueError):
            get_chunk_splitter("invalid_strategy")


class TestChunkSplitter(unittest.TestCase):
    """Tests for the base ChunkSplitter class."""

    def test_split_method(self):
        """Tests the split method of the base ChunkSplitter."""

        class TestSplitter(ChunkSplitter):
            def split_document(self, document):
                return [Node(document.text, document.metadata)]

        splitter = TestSplitter()
        docs = [
            Document("1", {"page": "1"}, "Doc 1"),
            Document("2", {"page": "2"}, "Doc 2"),
        ]
        nodes = splitter.split(docs)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].text, "Doc 1")
        self.assertEqual(nodes[1].text, "Doc 2")


if __name__ == "__main__":
    unittest.main()