import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from zrag.exceptions import VectorStoreError
from zrag.models import Node
from zrag.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Comprehensive tests for the VectorStore class."""

    @patch("zrag.vector_store.faiss")
    @patch("zrag.vector_store.chromadb")
    def setUp(self, mock_chroma, mock_faiss):
        """Sets up mocks for FAISS and ChromaDB."""
        self.mock_faiss = mock_faiss
        self.mock_faiss.IndexFlatL2.return_value = MagicMock()
        self.mock_faiss.write_index = MagicMock()
        self.mock_chroma = mock_chroma
        self.mock_chroma.PersistentClient.return_value = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_chroma.PersistentClient.return_value.get_or_create_collection.return_value = self.mock_collection

    def test_initialization_faiss(self):
        """Tests initialization with FAISS."""
        vs = VectorStore(vector_store_type="faiss")
        self.assertEqual(vs.vector_store_type, "faiss")
        self.assertIsNone(vs.index)

    def test_initialization_chroma(self):
        """Tests initialization with ChromaDB."""
        vs = VectorStore(vector_store_type="chroma")
        self.assertEqual(vs.vector_store_type, "chroma")
        self.assertIsNotNone(vs.client)
        self.assertIs(self.mock_collection, vs.collection)

    def test_initialization_invalid(self):
        """Tests initialization with an invalid vector store type."""
        with self.assertRaises(VectorStoreError):
            VectorStore(vector_store_type="invalid")

    def test_index_faiss(self):
        """Tests indexing with FAISS."""
        vs = VectorStore(vector_store_type="faiss")
        chunks = [
            Node("text1", embedding=np.array([0.1, 0.2])),
            Node("text2", embedding=np.array([0.3, 0.4])),
        ]
        vs.index(chunks)
        self.mock_faiss.IndexFlatL2.assert_called_once_with(2)
        self.mock_faiss.write_index.assert_called_once_with(vs.index, str(vs.index_path))
        self.assertEqual(len(vs._records), 2)

    def test_index_chroma(self):
        """Tests indexing with ChromaDB."""
        vs = VectorStore(vector_store_type="chroma")
        chunks = [
            Node("text1", metadata={"node_id": "1"}, embedding=np.array([0.1, 0.2])),
            Node("text2", metadata={"node_id": "2"}, embedding=np.array([0.3, 0.4])),
        ]
        vs.index(chunks)
        vs.collection.add.assert_called_once_with(
            ids=["1", "2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"node_id": "1"}, {"node_id": "2"}],
            documents=["text1", "text2"],
        )

    def test_search_faiss(self):
        """Tests searching with FAISS."""
        vs = VectorStore(vector_store_type="faiss")
        vs.index = MagicMock()
        vs._records = [
            {"node_id": "n1", "metadata": {"k": "v1"}, "text": "doc1"},
            {"node_id": "n2", "metadata": {"k": "v2"}, "text": "doc2"},
        ]
        vs.index.search.return_value = (
            np.array([[0.1, 0.2]], dtype="float32"),
            np.array([[0, 1]], dtype="int32"),
        )
        query_embedding = np.array([0.1, 0.2])
        results = vs.search(query_embedding)
        vs.index.search.assert_called_once_with(np.array([[0.1, 0.2]], dtype="float32"), 5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["node_id"], "n1")
        self.assertAlmostEqual(results[0]["score"], 1 / (1 + 0.1))

    def test_search_chroma(self):
        """Tests searching with ChromaDB."""
        vs = VectorStore(vector_store_type="chroma")
        vs.collection.query.return_value = {
            "ids": [["1", "2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
        }
        query_embedding = np.array([0.1, 0.2])
        results = vs.search(query_embedding)
        vs.collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]], n_results=5, include=["metadatas", "distances", "documents"]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["node_id"], "1")
        self.assertEqual(results[0]["metadata"], {"key": "value1"})
        self.assertEqual(results[0]["score"], 0.9)
        self.assertEqual(results[0]["document"], "doc1")

    def test_save_faiss(self):
        """Tests saving the FAISS index."""
        vs = VectorStore(vector_store_type="faiss")
        vs.index = MagicMock()
        vs._records = []
        vs.save()
        self.mock_faiss.write_index.assert_called_once_with(vs.index, str(vs.index_path))

    @patch("pathlib.Path.exists", return_value=True)
    def test_load_faiss(self, mock_exists):
        """Tests loading the FAISS index when the file exists."""
        vs = VectorStore(vector_store_type="faiss")
        with patch.object(vs, "_load_metadata", return_value=[{"node_id": "n1", "metadata": {}, "text": ""}]):
            vs.load()
        self.mock_faiss.read_index.assert_called_once_with(str(vs.index_path))

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_faiss_no_file(self, mock_exists):
        """Tests loading the FAISS index when the file doesn't exist."""
        vs = VectorStore(vector_store_type="faiss")
        loaded = vs.load()
        self.assertFalse(loaded)
        self.mock_faiss.read_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()