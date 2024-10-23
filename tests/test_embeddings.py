import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from zrag.embeddings import Embeddings
from zrag.chunk_node import Node


class TestEmbeddings(unittest.TestCase):
    """Comprehensive tests for the Embeddings class."""

    @patch("srag.embeddings.AutoModel.from_pretrained")
    @patch("srag.embeddings.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer, mock_model):
        """Sets up mocks for the Hugging Face model and tokenizer."""
        self.mock_model = mock_model.return_value
        self.mock_tokenizer = mock_tokenizer.return_value
        self.embeddings = Embeddings()  # Using default model name

    def test_initialization(self):
        """Tests initialization of the Embeddings class with default values."""
        self.assertEqual(self.embeddings.model_name, "nomic-ai/nomic-embed-text-v1.5")
        self.assertIn(self.embeddings.device, ["cuda", "cpu"])

    def test_embed_single_text(self):
        """Tests embedding a single text string."""
        text = "Hello world"
        self.mock_tokenizer.__call__.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        self.mock_model.__call__.return_value = MagicMock(
            last_hidden_state=MagicMock()
        )

        with patch.object(
            Embeddings,
            "mean_pooling",
            return_value=MagicMock(
                cpu=lambda: MagicMock(numpy=lambda: np.array([0.1, 0.2]))
            ),
        ):
            embedding = self.embeddings.embed([text])  # Pass as a list

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (1, 2))  # 1 text, 2-dimensional embedding

    def test_embed_multiple_texts(self):
        """Tests embedding multiple text strings."""
        texts = ["Hello world", "Test embedding"]
        self.mock_tokenizer.__call__.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        self.mock_model.__call__.return_value = MagicMock(
            last_hidden_state=MagicMock()
        )

        with patch.object(
            Embeddings,
            "mean_pooling",
            return_value=MagicMock(
                cpu=lambda: MagicMock(numpy=lambda: np.array([[0.1, 0.2], [0.3, 0.4]]))
            ),
        ):
            embeddings = self.embeddings.embed(texts)

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 2))  # 2 texts, 2-dimensional embeddings

    def test_embed_nodes(self):
        """Tests embedding a list of Node objects."""
        nodes = [Node("Hello world", {}), Node("Test embedding", {})]
        with patch.object(
            Embeddings, "embed", return_value=np.array([[0.1, 0.2], [0.3, 0.4]])
        ):
            result_nodes = self.embeddings.embed_nodes(nodes)

        self.assertEqual(len(result_nodes), 2)
        self.assertIsInstance(result_nodes[0].embedding, np.ndarray)
        self.assertEqual(result_nodes[0].embedding.shape, (2,))
        np.testing.assert_array_equal(result_nodes[0].embedding, [0.1, 0.2])
        np.testing.assert_array_equal(result_nodes[1].embedding, [0.3, 0.4])

    def test_cosine_similarity_single(self):
        """Tests cosine similarity between single embeddings."""
        embedding1 = np.array([1, 0])
        embedding2 = np.array([0, 1])
        similarity = Embeddings.cosine_similarity(embedding1, embedding2)
        self.assertAlmostEqual(similarity[0][0], 0.0)

        embedding3 = np.array([1, 1])
        similarity = Embeddings.cosine_similarity(embedding1, embedding3)
        self.assertAlmostEqual(similarity[0][0], 1.0 / np.sqrt(2))

    def test_cosine_similarity_batch(self):
        """Tests cosine similarity between batches of embeddings."""
        embeddings1 = np.array([[1, 0], [0, 1]])
        embeddings2 = np.array([[0, 1], [1, 0]])
        similarity_matrix = Embeddings.cosine_similarity(embeddings1, embeddings2)

        # Check the shape and values of the similarity matrix
        self.assertEqual(similarity_matrix.shape, (2, 2))
        # Add assertions to verify the expected similarity values

    def test_empty_text_list(self):
        """Tests embedding an empty list of texts."""
        with patch.object(
            Embeddings,
            "mean_pooling",
            return_value=MagicMock(
                cpu=lambda: MagicMock(numpy=lambda: np.array([]))
            ),
        ):
            embeddings = self.embeddings.embed([])  # Pass an empty list

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (0,))  # Expect an empty array


if __name__ == "__main__":
    unittest.main()