import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from srag.embeddings import Embeddings
from srag.chunk_node import Node

class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.patcher = patch('srag.embeddings.AutoModel.from_pretrained', return_value=self.mock_model)
        self.patcher.start()
        self.tokenizer_patcher = patch('srag.embeddings.AutoTokenizer.from_pretrained', return_value=self.mock_tokenizer)
        self.tokenizer_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tokenizer_patcher.stop()

    def test_initialization(self):
        embeddings = Embeddings()
        self.assertEqual(embeddings.model_name, "nomic-ai/nomic-embed-text-v1.5")
        self.assertIn(embeddings.device, ["cuda", "cpu"])

    def test_embed(self):
        embeddings = Embeddings()
        texts = ["Hello world", "Test embedding"]
        
        # Mock the tokenizer and model outputs
        self.mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        self.mock_model.return_value = MagicMock(last_hidden_state=MagicMock())

        # Mock the mean_pooling method
        with patch.object(Embeddings, 'mean_pooling', return_value=MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([[0.1, 0.2], [0.3, 0.4]])))):
            result = embeddings.embed(texts)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))  # 2 texts, 2-dimensional embeddings


if __name__ == '__main__':
    unittest.main()