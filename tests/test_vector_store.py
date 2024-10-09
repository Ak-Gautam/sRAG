import unittest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from srag.vector_store import VectorStore
from srag.chunk_node import Node

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.faiss_patcher = patch('srag.vector_store.faiss')
        self.mock_faiss = self.faiss_patcher.start()
        self.chroma_patcher = patch('srag.vector_store.chromadb')
        self.mock_chroma = self.chroma_patcher.start()

    def tearDown(self):
        self.faiss_patcher.stop()
        self.chroma_patcher.stop()

    def test_initialization_faiss(self):
        vs = VectorStore(vector_store_type="faiss")
        self.assertEqual(vs.vector_store_type, "faiss")
        self.assertIsNone(vs.index)

    def test_initialization_chroma(self):
        vs = VectorStore(vector_store_type="chroma")
        self.assertEqual(vs.vector_store_type, "chroma")
        self.assertIsNotNone(vs.client)
        self.assertIsNotNone(vs.collection)

    def test_initialization_invalid(self):
        with self.assertRaises(ValueError):
            VectorStore(vector_store_type="invalid")

    def test_index_faiss(self):
        vs = VectorStore(vector_store_type="faiss")
        chunks = [Node("text1", embedding=np.array([0.1, 0.2])), Node("text2", embedding=np.array([0.3, 0.4]))]
        vs.index(chunks)
        self.mock_faiss.IndexFlatL2.assert_called_once()
        self.mock_faiss.write_index.assert_called_once()

    def test_index_chroma(self):
        vs = VectorStore(vector_store_type="chroma")
        chunks = [Node("text1", embedding=np.array([0.1, 0.2])), Node("text2", embedding=np.array([0.3, 0.4]))]
        vs.index(chunks)
        vs.collection.add.assert_called_once()

    def test_search_faiss(self):
        vs = VectorStore(vector_store_type="faiss")
        vs.index = MagicMock()
        vs.index.search.return_value = (None, np.array([[0, 1]]))
        query_embedding = np.array([0.1, 0.2])
        results = vs.search(query_embedding)
        self.assertEqual(len(results), 2)
        self.assertIn('faiss_index', results[0])

    def test_search_chroma(self):
        vs = VectorStore(vector_store_type="chroma")
        vs.collection.query.return_value = {
            'ids': [['1', '2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]],
            'documents': [['doc1', 'doc2']]
        }
        query_embedding = np.array([0.1, 0.2])
        results = vs.search(query_embedding)
        self.assertEqual(len(results), 2)
        self.assertIn('node_id', results[0])
        self.assertIn('metadata', results[0])
        self.assertIn('score', results[0])
        self.assertIn('document', results[0])


if __name__ == '__main__':
    unittest.main()