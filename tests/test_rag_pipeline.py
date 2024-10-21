import unittest
from unittest.mock import Mock, patch

from srag.rag_pipeline import RAGPipeline
from srag.chunk_node import Node
from srag.doc_loader import DocumentLoader


class TestRAGPipeline(unittest.TestCase):
    """Comprehensive tests for the RAGPipeline class."""

    def setUp(self):
        """Sets up mocks for all the components of the RAGPipeline."""
        self.mock_file_loader = Mock(spec=DocumentLoader)
        self.mock_chunk_splitter = Mock()
        self.mock_embeddings = Mock()
        self.mock_vector_store = Mock()
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()

        self.rag_pipeline = RAGPipeline(
            file_loader=self.mock_file_loader,
            chunk_splitter=self.mock_chunk_splitter,
            embeddings=self.mock_embeddings,
            vector_store=self.mock_vector_store,
            llm=self.mock_llm,
            prompt_manager=self.mock_prompt_manager,
            default_prompt_template="rag_simple",
        )

    def test_initialization(self):
        """Tests initialization of the RAGPipeline."""
        self.assertIsInstance(self.rag_pipeline, RAGPipeline)
        self.assertEqual(self.rag_pipeline.default_prompt_template, "rag_simple")

    def test_load_and_index(self):
        """Tests the load_and_index method."""
        mock_documents = [
            Document("doc1", {"page_label": "1"}, "Document 1 content"),
            Document("doc2", {"page_label": "1"}, "Document 2 content"),
        ]
        mock_chunks = [
            Node("Chunk 1", {"document_id": "doc1", "start_index": 0, "end_index": 10}),
            Node("Chunk 2", {"document_id": "doc2", "start_index": 0, "end_index": 10}),
        ]

        self.mock_file_loader.load.return_value = mock_documents
        self.mock_chunk_splitter.split.return_value = mock_chunks

        self.rag_pipeline.load_and_index("test_directory")

        self.mock_file_loader.load.assert_called_once_with(
            directory_path="test_directory", recursive=False, ext=None, exc=None, filenames=None, preprocess_fn=None, max_workers=None
        )
        self.mock_chunk_splitter.split.assert_called_once_with(mock_documents)
        self.mock_embeddings.embed_nodes.assert_called_once_with(mock_chunks)
        self.mock_vector_store.index.assert_called_once_with(mock_chunks)

    def test_run(self):
        """Tests the run method with the default prompt template."""
        mock_query = "Test query"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_search_results = [
            {"document": "Result 1", "score": 0.8},
            {"document": "Result 2", "score": 0.7},
        ]  # Simulate ChromaDB output
        mock_prompt = "Generated prompt with context"
        mock_response = "Generated response"

        self.mock_embeddings.embed.return_value = [mock_embedding]
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_prompt_manager.create_prompt.return_value = mock_prompt
        self.mock_llm.generate.return_value = mock_response

        response = self.rag_pipeline.run(mock_query)

        self.mock_embeddings.embed.assert_called_once_with([mock_query])
        self.mock_vector_store.search.assert_called_once_with(
            mock_embedding, top_k=5
        )
        self.mock_prompt_manager.create_prompt.assert_called_once_with(
            template_name="rag_simple", query=mock_query, context=mock_search_results
        )
        self.mock_llm.generate.assert_called_once_with(mock_prompt)
        self.assertEqual(response, mock_response)

    def test_run_with_custom_prompt_template(self):
        """Tests the run method with a custom prompt template."""
        mock_query = "Test query"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_search_results = [
            {"document": "Result 1", "score": 0.8},
            {"document": "Result 2", "score": 0.7},
        ]
        mock_prompt = "Generated prompt with a custom template"
        mock_response = "Generated response"

        self.mock_embeddings.embed.return_value = [mock_embedding]
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_prompt_manager.create_prompt.return_value = mock_prompt
        self.mock_llm.generate.return_value = mock_response

        response = self.rag_pipeline.run(
            mock_query, prompt_template="custom_template"
        )

        self.mock_prompt_manager.create_prompt.assert_called_once_with(
            template_name="custom_template",
            query=mock_query,
            context=mock_search_results,
        )
        self.assertEqual(response, mock_response)

    def test_run_with_empty_context(self):
        """Tests the run method when no relevant context is found."""
        mock_query = "Test query"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_search_results = []  # Empty context
        mock_prompt = "Generated prompt without context"
        mock_response = "Generated response"

        self.mock_embeddings.embed.return_value = [mock_embedding]
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_prompt_manager.create_prompt.return_value = mock_prompt
        self.mock_llm.generate.return_value = mock_response

        response = self.rag_pipeline.run(mock_query)

        self.mock_prompt_manager.create_prompt.assert_called_once_with(
            template_name="rag_simple", query=mock_query, context=mock_search_results
        )
        self.assertEqual(response, mock_response)

    def test_run_with_failed_prompt_creation(self):
        """Tests the run method when prompt creation fails."""
        mock_query = "Test query"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_search_results = [
            {"document": "Result 1"},
            {"document": "Result 2"},
        ]

        self.mock_embeddings.embed.return_value = [mock_embedding]
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_prompt_manager.create_prompt.return_value = (
            ""  # Simulate failed prompt creation
        )

        response = self.rag_pipeline.run(mock_query)

        self.assertEqual(response, "")
        self.mock_llm.generate.assert_not_called()

    def test_save_index(self):
        """Tests saving the vector store index."""
        self.rag_pipeline.save_index()
        self.mock_vector_store.save.assert_called_once()

    def test_load_index(self):
        """Tests loading the vector store index."""
        self.rag_pipeline.load_index()
        self.mock_vector_store.load.assert_called_once()


if __name__ == "__main__":
    unittest.main()