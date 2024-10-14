import unittest
from unittest.mock import Mock, patch
from srag.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.mock_file_loader = Mock()
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
            default_prompt_template="rag_simple"
        )

    def test_initialization(self):
        self.assertIsInstance(self.rag_pipeline, RAGPipeline)
        self.assertEqual(self.rag_pipeline.default_prompt_template, "rag_simple")

    def test_load_and_index(self):
        mock_documents = [Mock(), Mock()]
        mock_chunks = [Mock(), Mock(), Mock()]
        
        self.mock_file_loader.load.return_value = mock_documents
        self.mock_chunk_splitter.split.return_value = mock_chunks

        self.rag_pipeline.load_and_index("test_directory")

        self.mock_file_loader.load.assert_called_once()
        self.mock_chunk_splitter.split.assert_called_once_with(mock_documents)
        self.mock_embeddings.embed_nodes.assert_called_once_with(mock_chunks)
        self.mock_vector_store.index.assert_called_once_with(mock_chunks)

    def test_run(self):
        mock_query = "Test query"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_search_results = [{"document": "Result 1"}, {"document": "Result 2"}]
        mock_prompt = "Generated prompt"
        mock_response = "Generated response"

        self.mock_embeddings.embed.return_value = [mock_embedding]
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_prompt_manager.create_prompt.return_value = mock_prompt
        self.mock_llm.generate.return_value = mock_response

        response = self.rag_pipeline.run(mock_query)

        self.mock_embeddings.embed.assert_called_once_with([mock_query])
        self.mock_vector_store.search.assert_called_once_with(mock_embedding, top_k=5)
        self.mock_prompt_manager.create_prompt.assert_called_once_with(
            template_name="rag_simple", query=mock_query, context=mock_search_results
        )
        self.mock_llm.generate.assert_called_once_with(mock_prompt)
        self.assertEqual(response, mock_response)

    def test_run_with_custom_prompt_template(self):
        mock_query = "Test query"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_search_results = [{"document": "Result 1"}, {"document": "Result 2"}]
        mock_prompt = "Generated prompt"
        mock_response = "Generated response"

        self.mock_embeddings.embed.return_value = [mock_embedding]
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_prompt_manager.create_prompt.return_value = mock_prompt
        self.mock_llm.generate.return_value = mock_response

        response = self.rag_pipeline.run(mock_query, prompt_template="custom_template")

        self.mock_prompt_manager.create_prompt.assert_called_once_with(
            template_name="custom_template", query=mock_query, context=mock_search_results
        )

    def test_run_with_failed_prompt_creation(self):
        mock_query = "Test query"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_search_results = [{"document": "Result 1"}, {"document": "Result 2"}]

        self.mock_embeddings.embed.return_value = [mock_embedding]
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_prompt_manager.create_prompt.return_value = ""  # Simulate failed prompt creation

        response = self.rag_pipeline.run(mock_query)

        self.assertEqual(response, "")
        self.mock_llm.generate.assert_not_called()

    def test_save_index(self):
        self.rag_pipeline.save_index()
        self.mock_vector_store.save.assert_called_once()

    def test_load_index(self):
        self.rag_pipeline.load_index()
        self.mock_vector_store.load.assert_called_once()

if __name__ == '__main__':
    unittest.main()