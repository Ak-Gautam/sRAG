import unittest
from unittest.mock import MagicMock

import numpy as np

from zrag.models import Document, Node, RAGConfig, RetrievalResult, StageTiming
from zrag.prompt_manager import PromptManager
from zrag.vector_store import VectorStore
from zrag.rag_pipeline import IngestionReport, PipelineTrace, RAGPipeline


class StubDocumentLoader:
    def load(self, **_: object):
        return [Document(document_id="doc-1", text="Sample text for testing", metadata={"page_label": "1"})]


class StubChunkSplitter:
    def split(self, documents):
        return [Node(text=doc.text, metadata={"document_id": doc.document_id}) for doc in documents]


class StubEmbeddings:
    def embed(self, texts, **_: object):
        return np.array([[0.1, 0.2] for _ in texts], dtype="float32")

    def embed_nodes(self, nodes, **_: object):
        for node in nodes:
            node.embedding = np.array([0.1, 0.2], dtype="float32")
        return nodes


class RAGPipelineTests(unittest.TestCase):
    def setUp(self):
        self.loader = StubDocumentLoader()
        self.splitter = StubChunkSplitter()
        self.embeddings = StubEmbeddings()
        self.vector_store = MagicMock(spec=VectorStore)
        self.vector_store.add_documents.return_value = None
        self.vector_store.similarity_search.return_value = [
            RetrievalResult(
                node_id="node-1",
                score=0.9,
                distance=0.1,
                metadata={"document_id": "doc-1"},
                text="Sample chunk text",
            )
        ]
        self.llm = MagicMock()
        self.llm.generate.return_value = "Generated answer"
        self.prompt_manager = PromptManager(default_template="rag_simple")
        self.pipeline = RAGPipeline(
            file_loader=self.loader,
            chunk_splitter=self.splitter,
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            llm=self.llm,
            prompt_manager=self.prompt_manager,
            config=RAGConfig(max_retrieval_docs=2),
        )

    def test_load_and_index_returns_report(self):
        report = self.pipeline.load_and_index("/tmp", capture_report=True)
        self.assertIsInstance(report, IngestionReport)
        self.assertEqual(report.documents_indexed, 1)
        self.assertEqual(report.chunks_indexed, 1)
        stage_names = [timing.name for timing in report.timings]
        self.assertListEqual(stage_names, ["load_documents", "chunk_documents", "embed_chunks", "index_chunks"])
        self.assertTrue(all(isinstance(timing, StageTiming) for timing in report.timings))

    def test_run_with_trace_returns_pipeline_trace(self):
        trace = self.pipeline.run_with_trace("What is the sample text?", top_k=2)
        self.vector_store.similarity_search.assert_called_once()
        self.assertIsInstance(trace, PipelineTrace)
        self.assertEqual(trace.response, "Generated answer")
        self.assertEqual(len(trace.results), 1)
        self.assertTrue(all(isinstance(timing, StageTiming) for timing in trace.timings))

    def test_retrieve_validates_query(self):
        with self.assertRaises(ValueError):
            self.pipeline.retrieve("")


if __name__ == "__main__":
    unittest.main()
