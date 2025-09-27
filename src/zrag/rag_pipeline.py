from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Generator, List, Optional, Union

from zrag.doc_loader import DocumentLoader

from .chunk_node import ChunkSplitter, get_chunk_splitter
from .embeddings import Embeddings
from .exceptions import RAGPipelineError
from .llm import LLM
from .models import RAGConfig, RetrievalResult, StageTiming
from .prompt_manager import PromptManager
from .vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionReport:
    """Summary of a corpus ingestion run."""

    documents_indexed: int
    chunks_indexed: int
    timings: List[StageTiming]


@dataclass(slots=True)
class PipelineTrace:
    """Represents the detailed output of a RAG query execution."""

    query: str
    response: Union[str, List[str], Generator]
    results: List[RetrievalResult]
    timings: List[StageTiming]


class RAGPipeline:
    """Sync-first orchestration coordinating ingestion, retrieval, and generation."""

    def __init__(
        self,
        file_loader: DocumentLoader,
        chunk_splitter: ChunkSplitter,
        embeddings: Embeddings,
        vector_store: VectorStore,
        llm: LLM,
        prompt_manager: PromptManager,
        default_prompt_template: str = "rag_simple",
        *,
        config: Optional[RAGConfig] = None,
        max_retrieval_docs: Optional[int] = None,
    ) -> None:
        self.file_loader = file_loader
        self.chunk_splitter = chunk_splitter
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.default_prompt_template = default_prompt_template
        self.config = config or RAGConfig()
        self.max_retrieval_docs = max_retrieval_docs or self.config.max_retrieval_docs
        self._last_ingestion_report: Optional[IngestionReport] = None

    @classmethod
    def from_config(
        cls,
        document_directory: str,
        llm: LLM,
        config: Optional[RAGConfig] = None,
        *,
        loader_kwargs: Optional[Dict[str, Any]] = None,
        embeddings_kwargs: Optional[Dict[str, Any]] = None,
        vector_store_kwargs: Optional[Dict[str, Any]] = None,
        prompt_manager: Optional[PromptManager] = None,
    ) -> "RAGPipeline":
        """Convenience constructor that wires components from a configuration."""

        cfg = config or RAGConfig()
        loader = DocumentLoader(document_directory, **(loader_kwargs or {}))
        splitter = get_chunk_splitter(cfg.chunker)
        embeddings = Embeddings(config=cfg.embeddings, **(embeddings_kwargs or {}))
        vector_store = VectorStore(config=cfg.vector_store, **(vector_store_kwargs or {}))
        prompt_mgr = prompt_manager or PromptManager(default_template="rag_simple")

        return cls(
            file_loader=loader,
            chunk_splitter=splitter,
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            prompt_manager=prompt_mgr,
            default_prompt_template=prompt_mgr.default_template_name or "rag_simple",
            config=cfg,
            max_retrieval_docs=cfg.max_retrieval_docs,
        )

    def load_and_index(
        self,
        directory_path: str,
        recursive: bool = False,
        ext: Optional[List[str]] = None,
        exc: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        preprocess_fn: Optional[Any] = None,
        max_workers: Optional[int] = None,
        *,
        capture_report: bool = False,
    ) -> Optional[IngestionReport]:
        """Load files, create chunks, generate embeddings, and index them."""

        timings: List[StageTiming] = []

        start = perf_counter()
        documents = self.file_loader.load(
            recursive=recursive,
            ext=ext,
            exc=exc,
            filenames=filenames,
            max_workers=max_workers,
            preprocess_fn=preprocess_fn,
        )
        timings.append(StageTiming("load_documents", (perf_counter() - start) * 1000))

        start = perf_counter()
        chunks = self.chunk_splitter.split(documents)
        timings.append(StageTiming("chunk_documents", (perf_counter() - start) * 1000))

        if not chunks:
            logger.warning("No chunks generated from %s; skipping embedding/indexing", directory_path)
            report = IngestionReport(len(documents), 0, timings)
            self._last_ingestion_report = report
            return report if capture_report else None

        start = perf_counter()
        self.embeddings.embed_nodes(chunks)
        timings.append(StageTiming("embed_chunks", (perf_counter() - start) * 1000))

        start = perf_counter()
        self.vector_store.add_documents(chunks)
        timings.append(StageTiming("index_chunks", (perf_counter() - start) * 1000))

        report = IngestionReport(len(documents), len(chunks), timings)
        self._last_ingestion_report = report

        logger.info(
            "Ingested %d documents -> %d chunks into %s",
            len(documents),
            len(chunks),
            directory_path,
        )

        return report if capture_report else None

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Embed a query and return similarity search results."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_embedding = self.embeddings.embed([query.strip()])[0]
        return self.vector_store.similarity_search(query_embedding, k=top_k or self.max_retrieval_docs)

    def run_with_trace(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        prompt_template: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> PipelineTrace:
        """Execute the pipeline while capturing retrieval results and timings."""

        timings: List[StageTiming] = []

        start = perf_counter()
        query_embedding = self.embeddings.embed([query])[0]
        timings.append(StageTiming("embed_query", (perf_counter() - start) * 1000))

        start = perf_counter()
        results = self.vector_store.similarity_search(query_embedding, k=top_k or self.max_retrieval_docs)
        timings.append(StageTiming("retrieve", (perf_counter() - start) * 1000))

        template_name = prompt_template or self.default_prompt_template
        prompt = self.prompt_manager.create_prompt(
            template_name=template_name,
            query=query,
            context=results,
        )

        if not prompt:
            raise RAGPipelineError("Prompt creation failed", details=template_name)

        start = perf_counter()
        response = self.llm.generate(prompt, **llm_kwargs)
        timings.append(StageTiming("generate", (perf_counter() - start) * 1000))

        return PipelineTrace(query=query, response=response, results=results, timings=timings)

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        prompt_template: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> Union[str, Generator]:
        """Processes a user query and returns the generated response."""

        trace = self.run_with_trace(
            query=query,
            top_k=top_k,
            prompt_template=prompt_template,
            **llm_kwargs,
        )
        return trace.response

    def save_index(self) -> None:
        """Persist the vector store state to disk."""

        self.vector_store.save()

    def load_index(self) -> bool:
        """Load the vector store state from disk."""

        return self.vector_store.load()

    @property
    def last_ingestion_report(self) -> Optional[IngestionReport]:
        """Return the summary of the most recent ingestion run, if any."""

        return self._last_ingestion_report