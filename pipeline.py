# pipeline.py
import logging
from typing import List, Dict, Any, Optional, Generator
from DocLoader import FileLoader
from ChunkNode import get_chunk_splitter, ChunkSplitter, Node  # Correct import
from embeddings import Embeddings
from indexing import VectorStore
from llm import LLM
from PromptManager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:  # More descriptive class name
    def __init__(
        self,
        file_loader: FileLoader,
        chunk_splitter: ChunkSplitter,
        embeddings: Embeddings,
        vector_store: VectorStore,
        llm: LLM,
        prompt_manager: PromptManager,
        default_prompt_template: str = "rag_simple",
    ):
        self.file_loader = file_loader
        self.chunk_splitter = chunk_splitter
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.default_prompt_template = default_prompt_template


    def load_and_index(
        self,
        directory_path: str,  # Moved directory_path here
        recursive: bool = False,
        ext: Optional[List[str]] = None,
        exc: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        preprocess_fn: Optional[Any] = None,
        max_workers: int = None,
    ):
        """Loads, chunks, embeds, and indexes documents from the specified directory."""
        documents = self.file_loader.load_files(recursive, ext, exc, filenames, max_workers, preprocess_fn)
        chunks = self.chunk_splitter.get_nodes_from_documents(documents)
        self.embeddings.embed_nodes(chunks)
        self.vector_store.index_chunks(chunks)
        logger.info(f"Loaded, chunked, embedded, and indexed {len(chunks)} chunks from {directory_path}")

    def run(
        self,
        query: str,
        top_k: int = 5,
        prompt_template: Optional[str] = None,
        **llm_kwargs  # Use **llm_kwargs
    ) -> Union[str, Generator]: # Changed return type to also allow for generator objects

        """
        Processes a user query and generates a response.

        Args:
            query (str): The user's query.
            top_k (int): The number of top similar chunks to retrieve.
            prompt_template (Optional[str]):  Name of the prompt template to use. If None, uses the default.

        Returns:
            Union[str, Generator]: The generated response string or a generator of strings for streaming.
                                    If 'stream_output' is True in 'llm_kwargs', returns a generator.
        """
        query_embedding = self.embeddings.embed([query])[0]  # Embed the query
        results = self.vector_store.search(query_embedding, top_k=top_k)  # Search vector store


        # Use prompt_template if provided, otherwise use default
        if not prompt_template:
            prompt_template = self.default_prompt_template


        # Create prompt (updated to use PromptManager correctly)
        prompt = self.prompt_manager.create_prompt(
            template_name=prompt_template, query=query, context=results
        )

        if not prompt: # Check if create_prompt did not return an empty string due to error
            logger.error("Prompt creation failed. Returning empty string.")
            return ""  # Return empty string on prompt failure

        # Generate response (updated to handle streaming)
        response = self.llm.generate(prompt, **llm_kwargs)

        return response



    def save_index(self):
        self.vector_store.save_index()

    def load_index(self):
        self.vector_store.load_index()