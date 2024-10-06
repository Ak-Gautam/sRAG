import logging
from typing import List, Dict, Any, Optional, Union, Generator
from document_loader import DocumentLoader
from chunk_node import get_chunk_splitter, ChunkSplitter, Node 
from embeddings import Embeddings
from vector_store import VectorStore
from llm import LLM
from prompt_manager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline: 
    """
    Implements a Retrieval Augmented Generation (RAG) pipeline.
    """
    def __init__(
        self,
        file_loader: DocumentLoader,
        chunk_splitter: ChunkSplitter,
        embeddings: Embeddings,
        vector_store: VectorStore,
        llm: LLM,
        prompt_manager: PromptManager,
        default_prompt_template: str = "rag_simple",
    ):
        """
        Initializes the RAGPipeline with necessary components.
        """
        self.file_loader = file_loader
        self.chunk_splitter = chunk_splitter
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.default_prompt_template = default_prompt_template

    def load_and_index(
        self,
        directory_path: str, 
        recursive: bool = False,
        ext: Optional[List[str]] = None,
        exc: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        preprocess_fn: Optional[Any] = None,
        max_workers: int = None,
    ):
        """Loads, chunks, embeds, and indexes documents from a directory."""
        documents = self.file_loader.load(
            recursive=recursive, 
            ext=ext, 
            exc=exc, 
            filenames=filenames, 
            max_workers=max_workers, 
            preprocess_fn=preprocess_fn
        )
        chunks = self.chunk_splitter.split(documents)
        self.embeddings.embed_nodes(chunks)
        self.vector_store.index(chunks)
        logger.info(f"Loaded, chunked, embedded, and indexed {len(chunks)} chunks from {directory_path}")

    def run(
        self,
        query: str,
        top_k: int = 5,
        prompt_template: Optional[str] = None,
        **llm_kwargs 
    ) -> Union[str, Generator]: 

        """
        Processes a user query and generates a response using the RAG pipeline.
        """
        query_embedding = self.embeddings.embed([query])[0]  
        results = self.vector_store.search(query_embedding, top_k=top_k)  

        # Use prompt_template if provided, otherwise use default
        if not prompt_template:
            prompt_template = self.default_prompt_template

        # Create prompt 
        prompt = self.prompt_manager.create_prompt(
            template_name=prompt_template, query=query, context=results
        )

        if not prompt: 
            logger.error("Prompt creation failed. Returning empty string.")
            return ""  

        # Generate response 
        response = self.llm.generate(prompt, **llm_kwargs)

        return response

    def save_index(self):
        """Saves the vector store index."""
        self.vector_store.save()

    def load_index(self):
        """Loads the vector store index."""
        self.vector_store.load()