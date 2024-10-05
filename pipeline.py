# pipeline.py

import logging
from typing import List, Dict, Any, Optional

from DocLoader import FileLoader, Document
from ChunkNode import get_chunk_splitter, ChunkSplitter
from embeddings import Embeddings
from indexing import VectorStore
from llm import LLM
from PromptManager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) workflow.
    Handles document processing, embedding, indexing, query handling, and response generation.
    """

    def __init__(
        self,
        document_directory: str,
        chunking_strategy: str = 'overlap',
        chunking_kwargs: Optional[Dict[str, Any]] = None,
        embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        vector_store_type: str = "faiss",
        vector_store_kwargs: Optional[Dict[str, Any]] = None,
        llm_model_name: str = "gpt2",
        llm_task: str = "text-generation",
        prompt_template: str = "rag_simple",
        use_gpu: bool = True,
    ):
        """
        Initializes the Pipeline with specified configurations.

        Args:
            document_directory (str): Path to the directory containing documents.
            chunking_strategy (str): Strategy for chunking ('token', 'overlap', 'paragraph').
            chunking_kwargs (Optional[Dict[str, Any]]): Additional arguments for the chunk splitter.
            embedding_model_name (str): Hugging Face model name for embeddings.
            vector_store_type (str): Type of vector store ('faiss' or 'chroma').
            vector_store_kwargs (Optional[Dict[str, Any]]): Additional arguments for the vector store.
            llm_model_name (str): Hugging Face model name for the LLM.
            llm_task (str): Task for the LLM ('text-generation', 'translation', etc.).
            prompt_template (str): Name of the prompt template to use.
            use_gpu (bool): Whether to use GPU for the LLM.
        """
        logger.info("Initializing Pipeline.")

        # Initialize DocLoader
        self.file_loader = FileLoader(document_directory)
        logger.info(f"FileLoader initialized for directory: {document_directory}")

        # Initialize ChunkSplitter
        if chunking_kwargs is None:
            chunking_kwargs = {}
        self.chunk_splitter: ChunkSplitter = get_chunk_splitter(
            strategy=chunking_strategy, **chunking_kwargs
        )
        logger.info(f"ChunkSplitter initialized with strategy: {chunking_strategy}")

        # Initialize Embeddings
        self.embeddings = Embeddings(model_name=embedding_model_name)
        logger.info(f"Embeddings initialized with model: {embedding_model_name}")

        # Initialize VectorStore
        if vector_store_kwargs is None:
            vector_store_kwargs = {}
        self.vector_store = VectorStore(
            vector_store_type=vector_store_type, **vector_store_kwargs
        )
        logger.info(f"VectorStore initialized with type: {vector_store_type}")

        # Initialize PromptManager
        self.prompt_manager = PromptManager()
        logger.info("PromptManager initialized.")

        # Initialize LLM
        self.llm = LLM(model_name=llm_model_name, use_gpu=use_gpu, task=llm_task)
        logger.info(f"LLM initialized with model: {llm_model_name} for task: {llm_task}")

        # Set prompt template
        self.prompt_template = prompt_template
        logger.info(f"Using prompt template: {prompt_template}")

    def process_documents(
        self,
        recursive: bool = False,
        ext: Optional[List[str]] = None,
        exc: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        preprocess_fn: Optional[Any] = None,
        max_workers: int = None,
    ) -> List[Any]:
        """
        Loads, chunks, embeds, and indexes documents.

        Args:
            recursive (bool): Whether to load files recursively.
            ext (Optional[List[str]]): List of file extensions to include.
            exc (Optional[List[str]]): List of file extensions to exclude.
            filenames (Optional[List[str]]): Specific filenames to include.
            preprocess_fn (Optional[Any]): Preprocessing function to apply to text.
            max_workers (int): Number of worker processes for parallel processing.

        Returns:
            List[Any]: List of indexed nodes.
        """
        logger.info("Starting document processing pipeline.")

        # Step 1: Load Documents
        documents: List[Document] = self.file_loader.load_files(
            recursive=recursive,
            ext=ext,
            exc=exc,
            filenames=filenames,
            max_workers=max_workers,
            preprocess_fn=preprocess_fn,
        )
        logger.info(f"Loaded {len(documents)} documents.")

        if not documents:
            logger.warning("No documents loaded. Exiting processing pipeline.")
            return []

        # Step 2: Chunk Documents
        nodes = self.chunk_splitter.get_nodes_from_documents(documents)
        logger.info(f"Chunked documents into {len(nodes)} nodes.")

        if not nodes:
            logger.warning("No nodes generated from documents.")
            return []

        # Step 3: Generate Embeddings
        nodes_with_embeddings = self.embeddings.embed_nodes(nodes)
        logger.info("Generated embeddings for all nodes.")

        # Step 4: Index Embeddings
        self.vector_store.index_chunks(nodes_with_embeddings)
        logger.info("Indexed all node embeddings.")

        logger.info("Document processing pipeline completed successfully.")
        return nodes_with_embeddings

    def handle_query(
        self,
        query: str,
        top_k: int = 5,
        prompt_template: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Processes a user query and generates a response using the RAG pipeline.

        Args:
            query (str): The user's query.
            top_k (int): Number of top relevant chunks to retrieve.
            prompt_template (Optional[str]): Specific prompt template to use.
            max_length (int): Maximum length of the generated response.
            temperature (float): Sampling temperature for the LLM.
            top_p (float): Nucleus sampling probability for the LLM.

        Returns:
            str: Generated response from the LLM.
        """
        logger.info(f"Handling query: {query}")

        # Step 1: Embed the Query
        query_embedding = self.embeddings.embed([query])[0]
        logger.info("Generated embedding for the query.")

        # Step 2: Retrieve Relevant Chunks
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

        if not retrieved_chunks:
            logger.warning("No relevant chunks found for the query.")
            return "I'm sorry, I couldn't find any relevant information to answer your question."

        # Extract context texts from retrieved chunks
        context_texts = [res['document'] for res in retrieved_chunks]

        # Step 3: Construct the Prompt
        template_to_use = prompt_template if prompt_template else self.prompt_template
        prompt = self.prompt_manager.get_prompt(
            template_name=template_to_use,
            variables={
                "context": "\n".join(context_texts),
                "query": query
            }
        )

        if not prompt:
            logger.error("Failed to construct prompt. Returning default response.")
            return "I'm sorry, I couldn't process your request at the moment."

        logger.info(f"Constructed prompt using template '{template_to_use}'.")

        # Step 4: Generate the Response
        response = self.llm.generate_rag_response(
            query=query,
            context=context_texts,
            prompt_template=None if template_to_use == "rag_simple" else prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )

        logger.info("Generated response from LLM.")
        return response

    def save_index(self):
        """
        Saves the current state of the vector index to disk.
        """
        logger.info("Saving the vector index.")
        self.vector_store.save_index()

    def load_index(self):
        """
        Loads the vector index from disk.
        """
        logger.info("Loading the vector index.")
        self.vector_store.load_index()