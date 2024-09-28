# EmbeddingsIndex.py

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import chromadb
from chromadb.config import Settings
from ChunkNode import Node

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingIndexer:
    """
    A class that handles embedding generation and indexing using FAISS and ChromaDB.

    Attributes:
        model_name (str): The name of the Hugging Face model to use for embeddings.
        device (str): The device to run the model on ('cpu' or 'cuda').
        tokenizer (AutoTokenizer): Tokenizer associated with the model.
        model (AutoModel): Pre-trained embedding model.
        index_faiss (faiss.Index): FAISS index for similarity search.
        use_faiss (bool): Flag to determine whether to use FAISS.
        use_chroma (bool): Flag to determine whether to use ChromaDB.
        index_path_faiss (str): Path to save/load the FAISS index.
        index_path_chroma (str): Path to save/load the ChromaDB index.
        batch_size (int): Number of nodes to process in each batch for embeddings.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: Optional[str] = None,
        use_faiss: bool = True,
        use_chroma: bool = True,
        index_path_faiss: str = "faiss_index.bin",
        index_path_chroma: str = "chromadb_index",
        batch_size: int = 4,
        embedding_dim: Optional[int] = None,
    ):
        """
        Initializes the EmbeddingIndexer with the specified parameters.

        Args:
            model_name (str, optional): Hugging Face model name for embeddings. Defaults to "Alibaba-NLP/gte-base-en-v1.5".
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cuda' if available.
            use_faiss (bool, optional): Whether to use FAISS for indexing. Defaults to True.
            use_chroma (bool, optional): Whether to use ChromaDB for indexing. Defaults to True.
            index_path_faiss (str, optional): File path to save/load FAISS index. Defaults to "faiss_index.bin".
            index_path_chroma (str, optional): Directory path to save/load ChromaDB index. Defaults to "chromadb_index".
            batch_size (int, optional): Batch size for embedding computation. Defaults to 32.
            embedding_dim (int, optional): Dimension of the embeddings. If None, inferred from the model.
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_faiss = use_faiss
        self.use_chroma = use_chroma
        self.index_path_faiss = index_path_faiss
        self.index_path_chroma = index_path_chroma
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # Initialize ChromaDB
        if self.use_chroma:
            self.chroma_client = chromadb.PersistentClient(path=self.index_path_chroma)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="embeddings",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Initialized ChromaDB client at: {self.index_path_chroma}")

        logger.info(f"Initializing EmbeddingIndexer with model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Determine embedding dimension if not provided
        if not self.embedding_dim:
            with torch.no_grad():
                sample_input = self.tokenizer("Sample text", return_tensors="pt").to(self.device)
                sample_output = self.model(**sample_input)
                self.embedding_dim = sample_output.last_hidden_state.size(-1)
                logger.info(f"Embedding dimension set to: {self.embedding_dim}")

        # Initialize FAISS index
        if self.use_faiss:
            self.index_faiss = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Initialized FAISS IndexFlatL2 for L2 similarity.")

    def embed_nodes(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generates embeddings for a list of nodes.

        Args:
            nodes (List[Dict[str, Any]]): A list of node dictionaries with 'text' and 'metadata'.

        Returns:
            np.ndarray: A NumPy array of embeddings.
        """
        logger.info(f"Generating embeddings for {len(nodes)} nodes.")
        embeddings = []
        texts = [node['text'] for node in nodes]

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                model_output = self.model(**encoded_input)
                # Mean pooling
                embeddings_batch = self.mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings.append(embeddings_batch.cpu().numpy())

        embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    @staticmethod
    def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mean pooling to the model outputs.

        Args:
            model_output (Any): The output from the embedding model.
            attention_mask (torch.Tensor): The attention mask from the tokenizer.

        Returns:
            torch.Tensor: Mean pooled embeddings.
        """
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Adds embeddings to the FAISS index.

        Args:
            embeddings (np.ndarray): A NumPy array of embeddings to add.
        """
        if not self.use_faiss:
            logger.warning("FAISS index is not enabled.")
            return
        logger.info(f"Adding {embeddings.shape[0]} embeddings to FAISS index.")
        self.index_faiss.add(embeddings)
        logger.info("FAISS index updated.")

    def save_faiss_index(self):
        """
        Saves the FAISS index to disk.
        """
        if not self.use_faiss:
            logger.warning("FAISS index is not enabled.")
            return
        faiss.write_index(self.index_faiss, self.index_path_faiss)
        logger.info(f"FAISS index saved to {self.index_path_faiss}.")

    def load_faiss_index(self):
        """
        Loads the FAISS index from disk.
        """
        if not self.use_faiss:
            logger.warning("FAISS index is not enabled.")
            return
        if os.path.exists(self.index_path_faiss):
            self.index_faiss = faiss.read_index(self.index_path_faiss)
            logger.info(f"FAISS index loaded from {self.index_path_faiss}.")
        else:
            logger.warning(f"FAISS index file {self.index_path_faiss} does not exist. Initializing a new index.")

    def search_faiss(self, query_embedding: np.ndarray, top_k: int = 5) -> List[int]:
        """
        Searches the FAISS index for the top_k nearest neighbors to the query embedding.

        Args:
            query_embedding (np.ndarray): A single query embedding.
            top_k (int, optional): Number of top similar vectors to retrieve. Defaults to 5.

        Returns:
            List[int]: List of indices of the top_k nearest neighbors.
        """
        if not self.use_faiss:
            logger.warning("FAISS index is not enabled.")
            return []
        logger.info("Starting FAISS search.")
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index_faiss.search(query_embedding, top_k)
        logger.info(f"FAISS search completed. Top-{top_k} indices: {indices[0]}")
        return indices[0].tolist()

    def build_chromadb_index(self, nodes: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds embeddings and metadata to the ChromaDB index.

        Args:
            nodes (List[Dict[str, Any]]): A list of node dictionaries with 'node_id' and 'metadata'.
            embeddings (np.ndarray): A NumPy array of embeddings corresponding to the nodes.
        """
        if not self.use_chroma:
            logger.warning("ChromaDB index is not enabled.")
            return
        logger.info(f"Adding {len(nodes)} nodes to ChromaDB index.")
        
        ids = [node['metadata'].get('node_id', str(idx)) for idx, node in enumerate(nodes)]
        documents = [node['text'] for node in nodes]
        metadatas = [node['metadata'] for node in nodes]
        
        self.chroma_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        logger.info("ChromaDB index updated.")

    def save_chromadb_index(self):
        """
        Saves the ChromaDB index to disk.
        """
        if not self.use_chroma:
            logger.warning("ChromaDB index is not enabled.")
            return
        # ChromaDB automatically persists data, so no explicit save is needed
        logger.info(f"ChromaDB index is already saved at {self.index_path_chroma}.")

    def load_chromadb_index(self):
        """
        Loads the ChromaDB index from disk.
        """
        if not self.use_chroma:
            logger.warning("ChromaDB index is not enabled.")
            return
        # ChromaDB automatically loads the existing index, so no explicit load is needed
        logger.info(f"ChromaDB index is already loaded from {self.index_path_chroma}.")

    def search_chromadb(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the ChromaDB index for the top_k nearest neighbors to the query embedding.

        Args:
            query_embedding (np.ndarray): A single query embedding.
            top_k (int, optional): Number of top similar vectors to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of metadata for the top_k nearest neighbors.
        """
        if not self.use_chroma:
            logger.warning("ChromaDB index is not enabled.")
            return []
        logger.info("Starting ChromaDB search.")

        results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
        
        # Process and return results
        return [
            {
                "node_id": result_id,
                "metadata": metadata,
                "score": 1 - distance,  # Convert distance to similarity score
                "document": document
            }
            for result_id, metadata, distance, document in zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0],
                results['documents'][0]
            )
        ]

    def index_nodes(self, nodes: List[Node]):
        """
        Processes nodes by generating embeddings and indexing them.

        Args:
            nodes (List[Dict[str, Any]]): A list of node dictionaries with 'text' and 'metadata'.
        """
        logger.info("Starting indexing process for nodes.")
        nodes = [{"text": node.text, "metadata": node.metadata} for node in nodes]
        embeddings = self.embed_nodes(nodes)
        # Check if node IDs exist in metadata; if not, assign unique IDs
        for idx, node in enumerate(nodes):
            if 'node_id' not in node['metadata']:
                node['metadata']['node_id'] = f"node_{idx}_{os.urandom(8).hex()}"

        if self.use_faiss:
            self.build_faiss_index(embeddings)
            self.save_faiss_index()

        if self.use_chroma:
            self.build_chromadb_index(nodes, embeddings)
            self.save_chromadb_index()

        logger.info("Indexing process completed.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches both FAISS and ChromaDB indices for the top_k nearest neighbors to the query.

        Args:
            query (str): The query text.
            top_k (int, optional): Number of top similar vectors to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of metadata for the top_k nearest neighbors.
        """
        logger.info(f"Searching for query: {query}")
        # Generate embedding for the query
        query_embedding = self.embed_nodes([{'text': query, 'metadata': {}}])[0]

        results = []

        if self.use_faiss:
            faiss_indices = self.search_faiss(query_embedding, top_k)
            # FAISS does not store metadata; user needs to map indices to nodes externally
            results.extend([{'faiss_index': idx} for idx in faiss_indices])

        if self.use_chroma:
            chroma_results = self.search_chromadb(query_embedding, top_k)
            results.extend(chroma_results)

        logger.info("Search completed.")
        return results

    def close(self):
        """
        Closes any open connections or resources.
        """
        if self.use_chroma:
            self.chroma_client.persist()
            logger.info("ChromaDB index persisted.")

    def __del__(self):
        self.close()