#wip

# This attempt is likely not going to work but well.

from Document import Document, FileLoader
from ChunkNode import Node, SentenceChunkSplitter

from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SummaryIndex:
    """
    A simple index for summarization, storing nodes sequentially.
    """
    def __init__(self, nodes: List[Node] = None):
        """
        Initializes the SummaryIndex.

        Args:
            nodes (List[Node], optional): An initial list of nodes for the index. 
                                         Defaults to None.
        """
        self.nodes = nodes if nodes is not None else []

    def add_nodes(self, nodes: List[Node]):
        """
        Adds a list of nodes to the index.

        Args:
            nodes (List[Node]): The list of nodes to add.
        """
        self.nodes.extend(nodes)

    def query(self, query_str: str = None) -> str: 
        """
        Generates a "summary" by concatenating the text of all nodes.

        Args:
            query_str (str, optional): The query string (not used in this implementation). 
                                     Defaults to None.

        Returns:
            str: The concatenated text of all nodes as a summary.
        """
        return " ".join([node.text for node in self.nodes])

class VectorIndex:
    """
    An index that uses embeddings and cosine similarity for retrieval.
    """
    def __init__(self, nodes: List[Node] = None, embedding_model_name='all-mpnet-base-v2'):
        """
        Initializes the VectorIndex.

        Args:
            nodes (List[Node], optional): An initial list of nodes for the index.
                                          Defaults to None.
            embedding_model_name (str): The name of the sentence-transformers model 
                                       to use for embedding. Defaults to 'all-mpnet-base-v2'.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.nodes = nodes if nodes is not None else []
        self.embeddings = []
        if nodes:  # Generate embeddings if nodes are provided during initialization
            self.embeddings = self.generate_embeddings(nodes)

    def add_nodes(self, nodes: List[Node]):
        """
        Adds nodes to the index and updates embeddings.

        Args:
            nodes (List[Node]): The list of nodes to add.
        """
        self.nodes.extend(nodes)
        self.embeddings.extend(self.generate_embeddings(nodes)) 

    def generate_embeddings(self, nodes: List[Node]) -> List[List[float]]:
        """
        Generates embeddings for the given nodes using the embedding model.

        Args:
            nodes (List[Node]): The list of nodes.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each node.
        """
        return self.embedding_model.encode([node.text for node in nodes])

    def query(self, query_str: str, top_k: int = 3) -> List[Node]:
        """
        Performs a query against the index to retrieve the most similar nodes.

        Args:
            query_str (str): The query string.
            top_k (int, optional): The number of top matching nodes to return. 
                                    Defaults to 3.

        Returns:
            List[Node]: A list of the top_k matching nodes.
        """
        query_embedding = self.embedding_model.encode(query_str)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.nodes[i] for i in top_indices]