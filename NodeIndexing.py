#wip

# This attempt is likely not going to work but well.

from Document import Document, FileLoader
from ChunkNode import Node, SentenceChunkSplitter

from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SummaryIndex:
    """A simple index that summarizes and stores nodes sequentially."""

    def __init__(self, nodes: List[Node] = None):
        """
        Initializes the SummaryIndex object.

        Args:
            nodes (List[Node], optional): A list of Node objects to initialize the index with. Defaults to None.
        """
        self.nodes = nodes if nodes is not None else []

    def add_node(self, nodes: List[Node]):
        """Adds a node to the index.

        Args:
            node (Node): The list of nodes to add to the index.
        """
        self.nodes.extend(nodes)

        def query(self, query: str, top_k: int = 5) -> List[Node]:
            """Performs a simple keyword search on the index.

            Args:
                query (str): The query string.
                top_k (int, optional): The number of results to return. Defaults to 5.

            Returns:
                List[Node]: A list of Node objects that match the query.
            """
            results = []
            for node in self.nodes:
                if query.lower() in node.text.lower():
                    results.append(node)

            return results[:top_k]
        
        def get_summary(self, query: str, top_k: int = 5) -> str:
            """Generates a summary of the most relevant nodes based on a query.

            Args:
                query (str): The query string.
                top_k (int, optional): The number of nodes to include in the summary. Defaults to 5.

            Returns:
                str: A summary of the most relevant nodes.
            """
            relevant_nodes = self.query(query, top_k=top_k)
            summary = " ".join([node.text for node in relevant_nodes])
            return summary
        