#wip

# This attempt is likely not going to work but well.

from DocLoader import Document, FileLoader
from ChunkNode import Node, SentenceChunkSplitter

from pathlib import Path
from typing import List, Dict, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re


class SummaryIndex:
    """
    An index for summarization using a pre-trained LLM (Qwen/Qwen2-1.5B-Instruct).
    """

    def __init__(self, nodes: List[Node] = None, model_name: str = "Qwen/Qwen-2-1.5B-Instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the SummaryIndex.

        Args:
            nodes (List[Node], optional): An initial list of nodes. Defaults to None.
            model_name (str, optional): The name of the Hugging Face Transformers model 
                                       to use for summarization. Defaults to "Qwen/Qwen-2-1.5B-Instruct".
            device (str, optional): The device to run the model on ('cuda' or 'cpu').
                                     Defaults to 'cuda' if available, else 'cpu'.
        """
        self.nodes = nodes if nodes is not None else []
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device) 

    def add_nodes(self, nodes: List[Node]):
        """
        Adds a list of nodes to the index.

        Args:
            nodes (List[Node]): The list of nodes to add.
        """
        self.nodes.extend(nodes)

    def query(self, query_str: str = None, max_length: int = 100) -> str:
        """
        Generates a summary using the loaded LLM.

        Args:
            query_str (str, optional): The query string. The text of all nodes is 
                                       summarized, regardless of the query.
                                       Defaults to None. 
            max_length (int, optional): The maximum length of the generated summary. 
                                        Defaults to 100.

        Returns:
            str: The generated summary.
        """
        all_text = " ".join([node.text for node in self.nodes])

        # Tokenize input text and generate summary
        inputs = self.tokenizer(all_text, return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(**inputs, max_length=max_length)
        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

        return summary