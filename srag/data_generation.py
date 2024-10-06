import logging
import json
import csv
import os
from typing import List, Dict, Any, Optional, Union, Iterator
from document_loader import DocumentLoader
from llm import LLM
from prompt_manager import PromptManager
from chunk_node import get_chunk_splitter, ChunkSplitter, Node
from embeddings import Embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:  
    """
    Generates datasets using LLMs, knowledge sources, and example datasets.
    """
    def __init__(
        self,
        file_loader: DocumentLoader,  
        chunk_splitter: ChunkSplitter,  
        embeddings: Embeddings,  
        llm: LLM, 
        prompt_manager: PromptManager,  
        example_dataset_path: str,  
        output_format: str = "json",
        output_path: str = "generated_dataset.json",
        batch_size: int = 8,  
        default_prompt_template: str = "dataset_instruction" 
    ):
        """
        Initializes the DataGenerator with necessary components.
        """
        self.file_loader = file_loader
        self.chunk_splitter = chunk_splitter
        self.embeddings = embeddings
        self.llm = llm
        self.prompt_manager = prompt_manager

        self.example_dataset_path = example_dataset_path
        self.output_format = output_format.lower()
        self.output_path = output_path
        self.batch_size = batch_size
        self.default_prompt_template = default_prompt_template

    def load_knowledge(self, directory_path: str, **kwargs) -> List[Node]:  
        """Loads, chunks, and embeds knowledge documents."""

        documents = self.file_loader.load(directory_path=directory_path, **kwargs)  
        chunks = self.chunk_splitter.split(documents)
        self.embeddings.embed_nodes(chunks) 
        return chunks 

    def _load_example_dataset(self) -> List[Dict[str, Any]]:  
        """Loads the example dataset."""
        if not os.path.exists(self.example_dataset_path): 
            raise FileNotFoundError(f"Example dataset file not found: {self.example_dataset_path}")

        try:
            if self.example_dataset_path.endswith(".json"):
                with open(self.example_dataset_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif self.example_dataset_path.endswith(".csv"):
                with open(self.example_dataset_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            else:
                raise ValueError("Unsupported example dataset format. Use JSON or CSV.")  
        except Exception as e:  
            raise RuntimeError(f"Error loading example dataset: {e}") from e  

    def generate_dataset(
        self,
        knowledge_nodes: List[Node],
        num_entries: int = 1000,
        prompt_template: Optional[str] = None,
        **llm_kwargs
    ):
        """Generates a dataset using knowledge nodes and the example dataset format."""

        examples = self._load_example_dataset() 

        if not examples:
            raise ValueError("Example dataset is empty.")  

        # Default prompt_template handling
        if not prompt_template:
            prompt_template = self.default_prompt_template

        # Generator to produce prompts in a streaming way
        def prompt_generator(num: int, knowledge: List[Node], template_name: str) -> Iterator[str]:
            """Generate prompts from knowledge nodes in a streaming way"""
            for _ in range(num):
                knowledge_node = knowledge[_ % len(knowledge)] # reusing knowledge nodes 
                variables = {"instruction": "Generate a question and answer pair based on the following context.",
                               "input_data": knowledge_node.text}
                prompt = self.prompt_manager.create_prompt(template_name=template_name, **variables)
                if prompt:
                    yield prompt
                else:
                    logger.warning("Failed to generate prompt for a dataset entry.")
                    continue

        # Generate dataset entries using LLM (streaming)
        generated_entries = self.llm.generate(list(prompt_generator(num_entries, knowledge_nodes, prompt_template)), stream_output=True, **llm_kwargs)  

        # Function to structure generated entries
        def structure_entry(entry: str):
            parts = entry.split('\n')  # Split by newline 
            if len(parts) >= 2:
                return {"question": parts[0].strip(), "answer": " ".join(parts[1:]).strip()}  
            else:
                logger.warning("Generated entry does not conform to expected format. Skipping.")
                return None  

        # Structure generated entries 
        structured_entries = [
            structured_entry(entry) for entry in generated_entries if structured_entry(entry)
        ]

        self._save_dataset(structured_entries)  

    def _save_dataset(self, dataset: List[Dict[str, Any]]):  
        """Saves the generated dataset."""

        if not dataset:
            logger.warning("Dataset is empty. Nothing to save.")
            return

        try:
            if self.output_format == "json":
                with open(self.output_path, "w", encoding="utf-8") as f:  
                    json.dump(dataset, f, ensure_ascii=False, indent=4)
            elif self.output_format == "csv":
                fieldnames = dataset[0].keys()  
                with open(self.output_path, "w", newline="", encoding="utf-8") as f: 
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(dataset)
            else:
                raise ValueError("Unsupported output format. Use 'json' or 'csv'.")  
        except Exception as e:  
            raise RuntimeError(f"Error saving dataset: {e}") from e 