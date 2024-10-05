# DataGen.py
import logging
import json
import csv
import os
from typing import List, Dict, Any, Optional, Union, Iterator

from DocLoader import FileLoader
from llm import LLM
from PromptManager import PromptManager
from ChunkNode import get_chunk_splitter, ChunkSplitter, Node
from embeddings import Embeddings
from indexing import VectorStore  # Import VectorStore if needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:  # Class name changed for consistency
    def __init__(
        self,
        file_loader: FileLoader,  # Using dependency injection
        chunk_splitter: ChunkSplitter,  # Using dependency injection
        embeddings: Embeddings,  # Using dependency injection
        llm: LLM,  # Using dependency injection
        prompt_manager: PromptManager,  # Using dependency injection
        example_dataset_path: str,  # More descriptive name
        output_format: str = "json",
        output_path: str = "generated_dataset.json",
        batch_size: int = 8,  # Keeping batch size for other potential uses
        default_prompt_template: str = "dataset_instruction" # For Data Generation Tasks
    ):

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


    def load_knowledge(self, directory_path: str, **kwargs) -> List[Node]:  # Added type hint
        """Loads, chunks, and embeds knowledge documents."""

        documents = self.file_loader.load_files(directory_path=directory_path, **kwargs)  # Passing directory_path
        chunks = self.chunk_splitter.get_nodes_from_documents(documents)
        self.embeddings.embed_nodes(chunks) # No need to return this
        return chunks # returning chunks for later use


    def _load_example_dataset(self) -> List[Dict[str, Any]]:  # Made private
        """Loads the example dataset."""
        if not os.path.exists(self.example_dataset_path):  # Check for file existence
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
                raise ValueError("Unsupported example dataset format. Use JSON or CSV.")  # Raise ValueError for incorrect format
        except Exception as e:  # Catch broader exceptions during file loading
            raise RuntimeError(f"Error loading example dataset: {e}") from e  # Chain exceptions for better debugging



    def generate_dataset(
        self,
        knowledge_nodes: List[Node],
        num_entries: int = 1000,
        prompt_template: Optional[str] = None,
        **llm_kwargs
    ):
        """Generates a dataset using knowledge nodes and the example dataset format."""

        examples = self._load_example_dataset() # Load examples

        if not examples:
            raise ValueError("Example dataset is empty.")  # Raise error if example dataset is empty


        # Default prompt_template handling
        if not prompt_template:
            prompt_template = self.default_prompt_template

        # Generator to produce prompts in a streaming way
        def prompt_generator(num: int, knowledge: List[Node], template_name: str) -> Iterator[str]:
            """Generate prompts from knowledge nodes in a streaming way"""
            for _ in range(num):
                knowledge_node = knowledge[_ % len(knowledge)] # reusing knowledge nodes to reach the desired size
                variables = {"instruction": "Generate a question and answer pair based on the following context.",
                               "input_data": knowledge_node.text}
                prompt = self.prompt_manager.create_prompt(template_name=template_name, **variables)
                if prompt:
                    yield prompt
                else:
                    logger.warning("Failed to generate prompt for a dataset entry.")
                    continue

        # Generate dataset entries using LLM (streaming)
        generated_entries = self.llm.generate(list(prompt_generator(num_entries, knowledge_nodes, prompt_template)), stream_output=True, **llm_kwargs)  # Corrected generate call


        # Function to structure generated entries
        def structure_entry(entry: str):
            parts = entry.split('\n')  # Split by newline. Handle this split according to your generation format
            if len(parts) >= 2:
                return {"question": parts[0].strip(), "answer": " ".join(parts[1:]).strip()} # Assuming output has minimum two newline-separated parts. Customize this if your output is different
            else:
                logger.warning("Generated entry does not conform to expected format. Skipping.")
                return None  # Indicate invalid entry

        # Structure generated entries (using a list comprehension for conciseness)
        structured_entries = [
            structured_entry(entry) for entry in generated_entries if structured_entry(entry)
        ]

        self._save_dataset(structured_entries)  # Private save method


    def _save_dataset(self, dataset: List[Dict[str, Any]]):  # Made private
        """Saves the generated dataset."""

        if not dataset:
            logger.warning("Dataset is empty. Nothing to save.")
            return

        try:
            if self.output_format == "json":
                with open(self.output_path, "w", encoding="utf-8") as f:  # Open in text mode for JSON
                    json.dump(dataset, f, ensure_ascii=False, indent=4)
            elif self.output_format == "csv":
                fieldnames = dataset[0].keys()  # Get fieldnames from the first entry
                with open(self.output_path, "w", newline="", encoding="utf-8") as f: # Added newline='' to prevent extra blank lines in CSV
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(dataset)
            else:
                raise ValueError("Unsupported output format. Use 'json' or 'csv'.")  # Raise error instead of logging
        except Exception as e:  # Catch broader exceptions during file saving
            raise RuntimeError(f"Error saving dataset: {e}") from e # Raise exception if saving fails