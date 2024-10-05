# DataGen.py

import logging
import json
import csv
import os
from typing import List, Dict, Any, Optional

from DocLoader import FileLoader, Document
from llm import LLM
from PromptManager import PromptManager
from ChunkNode import get_chunk_splitter, ChunkSplitter
from embeddings import Embeddings
from indexing import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGen:
    """
    Handles dataset generation using knowledge from documents and example dataset formats.
    Utilizes LLMs to generate new data entries based on prompts.
    """

    def __init__(
        self,
        knowledge_directory: str,
        example_dataset: str,
        output_format: str = "json",
        output_path: str = "generated_dataset.json",
        chunking_strategy: str = 'overlap',
        chunking_kwargs: Optional[Dict[str, Any]] = None,
        embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        llm_model_name: str = "gpt2",
        llm_task: str = "text-generation",
        prompt_template: str = "dataset_instruction",
        use_gpu: bool = True,
        batch_size: int = 8,
    ):
        """
        Initializes the DataGen with specified configurations.

        Args:
            knowledge_directory (str): Path to the directory containing knowledge documents.
            example_dataset (str): Path to the example dataset file (JSON or CSV).
            output_format (str): Desired output format for the generated dataset ('json' or 'csv').
            output_path (str): Path to save the generated dataset.
            chunking_strategy (str): Strategy for chunking ('token', 'overlap', 'paragraph').
            chunking_kwargs (Optional[Dict[str, Any]]): Additional arguments for the chunk splitter.
            embedding_model_name (str): Hugging Face model name for embeddings.
            llm_model_name (str): Hugging Face model name for the LLM.
            llm_task (str): Task for the LLM ('text-generation', 'translation', etc.).
            prompt_template (str): Name of the prompt template to use for dataset generation.
            use_gpu (bool): Whether to use GPU for the LLM.
            batch_size (int): Number of prompts to process per batch.
        """
        logger.info("Initializing DataGen.")

        # Initialize FileLoader for knowledge documents
        self.file_loader = FileLoader(knowledge_directory)
        logger.info(f"FileLoader initialized for directory: {knowledge_directory}")

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

        # Initialize LLM
        self.llm = LLM(
            model_name=llm_model_name,
            use_gpu=use_gpu,
            task=llm_task,
        )
        logger.info(f"LLM initialized with model: {llm_model_name} for task: {llm_task}")

        # Initialize PromptManager
        self.prompt_manager = PromptManager()
        logger.info("PromptManager initialized.")

        # Store example dataset and output configurations
        self.example_dataset = example_dataset
        self.output_format = output_format.lower()
        self.output_path = output_path
        self.batch_size = batch_size

        # Set prompt template
        self.prompt_template = prompt_template
        logger.info(f"Using prompt template: {prompt_template}")

    def load_knowledge(self,
                       recursive: bool = False,
                       ext: Optional[List[str]] = None,
                       exc: Optional[List[str]] = None,
                       filenames: Optional[List[str]] = None,
                       preprocess_fn: Optional[Any] = None,
                       max_workers: int = None
                       ) -> List[Any]:
        """
        Loads, chunks, and embeds knowledge documents.

        Args:
            recursive (bool): Whether to load files recursively.
            ext (Optional[List[str]]): List of file extensions to include.
            exc (Optional[List[str]]): List of file extensions to exclude.
            filenames (Optional[List[str]]): Specific filenames to include.
            preprocess_fn (Optional[Any]): Preprocessing function to apply to text.
            max_workers (int): Number of worker processes for parallel processing.

        Returns:
            List[Any]: List of nodes with embeddings.
        """
        logger.info("Starting knowledge loading pipeline.")

        # Step 1: Load Documents
        documents: List[Document] = self.file_loader.load_files(
            recursive=recursive,
            ext=ext,
            exc=exc,
            filenames=filenames,
            max_workers=max_workers,
            preprocess_fn=preprocess_fn
        )
        logger.info(f"Loaded {len(documents)} knowledge documents.")

        if not documents:
            logger.warning("No knowledge documents loaded. Exiting knowledge loading pipeline.")
            return []

        # Step 2: Chunk Documents
        nodes = self.chunk_splitter.get_nodes_from_documents(documents)
        logger.info(f"Chunked knowledge documents into {len(nodes)} nodes.")

        if not nodes:
            logger.warning("No nodes generated from knowledge documents.")
            return []

        # Step 3: Generate Embeddings
        nodes_with_embeddings = self.embeddings.embed_nodes(nodes)
        logger.info("Generated embeddings for all knowledge nodes.")

        logger.info("Knowledge loading pipeline completed successfully.")
        return nodes_with_embeddings

    def load_example_dataset(self) -> List[Dict[str, Any]]:
        """
        Loads the example dataset to determine the format and extract required fields.

        Returns:
            List[Dict[str, Any]]: List of examples from the dataset.
        """
        logger.info(f"Loading example dataset from: {self.example_dataset}")

        if not os.path.exists(self.example_dataset):
            logger.error(f"Example dataset file not found: {self.example_dataset}")
            return []

        examples = []
        try:
            if self.example_dataset.endswith('.json'):
                with open(self.example_dataset, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                logger.info(f"Loaded {len(examples)} examples from JSON dataset.")
            elif self.example_dataset.endswith('.csv'):
                with open(self.example_dataset, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    examples = [row for row in reader]
                logger.info(f"Loaded {len(examples)} examples from CSV dataset.")
            else:
                logger.error("Unsupported example dataset format. Use JSON or CSV.")
        except Exception as e:
            logger.error(f"Failed to load example dataset: {e}")

        return examples

    def generate_dataset(
        self,
        knowledge_nodes: List[Any],
        num_entries: int = 1000,
        output_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """
        Generates a dataset by leveraging knowledge from documents and example dataset format.

        Args:
            knowledge_nodes (List[Any]): List of knowledge nodes with embeddings.
            num_entries (int): Number of dataset entries to generate.
            output_format (Optional[str]): Desired output format ('json' or 'csv').
            output_path (Optional[str]): Path to save the generated dataset.
        """
        logger.info("Starting dataset generation process.")

        if output_format:
            self.output_format = output_format.lower()
        if output_path:
            self.output_path = output_path

        # Load example dataset to understand the format
        examples = self.load_example_dataset()
        if not examples:
            logger.error("No examples loaded from the example dataset. Cannot proceed with dataset generation.")
            return

        # Determine fields from the example dataset
        example_fields = examples[0].keys()
        logger.info(f"Example dataset fields: {list(example_fields)}")

        # Prepare prompts based on the example dataset
        prompts = []
        for _ in range(num_entries):
            # Randomly select a knowledge node to base the prompt on
            knowledge_node = knowledge_nodes[_ % len(knowledge_nodes)]
            # Construct variables for the prompt
            variables = {
                "instruction": "Generate a question and answer pair based on the following context.",
                "input_data": knowledge_node.text
            }
            # Render the prompt
            prompt = self.prompt_manager.get_prompt(
                template_name=self.prompt_template,
                variables=variables
            )
            if prompt:
                prompts.append(prompt)
            else:
                logger.warning("Failed to generate prompt for a dataset entry.")

        if not prompts:
            logger.error("No prompts generated. Cannot proceed with dataset generation.")
            return

        logger.info(f"Generated {len(prompts)} prompts for dataset generation.")

        # Generate dataset entries using LLM in batches
        generated_entries = self.llm.generate_dataset(
            prompts=prompts,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            batch_size=self.batch_size
        )

        logger.info(f"Generated {len(generated_entries)} dataset entries.")

        # Structure the generated entries according to the example dataset fields
        structured_entries = []
        for i, entry in enumerate(generated_entries):
            # Simple assumption: the generated entry contains both question and answer separated by a newline
            parts = entry.split('\n')
            if len(parts) >= 2:
                structured_entry = {
                    "question": parts[0].strip(),
                    "answer": " ".join(parts[1:]).strip()
                }
                structured_entries.append(structured_entry)
            else:
                logger.warning(f"Generated entry {i} does not conform to expected format.")

        logger.info(f"Structured {len(structured_entries)} dataset entries.")

        # Save the structured dataset
        self.save_dataset(structured_entries)
        logger.info("Dataset generation process completed successfully.")

    def save_dataset(self, dataset: List[Dict[str, Any]]):
        """
        Saves the generated dataset to the specified output path in the desired format.

        Args:
            dataset (List[Dict[str, Any]]): List of dataset entries to save.
        """
        logger.info(f"Saving dataset to {self.output_path} in {self.output_format.upper()} format.")

        try:
            if self.output_format == "json":
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=4)
                logger.info(f"Dataset saved successfully to {self.output_path}.")
            elif self.output_format == "csv":
                if not dataset:
                    logger.warning("Empty dataset. No CSV file created.")
                    return
                fieldnames = dataset[0].keys()
                with open(self.output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(dataset)
                logger.info(f"Dataset saved successfully to {self.output_path}.")
            else:
                logger.error("Unsupported output format. Use 'json' or 'csv'.")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")