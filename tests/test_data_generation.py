import unittest
from unittest.mock import Mock, patch, mock_open
import json
import csv
from zrag.data_generation import DataGenerator
from zrag.chunk_node import Node


class TestDataGenerator(unittest.TestCase):
    """Comprehensive tests for the DataGenerator class."""

    def setUp(self):
        """Sets up mocks for the components of the DataGenerator."""
        self.mock_file_loader = Mock()
        self.mock_chunk_splitter = Mock()
        self.mock_embeddings = Mock()
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()

        self.data_generator = DataGenerator(
            file_loader=self.mock_file_loader,
            chunk_splitter=self.mock_chunk_splitter,
            embeddings=self.mock_embeddings,
            llm=self.mock_llm,
            prompt_manager=self.mock_prompt_manager,
            example_dataset_path="example.json",
            output_format="json",
            output_path="output.json",
            batch_size=8,
            default_prompt_template="dataset_instruction",
        )

    def test_init(self):
        """Tests initialization of the DataGenerator."""
        self.assertIsInstance(self.data_generator, DataGenerator)
        self.assertEqual(self.data_generator.example_dataset_path, "example.json")
        self.assertEqual(self.data_generator.output_format, "json")
        self.assertEqual(self.data_generator.output_path, "output.json")
        self.assertEqual(self.data_generator.batch_size, 8)
        self.assertEqual(
            self.data_generator.default_prompt_template, "dataset_instruction"
        )

    def test_load_knowledge(self):
        """Tests loading and embedding knowledge data."""
        mock_documents = [Mock(), Mock()]
        mock_chunks = [Node(text="chunk1"), Node(text="chunk2")]

        self.mock_file_loader.load.return_value = mock_documents
        self.mock_chunk_splitter.split.return_value = mock_chunks

        result = self.data_generator.load_knowledge("test_dir")

        self.mock_file_loader.load.assert_called_once_with(directory_path="test_dir")
        self.mock_chunk_splitter.split.assert_called_once_with(mock_documents)
        self.mock_embeddings.embed_nodes.assert_called_once_with(mock_chunks)
        self.assertEqual(result, mock_chunks)

    def test_load_example_dataset_json(self):
        """Tests loading an example dataset from a JSON file."""
        file_content = '{"key": "value"}'
        expected = [{"key": "value"}]

        with patch("builtins.open", mock_open(read_data=file_content)):
            result = self.data_generator._load_example_dataset()
            self.assertEqual(result, expected)

    def test_load_example_dataset_csv(self):
        """Tests loading an example dataset from a CSV file."""
        self.data_generator.example_dataset_path = "example.csv"
        file_content = "key,value\ntest,data"
        expected = [{"key": "test", "value": "data"}]

        with patch("builtins.open", mock_open(read_data=file_content)):
            result = self.data_generator._load_example_dataset()
            self.assertEqual(result, expected)

    def test_load_example_dataset_file_not_found(self):
        """Tests handling of FileNotFoundError."""
        self.data_generator.example_dataset_path = "non_existent_file.json"
        with self.assertRaises(FileNotFoundError):
            self.data_generator._load_example_dataset()

    def test_load_example_dataset_unsupported_format(self):
        """Tests handling of unsupported file formats."""
        self.data_generator.example_dataset_path = "example.txt"
        with self.assertRaises(ValueError) as context:
            self.data_generator._load_example_dataset()
        self.assertTrue(
            "Unsupported example dataset format. Use JSON or CSV."
            in str(context.exception)
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load", side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_load_example_dataset_json_error(self, mock_json_load, mock_open_file):
        """Tests handling of JSON decode errors."""
        with self.assertRaises(RuntimeError) as context:
            self.data_generator._load_example_dataset()
        self.assertTrue("Error loading example dataset:" in str(context.exception))

    @patch("builtins.open", mock_open(read_data="invalid_csv_data"))
    def test_load_example_dataset_csv_error(self):
        """Tests handling of CSV read errors."""
        self.data_generator.example_dataset_path = "example.csv"
        with self.assertRaises(RuntimeError) as context:
            self.data_generator._load_example_dataset()
        self.assertTrue("Error loading example dataset:" in str(context.exception))

    def test_generate_dataset(self):
        """Tests generating a dataset using the default prompt template."""
        mock_knowledge_nodes = [Node(text="test knowledge 1", metadata={}), Node(text="test knowledge 2", metadata={})]
        mock_examples = [{"question": "q1", "answer": "a1"}]
        mock_generated_entries = ["Question: test1\nAnswer: response1", "Question: test2\nAnswer: response2"]

        with patch.object(
            DataGenerator, "_load_example_dataset", return_value=mock_examples
        ):
            with patch.object(DataGenerator, "_save_dataset") as mock_save:
                self.mock_prompt_manager.create_prompt.side_effect = ["test prompt 1", "test prompt 2"]
                self.mock_llm.generate.return_value = mock_generated_entries

                self.data_generator.generate_dataset(
                    mock_knowledge_nodes, num_entries=2
                )

                self.mock_prompt_manager.create_prompt.assert_called()
                self.mock_llm.generate.assert_called_once_with(['test prompt 1', 'test prompt 2'], stream_output=True)
                mock_save.assert_called_once_with(
                    [
                        {"question": "Question: test1", "answer": "response1"},
                        {"question": "Question: test2", "answer": "response2"},
                    ]
                )

    def test_generate_dataset_with_custom_prompt(self):
        """Tests generating a dataset with a custom prompt template."""
        mock_knowledge_nodes = [Node(text="test knowledge", metadata={})]
        mock_examples = [{"problem": "p1", "solution": "s1"}]
        mock_generated_entries = ["Problem: test\nSolution: response"]

        with patch.object(
            DataGenerator, "_load_example_dataset", return_value=mock_examples
        ):
            with patch.object(DataGenerator, "_save_dataset") as mock_save:
                self.mock_prompt_manager.create_prompt.return_value = "test prompt"
                self.mock_llm.generate.return_value = mock_generated_entries

                self.data_generator.generate_dataset(
                    mock_knowledge_nodes,
                    num_entries=1,
                    prompt_template="dataset_reasoning",
                )

                self.mock_prompt_manager.create_prompt.assert_called_with(
                    template_name="dataset_reasoning",
                    instruction="Generate a problem and solution pair based on the following context.",
                    input_data="test knowledge",
                )
                mock_save.assert_called_once_with(
                    [{"problem": "Problem: test", "solution": "response"}]
                )

    def test_generate_dataset_empty_example(self):
        """Tests handling of an empty example dataset."""
        with patch.object(DataGenerator, "_load_example_dataset", return_value=[]):
            with self.assertRaises(ValueError) as context:
                self.data_generator.generate_dataset([Node(text="test", metadata={})])
            self.assertTrue("Example dataset is empty" in str(context.exception))

    def test_save_dataset_json(self):
        """Tests saving a dataset to a JSON file."""
        mock_dataset = [{"question": "q1", "answer": "a1"}]

        with patch("builtins.open", mock_open()) as mock_file:
            self.data_generator._save_dataset(mock_dataset)
            mock_file.assert_called_once_with(
                self.data_generator.output_path, "w", encoding="utf-8"
            )
            mock_file().write.assert_called_once_with(
                json.dumps(mock_dataset, ensure_ascii=False, indent=4)
            )

    def test_save_dataset_csv(self):
        """Tests saving a dataset to a CSV file."""
        self.data_generator.output_format = "csv"
        mock_dataset = [{"question": "q1", "answer": "a1"}]

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("csv.DictWriter.writeheader") as mock_writeheader:
                with patch("csv.DictWriter.writerows") as mock_writerows:
                    self.data_generator._save_dataset(mock_dataset)
                    mock_file.assert_called_once_with(
                        self.data_generator.output_path,
                        "w",
                        newline="",
                        encoding="utf-8",
                    )
                    mock_writeheader.assert_called_once()
                    mock_writerows.assert_called_once_with(mock_dataset)

    def test_save_dataset_empty(self):
        """Tests saving an empty dataset."""
        with self.assertLogs() as captured:
            self.data_generator._save_dataset([])
            self.assertIn(
                "Dataset is empty. Nothing to save.", captured.records[0].getMessage()
            )

    def test_save_dataset_unsupported_format(self):
        """Tests handling of unsupported output formats."""
        self.data_generator.output_format = "unsupported"
        with self.assertRaises(ValueError) as context:
            self.data_generator._save_dataset([{"question": "q", "answer": "a"}])
        self.assertTrue("Unsupported output format" in str(context.exception))

if __name__ == "__main__":
    unittest.main()