import unittest
from unittest.mock import patch, MagicMock
import torch
from zrag.llm import LLM


class TestLLM(unittest.TestCase):
    """Comprehensive tests for the LLM class."""

    @patch("srag.llm.AutoTokenizer.from_pretrained")
    @patch("srag.llm.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("srag.llm.AutoModelForCausalLM.from_pretrained")
    @patch("srag.llm.torch.cuda.is_available")
    def setUp(self, mock_cuda, mock_causal, mock_seq2seq, mock_tokenizer):
        """Sets up mocks for tokenizer, models, and CUDA availability."""
        self.model_name = "test-model"
        self.mock_tokenizer = mock_tokenizer.return_value
        self.mock_seq2seq = mock_seq2seq.return_value
        self.mock_causal = mock_causal.return_value
        mock_cuda.return_value = True  # Assume GPU is available by default

    def test_initialization_seq2seq(self):
        """Tests initialization with a seq2seq model (encoder-decoder)."""
        llm = LLM(self.model_name)
        self.mock_tokenizer.assert_called_once_with(
            self.model_name, trust_remote_code=True
        )
        self.mock_seq2seq.assert_called_once_with(
            self.model_name, trust_remote_code=True
        )
        self.assertTrue(llm.is_encoder_decoder)
        self.assertEqual(llm.device, 0)  # GPU device

    def test_initialization_causal(self):
        """Tests initialization with a causal model (decoder-only)."""
        self.mock_seq2seq.side_effect = ValueError(
            "Fake error to force causal model loading"
        )
        llm = LLM(self.model_name)
        self.mock_causal.assert_called_once_with(
            self.model_name, trust_remote_code=True
        )
        self.assertFalse(llm.is_encoder_decoder)
        self.assertEqual(llm.device, 0)  # GPU device

    def test_initialization_cpu(self):
        """Tests initialization with CPU when GPU is not available."""
        self.mock_cuda.return_value = False  # Simulate no GPU
        llm = LLM(self.model_name, use_gpu=True)  # Still try to use GPU
        self.assertEqual(llm.device, -1)  # CPU device

    def test_initialization_cpu_forced(self):
        """Tests initialization with CPU when use_gpu is False."""
        llm = LLM(self.model_name, use_gpu=False)  # Force CPU usage
        self.assertEqual(llm.device, -1)  # CPU device

    @patch("srag.llm.LLM._generate_text")
    def test_generate_single_prompt(self, mock_generate):
        """Tests generating text for a single prompt."""
        llm = LLM(self.model_name)
        prompt = "Test prompt"
        expected_output = "Generated text"
        mock_generate.return_value = expected_output

        result = llm.generate(prompt)
        self.assertEqual(result, expected_output)
        mock_generate.assert_called_once_with(
            prompt, max_new_tokens=256, num_beams=1
        )  # Default args for seq2seq

    @patch("srag.llm.LLM._generate_text")
    def test_generate_multiple_prompts(self, mock_generate):
        """Tests generating text for multiple prompts."""
        llm = LLM(self.model_name)
        prompts = ["Prompt 1", "Prompt 2"]
        expected_outputs = ["Generated 1", "Generated 2"]
        mock_generate.return_value = expected_outputs

        result = llm.generate(prompts)
        self.assertEqual(result, expected_outputs)
        mock_generate.assert_called_once_with(
            prompts, max_new_tokens=256, num_beams=1
        )  # Default args for seq2seq

    @patch("srag.llm.LLM._generate_text")
    def test_generate_stream_single(self, mock_generate):
        """Tests generating text with streaming output for a single prompt."""
        llm = LLM(self.model_name)
        prompt = "Test prompt"
        expected_output = "Generated text"
        mock_generate.return_value = expected_output  # Simulate streaming output

        result = llm.generate(prompt, stream_output=True)
        self.assertEqual(result, expected_output)
        mock_generate.assert_called_once_with(
            prompt, max_new_tokens=256, num_beams=1
        )

    @patch("srag.llm.LLM._generate_text")
    def test_generate_stream_multiple(self, mock_generate):
        """Tests generating text with streaming output for multiple prompts."""
        llm = LLM(self.model_name)
        prompts = ["Prompt 1", "Prompt 2"]
        expected_outputs = ["Generated 1", "Generated 2"]
        mock_generate.side_effect = expected_outputs  # Simulate streaming for each

        result = llm.generate(prompts, stream_output=True)
        self.assertTrue(isinstance(result, type(self.llm.generate("", stream_output=True)))) # Check if generator
        self.assertEqual(list(result), expected_outputs)

    def test_generate_invalid_input(self):
        """Tests handling of invalid input type to generate()."""
        llm = LLM(self.model_name)
        with self.assertRaises(TypeError):
            llm.generate(123)  # Invalid input type

    @patch("srag.llm.LLM._generate_text")
    def test_generate_with_custom_kwargs_seq2seq(self, mock_generate):
        """Tests generating text with custom keyword arguments (seq2seq)."""
        llm = LLM(self.model_name, temperature=0.7)  # Custom kwargs in init
        prompt = "Test prompt"
        expected_output = "Generated text"
        mock_generate.return_value = expected_output

        result = llm.generate(
            prompt, max_new_tokens=100, top_k=50
        )  # Custom kwargs in generate
        self.assertEqual(result, expected_output)
        mock_generate.assert_called_once_with(
            prompt, max_new_tokens=100, num_beams=1, temperature=0.7, top_k=50
        )

    @patch("srag.llm.LLM._generate_text")
    def test_generate_with_custom_kwargs_causal(self, mock_generate):
        """Tests generating text with custom keyword arguments (causal)."""
        self.mock_seq2seq.side_effect = ValueError(
            "Fake error to force causal model loading"
        )  # Force causal model
        llm = LLM(self.model_name, temperature=0.7)  # Custom kwargs in init
        prompt = "Test prompt"
        expected_output = "Generated text"
        mock_generate.return_value = expected_output

        result = llm.generate(
            prompt, max_new_tokens=100, top_k=50
        )  # Custom kwargs in generate
        self.assertEqual(result, expected_output)
        mock_generate.assert_called_once_with(
            prompt, max_new_tokens=100, num_beams=None, temperature=0.7, top_k=50
        )  # num_beams=None for causal


if __name__ == "__main__":
    unittest.main()