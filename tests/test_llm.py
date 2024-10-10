import unittest
from unittest.mock import patch, MagicMock
import torch
from srag.llm import LLM

class TestLLM(unittest.TestCase):

    def setUp(self):
        self.model_name = "test-model"
        self.patcher_tokenizer = patch('srag.llm.AutoTokenizer.from_pretrained')
        self.mock_tokenizer = self.patcher_tokenizer.start()
        self.patcher_seq2seq = patch('srag.llm.AutoModelForSeq2SeqLM.from_pretrained')
        self.mock_seq2seq = self.patcher_seq2seq.start()
        self.patcher_causal = patch('srag.llm.AutoModelForCausalLM.from_pretrained')
        self.mock_causal = self.patcher_causal.start()

        # Mock torch.cuda.is_available
        self.patcher_cuda = patch('srag.llm.torch.cuda.is_available', return_value=True)
        self.mock_cuda = self.patcher_cuda.start()

    def tearDown(self):
        self.patcher_tokenizer.stop()
        self.patcher_seq2seq.stop()
        self.patcher_causal.stop()
        self.patcher_cuda.stop()

    def test_initialization_seq2seq(self):
        llm = LLM(self.model_name)
        self.mock_tokenizer.assert_called_once_with(self.model_name, trust_remote_code=True)
        self.mock_seq2seq.assert_called_once_with(self.model_name, trust_remote_code=True)
        self.assertTrue(llm.is_encoder_decoder)
        self.assertEqual(llm.device, 0)  # GPU device

    def test_initialization_causal(self):
        self.mock_seq2seq.side_effect = ValueError()  # Force fallback to causal model
        llm = LLM(self.model_name)
        self.mock_causal.assert_called_once_with(self.model_name, trust_remote_code=True)
        self.assertFalse(llm.is_encoder_decoder)

    @patch('srag.llm.LLM._generate_text')
    def test_generate_single_prompt(self, mock_generate):
        llm = LLM(self.model_name)
        prompt = "Test prompt"
        expected_output = "Generated text"
        mock_generate.return_value = expected_output

        result = llm.generate(prompt)
        self.assertEqual(result, expected_output)
        mock_generate.assert_called_once_with(prompt, max_new_tokens=256, num_beams=1)

    @patch('srag.llm.LLM._generate_text')
    def test_generate_multiple_prompts(self, mock_generate):
        llm = LLM(self.model_name)
        prompts = ["Prompt 1", "Prompt 2"]
        expected_outputs = ["Generated 1", "Generated 2"]
        mock_generate.return_value = expected_outputs

        result = llm.generate(prompts)
        self.assertEqual(result, expected_outputs)
        mock_generate.assert_called_once_with(prompts, max_new_tokens=256, num_beams=1)

    @patch('srag.llm.LLM._generate_text')
    def test_generate_stream(self, mock_generate):
        llm = LLM(self.model_name)
        prompts = ["Prompt 1", "Prompt 2"]
        expected_outputs = ["Generated 1", "Generated 2"]
        mock_generate.side_effect = expected_outputs

        result = llm.generate(prompts, stream_output=True)
        self.assertTrue(isinstance(result, Generator))
        self.assertEqual(list(result), expected_outputs)

    def test_generate_invalid_input(self):
        llm = LLM(self.model_name)
        with self.assertRaises(TypeError):
            llm.generate(123)  # Invalid input type

    @patch('srag.llm.LLM._generate_text')
    def test_generate_with_custom_kwargs(self, mock_generate):
        llm = LLM(self.model_name, temperature=0.7)  # Custom kwargs in init
        prompt = "Test prompt"
        expected_output = "Generated text"
        mock_generate.return_value = expected_output

        result = llm.generate(prompt, max_new_tokens=100, top_k=50)  # Custom kwargs in generate
        self.assertEqual(result, expected_output)
        mock_generate.assert_called_once_with(prompt, max_new_tokens=100, num_beams=1, temperature=0.7, top_k=50)

if __name__ == '__main__':
    unittest.main()