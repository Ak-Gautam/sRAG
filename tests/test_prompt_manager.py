import unittest
from srag.prompt_manager import PromptManager

class TestPromptManager(unittest.TestCase):

    def setUp(self):
        self.pm = PromptManager()

    def test_initialization(self):
        self.assertIsInstance(self.pm, PromptManager)
        self.assertIn("rag_cot", self.pm.templates)
        self.assertIn("rag_simple", self.pm.templates)
        self.assertIn("dataset_instruction", self.pm.templates)
        self.assertIn("dataset_reasoning", self.pm.templates)

    def test_initialization_with_default_template(self):
        pm = PromptManager(default_template="rag_cot")
        self.assertEqual(pm.default_template_name, "rag_cot")

    def test_initialization_with_invalid_default_template(self):
        with self.assertRaises(ValueError):
            PromptManager(default_template="non_existent_template")

    def test_add_template(self):
        self.pm.add_template("test_template", "This is a {{ test }} template")
        self.assertIn("test_template", self.pm.templates)
        self.assertEqual(self.pm.templates["test_template"], "This is a {{ test }} template")

    def test_remove_template(self):
        self.pm.add_template("test_template", "This is a test template")
        self.pm.remove_template("test_template")
        self.assertNotIn("test_template", self.pm.templates)

    def test_remove_non_existent_template(self):
        with self.assertLogs(level='WARNING') as cm:
            self.pm.remove_template("non_existent_template")
        self.assertIn("Template 'non_existent_template' not found.", cm.output[0])

    def test_create_prompt_with_default_template(self):
        pm = PromptManager(default_template="rag_simple")
        prompt = pm.create_prompt(query="What is the capital of France?", context="France is a country in Europe.")
        expected_prompt = "Context:\nFrance is a country in Europe.\n\nQuestion: What is the capital of France?\nAnswer:"
        self.assertEqual(prompt.strip(), expected_prompt.strip())

    def test_create_prompt_with_specified_template(self):
        prompt = self.pm.create_prompt("rag_cot", query="What is 2+2?", context="Basic arithmetic operations.")
        expected_prompt = "Context:\nBasic arithmetic operations.\n\nQuestion: What is 2+2?\nLet's think step by step:\nAnswer:"
        self.assertEqual(prompt.strip(), expected_prompt.strip())

    def test_create_prompt_with_list_context(self):
        context = ["France is in Europe.", "Paris is a city in France."]
        prompt = self.pm.create_prompt("rag_simple", query="What is the capital of France?", context=context)
        expected_prompt = "Context:\nFrance is in Europe.\nParis is a city in France.\n\nQuestion: What is the capital of France?\nAnswer:"
        self.assertEqual(prompt.strip(), expected_prompt.strip())

    def test_create_prompt_with_dict_context(self):
        context = [{"document": "France is in Europe."}, {"document": "Paris is a city in France."}]
        prompt = self.pm.create_prompt("rag_simple", query="What is the capital of France?", context=context)
        expected_prompt = "Context:\nFrance is in Europe.\nParis is a city in France.\n\nQuestion: What is the capital of France?\nAnswer:"
        self.assertEqual(prompt.strip(), expected_prompt.strip())

    def test_create_prompt_without_context(self):
        prompt = self.pm.create_prompt("rag_simple", query="What is 2+2?")
        expected_prompt = "Question: What is 2+2?\nAnswer:"
        self.assertEqual(prompt.strip(), expected_prompt.strip())

    def test_create_prompt_with_non_existent_template(self):
        with self.assertLogs(level='ERROR') as cm:
            prompt = self.pm.create_prompt("non_existent_template", query="Test query")
        self.assertIn("Template 'non_existent_template' not found.", cm.output[0])
        self.assertEqual(prompt, "")

    def test_list_templates(self):
        templates = self.pm.list_templates()
        self.assertIsInstance(templates, list)
        self.assertIn("rag_cot", templates)
        self.assertIn("rag_simple", templates)
        self.assertIn("dataset_instruction", templates)
        self.assertIn("dataset_reasoning", templates)

if __name__ == '__main__':
    unittest.main()