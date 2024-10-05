# PromptManager.py

import logging
from typing import Dict, Optional
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates for interacting with LLMs.
    Supports different templates for various tasks such as RAG and dataset generation.
    """

    def __init__(self):
        """
        Initializes the PromptManager with predefined templates.
        Templates are stored in a dictionary with keys as template names.
        """
        self.templates: Dict[str, str] = {}
        self._load_predefined_templates()
        logger.info("PromptManager initialized with predefined templates.")

    def _load_predefined_templates(self):
        """
        Loads a set of predefined templates into the manager.
        Users can extend this method to load templates from external sources or files.
        """
        # Template for RAG with Chain-of-Thought
        self.templates["rag_cot"] = (
            "Context:\n{{ context }}\n\n"
            "Question: {{ query }}\n"
            "Answer (with Chain-of-Thought):"
        )

        # Template for simple RAG
        self.templates["rag_simple"] = (
            "Context:\n{{ context }}\n\n"
            "Question: {{ query }}\n"
            "Answer:"
        )

        # Template for dataset generation (instruction following)
        self.templates["dataset_instruction"] = (
            "Instruction: {{ instruction }}\n"
            "Input: {{ input_data }}\n"
            "Output:"
        )

        # Template for dataset generation (reasoning)
        self.templates["dataset_reasoning"] = (
            "Problem: {{ problem_statement }}\n"
            "Solution:"
        )

        logger.info("Predefined templates loaded successfully.")

    def add_template(self, template_name: str, template_str: str):
        """
        Adds a new template to the manager.

        Args:
            template_name (str): Unique name for the template.
            template_str (str): The template string containing Jinja2 placeholders.
        """
        if template_name in self.templates:
            logger.warning(
                f"Template '{template_name}' already exists and will be overwritten."
            )
        self.templates[template_name] = template_str
        logger.info(f"Template '{template_name}' added successfully.")

    def remove_template(self, template_name: str):
        """
        Removes a template from the manager.

        Args:
            template_name (str): Name of the template to remove.
        """
        if template_name in self.templates:
            del self.templates[template_name]
            logger.info(f"Template '{template_name}' removed successfully.")
        else:
            logger.warning(
                f"Attempted to remove non-existent template '{template_name}'."
            )

    def get_prompt(
        self, template_name: str, variables: Dict[str, str]
    ) -> Optional[str]:
        """
        Renders a prompt based on the specified template and variables.

        Args:
            template_name (str): Name of the template to use.
            variables (Dict[str, str]): Dictionary of variables to substitute in the template.

        Returns:
            Optional[str]: The rendered prompt string, or None if template not found.
        """
        template_str = self.templates.get(template_name)
        if not template_str:
            logger.error(f"Template '{template_name}' not found.")
            return None

        try:
            template = Template(template_str)
            prompt = template.render(**variables)
            logger.info(f"Prompt generated using template '{template_name}'.")
            return prompt
        except Exception as e:
            logger.error(
                f"Error rendering template '{template_name}' with variables {variables}: {e}"
            )
            return None

    def list_templates(self) -> list:
        """
        Lists all available template names.

        Returns:
            list: List of template names.
        """
        return list(self.templates.keys())