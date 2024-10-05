# PromptManager.py
import logging
from typing import Dict, Optional, List, Any
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self, default_template: Optional[str] = None):  # Allow default template setting
        self.templates = {}
        self._load_predefined_templates()
        if default_template:  # Set and log default template
            if default_template not in self.templates:
                raise ValueError(f"Default template '{default_template}' not found in predefined templates.")
            self.default_template_name = default_template
            logger.info(f"Default template set to: {default_template}")
        else:
            self.default_template_name = None

    def _load_predefined_templates(self):
        # More concise templates using f-strings.  RAG templates include CoT prompting.
        self.templates["rag_cot"] = (
            "Context:\n{{ context }}\n\nQuestion: {{ query }}\nLet's think step by step:\nAnswer:"
        )
        self.templates["rag_simple"] = (
            "Context:\n{{ context }}\n\nQuestion: {{ query }}\nAnswer:"
        )
        self.templates["dataset_instruction"] = (
            "Instruction: {{ instruction }}\nInput: {{ input_data }}\nOutput:"
        )
        self.templates["dataset_reasoning"] = "Problem: {{ problem_statement }}\nSolution:"


    def add_template(self, template_name: str, template_str: str):
        self.templates[template_name] = template_str # Overwrites existing templates by design, so no need to check or warn


    def remove_template(self, template_name: str):
        try:
            del self.templates[template_name]
        except KeyError:  # Catch the specific error for missing keys
            logger.warning(f"Template '{template_name}' not found.") # Only warning, not error

    def create_prompt(self, template_name: Optional[str] = None, **kwargs) -> str:
        """
        Creates a prompt using a specified template or the default template.  Handles optional context.

        Args:
            template_name (Optional[str]): The name of the prompt template to use.
            **kwargs: Variables for the prompt template. Includes:
                context (Optional[List[Any]]): A list of context elements (e.g., Node objects, strings).
                    If not provided or empty, the context part of the prompt will be excluded.

        Returns:
            str: The rendered prompt.  Returns an empty string if the template is not found.
        """

        if template_name is None:
            if self.default_template_name is None:
                logger.error("No template name provided and no default template set.")
                return ""
            template_name = self.default_template_name  # Use default template

        template_str = self.templates.get(template_name)
        if template_str is None:
            logger.error(f"Template '{template_name}' not found.")
            return ""

        context = kwargs.get('context', [])
        if context:
            if all(isinstance(item, str) for item in context): # Determine if all context elements are strings.
                kwargs['context'] = "\n".join(context)
            else:
                kwargs['context'] = "\n".join([item.text for item in context])
        else:  # If context is not provided, modify the template to exclude it
            template_str = template_str.replace("Context:\n{{ context }}\n\n", "")

        try:  # Handle potential errors during rendering
            template = Template(template_str)  # Re-render here, as template_str might have been modified in line 107
            prompt = template.render(**kwargs)
            return prompt
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return ""



    def list_templates(self) -> List[str]:
        return list(self.templates.keys())