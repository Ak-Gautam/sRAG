import logging
from typing import Dict, Optional, List, Any
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates for generating prompts for LLMs.
    """
    def __init__(self, default_template: Optional[str] = None):  
        """Initializes the PromptManager with predefined templates."""
        self.templates = {}
        self._load_predefined_templates()
        if default_template:  
            if default_template not in self.templates:
                raise ValueError(f"Default template '{default_template}' not found in predefined templates.")
            self.default_template_name = default_template
            logger.info(f"Default template set to: {default_template}")
        else:
            self.default_template_name = None

    def _load_predefined_templates(self):
        """Loads predefined prompt templates."""
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
        """Adds a new prompt template."""
        self.templates[template_name] = template_str 

    def remove_template(self, template_name: str):
        """Removes a prompt template."""
        try:
            del self.templates[template_name]
        except KeyError:  
            logger.warning(f"Template '{template_name}' not found.") 

    def create_prompt(self, template_name: Optional[str] = None, **kwargs) -> str:
        """
        Creates a prompt using a specified template or the default template.
        """

        if template_name is None:
            if self.default_template_name is None:
                logger.error("No template name provided and no default template set.")
                return ""
            template_name = self.default_template_name  

        template_str = self.templates.get(template_name)
        if template_str is None:
            logger.error(f"Template '{template_name}' not found.")
            return ""

        context = kwargs.get('context', [])
        if context:
            if all(isinstance(item, str) for item in context): 
                kwargs['context'] = "\n".join(context)
            elif all(isinstance(item, dict) for item in context) and all('document' in item for item in context):
                # Handle ChromaDB results that include metadata (dictionaries)
                kwargs['context'] = "\n".join([item['document'] for item in context]) 
            else:
                kwargs['context'] = "\n".join([item.text for item in context])
        else:  # If context is not provided or is empty, modify the template to exclude it
            template_str = template_str.replace("Context:\n{{ context }}\n\n", "")

        try:  
            template = Template(template_str)  
            prompt = template.render(**kwargs)
            return prompt
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return ""

    def list_templates(self) -> List[str]:
        """Returns a list of available template names."""
        return list(self.templates.keys())