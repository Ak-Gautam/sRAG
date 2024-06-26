# WIP
# Code to load load pdf or md file as text for later use

# Markdown is the easiest to handle
import os
import markdown
from pathlib import Path
from typing import List, Dict

def load_markdown_files(directory_path: str, recursive: bool = False, encoding: str = "utf-8") -> List[Dict]:
    """Loads Markdown files from a directory.

    Args:
        directory_path (str): The path to the directory.
        recursive (bool, optional): Whether to recursively search subdirectories. Defaults to False.
        encoding (str, optional): The encoding of the files. Defaults to "utf-8".

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a Markdown file with its content.
    """
    directory = Path(directory_path)
    files: List[Dict] = []

    for file_path in directory.rglob("*") if recursive else directory.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() == '.md':
            try:
                with open(file_path, encoding=encoding) as f:
                    md_content = f.read()
                content = markdown.markdown(md_content)
                files.append({
                    "file_path": str(file_path),
                    "content": content,
                    "file_name": file_path.name
                })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return files
