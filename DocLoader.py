# WIP
# Code to load load pdf or md file as text for later use

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF for PDF reading
import markdown  # For Markdown parsing

def read_file(file_path: Path, encoding: str = "utf-8") -> str:
    """Reads the content of a file based on its MIME type.

    Args:
        file_path (Path): The path to the file.
        encoding (str, optional): The encoding of the file. Defaults to "utf-8".

    Returns:
        str: The content of the file.
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))

    if mime_type == "application/pdf":
        return read_pdf(file_path)
    elif mime_type == "text/markdown":
        return read_markdown(file_path, encoding)
    elif mime_type == "text/plain":
        return read_text(file_path, encoding)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

def read_pdf(file_path: Path) -> str:
    """Reads the text content of a PDF file.

    Args:
        file_path (Path): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    doc = fitz.open(str(file_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def read_markdown(file_path: Path, encoding: str) -> str:
    """Reads and parses the content of a Markdown file.

    Args:
        file_path (Path): The path to the Markdown file.
        encoding (str): The encoding of the file.

    Returns:
        str: The parsed Markdown text.
    """
    with open(file_path, encoding=encoding) as f:
        md_content = f.read()
    return markdown.markdown(md_content)

def read_text(file_path: Path, encoding: str) -> str:
    """Reads the content of a plain text file.

    Args:
        file_path (Path): The path to the text file.
        encoding (str): The encoding of the file.

    Returns:
        str: The content of the text file.
    """
    with open(file_path, encoding=encoding) as f:
        return f.read()

def load_directory(directory_path: str, 
                   recursive: bool = False, 
                   encoding: str = "utf-8", 
                   ext: Optional[str] = None,
                   exc: Optional[str] = None,
                   filenames: Optional[List[str]] = None) -> List[Dict]:
    """Loads files from a directory with optional filters.

    Args:
        directory_path (str): The path to the directory.
        recursive (bool, optional): Whether to recursively search subdirectories. Defaults to False.
        encoding (str, optional): The encoding of the files. Defaults to "utf-8".
        ext (Optional[str], optional): Only load files with this extension (e.g., "*.pdf"). Defaults to None.
        exc (Optional[str], optional): Exclude files with this extension (e.g., "*.txt"). Defaults to None.
        filenames (Optional[List[str]], optional): Load only these specific filenames. Defaults to None.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a file with its content.
    """
    directory = Path(directory_path)
    files: List[Dict] = []

    for file_path in directory.rglob("*") if recursive else directory.glob("*"):
        if file_path.is_file():
            if filenames is not None and file_path.name not in filenames:
                continue  # Skip if specific filenames are provided and current file is not in the list

            if ext is not None and not file_path.match(ext):
                continue  # Skip if a specific extension is provided and current file doesn't match

            if exc is not None and file_path.match(exc):
                continue  # Skip if a specific extension to exclude is provided and current file matches

            try:
                content = read_file(file_path, encoding)
                files.append({
                    "file_path": str(file_path),
                    "content": content,
                    "file_name": file_path.name
                })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return files


directory_path = "/content"  # Replace with the actual directory path
files = load_directory(directory_path)
for file_data in files:
    print(f"File: {file_data['file_name']}")
    print(f"Content: {file_data['content']}")
    print("-" * 20)