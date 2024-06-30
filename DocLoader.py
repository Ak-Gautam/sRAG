# WIP
# Code to load pdf or md file as text for later use

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional
import uuid
import datetime

import fitz  # PyMuPDF for PDF reading
import markdown  # For Markdown parsing


class Document:
    """Represents a single document with its content and metadata."""

    def __init__(self, id: str, metadata: Dict, text: str):
        """
        Initializes a Document object.

        Args:
            id (str): A unique identifier for the document.
            metadata (Dict): A dictionary containing metadata about the document.
            text (str): The content of the document.
        """
        self.id = id
        self.metadata = metadata
        self.text = text

    def __repr__(self):
        return f"Document(id='{self.id}', metadata={self.metadata}, text='{self.text[:20]}...')"


class FileLoader:
    """
    A class for loading and reading files from a directory,
    returning documents with metadata and text content.
    """

    def __init__(self, directory_path: str, encoding: str = "utf-8"):
        """
        Initializes the FileLoader object.

        Args:
            directory_path (str): The path to the directory.
            encoding (str, optional): The encoding of the files. Defaults to "utf-8".
        """
        self.directory_path = directory_path
        self.encoding = encoding

    def read_file(self, file_path: Path) -> str:
        """Reads the content of a file based on its MIME type.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The content of the file.
        """
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type == "application/pdf":
            return self.read_pdf(file_path)
        elif mime_type == "text/markdown":
            return self.read_markdown(file_path)
        elif mime_type == "text/plain":
            return self.read_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

    def read_pdf(self, file_path: Path) -> List[Document]:
        """Reads the text content of a PDF file, returning a list of documents for each page.

        Args:
            file_path (Path): The path to the PDF file.

        Returns:
            List[Document]: A list of Document objects, one for each page in the PDF.
        """
        documents = []
        doc = fitz.open(str(file_path))
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        creation_date = datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d')
        last_modified_date = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d')
        for page_num, page in enumerate(doc):
            page_label = str(page_num + 1)
            text = page.get_text()
            metadata = {
                'page_label': page_label,
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_type': 'application/pdf',
                'file_size': file_size,
                'creation_date': creation_date,
                'last_modified_date': last_modified_date
            }
            document_id = str(uuid.uuid4())
            documents.append(Document(document_id, metadata, text))
        doc.close()
        return documents

    def read_markdown(self, file_path: Path) -> Document:
        """Reads and parses the content of a Markdown file.

        Args:
            file_path (Path): The path to the Markdown file.

        Returns:
            Document: A Document object containing the parsed Markdown text.
        """
        with open(file_path, encoding=self.encoding) as f:
            md_content = f.read()
        text = markdown.markdown(md_content)
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        creation_date = datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d')
        last_modified_date = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d')
        metadata = {
            'page_label': '1',
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': 'text/markdown',
            'file_size': file_size,
            'creation_date': creation_date,
            'last_modified_date': last_modified_date
        }
        document_id = str(uuid.uuid4())
        return Document(document_id, metadata, text)

    def read_text(self, file_path: Path) -> Document:
        """Reads the content of a plain text file.

        Args:
            file_path (Path): The path to the text file.

        Returns:
            Document: A Document object containing the text content.
        """
        with open(file_path, encoding=self.encoding) as f:
            text = f.read()
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        creation_date = datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d')
        last_modified_date = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d')
        metadata = {
            'page_label': '1',
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': 'text/plain',
            'file_size': file_size,
            'creation_date': creation_date,
            'last_modified_date': last_modified_date
        }
        document_id = str(uuid.uuid4())
        return Document(document_id, metadata, text)

    def load_files(self, 
                   recursive: bool = False, 
                   ext: Optional[str] = None,
                   exc: Optional[str] = None,
                   filenames: Optional[List[str]] = None) -> List[Document]:
        """Loads files from the directory with optional filters, returning a list of Document objects.

        Args:
            recursive (bool, optional): Whether to recursively search subdirectories. Defaults to False.
            ext (Optional[str], optional): Only load files with this extension (e.g., "*.pdf"). Defaults to None.
            exc (Optional[str], optional): Exclude files with this extension (e.g., "*.txt"). Defaults to None.
            filenames (Optional[List[str]], optional): Load only these specific filenames. Defaults to None.

        Returns:
            List[Document]: A list of Document objects, one for each loaded file.
        """
        directory = Path(self.directory_path)
        documents: List[Document] = []

        for file_path in directory.rglob("*") if recursive else directory.glob("*"):
            if file_path.is_file():
                if filenames is not None and file_path.name not in filenames:
                    continue  # Skip if specific filenames are provided and current file is not in the list

                if ext is not None and not file_path.match(ext):
                    continue  # Skip if a specific extension is provided and current file doesn't match

                if exc is not None and file_path.match(exc):
                    continue  # Skip if a specific extension to exclude is provided and current file matches

                try:
                    if file_path.suffix.lower() == '.pdf':
                        documents.extend(self.read_pdf(file_path))
                    else:
                        document = self.read_file(file_path)
                        documents.append(document)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

        return documents