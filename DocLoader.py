# DocLoader.py
# This is a file loader for pdf or md files
# It can be used to load files from a directory and return a list of documents
# Each document is a dictionary with the following keys:
# - id: a unique identifier for the document Needs fixing.
# - metadata: a dictionary containing metadata about the document
# - text: the content of the document

# DocLoader.py
import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional
import uuid
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF for PDF reading
import markdown  # For Markdown parsing

class Document:
    """Represents a single document with its content and metadata."""

    def __init__(self, id: str, metadata: Dict, text: str):
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
        self.directory_path = directory_path
        self.encoding = encoding

    @staticmethod
    def read_file(file_path: Path, encoding: str) -> List[Document]:
        """Reads the content of a file based on its MIME type."""
        mime_type, _ = mimetypes.guess_type(str(file_path))

        read_methods = {
            "application/pdf": FileLoader.read_pdf,
            "text/markdown": FileLoader.read_markdown,
            "text/plain": FileLoader.read_text
        }

        read_method = read_methods.get(mime_type, FileLoader.read_text)
        return read_method(file_path, encoding)

    @staticmethod
    def read_pdf(file_path: Path, encoding: str) -> List[Document]:
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

    @staticmethod
    def read_markdown(file_path: Path, encoding: str) -> List[Document]:
        with open(file_path, encoding=encoding) as f:
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
        return [Document(document_id, metadata, text)]

    @staticmethod
    def read_text(file_path: Path, encoding: str) -> List[Document]:
        with open(file_path, encoding=encoding) as f:
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
        return [Document(document_id, metadata, text)]

    @staticmethod
    def process_file(file_path: Path, encoding: str, ext: Optional[str], exc: Optional[str], filenames: Optional[List[str]]) -> List[Document]:
        if filenames is not None and file_path.name not in filenames:
            return []
        if ext is not None and not file_path.match(ext):
            return []
        if exc is not None and file_path.match(exc):
            return []
        try:
            return FileLoader.read_file(file_path, encoding)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    def load_files(self,
                   recursive: bool = False,
                   ext: Optional[str] = None,
                   exc: Optional[str] = None,
                   filenames: Optional[List[str]] = None,
                   max_workers: int = os.cpu_count()) -> List[Document]:
        """Loads files from the directory with optional filters, returning a list of Document objects."""
        directory = Path(self.directory_path)
        documents: List[Document] = []

        file_paths = list(directory.rglob("*") if recursive else directory.glob("*"))
        file_paths = [fp for fp in file_paths if fp.is_file()]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, file_path, self.encoding, ext, exc, filenames): file_path for file_path in file_paths}
            for future in as_completed(future_to_file):
                documents.extend(future.result())

        return documents