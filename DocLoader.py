# DocLoader.py
import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Callable
import uuid
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz
import markdown
import logging

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    # Define supported code MIME types
    CODE_MIME_TYPES = {
        'text/x-python': '.py',
        'text/javascript': '.js',
        'text/x-java-source': '.java',
        'text/x-c++src': '.cpp',
        'text/x-csrc': '.c',
        'text/x-ruby': '.rb',
        'text/x-go': '.go',
        'text/x-shellscript': '.sh',
        'application/typescript': '.ts',
        # Add more as needed
    }

    def __init__(self, directory_path: str, encoding: str = "utf-8"):
        self.directory_path = directory_path
        self.encoding = encoding
        # Extend mimetypes with additional code types
        self._initialize_mimetypes()

    def _initialize_mimetypes(self):
        # Ensure code MIME types are recognized
        for mime, ext in self.CODE_MIME_TYPES.items():
            mimetypes.add_type(mime, ext)

    @staticmethod
    def read_file(file_path: Path, encoding: str, preprocess_fn: Optional[Callable[[str], str]] = None) -> List[Document]:
        mime_type, _ = mimetypes.guess_type(str(file_path))

        read_methods = {
            "application/pdf": FileLoader.read_pdf,
            "text/markdown": FileLoader.read_markdown,
            "text/plain": FileLoader.read_text,
            # code MIME types
            "text/x-python": FileLoader.read_code,
            "text/javascript": FileLoader.read_code,
            "application/typescript": FileLoader.read_code,
            "text/x-java-source": FileLoader.read_code,
            "text/x-c++src": FileLoader.read_code,
            "text/x-csrc": FileLoader.read_code,
            "text/x-ruby": FileLoader.read_code,
            "text/x-go": FileLoader.read_code,
            "text/x-shellscript": FileLoader.read_code,

        }

        read_method = read_methods.get(mime_type, FileLoader.read_text)
        return read_method(file_path, encoding, preprocess_fn)

    @staticmethod
    def read_pdf(file_path: Path, encoding: str, preprocess_fn: Optional[Callable[[str], str]] = None) -> List[Document]:
        documents = []
        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            logger.error(f"Failed to open PDF file {file_path}: {e}")
            return documents

        file_stats = file_path.stat()
        metadata_common = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': 'application/pdf',
            'file_size': file_stats.st_size,
            'creation_date': datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d'),
            'last_modified_date': datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d')
        }

        for page_num, page in enumerate(doc):
            try:
                text = page.get_text()
                if preprocess_fn:
                    text = preprocess_fn(text)
            except Exception as e:
                logger.warning(f"Failed to extract or preprocess text from page {page_num+1} in {file_path}: {e}")
                text = ""

            metadata = metadata_common.copy()
            metadata.update({
                'page_label': str(page_num + 1)
            })

            document_id = str(uuid.uuid4())
            documents.append(Document(document_id, metadata, text))

        doc.close()
        return documents

    @staticmethod
    def read_markdown(file_path: Path, encoding: str, preprocess_fn: Optional[Callable[[str], str]] = None) -> List[Document]:
        try:
            with open(file_path, encoding=encoding) as f:
                md_content = f.read()
            # Convert Markdown to plain text for consistency
            text = markdown.markdown(md_content)
            if preprocess_fn:
                text = preprocess_fn(text)
        except Exception as e:
            logger.error(f"Failed to read or preprocess Markdown file {file_path}: {e}")
            text = ""

        return [FileLoader._create_single_document(file_path, text, 'text/markdown')]

    @staticmethod
    def read_text(file_path: Path, encoding: str) -> List[Document]:
        """Reads a plain text file and returns a list containing a single Document object."""
        try:
            with open(file_path, encoding=encoding) as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            text = ""

        return [FileLoader._create_single_document(file_path, text, 'text/plain')]

    @staticmethod
    def process_file(file_path: Path, encoding: str, ext: Optional[str], exc: Optional[str], filenames: Optional[List[str]]) -> List[Document]:
        """Processes a single file, applying filters and reading its content."""
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
        """
        Loads files from the directory with optional filters, returning a list of Document objects.
        
        Args:
            recursive: If True, search for files recursively in subdirectories.
            ext: File extension filter (e.g., "*.txt").
            exc: Exclusion pattern for files to ignore.
            filenames: List of specific filenames to include.
            max_workers: Maximum number of worker processes for parallel execution.
        """
        directory = Path(self.directory_path)
        documents: List[Document] = []

        file_paths = list(directory.rglob("*") if recursive else directory.glob("*"))
        file_paths = [fp for fp in file_paths if fp.is_file()]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, file_path, self.encoding, ext, exc, filenames): file_path for file_path in file_paths}
            for future in as_completed(future_to_file):
                documents.extend(future.result())

        return documents
    

# Requirements outline:
from DocLoader import FileLoader
import os

def simple_preprocess(text: str) -> str:
    # This is a simple preprocessing function that removes extra whitespace
    return ' '.join(text.split())

# Initialize the FileLoader
loader = FileLoader(directory_path="/path/to/your/documents")

# Example 1: Load all files recursively with default settings
documents = loader.load_files(recursive=True)

# Example 2: Load only PDF and Markdown files
documents = loader.load_files(recursive=True, ext="*.{pdf,md}")

# Example 3: Load files excluding certain patterns
documents = loader.load_files(recursive=True, exc="*{temp,draft}*")

# Example 4: Load specific files by name
documents = loader.load_files(filenames=["document1.pdf", "document2.md"])

# Example 5: Load files with a preprocessing function
documents = loader.load_files(recursive=True, preprocess_fn=simple_preprocess)

# Example 6: Load files with custom maximum workers
documents = loader.load_files(recursive=True, max_workers=4)

# Example 7: Combine multiple options
documents = loader.load_files(
    recursive=True,
    ext="*.{pdf,md}",
    exc="*draft*",
    preprocess_fn=simple_preprocess,
    max_workers=os.cpu_count() // 2  # Use half of available CPU cores
)

# Process the loaded documents
for doc in documents:
    print(f"Document ID: {doc.id}")
    print(f"File name: {doc.metadata['file_name']}")
    print(f"File type: {doc.metadata['file_type']}")
    print(f"Text preview: {doc.text[:100]}...")  # Print first 100 characters
    print("---")