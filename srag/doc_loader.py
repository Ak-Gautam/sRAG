import os
import uuid
import fitz
import logging
import datetime
import markdown
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Represents a single document with its content and metadata."""

    def __init__(self, document_id: str, metadata: Dict, text: str):
        self.document_id = document_id
        self.metadata = metadata
        self.text = text

    def __repr__(self):
        return f"Document(document_id='{self.document_id}', metadata={self.metadata}, text='{self.text[:20]}...')"


class DocumentLoader:
    """
    Loads and reads files from a directory, returning documents with 
    metadata and text content.
    """

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
        self._initialize_mimetypes()

    def _initialize_mimetypes(self):
        """Ensures code MIME types are recognized."""
        for mime, ext in self.CODE_MIME_TYPES.items():
            mimetypes.add_type(mime, ext)

    @staticmethod
    def _read_file(
        file_path: Path, 
        encoding: str, 
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """Reads a file based on its MIME type."""
        mime_type, _ = mimetypes.guess_type(str(file_path))

        read_methods = {
            "application/pdf": DocumentLoader._read_pdf,
            "text/markdown": DocumentLoader._read_markdown,
            "text/plain": DocumentLoader._read_text,
            # code MIME types
            "text/x-python": DocumentLoader._read_code,
            "text/javascript": DocumentLoader._read_code,
            "application/typescript": DocumentLoader._read_code,
            "text/x-java-source": DocumentLoader._read_code,
            "text/x-c++src": DocumentLoader._read_code,
            "text/x-csrc": DocumentLoader._read_code,
            "text/x-ruby": DocumentLoader._read_code,
            "text/x-go": DocumentLoader._read_code,
            "text/x-shellscript": DocumentLoader._read_code,
        }

        read_method = read_methods.get(mime_type, DocumentLoader._read_text)
        return read_method(file_path, encoding, preprocess_fn)

    @staticmethod
    def _read_pdf(
        file_path: Path, 
        encoding: str, 
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """Reads a PDF file."""
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
    def _read_markdown(
        file_path: Path, 
        encoding: str, 
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """Reads a Markdown file."""
        try:
            with open(file_path, encoding=encoding) as f:
                md_content = f.read()
            # Convert Markdown to plain text 
            text = markdown.markdown(md_content)
            if preprocess_fn:
                text = preprocess_fn(text)
        except Exception as e:
            logger.error(f"Failed to read or preprocess Markdown file {file_path}: {e}")
            text = ""

        return [DocumentLoader._create_single_document(file_path, text, 'text/markdown')]

    @staticmethod
    def _read_text(file_path: Path, encoding: str, preprocess_fn: Optional[Callable[[str], str]] = None) -> List[Document]:
        """Reads a plain text file."""
        try:
            with open(file_path, encoding=encoding) as f:
                text = f.read()
            if preprocess_fn:
                text = preprocess_fn(text)
        except Exception as e:
            logger.error(f"Failed to read or preprocess text file {file_path}: {e}")
            text = ""

        return [DocumentLoader._create_single_document(file_path, text, 'text/plain')]


    @staticmethod
    def _read_code(
        file_path: Path, 
        encoding: str, 
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """Reads a code file."""
        try:
            with open(file_path, encoding=encoding) as f:
                code = f.read()
            if preprocess_fn:
                code = preprocess_fn(code)
        except Exception as e:
            logger.error(f"Failed to read code file {file_path}: {e}")
            code = ""

        mime_type = mimetypes.guess_type(str(file_path))[0] or 'text/plain'
        return [DocumentLoader._create_single_document(file_path, code, mime_type)]

    @staticmethod
    def _create_single_document(file_path: Path, text: str, mime_type: str) -> Document:
        """Creates a Document object with metadata."""
        file_stats = file_path.stat()
        metadata = {
            'page_label': '1',
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': mime_type,
            'file_size': file_stats.st_size,
            'creation_date': datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d'),
            'last_modified_date': datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d')
        }
        document_id = str(uuid.uuid4())
        return Document(document_id, metadata, text)

    @staticmethod
    def _process_file(
        file_path: Path, 
        encoding: str, 
        ext: Optional[List[str]], 
        exc: Optional[List[str]], 
        filenames: Optional[List[str]], 
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Helper function to process a single file based on filtering criteria.
        """
        # Check if the file should be processed based on filename
        if filenames is not None and file_path.name not in filenames:
            return []
        
        if ext is not None:
            if not any(file_path.match(pattern) for pattern in ext):
                return []
        
        if exc is not None:
            if any(file_path.match(pattern) for pattern in exc):
                return []
        
        try:
            return DocumentLoader._read_file(file_path, encoding, preprocess_fn)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []

    def load(
        self,
        recursive: bool = False,
        ext: Optional[List[str]] = None,
        exc: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        max_workers: int = os.cpu_count(),
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Loads documents from the directory, applying optional filtering and preprocessing.
        """
        directory = Path(self.directory_path)
        documents: List[Document] = []

        file_generator = directory.rglob("*") if recursive else directory.glob("*")
        file_paths = (fp for fp in file_generator if fp.is_file())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_file,
                    file_path,
                    self.encoding,
                    ext,
                    exc,
                    filenames,
                    preprocess_fn
                ): file_path for file_path in file_paths
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    documents.extend(result)
                except Exception as e:
                    logger.error(f"Error processing file {futures[future]}: {e}")

        return documents