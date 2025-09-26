import os
import uuid
import fitz
import logging
import datetime
import markdown
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from .exceptions import DocumentLoadError, UnsupportedFileFormatError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a single document with its content and metadata.
    
    Attributes:
        document_id: Unique identifier for the document
        metadata: Dictionary containing document metadata (file info, page info, etc.)
        text: The actual text content of the document
    """
    document_id: str
    metadata: Dict[str, str]
    text: str

    def __repr__(self) -> str:
        return f"Document(document_id='{self.document_id}', metadata={self.metadata}, text='{self.text[:20]}...')"


def _process_file(
    file_path: Path,
    encoding: str,
    ext: Optional[List[str]],
    exc: Optional[List[str]],
    filenames: Optional[List[str]],
    preprocess_fn: Optional[Callable[[str], str]] = None,
    code_mime_types: Optional[Dict[str, str]] = None
) -> List[Document]:
    """
    Helper function to process a single file based on filtering criteria.
    
    Args:
        file_path: Path to the file to process
        encoding: Text encoding to use when reading files
        ext: List of file extensions to include (glob patterns)
        exc: List of file extensions to exclude (glob patterns)  
        filenames: List of specific filenames to include
        preprocess_fn: Optional preprocessing function to apply to text
        code_mime_types: Dictionary mapping MIME types to file extensions
        
    Returns:
        List of Document objects created from the file
        
    Raises:
        DocumentLoadError: If file processing fails
    """
    try:
        from zrag.doc_loader import DocumentLoader  # Import inside the function

        # Check if the file should be processed based on filename
        if filenames is not None and file_path.name not in filenames:
            return []

        if ext is not None:
            if not any(file_path.match(pattern) for pattern in ext):
                return []

        if exc is not None:
            if any(file_path.match(pattern) for pattern in exc):
                return []

        mime_type, _ = mimetypes.guess_type(str(file_path))

        # Check for code MIME types using the passed dictionary
        if code_mime_types is not None and mime_type in code_mime_types:
            return DocumentLoader._read_code(file_path, encoding, preprocess_fn)

        return DocumentLoader._read_file(file_path, encoding, preprocess_fn)
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise DocumentLoadError(f"Failed to process file {file_path}", details=str(e))


class DocumentLoader:
    """
    Loads and reads files from a directory, returning documents with
    metadata and text content.
    
    Supports multiple file formats including PDF, text, markdown, and code files.
    Provides concurrent processing capabilities for improved performance.
    
    Attributes:
        directory_path: Path to the directory containing documents to load
        encoding: Text encoding to use when reading files (default: utf-8)
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
        """
        Initialize the DocumentLoader.
        
        Args:
            directory_path: Path to the directory containing documents
            encoding: Text encoding to use when reading files
            
        Raises:
            DocumentLoadError: If directory path is invalid
        """
        if not os.path.exists(directory_path):
            raise DocumentLoadError(f"Directory does not exist: {directory_path}")
            
        if not os.path.isdir(directory_path):
            raise DocumentLoadError(f"Path is not a directory: {directory_path}")
            
        self.directory_path = directory_path
        self.encoding = encoding
        self._initialize_mimetypes()
        
        logger.info(f"DocumentLoader initialized for directory: {directory_path}")

    def _initialize_mimetypes(self) -> None:
        """Ensures code MIME types are recognized."""
        for mime, ext in self.CODE_MIME_TYPES.items():
            mimetypes.add_type(mime, ext)

    @staticmethod
    def _read_file(
        file_path: Path,
        encoding: str,
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Reads a file based on its MIME type.
        
        Args:
            file_path: Path to the file to read
            encoding: Text encoding to use
            preprocess_fn: Optional preprocessing function
            
        Returns:
            List of Document objects
            
        Raises:
            UnsupportedFileFormatError: If file format is not supported
            DocumentLoadError: If file reading fails
        """
        if not file_path.exists():
            raise DocumentLoadError(f"File does not exist: {file_path}")
            
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
        
        try:
            return read_method(file_path, encoding, preprocess_fn)
        except Exception as e:
            raise DocumentLoadError(f"Failed to read file {file_path}", details=str(e))

    @staticmethod
    def _read_pdf(
        file_path: Path,
        encoding: str,
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Reads a PDF file and extracts text from all pages.
        
        Args:
            file_path: Path to the PDF file
            encoding: Text encoding (not used for PDF)
            preprocess_fn: Optional preprocessing function
            
        Returns:
            List of Document objects, one per page
            
        Raises:
            DocumentLoadError: If PDF reading fails
        """
        documents = []
        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            logger.error(f"Failed to open PDF file {file_path}: {e}")
            raise DocumentLoadError(f"Failed to open PDF file {file_path}", details=str(e))

        try:
            file_stats = file_path.stat()
            metadata_common = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_type': 'application/pdf',
                'file_size': str(file_stats.st_size),
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
        finally:
            doc.close()
            
        return documents

    @staticmethod
    def _read_markdown(
        file_path: Path,
        encoding: str,
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Reads a Markdown file and converts it to plain text.
        
        Args:
            file_path: Path to the markdown file
            encoding: Text encoding to use
            preprocess_fn: Optional preprocessing function
            
        Returns:
            List containing a single Document object
            
        Raises:
            DocumentLoadError: If markdown reading fails
        """
        try:
            with open(file_path, encoding=encoding) as f:
                md_content = f.read()
            # Convert Markdown to plain text
            text = markdown.markdown(md_content)
            if preprocess_fn:
                text = preprocess_fn(text)
        except Exception as e:
            logger.error(f"Failed to read or preprocess Markdown file {file_path}: {e}")
            raise DocumentLoadError(f"Failed to read markdown file {file_path}", details=str(e))

        return [DocumentLoader._create_single_document(file_path, text, 'text/markdown')]

    @staticmethod
    def _read_text(
        file_path: Path,
        encoding: str,
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Reads a plain text file.
        
        Args:
            file_path: Path to the text file
            encoding: Text encoding to use
            preprocess_fn: Optional preprocessing function
            
        Returns:
            List containing a single Document object
            
        Raises:
            DocumentLoadError: If text reading fails
        """
        try:
            with open(file_path, encoding=encoding) as f:
                text = f.read()
            if preprocess_fn:
                text = preprocess_fn(text)
        except Exception as e:
            logger.error(f"Failed to read or preprocess text file {file_path}: {e}")
            raise DocumentLoadError(f"Failed to read text file {file_path}", details=str(e))

        return [DocumentLoader._create_single_document(file_path, text, 'text/plain')]


    @staticmethod
    def _read_code(
        file_path: Path,
        encoding: str,
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Reads a code file.
        
        Args:
            file_path: Path to the code file
            encoding: Text encoding to use
            preprocess_fn: Optional preprocessing function
            
        Returns:
            List containing a single Document object
            
        Raises:
            DocumentLoadError: If code reading fails
        """
        try:
            with open(file_path, encoding=encoding) as f:
                code = f.read()
            if preprocess_fn:
                code = preprocess_fn(code)
        except Exception as e:
            logger.error(f"Failed to read code file {file_path}: {e}")
            raise DocumentLoadError(f"Failed to read code file {file_path}", details=str(e))

        mime_type = mimetypes.guess_type(str(file_path))[0] or 'text/plain'
        return [DocumentLoader._create_single_document(file_path, code, mime_type)]

    @staticmethod
    def _create_single_document(file_path: Path, text: str, mime_type: str) -> Document:
        """
        Creates a Document object with metadata.
        
        Args:
            file_path: Path to the file
            text: Text content
            mime_type: MIME type of the file
            
        Returns:
            Document object with populated metadata
            
        Raises:
            DocumentLoadError: If metadata creation fails
        """
        try:
            file_stats = file_path.stat()
            metadata = {
                'page_label': '1',
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_type': mime_type,
                'file_size': str(file_stats.st_size),
                'creation_date': datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d'),
                'last_modified_date': datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d')
            }
            document_id = str(uuid.uuid4())
            return Document(document_id, metadata, text)
        except Exception as e:
            raise DocumentLoadError(f"Failed to create document metadata for {file_path}", details=str(e))

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
        max_workers: Optional[int] = None,
        preprocess_fn: Optional[Callable[[str], str]] = None
    ) -> List[Document]:
        """
        Loads documents from the directory, applying optional filtering and preprocessing.
        
        Args:
            recursive: Whether to search subdirectories recursively
            ext: List of file extensions to include (glob patterns)
            exc: List of file extensions to exclude (glob patterns)
            filenames: List of specific filenames to include
            max_workers: Maximum number of worker processes (defaults to CPU count)
            preprocess_fn: Optional preprocessing function to apply to text content
            
        Returns:
            List of Document objects loaded from the directory
            
        Raises:
            DocumentLoadError: If directory loading fails
        """
        if max_workers is None:
            max_workers = os.cpu_count() or 1
            
        if max_workers <= 0:
            raise DocumentLoadError("max_workers must be positive")
            
        logger.info(
            f"Loading documents from {self.directory_path}, "
            f"recursive={recursive}, max_workers={max_workers}"
        )
        
        try:
            directory = Path(self.directory_path)
            documents: List[Document] = []

            file_generator = directory.rglob("*") if recursive else directory.glob("*")
            file_paths = [fp for fp in file_generator if fp.is_file()]
            
            if not file_paths:
                logger.warning(f"No files found in directory: {self.directory_path}")
                return documents

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_file,  # Call the external function
                        file_path,
                        self.encoding,
                        ext,
                        exc,
                        filenames,
                        preprocess_fn,
                        self.CODE_MIME_TYPES  # Pass CODE_MIME_TYPES
                    ): file_path for file_path in file_paths
                }

                processed_count = 0
                error_count = 0
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        documents.extend(result)
                        processed_count += len(result)
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing file {futures[future]}: {e}")

            logger.info(
                f"Document loading completed. Processed: {processed_count} documents, "
                f"Errors: {error_count}"
            )
            return documents
            
        except Exception as e:
            raise DocumentLoadError(f"Failed to load documents from {self.directory_path}", details=str(e))