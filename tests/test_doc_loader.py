import os
import uuid
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from zrag.doc_loader import DocumentLoader, Document

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_files(temp_dir):
    """Create sample files of different types for testing."""
    # Create a text file
    text_file = temp_dir / "sample.txt"
    text_file.write_text("Hello, World!", encoding="utf-8")
    
    # Create a markdown file
    md_file = temp_dir / "sample.md"
    md_file.write_text("# Header\nSome *markdown* content", encoding="utf-8")
    
    # Create a Python file
    py_file = temp_dir / "sample.py"
    py_file.write_text("def hello():\n    print('Hello')", encoding="utf-8")
    
    # Create a directory with nested files for recursive testing
    nested_dir = temp_dir / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "nested.txt"
    nested_file.write_text("Nested content", encoding="utf-8")
    
    return temp_dir

def test_document_initialization():
    """Test Document class initialization."""
    doc_id = str(uuid.uuid4())
    metadata = {"file_name": "test.txt", "page_label": "1"}
    text = "Sample text"
    
    doc = Document(doc_id, metadata, text)
    
    assert doc.document_id == doc_id
    assert doc.metadata == metadata
    assert doc.text == text

def test_document_loader_initialization():
    """Test DocumentLoader initialization."""
    loader = DocumentLoader("/tmp")
    assert loader.directory_path == "/tmp"
    assert loader.encoding == "utf-8"
    
    loader_with_encoding = DocumentLoader("/tmp", encoding="latin-1")
    assert loader_with_encoding.encoding == "latin-1"

def test_load_text_file(temp_dir):
    """Test loading a simple text file."""
    file_path = temp_dir / "test.txt"
    content = "Hello, World!"
    file_path.write_text(content, encoding="utf-8")
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].text == content
    assert docs[0].metadata["file_name"] == "test.txt"
    assert docs[0].metadata["file_type"] == "text/plain"

def test_load_markdown_file(temp_dir):
    """Test loading a markdown file."""
    file_path = temp_dir / "test.md"
    content = "# Header\nSome *markdown* content"
    file_path.write_text(content, encoding="utf-8")
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load()
    
    assert len(docs) == 1
    assert "Header" in docs[0].text
    assert docs[0].metadata["file_type"] == "text/markdown"

def test_load_python_file(temp_dir):
    """Test loading a Python source file."""
    file_path = temp_dir / "test.py"
    content = "def test():\n    return True"
    file_path.write_text(content, encoding="utf-8")
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].text == content
    assert docs[0].metadata["file_type"] == "text/x-python"

@patch('fitz.open')
def test_load_pdf_file(mock_fitz_open, temp_dir):
    """Test loading a PDF file."""
    # Create a mock PDF document
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "PDF content"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc
    
    # Create a dummy PDF file
    file_path = temp_dir / "test.pdf"
    file_path.touch()
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].text == "PDF content"
    assert docs[0].metadata["file_type"] == "application/pdf"
    mock_fitz_open.assert_called_once_with(str(file_path))

def test_recursive_loading(sample_files):
    """Test recursive loading of files."""
    loader = DocumentLoader(str(sample_files))
    
    # Test without recursive flag
    non_recursive_docs = loader.load(recursive=False)
    assert len(non_recursive_docs) == 3  # Only top-level files
    
    # Test with recursive flag
    recursive_docs = loader.load(recursive=True)
    assert len(recursive_docs) == 4  # Including nested file

def test_file_filtering(sample_files):
    """Test file filtering options."""
    loader = DocumentLoader(str(sample_files))
    
    # Test extension filtering
    py_docs = loader.load(ext=["*.py"])
    assert len(py_docs) == 1
    assert py_docs[0].metadata["file_name"] == "sample.py"
    
    # Test exclusion
    no_py_docs = loader.load(exc=["*.py"])
    assert all(doc.metadata["file_name"] != "sample.py" for doc in no_py_docs)
    
    # Test filename filtering
    specific_docs = loader.load(filenames=["sample.txt"])
    assert len(specific_docs) == 1
    assert specific_docs[0].metadata["file_name"] == "sample.txt"

def test_preprocessing(temp_dir):
    """Test preprocessing function application."""
    file_path = temp_dir / "test.txt"
    content = "hello world"
    file_path.write_text(content, encoding="utf-8")
    
    def preprocess_fn(text):
        return text.upper()
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load(preprocess_fn=preprocess_fn)
    
    assert len(docs) == 1
    assert docs[0].text == "HELLO WORLD"

def test_error_handling(temp_dir):
    """Test error handling for invalid files and operations."""
    # Create an unreadable file
    file_path = temp_dir / "unreadable.txt"
    file_path.write_text("content", encoding="utf-8")
    os.chmod(file_path, 0o000)  # Remove read permissions
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load()  # Should not raise exception but log error
    
    assert len(docs) == 0  # No documents should be loaded
    os.chmod(file_path, 0o644)  # Restore permissions for cleanup

def test_metadata_consistency():
    """Test metadata consistency across different file types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = Path(tmp_dir)
        
        # Create files of different types
        text_file = temp_dir / "test.txt"
        text_file.write_text("content", encoding="utf-8")
        
        md_file = temp_dir / "test.md"
        md_file.write_text("# content", encoding="utf-8")
        
        loader = DocumentLoader(str(temp_dir))
        docs = loader.load()
        
        required_metadata = {
            'file_name',
            'file_path',
            'file_type',
            'file_size',
            'creation_date',
            'last_modified_date',
            'page_label'
        }
        
        for doc in docs:
            assert all(key in doc.metadata for key in required_metadata)
            assert isinstance(doc.metadata['creation_date'], str)
            assert isinstance(doc.metadata['last_modified_date'], str)