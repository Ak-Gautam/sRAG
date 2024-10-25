import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from zrag.doc_loader import DocumentLoader, Document

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

def test_document_initialization():
    """Test basic Document class initialization."""
    doc = Document("test-id", {"file_name": "test.txt"}, "Sample text")
    assert doc.document_id == "test-id"
    assert doc.metadata["file_name"] == "test.txt"
    assert doc.text == "Sample text"

def test_document_loader_initialization():
    """Test DocumentLoader initialization."""
    loader = DocumentLoader("/tmp")
    assert loader.directory_path == "/tmp"
    assert loader.encoding == "utf-8"

def test_load_basic_text_file(temp_dir):
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

def test_load_with_file_filtering(temp_dir):
    """Test basic file filtering."""
    # Create two different file types
    txt_file = temp_dir / "test.txt"
    txt_file.write_text("text content", encoding="utf-8")
    
    md_file = temp_dir / "test.md"
    md_file.write_text("# markdown content", encoding="utf-8")
    
    loader = DocumentLoader(str(temp_dir))
    
    # Test extension filtering
    txt_docs = loader.load(ext=["*.txt"])
    assert len(txt_docs) == 1
    assert txt_docs[0].metadata["file_name"] == "test.txt"

def test_basic_preprocessing(temp_dir):
    """Test basic preprocessing functionality."""
    file_path = temp_dir / "test.txt"
    content = "hello world"
    file_path.write_text(content, encoding="utf-8")
    
    def preprocess_fn(text):
        return text.upper()
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load(preprocess_fn=preprocess_fn)
    
    assert len(docs) == 1
    assert docs[0].text == "HELLO WORLD"

@patch('fitz.open')
def test_basic_pdf_handling(mock_fitz_open, temp_dir):
    """Test basic PDF handling with mocks."""
    # Simple mock setup
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "PDF content"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc
    
    file_path = temp_dir / "test.pdf"
    file_path.touch()
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].text == "PDF content"
    assert docs[0].metadata["file_type"] == "application/pdf"

def test_load_code_file(temp_dir):
    """Test loading a code file."""
    file_path = temp_dir / "test.py"
    content = "def test():\n    return True"
    file_path.write_text(content, encoding="utf-8")
    
    loader = DocumentLoader(str(temp_dir))
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].text == content
    assert docs[0].metadata["file_type"] == "text/x-python"

def test_basic_error_handling(temp_dir):
    """Test basic error handling with an invalid directory."""
    invalid_dir = temp_dir / "nonexistent"
    loader = DocumentLoader(str(invalid_dir))
    docs = loader.load()  # Should not raise exception
    assert len(docs) == 0