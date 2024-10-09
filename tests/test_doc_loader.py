import unittest
import tempfile
import os
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock
from srag.document_loader import DocumentLoader, Document

class TestDocumentLoader(unittest.TestCase):
    """Comprehensive tests for the DocumentLoader class."""

    def setUp(self):
        """Creates a temporary directory and test files for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_files()
        self.loader = DocumentLoader(self.test_dir)

    def tearDown(self):
        """Removes the temporary directory after the test."""
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Creates various test files (text, Markdown, Python) in the temporary directory."""
        with open(os.path.join(self.test_dir, 'test.txt'), 'w') as f:
            f.write('This is a test text file.')
        
        # Create a markdown file
        with open(os.path.join(self.test_dir, 'test.md'), 'w') as f:
            f.write('# This is a test markdown file')
        
        # Create a Python file
        with open(os.path.join(self.test_dir, 'test.py'), 'w') as f:
            f.write('print("This is a test Python file")')

    def test_load_text_file(self):
        """Tests loading a simple text file."""
        documents = self.loader.load(ext=['.txt'])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, 'This is a test text file.')
        self.assertEqual(documents[0].metadata['file_type'], 'text/plain')

    # ... (Other test methods similar to your example)

    def test_empty_file(self):
        """Tests loading an empty text file."""
        with open(os.path.join(self.test_dir, "empty.txt"), "w") as f:
            pass  # Create an empty file

        documents = self.loader.load(ext=[".txt"])
        # Check that the empty file is handled correctly (e.g., logged and skipped)

    def test_preprocess_function_exception(self):
        """Tests that exceptions in the preprocess function are handled gracefully."""
        def failing_preprocess(text):
            raise ValueError("Preprocessing failed!")

        documents = self.loader.load(ext=['.txt'], preprocess_fn=failing_preprocess)
        # Check that the file is skipped and an error is logged

    def test_load_markdown_file(self):
        documents = self.loader.load(ext=['.md'])
        self.assertEqual(len(documents), 1)
        self.assertIn('This is a test markdown file', documents[0].text)
        self.assertEqual(documents[0].metadata['file_type'], 'text/markdown')

    def test_load_python_file(self):
        documents = self.loader.load(ext=['.py'])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, 'print("This is a test Python file")')
        self.assertEqual(documents[0].metadata['file_type'], 'text/x-python')

    def test_load_all_files(self):
        documents = self.loader.load()
        self.assertEqual(len(documents), 3)

    def test_load_with_exclusion(self):
        documents = self.loader.load(exc=['.md'])
        self.assertEqual(len(documents), 2)

    def test_load_specific_filenames(self):
        documents = self.loader.load(filenames=['test.txt'])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata['file_name'], 'test.txt')

    def test_metadata(self):
        documents = self.loader.load(ext=['.txt'])
        metadata = documents[0].metadata
        self.assertIn('file_name', metadata)
        self.assertIn('file_path', metadata)
        self.assertIn('file_type', metadata)
        self.assertIn('file_size', metadata)
        self.assertIn('creation_date', metadata)
        self.assertIn('last_modified_date', metadata)

    @patch('srag.document_loader.fitz.open')
    def test_load_pdf(self, mock_fitz_open):
        # Mock PDF document
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is a test PDF page."
        mock_pdf.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_pdf

        # Create a mock PDF file
        pdf_path = os.path.join(self.test_dir, 'test.pdf')
        Path(pdf_path).touch()

        documents = self.loader.load(ext=['.pdf'])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, "This is a test PDF page.")
        self.assertEqual(documents[0].metadata['file_type'], 'application/pdf')

    def test_preprocess_function(self):
        def preprocess(text):
            return text.upper()

        documents = self.loader.load(ext=['.txt'], preprocess_fn=preprocess)
        self.assertEqual(documents[0].text, 'THIS IS A TEST TEXT FILE.')

    @patch('srag.document_loader.ProcessPoolExecutor')
    def test_concurrent_loading(self, mock_executor):
        mock_executor.return_value.__enter__.return_value.submit.side_effect = lambda f, *args, **kwargs: MagicMock(result=lambda: f(*args, **kwargs))
        
        documents = self.loader.load()
        self.assertEqual(len(documents), 3)
        mock_executor.assert_called_once()

    def test_recursive_loading(self):
        # Create a subdirectory with a file
        subdir = os.path.join(self.test_dir, 'subdir')
        os.mkdir(subdir)
        with open(os.path.join(subdir, 'subfile.txt'), 'w') as f:
            f.write('This is a file in a subdirectory.')

        documents = self.loader.load(recursive=True)
        self.assertEqual(len(documents), 4)  # 3 original files + 1 in subdirectory

    def test_error_handling(self):
        # Create a file with no read permissions
        no_access_file = os.path.join(self.test_dir, 'no_access.txt')
        with open(no_access_file, 'w') as f:
            f.write('You should not be able to read this.')
        os.chmod(no_access_file, 0o000)

        # This should not raise an exception, but log an error
        documents = self.loader.load()
        self.assertEqual(len(documents), 3)  # Still loads the other 3 files

        # Restore permissions for cleanup
        os.chmod(no_access_file, 0o666)

if __name__ == "__main__":
    unittest.main()