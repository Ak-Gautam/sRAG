import unittest
from pathlib import Path
from srag.document_loader import DocumentLoader, Document

class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent / "test_files"
        self.test_dir.mkdir(exist_ok=True)
        # Create dummy files here (PDF, txt, md, .py, etc.)
        (self.test_dir / "test.txt").write_text("This is a test text file.\nSecond Line")
        (self.test_dir / "test.md").write_text("# Test Markdown\nThis is a test markdown file.")
        (self.test_dir / "test.py").write_text("print('Hello from Python')")


    def test_load_text(self):
        loader = DocumentLoader(str(self.test_dir))
        docs = loader.load(ext=['*.txt'])
        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)
        self.assertEqual(docs[0].text, "This is a test text file.\nSecond Line")
        self.assertEqual(docs[0].metadata['file_type'], 'text/plain')


    def test_load_markdown(self):
        loader = DocumentLoader(str(self.test_dir))
        docs = loader.load(ext=['*.md'])
        self.assertEqual(len(docs), 1)
        self.assertIn("<h1>Test Markdown</h1>", docs[0].text) # Converted markdown


    def test_load_code(self):
        loader = DocumentLoader(str(self.test_dir))
        docs = loader.load(ext=['*.py'])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "print('Hello from Python')")
        self.assertEqual(docs[0].metadata['file_type'], 'text/x-python')



    def test_load_recursive(self):
        # Create a subdirectory and a file inside.
        (self.test_dir / "subdir").mkdir(exist_ok=True)
        (self.test_dir / "subdir" / "test_recursive.txt").write_text("Recursive Test.")

        loader = DocumentLoader(str(self.test_dir))
        docs = loader.load(recursive=True, ext=['*.txt'])
        self.assertEqual(len(docs), 2)  # Expect 2 text files.


    def test_load_filenames(self):
        loader = DocumentLoader(str(self.test_dir))
        docs = loader.load(filenames = ['test.txt'])
        self.assertEqual(len(docs), 1)


    def test_load_exc(self):
        loader = DocumentLoader(str(self.test_dir))
        docs = loader.load(exc=['*.md'])
        self.assertEqual(len(docs), 2)  # Two files are not markdown.


    def test_empty_dir(self):
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir(exist_ok=True)
        loader = DocumentLoader(str(empty_dir))
        docs = loader.load()
        self.assertEqual(len(docs), 0)


    def test_invalid_file(self):
        # Create a dummy binary file.
        (self.test_dir / "invalid.bin").write_bytes(b'\x00\x01\x02')
        loader = DocumentLoader(str(self.test_dir))
        docs = loader.load()
        self.assertEqual(len([d for d in docs if "invalid.bin" not in d.metadata['file_name']]), 3)
        # binary support?

    def tearDown(self):
        # Clean up dummy files
        import shutil
        shutil.rmtree(self.test_dir)